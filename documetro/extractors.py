from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable
from xml.etree import ElementTree as ET
import csv
import io
import json
import mimetypes
import os
import re
import zipfile

from bs4 import BeautifulSoup
import xlrd

from .config import AppConfig
from .models import DocumentSection, ExtractedDocument
from .openrouter import OpenRouterClient, OpenRouterError
from .utils import digest_file, normalize_whitespace, run_command


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
XLSX_NS = {
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}
PPTX_NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


class ExtractionError(RuntimeError):
    pass


@dataclass(slots=True)
class ExtractedPayload:
    sections: list[DocumentSection]
    metadata: dict[str, str]


class DocumentExtractor:
    def __init__(self, config: AppConfig | None = None, provider: OpenRouterClient | None = None) -> None:
        self.config = config or AppConfig()
        self.provider = provider

    def extract(self, path: Path) -> ExtractedDocument:
        extension = path.suffix.lower()
        checksum = digest_file(path)
        payload = self._dispatch(path, extension)
        sections = [section for section in payload.sections if normalize_whitespace(section.text)]
        if not sections:
            raise ExtractionError(f"No readable text found in {path.name}")
        return ExtractedDocument(
            document_id=checksum[:16],
            name=path.name,
            extension=extension or ".bin",
            path=path,
            size_bytes=path.stat().st_size,
            checksum=checksum,
            sections=sections,
            metadata=payload.metadata,
        )

    def _dispatch(self, path: Path, extension: str) -> ExtractedPayload:
        if extension in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return ExtractedPayload(
                [DocumentSection("Media analysis", self._multimodal_to_text(path), "media:1")],
                {"engine": "openrouter-multimodal", "source_modality": "image"},
            )
        if extension in {".wav", ".mp3", ".aiff", ".aac", ".ogg", ".flac", ".m4a"}:
            return ExtractedPayload(
                [DocumentSection("Media analysis", self._multimodal_to_text(path), "media:1")],
                {"engine": "openrouter-multimodal", "source_modality": "audio"},
            )
        if extension in {".mp4", ".mpeg", ".mov", ".webm"}:
            return ExtractedPayload(
                [DocumentSection("Media analysis", self._multimodal_to_text(path), "media:1")],
                {"engine": "openrouter-multimodal", "source_modality": "video"},
            )
        if extension in {".txt", ".md", ".rst", ".log", ".ini", ".cfg", ".toml", ".py", ".js", ".ts"}:
            return ExtractedPayload([DocumentSection("Text", path.read_text("utf-8", errors="ignore"), "full")], {"source_modality": "text"})
        if extension in {".vtt", ".srt"}:
            return ExtractedPayload([DocumentSection("Transcript", self._subtitle_to_text(path), "full")], {"source_modality": "transcript"})
        if extension in {".json"}:
            return ExtractedPayload([DocumentSection("JSON", self._json_to_text(path), "full")], {"source_modality": "text"})
        if extension in {".csv", ".tsv"}:
            return ExtractedPayload([DocumentSection("Table", self._delimited_to_text(path, extension), "sheet:1")], {"source_modality": "tabular"})
        if extension in {".html", ".htm", ".xml"}:
            return ExtractedPayload([DocumentSection("Markup", self._markup_to_text(path), "full")], {"source_modality": "text"})
        if extension == ".pdf":
            return ExtractedPayload(self._pdf_sections(path), {"engine": "pdftotext", "source_modality": "pdf"})
        if extension == ".doc":
            return ExtractedPayload([DocumentSection("Document", self._legacy_doc_to_text(path), "body")], {"engine": "catdoc", "source_modality": "text"})
        if extension == ".docx":
            return ExtractedPayload(self._docx_sections(path), {"engine": "docx-xml", "source_modality": "text"})
        if extension == ".xlsx":
            return ExtractedPayload(self._xlsx_sections(path), {"engine": "xlsx-xml", "source_modality": "tabular"})
        if extension == ".xls":
            return ExtractedPayload(self._xls_sections(path), {"engine": "xlrd", "source_modality": "tabular"})
        if extension == ".pptx":
            return ExtractedPayload(self._pptx_sections(path), {"engine": "pptx-xml", "source_modality": "text"})
        if extension in {".odt", ".ods", ".odp"}:
            return ExtractedPayload([DocumentSection("Converted", self._libreoffice_to_text(path), "converted")], {"engine": "libreoffice", "source_modality": "text"})

        guessed, _ = mimetypes.guess_type(str(path))
        if guessed and guessed.startswith("text/"):
            return ExtractedPayload([DocumentSection("Text", path.read_text("utf-8", errors="ignore"), "full")], {"source_modality": "text"})
        return ExtractedPayload([DocumentSection("Converted", self._libreoffice_to_text(path), "converted")], {"engine": "libreoffice", "source_modality": "text"})

    def _multimodal_to_text(self, path: Path) -> str:
        if self.provider is None or not self.provider.enabled:
            raise ExtractionError(f"{path.name} requires OpenRouter multimodal processing. Set OPENROUTER_API_KEY.")
        try:
            return self.provider.summarize_file(path)
        except OpenRouterError as exc:
            raise ExtractionError(f"Unable to analyze media {path.name}: {exc}") from exc

    def _json_to_text(self, path: Path) -> str:
        content = json.loads(path.read_text("utf-8", errors="ignore"))
        return json.dumps(content, indent=2, ensure_ascii=True)

    def _subtitle_to_text(self, path: Path) -> str:
        lines = path.read_text("utf-8", errors="ignore").splitlines()
        cleaned: list[str] = []
        for raw in lines:
            line = raw.strip()
            if not line or line == "WEBVTT":
                continue
            if re.fullmatch(r"\d+", line):
                continue
            if "-->" in line:
                continue
            line = re.sub(r"<[^>]+>", " ", line)
            line = normalize_whitespace(line)
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)

    def _delimited_to_text(self, path: Path, extension: str) -> str:
        delimiter = "\t" if extension == ".tsv" else ","
        raw = path.read_text("utf-8", errors="ignore")
        sample = raw[:2048]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            pass
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            return ""
        header = rows[0]
        lines = ["Columns: " + " | ".join(cell.strip() or f"column_{index + 1}" for index, cell in enumerate(header))]
        for row_index, row in enumerate(rows[1:], start=2):
            pairs = []
            for index, value in enumerate(row):
                label = header[index].strip() if index < len(header) and header[index].strip() else f"column_{index + 1}"
                if value.strip():
                    pairs.append(f"{label}: {value.strip()}")
            if pairs:
                lines.append(f"Row {row_index}: " + " | ".join(pairs))
        return "\n".join(lines)

    def _markup_to_text(self, path: Path) -> str:
        raw = path.read_text("utf-8", errors="ignore")
        if path.suffix.lower() == ".xml":
            return normalize_whitespace(re.sub(r"<[^>]+>", " ", raw))
        soup = BeautifulSoup(raw, "lxml")
        return normalize_whitespace(soup.get_text("\n", strip=True))

    def _pdf_sections(self, path: Path) -> list[DocumentSection]:
        command = ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"]
        result = run_command(command, timeout=180)
        if result.returncode != 0:
            fallback = run_command(["mutool", "draw", "-F", "txt", "-o", "-", str(path)], timeout=180)
            if fallback.returncode != 0:
                stderr = (result.stderr or fallback.stderr).strip()
                raise ExtractionError(f"Unable to parse PDF {path.name}: {stderr}")
            content = fallback.stdout
        else:
            content = result.stdout
        pages = [normalize_whitespace(page) for page in content.split("\f")]
        sections = [
            DocumentSection(title=f"Page {index}", text=page, locator=f"page:{index}")
            for index, page in enumerate(pages, start=1)
            if page
        ]
        return sections

    def _legacy_doc_to_text(self, path: Path) -> str:
        result = run_command(["catdoc", str(path)], timeout=90)
        if result.returncode != 0:
            raise ExtractionError(f"Unable to parse DOC {path.name}: {(result.stderr or '').strip()}")
        return result.stdout

    def _docx_sections(self, path: Path) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            ordered_parts = ["word/document.xml"]
            ordered_parts.extend(sorted(name for name in names if re.fullmatch(r"word/header\d+\.xml", name)))
            ordered_parts.extend(sorted(name for name in names if re.fullmatch(r"word/footer\d+\.xml", name)))
            ordered_parts.extend(name for name in ["word/comments.xml", "word/footnotes.xml", "word/endnotes.xml"] if name in names)
            for part in ordered_parts:
                if part not in names:
                    continue
                root = ET.fromstring(archive.read(part))
                paragraphs = []
                for paragraph in root.findall(".//w:p", WORD_NS):
                    text = "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS))
                    text = normalize_whitespace(text)
                    if text:
                        paragraphs.append(text)
                joined = "\n\n".join(paragraphs)
                if joined:
                    label = self._word_part_label(part)
                    sections.append(DocumentSection(title=label, text=joined, locator=label.lower().replace(" ", ":")))
        if not sections:
            raise ExtractionError(f"No readable XML text in {path.name}")
        return sections

    def _xlsx_sections(self, path: Path) -> list[DocumentSection]:
        with zipfile.ZipFile(path) as archive:
            shared_strings = self._xlsx_shared_strings(archive)
            sheets = self._xlsx_sheet_map(archive)
            sections: list[DocumentSection] = []
            for name, target in sheets:
                xml_path = f"xl/{target}" if not target.startswith("xl/") else target
                root = ET.fromstring(archive.read(xml_path))
                rows = self._xlsx_rows(root, shared_strings)
                text = self._table_rows_to_text(name, rows)
                if text:
                    sections.append(DocumentSection(title=name, text=text, locator=f"sheet:{name.lower()}"))
        if not sections:
            raise ExtractionError(f"No readable worksheets found in {path.name}")
        return sections

    def _xls_sections(self, path: Path) -> list[DocumentSection]:
        workbook = xlrd.open_workbook(str(path), on_demand=True)
        sections: list[DocumentSection] = []
        for sheet in workbook.sheets():
            rows: list[list[str]] = []
            for row_index in range(sheet.nrows):
                row_values = []
                for column_index in range(sheet.ncols):
                    value = str(sheet.cell_value(row_index, column_index)).strip()
                    row_values.append(value)
                rows.append(row_values)
            text = self._table_rows_to_text(sheet.name, rows)
            if text:
                sections.append(DocumentSection(title=sheet.name, text=text, locator=f"sheet:{sheet.name.lower()}"))
        workbook.release_resources()
        if not sections:
            raise ExtractionError(f"No readable worksheets found in {path.name}")
        return sections

    def _pptx_sections(self, path: Path) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        with zipfile.ZipFile(path) as archive:
            slides = sorted(name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name))
            for index, slide in enumerate(slides, start=1):
                root = ET.fromstring(archive.read(slide))
                text_runs = [normalize_whitespace(node.text or "") for node in root.findall(".//a:t", PPTX_NS)]
                text = "\n".join(run for run in text_runs if run)
                if text:
                    sections.append(DocumentSection(title=f"Slide {index}", text=text, locator=f"slide:{index}"))
        if not sections:
            raise ExtractionError(f"No readable slides found in {path.name}")
        return sections

    def _libreoffice_to_text(self, path: Path) -> str:
        with TemporaryDirectory(prefix="documetro-convert-") as outdir:
            result = run_command(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    outdir,
                    str(path),
                ],
                timeout=180,
            )
            if result.returncode != 0:
                raise ExtractionError(f"LibreOffice could not convert {path.name}: {(result.stderr or '').strip()}")
            target = Path(outdir) / f"{path.stem}.txt"
            if not target.exists():
                raise ExtractionError(f"LibreOffice did not produce text output for {path.name}")
            return target.read_text("utf-8", errors="ignore")

    def _word_part_label(self, part: str) -> str:
        if part == "word/document.xml":
            return "Body"
        stem = Path(part).stem
        return stem.replace("_", " ").title()

    def _xlsx_shared_strings(self, archive: zipfile.ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in archive.namelist():
            return []
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        values: list[str] = []
        for item in root.findall(".//x:si", XLSX_NS):
            parts = [node.text or "" for node in item.findall(".//x:t", XLSX_NS)]
            values.append("".join(parts))
        return values

    def _xlsx_sheet_map(self, archive: zipfile.ZipFile) -> list[tuple[str, str]]:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.attrib.get("Id", ""): rel.attrib.get("Target", "")
            for rel in rels.findall(".//pkg:Relationship", XLSX_NS)
        }
        sheets: list[tuple[str, str]] = []
        for sheet in workbook.findall(".//x:sheets/x:sheet", XLSX_NS):
            rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
            name = sheet.attrib.get("name", "Sheet")
            target = targets.get(rel_id)
            if target:
                sheets.append((name, target))
        return sheets

    def _xlsx_rows(self, root: ET.Element, shared_strings: list[str]) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in root.findall(".//x:sheetData/x:row", XLSX_NS):
            row_cells: dict[int, str] = {}
            for cell in row.findall("x:c", XLSX_NS):
                ref = cell.attrib.get("r", "A1")
                column = self._column_index(ref)
                value = self._xlsx_cell_value(cell, shared_strings)
                if value:
                    row_cells[column] = value
            if row_cells:
                max_column = max(row_cells)
                rows.append([row_cells.get(index, "") for index in range(max_column + 1)])
        return rows

    def _xlsx_cell_value(self, cell: ET.Element, shared_strings: list[str]) -> str:
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            return normalize_whitespace("".join(node.text or "" for node in cell.findall(".//x:t", XLSX_NS)))
        value_node = cell.find("x:v", XLSX_NS)
        if value_node is None or value_node.text is None:
            return ""
        value = value_node.text.strip()
        if cell_type == "s":
            index = int(value)
            return shared_strings[index] if 0 <= index < len(shared_strings) else value
        return value

    def _column_index(self, reference: str) -> int:
        letters = "".join(character for character in reference if character.isalpha()).upper()
        index = 0
        for character in letters:
            index = index * 26 + (ord(character) - 64)
        return max(index - 1, 0)

    def _table_rows_to_text(self, sheet_name: str, rows: Iterable[list[str]]) -> str:
        prepared = [list(row) for row in rows if any(str(cell).strip() for cell in row)]
        if not prepared:
            return ""
        header_index = self._header_row_index(prepared)
        header = prepared[header_index] if header_index is not None else []
        lines = [f"Sheet: {sheet_name}"]
        if header:
            labels = [cell.strip() or f"column_{index + 1}" for index, cell in enumerate(header)]
            lines.append("Columns: " + " | ".join(labels))
        start = header_index + 1 if header_index is not None else 0
        for row_number, row in enumerate(prepared[start:], start=start + 1):
            pairs = []
            for index, value in enumerate(row):
                cleaned = str(value).strip()
                if not cleaned:
                    continue
                label = header[index].strip() if header and index < len(header) and header[index].strip() else f"column_{index + 1}"
                pairs.append(f"{label}: {cleaned}")
            if pairs:
                lines.append(f"Row {row_number}: " + " | ".join(pairs))
        return "\n".join(lines)

    def _header_row_index(self, rows: list[list[str]]) -> int | None:
        for index, row in enumerate(rows[:5]):
            populated = [cell.strip() for cell in row if str(cell).strip()]
            if len(populated) >= 2 and sum(cell.replace(".", "", 1).isdigit() for cell in populated) <= 1:
                return index
        return None
