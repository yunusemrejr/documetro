from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from documetro.config import AppConfig
from documetro.extractors import DocumentExtractor


class FakeProvider:
    enabled = True

    def summarize_file(self, path: Path) -> str:
        return f"Media summary for {path.name}: a dashboard screenshot with a revenue label."


class ExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.extractor = DocumentExtractor()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_docx_xml_extraction(self) -> None:
        target = self.root / "report.docx"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr(
                "word/document.xml",
                """
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body>
                    <w:p><w:r><w:t>Quarterly revenue improved by 12 percent.</w:t></w:r></w:p>
                    <w:p><w:r><w:t>Retention stayed above 95 percent.</w:t></w:r></w:p>
                  </w:body>
                </w:document>
                """,
            )
        document = self.extractor.extract(target)
        self.assertEqual(document.extension, ".docx")
        joined = "\n".join(section.text for section in document.sections)
        self.assertIn("Quarterly revenue improved", joined)
        self.assertIn("Retention stayed above 95 percent", joined)

    def test_xlsx_xml_extraction(self) -> None:
        target = self.root / "regions.xlsx"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr(
                "xl/workbook.xml",
                """
                <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                  <sheets>
                    <sheet name="Pipeline" sheetId="1" r:id="rId1"/>
                  </sheets>
                </workbook>
                """,
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                """
                <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                  <Relationship Id="rId1"
                                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
                                Target="worksheets/sheet1.xml"/>
                </Relationships>
                """,
            )
            archive.writestr(
                "xl/sharedStrings.xml",
                """
                <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <si><t>Region</t></si>
                  <si><t>Bookings</t></si>
                  <si><t>East</t></si>
                </sst>
                """,
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                """
                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <sheetData>
                    <row r="1">
                      <c r="A1" t="s"><v>0</v></c>
                      <c r="B1" t="s"><v>1</v></c>
                    </row>
                    <row r="2">
                      <c r="A2" t="s"><v>2</v></c>
                      <c r="B2"><v>48</v></c>
                    </row>
                  </sheetData>
                </worksheet>
                """,
            )
        document = self.extractor.extract(target)
        joined = "\n".join(section.text for section in document.sections)
        self.assertIn("Sheet: Pipeline", joined)
        self.assertIn("Region: East", joined)
        self.assertIn("Bookings: 48", joined)

    def test_image_uses_multimodal_provider(self) -> None:
        target = self.root / "chart.png"
        target.write_bytes(b"\x89PNG\r\n\x1a\nbinary")
        extractor = DocumentExtractor(AppConfig(openrouter_api_key="x"), provider=FakeProvider())
        document = extractor.extract(target)
        self.assertEqual(document.sections[0].title, "Media analysis")
        self.assertIn("dashboard screenshot", document.sections[0].text)

    def test_vtt_transcript_strips_timestamps(self) -> None:
        target = self.root / "meeting.vtt"
        target.write_text(
            "WEBVTT\n\n"
            "1\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Welcome everyone.\n\n"
            "2\n"
            "00:00:03.500 --> 00:00:05.000\n"
            "<v Speaker>We need faster onboarding.</v>\n",
            encoding="utf-8",
        )
        document = self.extractor.extract(target)
        self.assertEqual(document.metadata["source_modality"], "transcript")
        self.assertIn("Welcome everyone.", document.sections[0].text)
        self.assertIn("We need faster onboarding.", document.sections[0].text)
        self.assertNotIn("-->", document.sections[0].text)


if __name__ == "__main__":
    unittest.main()
