from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re
import string
import subprocess


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "me",
    "more",
    "most",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "this",
    "those",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:%+-]*")
ANCHOR_RE = re.compile(
    r"\b(?:\d{1,4}(?:[./-]\d{1,2}){1,2}|\d+(?:\.\d+)?%|\$?\d[\d,]*(?:\.\d+)?|[A-Z]{2,}[A-Z0-9-]*)\b"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_whitespace(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def safe_filename(name: str) -> str:
    stem = Path(name).name
    allowed = f"-_.(){string.ascii_letters}{string.digits}"
    cleaned = "".join(char if char in allowed else "_" for char in stem)
    return cleaned[:160] or "upload"


def slugify(value: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return lowered.strip("-") or "item"


def digest_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_hash_index(value: str, dimensions: int) -> int:
    raw = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(raw, "little") % dimensions


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value)]


def weighted_tokens(value: str, title: str = "") -> dict[str, float]:
    counts: Counter[str] = Counter()
    tokens = tokenize(value)
    filtered = [token for token in tokens if len(token) > 1]
    for token in filtered:
        counts[token] += 1.0 if token not in STOPWORDS else 0.2
    for left, right in zip(filtered, filtered[1:]):
        if left in STOPWORDS or right in STOPWORDS:
            continue
        counts[f"{left}_{right}"] += 0.45
    for token in tokenize(title):
        counts[token] += 2.25
    return dict(counts)


def extract_keywords(token_counts: dict[str, float], limit: int = 6) -> list[str]:
    ranked = sorted(
        (
            (token, score)
            for token, score in token_counts.items()
            if len(token) > 2 and "_" not in token and token not in STOPWORDS
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    keywords: list[str] = []
    for token, _ in ranked:
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def extract_anchors(value: str) -> set[str]:
    return {match.group(0).lower() for match in ANCHOR_RE.finditer(value)}


def split_sentences(value: str) -> list[str]:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return []
    collapsed = cleaned.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", collapsed)
    output: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip(" -")
        if len(sentence) >= 18:
            output.append(sentence)
    return output or [collapsed]


def trim_excerpt(value: str, terms: list[str], width: int = 280) -> str:
    text = normalize_whitespace(value)
    lowered = text.lower()
    for term in terms:
        position = lowered.find(term.lower())
        if position >= 0:
            start = max(0, position - width // 3)
            end = min(len(text), start + width)
            excerpt = text[start:end].strip()
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt += "..."
            return excerpt
    return text[:width].rstrip() + ("..." if len(text) > width else "")


def chunk_text(value: str, target_chars: int = 720, overlap_chars: int = 120) -> list[str]:
    paragraphs = [part.strip() for part in normalize_whitespace(value).split("\n\n") if part.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        if length and length + len(paragraph) > target_chars:
            joined = "\n\n".join(current)
            chunks.append(joined)
            if overlap_chars > 0:
                overlap: list[str] = []
                overlap_length = 0
                for item in reversed(current):
                    overlap.insert(0, item)
                    overlap_length += len(item)
                    if overlap_length >= overlap_chars:
                        break
                current = overlap
                length = sum(len(item) for item in current)
            else:
                current = []
                length = 0
        current.append(paragraph)
        length += len(paragraph)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def run_command(command: list[str], timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

