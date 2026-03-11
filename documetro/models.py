from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentSection:
    title: str
    text: str
    locator: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedDocument:
    document_id: str
    name: str
    extension: str
    path: Path
    size_bytes: int
    checksum: str
    sections: list[DocumentSection]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    section_title: str
    locator: str
    text: str
    token_counts: dict[str, float]
    anchors: set[str]
    position: int
    source_path: Path | None = None
    source_mime_type: str = ""
    source_modality: str = "text"


@dataclass(slots=True)
class TopicSummary:
    label: str
    keywords: list[str]
    snippet: str
    document_names: list[str]
    chunk_count: int
    score: float


@dataclass(slots=True)
class DocumentSummary:
    document_id: str
    name: str
    extension: str
    size_bytes: int
    section_count: int
    chunk_count: int
    keywords: list[str]


@dataclass(slots=True)
class RelationshipSummary:
    left: str
    right: str
    similarity: float


@dataclass(slots=True)
class Evidence:
    document_name: str
    locator: str
    section_title: str
    excerpt: str
    score: float


@dataclass(slots=True)
class AnswerResult:
    answer: str
    confidence: float
    mode: str
    template: str
    title: str
    lead: str
    bullets: list[str]
    note: str
    quote: str
    quote_source: str
    steps: list[str]
    evidence: list[Evidence]
    related_topics: list[str]
