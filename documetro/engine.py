from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import mimetypes
import re

import numpy as np
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import svds

from .config import AppConfig
from .models import (
    AnswerResult,
    Chunk,
    DocumentSummary,
    Evidence,
    ExtractedDocument,
    RelationshipSummary,
    TopicSummary,
)
from .nous import NousClient, NousError
from .openrouter import OpenRouterClient, OpenRouterError
from .utils import (
    STOPWORDS,
    chunk_text,
    extract_anchors,
    extract_keywords,
    normalize_whitespace,
    split_sentences,
    stable_hash_index,
    tokenize,
    trim_excerpt,
    weighted_tokens,
)


@dataclass(slots=True)
class SentenceCandidate:
    chunk_index: int
    sentence_index: int
    sentence: str
    score: float
    tokens: set[str]
    actionable: bool


@dataclass(slots=True)
class QueryPlan:
    search_text: str
    keywords: list[str]
    related_terms: list[str]


@dataclass(slots=True)
class EmbeddingStrategy:
    mode: str
    model: str


class CorpusEngine:
    OPENINGS = {
        "how": [
            "According to the uploaded documents,",
            "Based on the files you uploaded,",
            "From the documents you provided,",
        ],
        "fact": [
            "According to the uploaded documents,",
            "The uploaded files indicate that",
            "Based on the material you uploaded,",
        ],
        "compare": [
            "According to the uploaded documents,",
            "Based on the files you uploaded,",
            "From the material you provided,",
        ],
        "list": [
            "According to the uploaded documents,",
            "The relevant items in the uploaded files are:",
            "The uploaded material points to these items:",
        ],
        "closest": [
            "I could not find a fully direct instruction, but the closest matching material says that",
            "The closest strong match in the uploaded documents says that",
            "I did not find an exact procedural match, but the nearest documented guidance says that",
        ],
    }

    def __init__(self, config: AppConfig, documents: list[ExtractedDocument]) -> None:
        self.config = config
        self.documents = documents
        self.provider: OpenRouterClient | None = None
        self.reasoning_provider: NousClient | None = None
        self.chunks: list[Chunk] = []
        self.document_summaries: list[DocumentSummary] = []
        self.topics: list[TopicSummary] = []
        self.relationships: list[RelationshipSummary] = []
        self.total_sections = sum(len(document.sections) for document in documents)
        self._idf = np.array([], dtype=np.float64)
        self._count_matrix = sparse.csr_matrix((0, config.vector_dimensions), dtype=np.float64)
        self._tfidf = sparse.csr_matrix((0, config.vector_dimensions), dtype=np.float64)
        self._latent_docs: np.ndarray | None = None
        self._latent_vt: np.ndarray | None = None
        self._latent_s: np.ndarray | None = None
        self._embedding_matrix: np.ndarray | None = None
        self._embedding_strategy = EmbeddingStrategy(mode="disabled", model="")
        self._topic_for_chunk: dict[str, str] = {}

    @classmethod
    def build(
        cls,
        config: AppConfig,
        documents: list[ExtractedDocument],
        provider: OpenRouterClient | None = None,
        reasoning_provider: NousClient | None = None,
    ) -> "CorpusEngine":
        engine = cls(config, documents)
        engine.provider = provider
        engine.reasoning_provider = reasoning_provider
        engine._fit()
        return engine

    def _fit(self) -> None:
        doc_token_rollup: dict[str, Counter[str]] = defaultdict(Counter)
        doc_chunk_counts: Counter[str] = Counter()
        for document in self.documents:
            for section_index, section in enumerate(document.sections):
                chunks = chunk_text(section.text)
                for chunk_index, text in enumerate(chunks):
                    token_counts = weighted_tokens(text, title=f"{document.name} {section.title}")
                    if not token_counts:
                        continue
                    chunk_id = f"{document.document_id}-{section_index}-{chunk_index}"
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document_name=document.name,
                        section_title=section.title,
                        locator=section.locator,
                        text=normalize_whitespace(text),
                        token_counts=token_counts,
                        anchors=extract_anchors(text),
                        position=len(self.chunks),
                        source_path=document.path if self._is_image_document(document.path) else None,
                        source_mime_type=mimetypes.guess_type(document.path.name)[0] or "",
                        source_modality=str(document.metadata.get("source_modality", "text")),
                    )
                    self.chunks.append(chunk)
                    doc_token_rollup[document.document_id].update(token_counts)
                    doc_chunk_counts[document.document_id] += 1

        self._count_matrix = self._build_count_matrix(chunk.token_counts for chunk in self.chunks)
        self._tfidf = self._to_tfidf(self._count_matrix)
        self._fit_latent_space()
        self._fit_embeddings()
        self.document_summaries = self._build_document_summaries(doc_token_rollup, doc_chunk_counts)
        self.topics = self._build_topics()
        self.relationships = self._build_relationships()

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def snapshot(self) -> dict[str, object]:
        return {
            "document_count": len(self.documents),
            "section_count": self.total_sections,
            "chunk_count": self.total_chunks,
            "embedding_strategy": self._embedding_strategy.mode,
            "embedding_model": self._embedding_strategy.model,
            "documents": [asdict(summary) for summary in self.document_summaries],
            "topics": [asdict(topic) for topic in self.topics],
            "relationships": [asdict(relationship) for relationship in self.relationships],
        }

    def answer(self, question: str) -> AnswerResult:
        question = normalize_whitespace(question)
        if not question:
            return AnswerResult(
                answer="Ask a specific question about the uploaded documents.",
                confidence=0.0,
                mode="empty",
                template="empty",
                title="No question yet",
                lead="Ask a specific question about the uploaded documents.",
                bullets=[],
                note="",
                quote="",
                quote_source="",
                steps=[],
                evidence=[],
                related_topics=[],
            )
        if not self.chunks:
            return AnswerResult(
                answer="Upload documents before asking a question.",
                confidence=0.0,
                mode="no-index",
                template="empty",
                title="No corpus yet",
                lead="Upload documents before asking a question.",
                bullets=[],
                note="",
                quote="",
                quote_source="",
                steps=[],
                evidence=[],
                related_topics=[],
            )

        query_plan = self._query_plan(question)
        retrieval_text = query_plan.search_text or question
        query_counts = weighted_tokens(retrieval_text)
        query_tokens = [token for token in tokenize(retrieval_text) if token not in STOPWORDS and len(token) > 1]
        query_anchors = extract_anchors(question)
        query_vector = self._vectorize_single(query_counts)
        lexical_scores = (self._tfidf @ query_vector.T).toarray().ravel()
        latent_scores = self._latent_scores(query_vector)
        embedding_scores = self._embedding_scores(retrieval_text)
        title_bonus = self._title_bonus(query_tokens + query_plan.keywords + query_plan.related_terms)
        anchor_bonus = self._anchor_bonus(query_anchors)
        type_bonus = self._type_bonus(question)

        scores = (
            (lexical_scores * 0.18)
            + (latent_scores * 0.10)
            + (embedding_scores * 0.62)
            + title_bonus
            + anchor_bonus
            + type_bonus
        )
        top_indices = np.argsort(scores)[::-1][: min(8, len(scores))]
        top_scores = [float(scores[index]) for index in top_indices if scores[index] > 0]
        if not top_scores:
            top_indices = np.argsort(lexical_scores)[::-1][: min(5, len(lexical_scores))]
        top_indices = self._rerank_indices(question, top_indices)

        question_type = self._question_type(question)
        candidates = self._sentence_candidates(question, query_tokens, query_anchors, scores, top_indices)
        selected = self._select_sentences(candidates, limit=4 if question_type == "list" else 3)
        if not selected:
            selected = [
                SentenceCandidate(
                    chunk_index=index,
                    sentence_index=0,
                    sentence=self.chunks[index].text,
                    score=float(scores[index]),
                    tokens=set(tokenize(self.chunks[index].text)),
                    actionable=self._is_actionable(self.chunks[index].text),
                )
                for index in top_indices[:2]
            ]

        confidence = self._confidence(question, scores, selected)
        template, title, lead, bullets, note, quote, quote_source, steps, answer_text = self._compose_answer(
            question,
            selected,
            confidence,
        )
        evidence = self._build_evidence(question, selected, top_indices, query_tokens)
        related_topics = self._related_topics(selected, top_indices)
        llm_answer = self._llm_answer(question, question_type, evidence, related_topics)
        if llm_answer:
            answer_text = llm_answer
            title = "Answer"
            lead = llm_answer.splitlines()[0][:240]
            template = "llm"
            bullets = []
            note = ""
            quote = ""
            quote_source = ""
            steps = []
        return AnswerResult(
            answer=answer_text,
            confidence=confidence,
            mode="llm-rag" if llm_answer else "templated-extractive",
            template=template,
            title=title,
            lead=lead,
            bullets=bullets,
            note=note,
            quote=quote,
            quote_source=quote_source,
            steps=steps,
            evidence=evidence,
            related_topics=related_topics,
        )

    def _fit_embeddings(self) -> None:
        self._embedding_matrix = None
        self._embedding_strategy = EmbeddingStrategy(mode="disabled", model="")
        if not self.provider or not self.provider.enabled or not self.chunks:
            return
        strategy = self._select_embedding_strategy()
        try:
            rows = self.provider.embed([self._embedding_input(chunk, strategy) for chunk in self.chunks], model=strategy.model)
        except OpenRouterError:
            return
        matrix = np.array(rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.chunks):
            return
        self._embedding_matrix = self._normalize_dense_rows(matrix)
        self._embedding_strategy = strategy

    def _embedding_scores(self, question: str) -> np.ndarray:
        if self._embedding_matrix is None or not self.provider or not self.provider.enabled:
            return np.zeros(self.total_chunks, dtype=np.float64)
        try:
            rows = self.provider.embed([question], model=self._embedding_strategy.model or None)
        except OpenRouterError:
            return np.zeros(self.total_chunks, dtype=np.float64)
        query = np.array(rows[0], dtype=np.float64)
        norm = np.linalg.norm(query)
        if norm == 0:
            return np.zeros(self.total_chunks, dtype=np.float64)
        query /= norm
        return self._embedding_matrix @ query

    def _embedding_input(self, chunk: Chunk, strategy: EmbeddingStrategy) -> object:
        if strategy.mode == "multimodal-unified" and chunk.source_path and chunk.source_path.exists() and chunk.source_modality == "image":
            return {
                "content": [
                    {"type": "text", "text": chunk.text},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.provider._data_url(chunk.source_path)},
                    },
                ]
            }
        return chunk.text

    def _select_embedding_strategy(self) -> EmbeddingStrategy:
        has_visual = any(chunk.source_modality == "image" for chunk in self.chunks)
        if has_visual and self.config.openrouter_multimodal_embedding_model:
            return EmbeddingStrategy(
                mode="multimodal-unified",
                model=self.config.openrouter_multimodal_embedding_model,
            )
        return EmbeddingStrategy(
            mode="text",
            model=self.config.openrouter_embedding_model,
        )

    def _llm_answer(self, question: str, question_type: str, evidence: list[Evidence], related_topics: list[str]) -> str:
        if not evidence:
            return ""
        if question_type in {"summary", "general", "compare", "how"} and self.reasoning_provider and self.reasoning_provider.enabled:
            try:
                return self.reasoning_provider.answer_question(question, evidence, related_topics, analysis_mode=question_type)
            except NousError:
                pass
        if not self.provider or not self.provider.enabled:
            return ""
        try:
            return self.provider.answer_question(question, evidence, related_topics)
        except OpenRouterError:
            return ""

    def _query_plan(self, question: str) -> QueryPlan:
        fallback = QueryPlan(search_text=question, keywords=[], related_terms=[])
        if not self.reasoning_provider or not self.reasoning_provider.enabled:
            return fallback
        try:
            payload = self.reasoning_provider.plan_query(
                question,
                [document.name for document in self.documents],
                [topic.label for topic in self.topics],
            )
        except NousError:
            return fallback
        search_text = normalize_whitespace(str(payload.get("search_query", "") or question))
        keywords = self._clean_plan_terms(payload.get("keywords", []))
        related_terms = self._clean_plan_terms(payload.get("related_terms", []))
        enriched = " ".join(part for part in [search_text, *keywords, *related_terms] if part).strip()
        return QueryPlan(
            search_text=enriched or question,
            keywords=keywords,
            related_terms=related_terms,
        )

    def _clean_plan_terms(self, raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []
        terms: list[str] = []
        seen: set[str] = set()
        for item in raw[:8]:
            text = normalize_whitespace(str(item))
            if len(text) < 2:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(text)
        return terms

    def _rerank_indices(self, question: str, indices: np.ndarray) -> np.ndarray:
        if not len(indices) or not self.reasoning_provider or not self.reasoning_provider.enabled:
            return indices
        candidates = []
        for index in indices[:12]:
            chunk = self.chunks[int(index)]
            candidates.append(
                {
                    "id": chunk.chunk_id,
                    "title": f"{chunk.document_name} / {chunk.section_title}",
                    "locator": chunk.locator,
                    "text": trim_excerpt(chunk.text, [], 240),
                }
            )
        try:
            ranked_ids = self.reasoning_provider.rerank_evidence(question, candidates)
        except NousError:
            return indices
        if not ranked_ids:
            return indices
        rank_order = {chunk_id: position for position, chunk_id in enumerate(ranked_ids)}
        reordered = sorted(
            [int(index) for index in indices],
            key=lambda index: rank_order.get(self.chunks[index].chunk_id, len(rank_order) + index),
        )
        return np.array(reordered, dtype=np.int64)

    def _is_image_document(self, path: Path) -> bool:
        return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}

    def _build_count_matrix(self, token_sets: Iterable[dict[str, float]]) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row_index, token_counts in enumerate(token_sets):
            buckets: dict[int, float] = {}
            for token, value in token_counts.items():
                column = stable_hash_index(token, self.config.vector_dimensions)
                buckets[column] = buckets.get(column, 0.0) + float(value)
            for column, value in buckets.items():
                rows.append(row_index)
                cols.append(column)
                data.append(value)
        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.chunks), self.config.vector_dimensions),
            dtype=np.float64,
        )
        return matrix

    def _to_tfidf(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        if matrix.shape[0] == 0:
            self._idf = np.ones(self.config.vector_dimensions, dtype=np.float64)
            return matrix
        binary = matrix.copy()
        binary.data = np.ones_like(binary.data)
        df = np.asarray(binary.sum(axis=0)).ravel()
        self._idf = np.log((1.0 + matrix.shape[0]) / (1.0 + df)) + 1.0
        tfidf = matrix.copy().astype(np.float64)
        tfidf.data = np.log1p(tfidf.data)
        tfidf.data *= self._idf[tfidf.indices]
        return self._normalize_sparse_rows(tfidf)

    def _normalize_sparse_rows(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        if matrix.shape[0] == 0:
            return matrix
        squared = matrix.multiply(matrix)
        norms = np.sqrt(np.asarray(squared.sum(axis=1)).ravel())
        normalized = matrix.tocsr(copy=True)
        for row_index in range(normalized.shape[0]):
            start = normalized.indptr[row_index]
            end = normalized.indptr[row_index + 1]
            norm = norms[row_index]
            if norm > 0:
                normalized.data[start:end] /= norm
        return normalized

    def _normalize_dense_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _vectorize_single(self, token_counts: dict[str, float]) -> sparse.csr_matrix:
        if not token_counts:
            return sparse.csr_matrix((1, self.config.vector_dimensions), dtype=np.float64)
        buckets: dict[int, float] = {}
        for token, value in token_counts.items():
            column = stable_hash_index(token, self.config.vector_dimensions)
            buckets[column] = buckets.get(column, 0.0) + value
        cols = list(buckets)
        data = np.array([math.log1p(buckets[column]) * self._idf[column] for column in cols], dtype=np.float64)
        row = sparse.csr_matrix((data, ([0] * len(cols), cols)), shape=(1, self.config.vector_dimensions))
        return self._normalize_sparse_rows(row)

    def _fit_latent_space(self) -> None:
        self._latent_docs = None
        self._latent_vt = None
        self._latent_s = None
        rows, columns = self._tfidf.shape
        if rows < 3 or columns < 3 or self._tfidf.nnz == 0:
            return
        dimensions = min(self.config.latent_dimensions, rows - 1, columns - 1)
        if dimensions < 2:
            return
        try:
            u, singular_values, vt = svds(self._tfidf, k=dimensions)
        except Exception:
            return
        order = np.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        vt = vt[order]
        u = u[:, order]
        doc_vectors = u * singular_values
        self._latent_docs = self._normalize_dense_rows(doc_vectors)
        self._latent_vt = vt
        self._latent_s = singular_values

    def _latent_scores(self, query_vector: sparse.csr_matrix) -> np.ndarray:
        if self._latent_docs is None or self._latent_vt is None or self._latent_s is None:
            return np.zeros(self.total_chunks, dtype=np.float64)
        dense = query_vector.toarray()
        projected = dense @ self._latent_vt.T
        safe_s = np.where(self._latent_s == 0, 1.0, self._latent_s)
        query_latent = projected / safe_s
        norm = np.linalg.norm(query_latent)
        if norm == 0:
            return np.zeros(self.total_chunks, dtype=np.float64)
        query_latent /= norm
        return self._latent_docs @ query_latent.T[:, 0]

    def _build_document_summaries(
        self,
        token_rollup: dict[str, Counter[str]],
        chunk_counts: Counter[str],
    ) -> list[DocumentSummary]:
        summaries: list[DocumentSummary] = []
        for document in self.documents:
            summaries.append(
                DocumentSummary(
                    document_id=document.document_id,
                    name=document.name,
                    extension=document.extension,
                    size_bytes=document.size_bytes,
                    section_count=len(document.sections),
                    chunk_count=chunk_counts[document.document_id],
                    keywords=extract_keywords(dict(token_rollup[document.document_id]), limit=5),
                )
            )
        return summaries

    def _build_topics(self) -> list[TopicSummary]:
        if not self.chunks:
            return []
        labels = np.zeros(self.total_chunks, dtype=int)
        if self._latent_docs is not None and self.total_chunks >= 4:
            cluster_count = max(1, min(6, int(round(math.sqrt(self.total_chunks)))))
            cluster_count = min(cluster_count, self.total_chunks)
            if cluster_count > 1:
                try:
                    _, labels = kmeans2(self._latent_docs, cluster_count, minit="points", iter=20)
                except Exception:
                    labels = np.zeros(self.total_chunks, dtype=int)
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped[int(label)].append(index)
        topics: list[TopicSummary] = []
        for member_indices in grouped.values():
            aggregate = Counter()
            document_names = sorted({self.chunks[index].document_name for index in member_indices})
            for index in member_indices:
                aggregate.update(self.chunks[index].token_counts)
            keywords = extract_keywords(dict(aggregate), limit=4)
            label = " / ".join(keywords[:3]) or "General context"
            representative = max(member_indices, key=lambda item: len(self.chunks[item].text))
            snippet = trim_excerpt(self.chunks[representative].text, keywords[:2], width=200)
            topic = TopicSummary(
                label=label,
                keywords=keywords,
                snippet=snippet,
                document_names=document_names,
                chunk_count=len(member_indices),
                score=round(len(member_indices) / max(self.total_chunks, 1), 3),
            )
            topics.append(topic)
            for member_index in member_indices:
                self._topic_for_chunk[self.chunks[member_index].chunk_id] = topic.label
        topics.sort(key=lambda item: (item.chunk_count, item.score), reverse=True)
        return topics[:6]

    def _build_relationships(self) -> list[RelationshipSummary]:
        if len(self.documents) < 2 or self.total_chunks == 0:
            return []
        doc_indices: dict[str, list[int]] = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            doc_indices[chunk.document_id].append(index)
        rows = []
        doc_ids: list[str] = []
        for document in self.documents:
            indices = doc_indices.get(document.document_id, [])
            if not indices:
                continue
            profile = self._tfidf[indices].mean(axis=0)
            profile = sparse.csr_matrix(profile)
            profile = self._normalize_sparse_rows(profile)
            rows.append(profile)
            doc_ids.append(document.document_id)
        if len(rows) < 2:
            return []
        doc_matrix = sparse.vstack(rows)
        similarity = (doc_matrix @ doc_matrix.T).toarray()
        names = {document.document_id: document.name for document in self.documents}
        relationships: list[RelationshipSummary] = []
        for left_index, left_id in enumerate(doc_ids):
            for right_index in range(left_index + 1, len(doc_ids)):
                score = float(similarity[left_index, right_index])
                if score >= 0.18:
                    relationships.append(
                        RelationshipSummary(
                            left=names[left_id],
                            right=names[doc_ids[right_index]],
                            similarity=round(score, 3),
                        )
                    )
        relationships.sort(key=lambda item: item.similarity, reverse=True)
        return relationships[:8]

    def _title_bonus(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.total_chunks, dtype=np.float64)
        if not query_tokens:
            return scores
        denominator = max(len(set(query_tokens)), 1)
        for index, chunk in enumerate(self.chunks):
            haystack = f"{chunk.document_name} {chunk.section_title}".lower()
            overlap = sum(1 for token in set(query_tokens) if token in haystack)
            scores[index] = 0.08 * overlap / denominator
        return scores

    def _anchor_bonus(self, query_anchors: set[str]) -> np.ndarray:
        scores = np.zeros(self.total_chunks, dtype=np.float64)
        if not query_anchors:
            return scores
        denominator = max(len(query_anchors), 1)
        for index, chunk in enumerate(self.chunks):
            overlap = len(chunk.anchors & query_anchors)
            scores[index] = 0.11 * overlap / denominator
        return scores

    def _type_bonus(self, question: str) -> np.ndarray:
        scores = np.zeros(self.total_chunks, dtype=np.float64)
        question_type = self._question_type(question)
        if question_type not in {"count", "date", "list", "compare"}:
            return scores
        for index, chunk in enumerate(self.chunks):
            text = chunk.text
            if question_type in {"count", "date"} and re.search(r"\d", text):
                scores[index] += 0.04
            if question_type == "list" and ("|" in text or "\n" in text):
                scores[index] += 0.03
            if question_type == "compare" and any(marker in text.lower() for marker in ["versus", "vs", "difference", "compare"]):
                scores[index] += 0.04
        return scores

    def _sentence_candidates(
        self,
        question: str,
        query_tokens: list[str],
        query_anchors: set[str],
        chunk_scores: np.ndarray,
        top_indices: np.ndarray,
    ) -> list[SentenceCandidate]:
        question_type = self._question_type(question)
        unique_query_tokens = set(query_tokens)
        total_weight = sum(self._idf[stable_hash_index(token, self.config.vector_dimensions)] for token in unique_query_tokens) or 1.0
        candidates: list[SentenceCandidate] = []
        seen_sentences: set[tuple[int, str]] = set()
        for chunk_index in top_indices:
            chunk = self.chunks[int(chunk_index)]
            sentences = split_sentences(chunk.text)
            if not sentences:
                continue
            for sentence_index, original_sentence in enumerate(sentences[:12]):
                sentence = self._candidate_sentence_text(sentences, sentence_index)
                normalized_sentence = normalize_whitespace(sentence)
                key = (int(chunk_index), normalized_sentence.lower())
                if not normalized_sentence or key in seen_sentences:
                    continue
                seen_sentences.add(key)
                sentence_tokens = {token for token in tokenize(sentence) if token not in STOPWORDS and len(token) > 1}
                if not sentence_tokens:
                    continue
                actionable = self._is_actionable(sentence)
                overlap_weight = sum(
                    self._idf[stable_hash_index(token, self.config.vector_dimensions)]
                    for token in (unique_query_tokens & sentence_tokens)
                )
                overlap_score = overlap_weight / total_weight
                anchor_overlap = len(query_anchors & extract_anchors(sentence))
                anchor_score = 0.08 * anchor_overlap / max(len(query_anchors), 1) if query_anchors else 0.0
                position_bonus = max(0.06 - (sentence_index * 0.008), 0.0)
                type_bonus = 0.0
                if question_type in {"count", "date"} and re.search(r"\d", sentence):
                    type_bonus += 0.04
                if question_type == "list" and any(marker in sentence for marker in [":", ";", "|"]):
                    type_bonus += 0.03
                if question_type == "how":
                    type_bonus += 0.07 if actionable else -0.06
                score_penalty = 0.0
                if self._is_heading_like(original_sentence):
                    score_penalty += 0.12
                if normalize_whitespace(original_sentence).endswith("?"):
                    score_penalty += 0.16
                if len(sentence_tokens) < 4:
                    score_penalty += 0.05
                if self._is_question_echo(sentence, question):
                    score_penalty += 0.14
                score = (chunk_scores[int(chunk_index)] * 0.58) + (overlap_score * 0.3) + anchor_score + position_bonus + type_bonus
                score -= score_penalty
                candidates.append(
                    SentenceCandidate(
                        chunk_index=int(chunk_index),
                        sentence_index=sentence_index,
                        sentence=sentence,
                        score=float(score),
                        tokens=sentence_tokens,
                        actionable=actionable,
                    )
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:24]

    def _select_sentences(self, candidates: list[SentenceCandidate], limit: int) -> list[SentenceCandidate]:
        selected: list[SentenceCandidate] = []
        remaining = candidates[:]
        while remaining and len(selected) < limit:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_index = 0
            best_value = -1.0
            for index, candidate in enumerate(remaining):
                novelty = max(self._token_jaccard(candidate.tokens, current.tokens) for current in selected)
                value = (candidate.score * 0.78) - (novelty * 0.22)
                if value > best_value:
                    best_value = value
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected

    def _compose_answer(
        self,
        question: str,
        sentences: list[SentenceCandidate],
        confidence: float,
    ) -> tuple[str, str, str, list[str], str, str, str, list[str], str]:
        if not sentences:
            lead = "The uploaded documents do not contain a confident answer yet."
            return ("empty", "No strong answer", lead, [], "", "", "", [], lead)
        question_type = self._question_type(question)
        excerpts = [self._compact_statement(candidate.sentence, max_chars=220) for candidate in sentences]
        support_candidates = self._supporting_candidates(sentences, question_type)
        support_excerpts = [self._compact_statement(candidate.sentence, max_chars=180) for candidate in support_candidates]
        support_documents = list(dict.fromkeys(self.chunks[candidate.chunk_index].document_name for candidate in support_candidates))
        unique_documents = support_documents or list(
            dict.fromkeys(self.chunks[candidate.chunk_index].document_name for candidate in sentences)
        )
        note = self._answer_note(unique_documents, confidence)
        quote, quote_source = self._build_quote(sentences, support_candidates)
        if question_type == "list":
            bullets = self._dedupe_lines(self._list_fragments(excerpts))[:4]
            lead = bullets[0] if len(bullets) == 1 else self._lead_with_opening("list", "the relevant items are listed below.", question)
            answer = lead if len(bullets) <= 1 else "\n".join([lead, *[f"- {item}" for item in bullets]])
            return ("list", "Relevant items", lead, bullets, note, quote, quote_source, [], answer)
        if question_type == "compare":
            lead = self._lead_with_opening("compare", excerpts[0], question)
            bullets = self._dedupe_lines(excerpts[1:4])
            answer = lead if not bullets else "\n".join([lead, *[f"- {item}" for item in bullets]])
            return ("compare", "Comparison", lead, bullets, note, quote, quote_source, [], answer)
        if question_type in {"count", "date", "who", "what", "where"}:
            lead_excerpt = self._preferred_fact_excerpt(question, excerpts[0], support_excerpts)
            lead = self._lead_with_opening("fact", lead_excerpt, question)
            bullets = self._dedupe_lines([item for item in support_excerpts if item != lead_excerpt][:2])
            answer = lead if not bullets else "\n".join([lead, *[f"- {item}" for item in bullets]])
            return ("fact", "Best match", lead, bullets, note, quote, quote_source, [], answer)
        if question_type == "how":
            lead_excerpt = self._preferred_how_excerpt(sentences, support_candidates)
            lead = self._lead_with_opening("how", lead_excerpt, question)
            steps = self._dedupe_lines(self._how_steps(support_candidates, support_excerpts))[:3]
            answer_lines = [lead]
            if steps:
                answer_lines.extend(f"{index + 1}. {step}" for index, step in enumerate(steps))
            if quote:
                answer_lines.append(f'One relevant passage says: "{quote}"')
            return ("how_to", "How to do it", lead, [], note, quote, quote_source, steps, "\n".join(answer_lines))
        lead = self._lead_with_opening("fact", excerpts[0], question)
        bullets = self._dedupe_lines(support_excerpts[:3])
        if confidence < 0.31:
            title = "Closest match"
            note = note or "Confidence is limited. Check the evidence."
            answer = lead if not bullets else "\n".join([lead, *[f"- {item}" for item in bullets]])
            return ("closest", title, lead, bullets, note, quote, quote_source, [], answer)
        title = "Answer"
        answer = lead if not bullets else "\n".join([lead, *[f"- {item}" for item in bullets]])
        return ("brief", title, lead, bullets, note, quote, quote_source, [], answer)

    def _confidence(self, question: str, chunk_scores: np.ndarray, sentences: list[SentenceCandidate]) -> float:
        best = float(np.max(chunk_scores)) if len(chunk_scores) else 0.0
        ordered = sorted((float(value) for value in chunk_scores), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0] if ordered else 0.0
        query_tokens = {token for token in tokenize(question) if token not in STOPWORDS and len(token) > 1}
        coverage = len(set().union(*(candidate.tokens for candidate in sentences)) & query_tokens) / max(len(query_tokens), 1)
        confidence = (best * 0.48) + (margin * 0.16) + (coverage * 0.36)
        return round(float(min(max(confidence, 0.12), 0.98)), 3)

    def _build_evidence(
        self,
        question: str,
        sentences: list[SentenceCandidate],
        top_indices: np.ndarray,
        query_tokens: list[str],
    ) -> list[Evidence]:
        question_type = self._question_type(question)
        if question_type in {"summary", "general", "compare"}:
            evidence = self._chunk_evidence(top_indices, query_tokens, limit=8)
            if evidence:
                return evidence
        evidence: list[Evidence] = []
        for candidate in sentences[:6]:
            chunk = self.chunks[candidate.chunk_index]
            excerpt = trim_excerpt(candidate.sentence, query_tokens, width=320)
            if len(excerpt) < 150 and candidate.sentence_index > 0:
                sentences_list = split_sentences(chunk.text)
                if candidate.sentence_index < len(sentences_list):
                    next_sent = sentences_list[candidate.sentence_index + 1] if candidate.sentence_index + 1 < len(sentences_list) else ""
                    excerpt = trim_excerpt(candidate.sentence + " " + next_sent, query_tokens, width=320)
            evidence.append(
                Evidence(
                    document_name=chunk.document_name,
                    locator=chunk.locator,
                    section_title=chunk.section_title,
                    excerpt=excerpt,
                    score=round(candidate.score, 3),
                )
            )
        return evidence

    def _chunk_evidence(self, top_indices: np.ndarray, query_tokens: list[str], limit: int) -> list[Evidence]:
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for raw_index in top_indices[:limit]:
            chunk = self.chunks[int(raw_index)]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            evidence.append(
                Evidence(
                    document_name=chunk.document_name,
                    locator=chunk.locator,
                    section_title=chunk.section_title,
                    excerpt=trim_excerpt(chunk.text, query_tokens, width=520),
                    score=1.0,
                )
            )
        return evidence

    def _related_topics(self, sentences: list[SentenceCandidate], top_indices: np.ndarray | None = None) -> list[str]:
        labels: list[str] = []
        for candidate in sentences:
            chunk = self.chunks[candidate.chunk_index]
            label = self._topic_for_chunk.get(chunk.chunk_id)
            if label and label not in labels:
                labels.append(label)
        if top_indices is not None:
            for raw_index in top_indices[:8]:
                chunk = self.chunks[int(raw_index)]
                label = self._topic_for_chunk.get(chunk.chunk_id)
                if label and label not in labels:
                    labels.append(label)
        return labels[:3]

    def _question_type(self, question: str) -> str:
        lowered = question.lower()
        if any(marker in lowered for marker in ["summarize", "summary", "overview", "recap", "what was discussed", "what has been talked about"]):
            return "summary"
        if lowered.startswith("how many") or " number of " in f" {lowered} ":
            return "count"
        if lowered.startswith("how do i") or lowered.startswith("how can i") or lowered.startswith("how to") or lowered.startswith("how "):
            return "how"
        if lowered.startswith("when") or " date " in f" {lowered} ":
            return "date"
        if lowered.startswith("who"):
            return "who"
        if lowered.startswith("where"):
            return "where"
        if lowered.startswith("what"):
            return "what"
        if any(marker in lowered for marker in [" list ", " what are ", " which ", " identify ", " name the "]):
            return "list"
        if any(marker in lowered for marker in [" compare ", " difference ", " versus ", " vs "]):
            return "compare"
        return "general"

    def _token_jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _answer_note(self, documents: list[str], confidence: float) -> str:
        if confidence < 0.32:
            return "This appears to be the closest documented match."
        if len(documents) > 1:
            return f"Evidence spans {len(documents)} files."
        if documents:
            return f"Evidence comes from {documents[0]}."
        return ""

    def _dedupe_lines(self, values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = self._compact_statement(value, max_chars=160).strip().rstrip(".")
            key = cleaned.lower()
            if len(cleaned) < 12 or key in seen:
                continue
            seen.add(key)
            output.append(cleaned)
        return output

    def _list_fragments(self, excerpts: list[str]) -> list[str]:
        fragments: list[str] = []
        for excerpt in excerpts:
            parts = re.split(r"\s*[;|]\s*|\s{2,}", excerpt)
            for part in parts:
                cleaned = normalize_whitespace(part).strip(" -")
                if cleaned:
                    fragments.append(cleaned)
        return fragments or excerpts

    def _compact_statement(self, value: str, max_chars: int = 160) -> str:
        cleaned = normalize_whitespace(value).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        window = cleaned[: max_chars + 1]
        for marker in [". ", "; ", ", "]:
            cut = window.rfind(marker)
            if cut >= max_chars // 2:
                return window[:cut].rstrip(" .,;") + "…"
        return window[:max_chars].rstrip() + "…"

    def _supporting_candidates(self, sentences: list[SentenceCandidate], question_type: str) -> list[SentenceCandidate]:
        if len(sentences) <= 1:
            return []
        if question_type in {"list", "compare"}:
            return sentences[1:]
        lead = sentences[0]
        lead_document = self.chunks[lead.chunk_index].document_name
        threshold = lead.score * 0.74
        filtered = [
            candidate
            for candidate in sentences[1:]
            if self.chunks[candidate.chunk_index].document_name == lead_document or candidate.score >= threshold
        ]
        return filtered or sentences[1:2]

    def _candidate_sentence_text(self, sentences: list[str], sentence_index: int) -> str:
        current = normalize_whitespace(sentences[sentence_index])
        if self._is_heading_like(current) and sentence_index + 1 < len(sentences):
            next_sentence = normalize_whitespace(sentences[sentence_index + 1])
            if next_sentence and not self._is_heading_like(next_sentence):
                return next_sentence
        return current

    def _is_heading_like(self, sentence: str) -> bool:
        normalized = normalize_whitespace(sentence)
        tokens = tokenize(normalized)
        if not normalized:
            return True
        if normalized.endswith("?"):
            return True
        if len(tokens) <= 5 and len(normalized) <= 48:
            return True
        if ":" not in normalized and normalized == normalized.title() and len(tokens) <= 7:
            return True
        return False

    def _is_actionable(self, sentence: str) -> bool:
        lowered = normalize_whitespace(sentence).lower()
        action_markers = [
            "open",
            "go to",
            "click",
            "select",
            "use",
            "edit",
            "create",
            "configure",
            "change",
            "set",
            "enable",
            "disable",
            "update",
            "add",
            "remove",
        ]
        return any(marker in lowered for marker in action_markers)

    def _is_question_echo(self, sentence: str, question: str) -> bool:
        sentence_tokens = {token for token in tokenize(sentence) if token not in STOPWORDS and len(token) > 1}
        question_tokens = {token for token in tokenize(question) if token not in STOPWORDS and len(token) > 1}
        if not sentence_tokens or not question_tokens:
            return False
        overlap = len(sentence_tokens & question_tokens) / max(len(sentence_tokens), 1)
        return overlap >= 0.8 and len(sentence_tokens) <= len(question_tokens) + 1

    def _pick_opening(self, intent: str, seed: str) -> str:
        options = self.OPENINGS.get(intent, self.OPENINGS["fact"])
        return options[stable_hash_index(seed, len(options))]

    def _lead_with_opening(self, intent: str, sentence: str, seed: str) -> str:
        opening = self._pick_opening(intent, seed)
        sentence = sentence.strip()
        if sentence.lower().startswith(opening.lower().rstrip(", ")):
            return sentence
        if opening.endswith(","):
            return f"{opening} {sentence}"
        return f"{opening} {sentence}"

    def _build_quote(
        self,
        sentences: list[SentenceCandidate],
        support_candidates: list[SentenceCandidate],
    ) -> tuple[str, str]:
        pool = support_candidates or sentences
        if not pool:
            return "", ""
        candidate = pool[0]
        quote = self._compact_statement(candidate.sentence, max_chars=150)
        chunk = self.chunks[candidate.chunk_index]
        source = f"{chunk.document_name} · {chunk.locator}"
        return quote, source

    def _how_steps(self, support_candidates: list[SentenceCandidate], support_excerpts: list[str]) -> list[str]:
        actionable = [candidate.sentence for candidate in support_candidates if candidate.actionable]
        if actionable:
            return actionable
        return support_excerpts

    def _preferred_how_excerpt(
        self,
        sentences: list[SentenceCandidate],
        support_candidates: list[SentenceCandidate],
    ) -> str:
        pool = support_candidates + sentences
        for candidate in pool:
            if candidate.actionable:
                return self._compact_statement(candidate.sentence, max_chars=220)
        for candidate in pool:
            lowered = candidate.sentence.lower()
            if any(marker in lowered for marker in [" through ", " using ", " via ", " links ", " linked ", " because ", " due to "]):
                return self._compact_statement(candidate.sentence, max_chars=220)
        if support_candidates:
            return self._compact_statement(support_candidates[0].sentence, max_chars=220)
        return self._compact_statement(sentences[0].sentence, max_chars=220)

    def _preferred_fact_excerpt(self, question: str, default_excerpt: str, support_excerpts: list[str]) -> str:
        lowered = question.lower()
        if support_excerpts and lowered.startswith("what improved "):
            rewritten = self._rewrite_subject_answer(question, support_excerpts[0], "improved")
            return rewritten or support_excerpts[0]
        if support_excerpts and lowered.startswith("what caused "):
            rewritten = self._rewrite_subject_answer(question, support_excerpts[0], "caused")
            return rewritten or support_excerpts[0]
        if support_excerpts and any(marker in lowered for marker in ["what changed", "what allows", "what enables"]):
            return support_excerpts[0]
        return default_excerpt

    def _rewrite_subject_answer(self, question: str, excerpt: str, verb: str) -> str:
        lowered_question = question.lower().strip(" ?.")
        prefix = f"what {verb} "
        if not lowered_question.startswith(prefix):
            return ""
        subject = question[len(prefix):].strip(" ?.")
        if not subject:
            return ""
        cause = self._extract_after_markers(
            excerpt,
            [
                "links the improvement to ",
                "linked the improvement to ",
                "improvement comes from ",
                "improved by ",
                "because of ",
                "due to ",
            ],
        )
        if not cause:
            return ""
        return f"{cause} {verb} {subject}."

    def _extract_after_markers(self, excerpt: str, markers: list[str]) -> str:
        lowered = excerpt.lower()
        for marker in markers:
            position = lowered.find(marker)
            if position >= 0:
                value = excerpt[position + len(marker):].strip(" .")
                if value:
                    return value[0].upper() + value[1:]
        return ""
