from __future__ import annotations

from pathlib import Path
import unittest

from documetro.config import AppConfig
from documetro.engine import CorpusEngine
from documetro.models import DocumentSection, Evidence, ExtractedDocument


class FakeProvider:
    enabled = True

    def embed(self, inputs: list[object], model: str | None = None) -> list[list[float]]:
        rows = []
        for item in inputs:
            text = item if isinstance(item, str) else str(item)
            lowered = text.lower()
            rows.append(
                [
                    1.0 if "retention" in lowered else 0.0,
                    1.0 if "onboarding" in lowered or "support" in lowered else 0.0,
                    1.0 if "revenue" in lowered else 0.0,
                ]
            )
        return rows

    def answer_question(self, question: str, evidence: list[Evidence], related_topics: list[str]) -> str:
        return "Retention improved because onboarding got faster and support coverage improved."


class FakeReasoningProvider:
    enabled = True

    def plan_query(self, question: str, document_names: list[str], topic_labels: list[str]) -> dict[str, object]:
        return {
            "intent": "causal",
            "search_query": "customer retention support coverage onboarding",
            "keywords": ["support coverage"],
            "related_terms": ["onboarding"],
        }

    def rerank_evidence(self, question: str, candidates: list[dict[str, str]]) -> list[str]:
        return [item["id"] for item in candidates]

    def answer_question(
        self,
        question: str,
        evidence: list[Evidence],
        related_topics: list[str],
        analysis_mode: str = "general",
    ) -> str:
        if analysis_mode == "summary":
            return "The meeting focused on onboarding delays, support coverage, and how both affected retention."
        return "Retention improved because onboarding got faster and support coverage improved."


class EngineTests(unittest.TestCase):
    def test_answers_with_evidence(self) -> None:
        documents = [
            ExtractedDocument(
                document_id="alpha",
                name="retention.txt",
                extension=".txt",
                path=Path("retention.txt"),
                size_bytes=10,
                checksum="alpha",
                sections=[
                    DocumentSection(
                        title="Body",
                        locator="full",
                        text=(
                            "Customer retention reached 97 percent in the second half. "
                            "The report links the improvement to faster onboarding and better support coverage."
                        ),
                    )
                ],
            ),
            ExtractedDocument(
                document_id="beta",
                name="finance.txt",
                extension=".txt",
                path=Path("finance.txt"),
                size_bytes=10,
                checksum="beta",
                sections=[
                    DocumentSection(
                        title="Body",
                        locator="full",
                        text="Revenue increased by 12 percent while infrastructure spend stayed flat.",
                    )
                ],
            ),
        ]
        engine = CorpusEngine.build(AppConfig(), documents, provider=FakeProvider())
        result = engine.answer("What improved customer retention?")
        self.assertIn("onboarding", result.answer.lower())
        self.assertEqual(result.template, "llm")
        self.assertEqual(result.title, "Answer")
        self.assertFalse(result.bullets)
        self.assertFalse(result.steps)
        self.assertTrue(result.lead)
        self.assertTrue(result.evidence)
        self.assertEqual(result.evidence[0].document_name, "retention.txt")
        self.assertEqual(engine.snapshot()["embedding_strategy"], "text")

    def test_uses_reasoning_provider_for_query_planning(self) -> None:
        documents = [
            ExtractedDocument(
                document_id="alpha",
                name="retention.txt",
                extension=".txt",
                path=Path("retention.txt"),
                size_bytes=10,
                checksum="alpha",
                sections=[
                    DocumentSection(
                        title="Body",
                        locator="full",
                        text=(
                            "Customer retention reached 97 percent in the second half. "
                            "The report links the improvement to faster onboarding and better support coverage."
                        ),
                    )
                ],
            ),
            ExtractedDocument(
                document_id="beta",
                name="finance.txt",
                extension=".txt",
                path=Path("finance.txt"),
                size_bytes=10,
                checksum="beta",
                sections=[
                    DocumentSection(
                        title="Body",
                        locator="full",
                        text="Revenue increased by 12 percent while infrastructure spend stayed flat.",
                    )
                ],
            ),
        ]
        engine = CorpusEngine.build(
            AppConfig(),
            documents,
            provider=FakeProvider(),
            reasoning_provider=FakeReasoningProvider(),
        )
        result = engine.answer("Why did users stay longer?")
        self.assertIn("onboarding", result.answer.lower())
        self.assertEqual(result.evidence[0].document_name, "retention.txt")

    def test_summary_questions_use_reasoning_synthesis(self) -> None:
        documents = [
            ExtractedDocument(
                document_id="meeting",
                name="meeting.vtt",
                extension=".vtt",
                path=Path("meeting.vtt"),
                size_bytes=10,
                checksum="meeting",
                sections=[
                    DocumentSection(
                        title="Text",
                        locator="full",
                        text=(
                            "00:01 Team discussed onboarding delays and support queue coverage. "
                            "00:02 They linked those issues to retention and planned fixes."
                        ),
                    )
                ],
            )
        ]
        engine = CorpusEngine.build(
            AppConfig(),
            documents,
            provider=FakeProvider(),
            reasoning_provider=FakeReasoningProvider(),
        )
        result = engine.answer("Summarize what was discussed in the meeting")
        self.assertIn("support coverage", result.answer.lower())
        self.assertEqual(result.title, "Answer")
        self.assertEqual(result.template, "llm")

    def test_uses_multimodal_embedding_strategy_when_images_exist(self) -> None:
        documents = [
            ExtractedDocument(
                document_id="img",
                name="chart.png",
                extension=".png",
                path=Path("chart.png"),
                size_bytes=10,
                checksum="img",
                sections=[
                    DocumentSection(
                        title="Media analysis",
                        locator="media:1",
                        text="A chart showing retention rising after onboarding updates.",
                    )
                ],
                metadata={"source_modality": "image"},
            )
        ]
        engine = CorpusEngine.build(
            AppConfig(openrouter_multimodal_embedding_model="nvidia/llama-nemotron-embed-vl-1b-v2:free"),
            documents,
            provider=FakeProvider(),
            reasoning_provider=FakeReasoningProvider(),
        )
        snapshot = engine.snapshot()
        self.assertEqual(snapshot["embedding_strategy"], "multimodal-unified")
        self.assertEqual(snapshot["embedding_model"], "nvidia/llama-nemotron-embed-vl-1b-v2:free")


if __name__ == "__main__":
    unittest.main()
