from __future__ import annotations

import io
import time
import unittest

from fastapi import UploadFile

from documetro.config import AppConfig
from documetro.service import CorpusService


class FakeProvider:
    enabled = True

    def summarize_file(self, path):  # pragma: no cover - not used here
        return ""

    def embed(self, inputs, model=None):
        return [[1.0 if "launch" in str(item).lower() else 0.0, 1.0 if "maya" in str(item).lower() else 0.0] for item in inputs]

    def answer_question(self, question, evidence, related_topics):
        return "The launch window opens on 2026-04-12, and the owner is Maya."


class FakeReasoningProvider:
    enabled = True

    def plan_query(self, question, document_names, topic_labels):
        return {
            "intent": "fact",
            "search_query": "launch window schedule owner",
            "keywords": ["launch window"],
            "related_terms": ["owner"],
        }

    def rerank_evidence(self, question, candidates):
        return [item["id"] for item in candidates]


class ApiTests(unittest.TestCase):
    def test_upload_query_reset_flow(self) -> None:
        service = CorpusService(AppConfig(), provider=FakeProvider(), reasoning_provider=FakeReasoningProvider())
        try:
            result = service.stage_uploads(
                [
                    UploadFile(
                        filename="brief.txt",
                        file=io.BytesIO(b"The launch window opens on 2026-04-12. The owner is Maya."),
                    )
                ]
            )
            self.assertEqual(result["accepted"], ["brief.txt"])

            ready = False
            for _ in range(40):
                status = service.snapshot()
                if status["state"] == "ready" and status["document_count"] == 1:
                    ready = True
                    break
                time.sleep(0.05)
            self.assertTrue(ready)
            status = service.snapshot()
            self.assertEqual(status["reasoning_provider"], "nous")
            self.assertTrue(status["reasoning_model"])
            self.assertTrue(status["embedding_model"])

            answer = service.ask("When does the launch window open?")
            self.assertIn("2026-04-12", answer.answer)
            self.assertEqual(answer.title, "Answer")
            self.assertEqual(answer.template, "llm")

            reset = service.reset()
            self.assertEqual(reset["document_count"], 0)
        finally:
            service.shutdown()


if __name__ == "__main__":
    unittest.main()
