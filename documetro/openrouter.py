from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
import json
import mimetypes
import urllib.error
import urllib.request

from .config import AppConfig
from .models import Evidence


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base_url = config.openrouter_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return self.config.openrouter_enabled

    def embed(self, inputs: list[object], model: str | None = None) -> list[list[float]]:
        payload = {
            "model": model or self.config.openrouter_embedding_model,
            "input": inputs,
            "encoding_format": "float",
        }
        data = self._post_json("/embeddings", payload)
        rows = data.get("data", [])
        embeddings = [item.get("embedding", []) for item in rows]
        if len(embeddings) != len(inputs):
            raise OpenRouterError("Embedding response size did not match the request.")
        return embeddings

    def summarize_file(self, path: Path) -> str:
        prompt = (
            "Convert this file into retrieval-ready notes. "
            "Extract visible or audible facts, entities, numbers, labels, tables, timestamps, scenes, and any user-facing text. "
            "Return plain text only."
        )
        content = [{"type": "text", "text": prompt}]
        content.extend(self._file_content(path))
        plugins = None
        if path.suffix.lower() == ".pdf":
            plugins = [{"id": "file-parser", "pdf": {"engine": "pdf-text"}}]
        return self.chat(
            model=self.config.openrouter_multimodal_model,
            system_prompt="You are a precise multimodal extraction engine.",
            user_content=content,
            max_tokens=900,
            plugins=plugins,
        )

    def answer_question(self, question: str, evidence: list[Evidence], related_topics: list[str]) -> str:
        evidence_lines = []
        for item in evidence[:8]:
            evidence_lines.append(
                f"[{item.document_name}] ({item.locator}) - {item.section_title}: {item.excerpt}"
            )
        topics = ", ".join(related_topics[:4]) if related_topics else "None"
        prompt = "\n".join(
            [
                f"Question: {question}",
                f"Related Topics: {topics}",
                "",
                "=== EVIDENCE FROM DOCUMENTS ===",
                *evidence_lines,
                "",
                "=== INSTRUCTIONS ===",
                "Analyze the evidence above and answer the question.",
                "- Base your answer ONLY on the provided evidence.",
                "- Be specific and cite document names when relevant.",
                "- If evidence is insufficient, clearly state what information is missing.",
                "- Keep the answer natural and readable rather than template-like.",
            ]
        )
        return self.chat(
            model=self.config.openrouter_generation_model,
            system_prompt=(
                "You are an expert document analyst. Answer naturally, like a strong human assistant, not like a template engine. "
                "Use only the provided evidence from the documents. "
                "If the evidence is incomplete or insufficient, say that plainly. "
                "Be specific, avoid filler, avoid mechanical lead-ins, and avoid listing timestamps unless they matter."
            ),
            user_content=prompt,
            max_tokens=600,
        )

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        max_tokens: int,
        plugins: list[dict[str, Any]] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "max_tokens": max_tokens,
        }
        if plugins:
            payload["plugins"] = plugins
        data = self._post_json("/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("OpenRouter returned no choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part.strip() for part in parts if part.strip()).strip()
        raise OpenRouterError("OpenRouter returned an unsupported message payload.")

    def _file_content(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return [{"type": "image_url", "image_url": {"url": self._data_url(path)}}]
        if suffix in {".wav", ".mp3", ".aiff", ".aac", ".ogg", ".flac", ".m4a"}:
            return [{"type": "input_audio", "input_audio": {"data": self._base64(path), "format": suffix.lstrip(".")}}]
        if suffix in {".mp4", ".mpeg", ".mov", ".webm"}:
            return [{"type": "video_url", "video_url": {"url": self._data_url(path)}}]
        if suffix == ".pdf":
            return [{"type": "file", "file": {"filename": path.name, "file_data": self._data_url(path)}}]
        return [{"type": "file", "file": {"filename": path.name, "file_data": self._data_url(path)}}]

    def _data_url(self, path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return f"data:{mime_type};base64,{self._base64(path)}"

    def _base64(self, path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _post_json(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise OpenRouterError("OpenRouter is not configured.")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{route}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://documetro.local",
                "X-Title": self.config.app_name,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.openrouter_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc.reason}") from exc
