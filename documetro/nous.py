from __future__ import annotations

from typing import Any
import json
import re
import urllib.error
import urllib.request

from .config import AppConfig
from .models import Evidence


class NousError(RuntimeError):
    pass


class NousClient:
    THINKING_SYSTEM_PROMPT = (
        "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem "
        "and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior "
        "to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and "
        "then provide your solution or response to the problem."
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base_url = config.nous_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return self.config.nous_enabled

    def plan_query(self, question: str, document_names: list[str], topic_labels: list[str]) -> dict[str, object]:
        prompt = "\n".join(
            [
                "Return JSON only.",
                'Schema: {"intent":"", "search_query":"", "keywords":[], "related_terms":[], "filters":[]}.',
                "Keep keywords and related_terms short, specific, and retrieval-oriented.",
                "Do not invent facts that are not implied by the question.",
                f"Question: {question}",
                f"Document names: {', '.join(document_names[:12]) or 'None'}",
                f"Known topics: {', '.join(topic_labels[:12]) or 'None'}",
            ]
        )
        raw = self.chat(
            system_prompt=(
                "You are a retrieval planner for document question answering. "
                "Rewrite user questions into precise search intents and semantic retrieval hints."
            ),
            user_content=prompt,
            max_tokens=220,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return self._parse_json_object(raw)

    def rerank_evidence(self, question: str, candidates: list[dict[str, str]]) -> list[str]:
        if not candidates:
            return []
        prompt_lines = [
            "Return JSON only.",
            'Schema: {"ranked_ids":[]}.',
            "Rank the candidates by their usefulness for answering the question from the provided evidence only.",
            f"Question: {question}",
            "Candidates:",
        ]
        for item in candidates[:12]:
            prompt_lines.append(
                f'- id={item["id"]} | title={item["title"]} | locator={item["locator"]} | text={item["text"]}'
            )
        raw = self.chat(
            system_prompt="You are a strict reranker for evidence-grounded retrieval.",
            user_content="\n".join(prompt_lines),
            max_tokens=180,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        payload = self._parse_json_object(raw)
        ranked = payload.get("ranked_ids", [])
        if not isinstance(ranked, list):
            return []
        return [str(item).strip() for item in ranked if str(item).strip()]

    def answer_question(
        self,
        question: str,
        evidence: list[Evidence],
        related_topics: list[str],
        analysis_mode: str = "general",
    ) -> str:
        evidence_lines = []
        for item in evidence[:10]:
            evidence_lines.append(
                f"[{item.document_name}] ({item.locator}) - {item.section_title}: {item.excerpt}"
            )
        topics = ", ".join(related_topics[:5]) if related_topics else "None"
        prompt = "\n".join(
            [
                f"Question: {question}",
                f"Mode: {analysis_mode}",
                f"Related Topics: {topics}",
                "",
                "Evidence:",
                *evidence_lines,
                "",
                "Instructions:",
                "- Answer only from the supplied evidence.",
                "- For meeting or transcript summaries, synthesize the main discussion points instead of echoing timestamps.",
                "- Group related points together and keep the answer readable.",
                "- Mention uncertainty or missing context clearly.",
            ]
        )
        return self.chat(
            system_prompt=(
                "You are a high-precision document reasoning model. "
                "Write like a natural, concise expert assistant rather than a retrieval template. "
                "Synthesize grounded answers from evidence without copying irrelevant timestamp noise."
            ),
            user_content=prompt,
            max_tokens=700,
            temperature=0.2,
        )

    def chat(
        self,
        *,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.nous_reasoning_model,
            "messages": [
                {"role": "system", "content": f"{self.THINKING_SYSTEM_PROMPT}\n\n{system_prompt}"},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        data = self._post_json("/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            raise NousError("Nous returned no choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return self._strip_think_blocks(content)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return self._strip_think_blocks("\n".join(part.strip() for part in parts if part.strip()))
        raise NousError("Nous returned an unsupported message payload.")

    def _parse_json_object(self, raw: str) -> dict[str, object]:
        text = raw.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {}
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                return {}

    def _strip_think_blocks(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    def _post_json(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise NousError("Nous is not configured.")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{route}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.nous_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.nous_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise NousError(f"Nous HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise NousError(f"Nous request failed: {exc.reason}") from exc
