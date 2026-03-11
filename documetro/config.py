from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


@dataclass(slots=True)
class AppConfig:
    app_name: str = "Documetro"
    runtime_prefix: str = "documetro-session-"
    max_files_per_upload: int = 24
    max_upload_bytes: int = 64 * 1024 * 1024
    max_question_chars: int = 1200
    vector_dimensions: int = 768
    latent_dimensions: int = 48
    host: str = "127.0.0.1"
    port: int = 8421
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embedding_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2:free"
    openrouter_multimodal_embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    openrouter_multimodal_model: str = "openai/gpt-4o-mini"
    openrouter_generation_model: str = "liquid/lfm2-8b-a1b"
    openrouter_timeout_seconds: int = 90
    nous_api_key: str = ""
    nous_base_url: str = "https://inference-api.nousresearch.com/v1"
    nous_reasoning_model: str = "Hermes-4-70B"
    nous_timeout_seconds: int = 90

    @property
    def env_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / ".env"

    @property
    def temp_root(self) -> Path:
        return Path(tempfile.gettempdir())

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key.strip())

    @property
    def nous_enabled(self) -> bool:
        return bool(self.nous_api_key.strip())

    def persist_runtime_settings(self, updates: dict[str, str]) -> None:
        allowed = {
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            "OPENROUTER_EMBEDDING_MODEL",
            "OPENROUTER_MULTIMODAL_EMBEDDING_MODEL",
            "OPENROUTER_MULTIMODAL_MODEL",
            "OPENROUTER_GENERATION_MODEL",
            "NOUS_API_KEY",
            "NOUS_BASE_URL",
            "NOUS_REASONING_MODEL",
        }
        current = _read_dotenv(self.env_path)
        for key, value in updates.items():
            if key in allowed:
                current[key] = str(value).strip()
        lines = [f"{key}={value}" for key, value in sorted(current.items()) if str(value).strip()]
        self.env_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @classmethod
    def from_env(cls) -> "AppConfig":
        env_path = Path(__file__).resolve().parent.parent / ".env"
        dotenv = _read_dotenv(env_path)
        host = os.getenv("DOCUMETRO_HOST", os.getenv("HOST", "127.0.0.1"))
        port = int(os.getenv("DOCUMETRO_PORT", os.getenv("PORT", "8421")))
        return cls(
            host=host,
            port=port,
            openrouter_api_key=dotenv.get("OPENROUTER_API_KEY", "").strip(),
            openrouter_base_url=dotenv.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            openrouter_embedding_model=dotenv.get(
                "OPENROUTER_EMBEDDING_MODEL",
                "nvidia/llama-3.2-nv-embedqa-1b-v2:free",
            ).strip(),
            openrouter_multimodal_embedding_model=dotenv.get(
                "OPENROUTER_MULTIMODAL_EMBEDDING_MODEL",
                "nvidia/llama-nemotron-embed-vl-1b-v2:free",
            ).strip(),
            openrouter_multimodal_model=dotenv.get("OPENROUTER_MULTIMODAL_MODEL", "openai/gpt-4o-mini").strip(),
            openrouter_generation_model=dotenv.get("OPENROUTER_GENERATION_MODEL", "liquid/lfm2-8b-a1b").strip(),
            openrouter_timeout_seconds=int(dotenv.get("OPENROUTER_TIMEOUT_SECONDS", "90")),
            nous_api_key=dotenv.get("NOUS_API_KEY", "").strip(),
            nous_base_url=dotenv.get("NOUS_BASE_URL", "https://inference-api.nousresearch.com/v1").rstrip("/"),
            nous_reasoning_model=dotenv.get("NOUS_REASONING_MODEL", "Hermes-4-70B").strip(),
            nous_timeout_seconds=int(dotenv.get("NOUS_TIMEOUT_SECONDS", "90")),
        )
