from __future__ import annotations

from collections import deque
from pathlib import Path
import os
import shutil
import tempfile
import threading
import time

from fastapi import UploadFile

from .config import AppConfig
from .engine import CorpusEngine
from .extractors import DocumentExtractor, ExtractionError
from .models import AnswerResult, ExtractedDocument
from .nous import NousClient
from .openrouter import OpenRouterClient
from .utils import digest_file, safe_filename, utc_now


class CorpusService:
    def __init__(
        self,
        config: AppConfig,
        provider: OpenRouterClient | None = None,
        reasoning_provider: NousClient | None = None,
    ) -> None:
        self.config = config
        self.provider = provider if provider is not None else (OpenRouterClient(config) if config.openrouter_enabled else None)
        self.reasoning_provider = (
            reasoning_provider if reasoning_provider is not None else (NousClient(config) if config.nous_enabled else None)
        )
        self.extractor = DocumentExtractor(config, provider=self.provider)
        self._lock = threading.RLock()
        self._queue: deque[tuple[int, list[Path]]] = deque()
        self._worker: threading.Thread | None = None
        self._generation = 0
        self._documents: dict[str, ExtractedDocument] = {}
        self._index: CorpusEngine | None = None
        self._recent_errors: list[str] = []
        self.runtime_dir = self._new_runtime_dir()
        self.upload_dir = self.runtime_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.status: dict[str, object] = self._base_status("Add files to build a temporary local index.")

    def _base_status(self, message: str) -> dict[str, object]:
        return {
            "state": "idle",
            "phase": "ready",
            "progress": 0.0,
            "message": message,
            "queued_files": 0,
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "provider": "openrouter" if self.provider else "local",
            "embedding_model": self.config.openrouter_embedding_model if self.provider else "",
            "embedding_strategy": "pending" if self.provider else "disabled",
            "generation_model": self.config.openrouter_generation_model if self.provider else "",
            "reasoning_provider": "nous" if self.reasoning_provider else "",
            "reasoning_model": self.config.nous_reasoning_model if self.reasoning_provider else "",
        }

    def config_snapshot(self) -> dict[str, str]:
        return {
            "openrouter_api_key": self.config.openrouter_api_key,
            "openrouter_base_url": self.config.openrouter_base_url,
            "openrouter_embedding_model": self.config.openrouter_embedding_model,
            "openrouter_multimodal_embedding_model": self.config.openrouter_multimodal_embedding_model,
            "openrouter_multimodal_model": self.config.openrouter_multimodal_model,
            "openrouter_generation_model": self.config.openrouter_generation_model,
            "nous_api_key": self.config.nous_api_key,
            "nous_base_url": self.config.nous_base_url,
            "nous_reasoning_model": self.config.nous_reasoning_model,
        }

    def update_runtime_config(self, updates: dict[str, str]) -> dict[str, object]:
        mapping = {
            "openrouter_api_key": "OPENROUTER_API_KEY",
            "openrouter_base_url": "OPENROUTER_BASE_URL",
            "openrouter_embedding_model": "OPENROUTER_EMBEDDING_MODEL",
            "openrouter_multimodal_embedding_model": "OPENROUTER_MULTIMODAL_EMBEDDING_MODEL",
            "openrouter_multimodal_model": "OPENROUTER_MULTIMODAL_MODEL",
            "openrouter_generation_model": "OPENROUTER_GENERATION_MODEL",
            "nous_api_key": "NOUS_API_KEY",
            "nous_base_url": "NOUS_BASE_URL",
            "nous_reasoning_model": "NOUS_REASONING_MODEL",
        }
        persisted: dict[str, str] = {}
        for field, env_key in mapping.items():
            if field in updates:
                value = str(updates.get(field, "")).strip()
                setattr(self.config, field, value)
                persisted[env_key] = value
        self.config.persist_runtime_settings(persisted)
        runtime_dir = self.runtime_dir
        with self._lock:
            self._generation += 1
            self._queue.clear()
            self._documents.clear()
            self._index = None
            self._recent_errors.clear()
            self.provider = OpenRouterClient(self.config) if self.config.openrouter_enabled else None
            self.reasoning_provider = NousClient(self.config) if self.config.nous_enabled else None
            self.extractor = DocumentExtractor(self.config, provider=self.provider)
            self.runtime_dir = self._new_runtime_dir()
            self.upload_dir = self.runtime_dir / "uploads"
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            self.status = self._base_status("Provider settings updated. Add files to build a new index.")
        shutil.rmtree(runtime_dir, ignore_errors=True)
        return {"config": self.config_snapshot(), "status": self.snapshot()}

    def _new_runtime_dir(self) -> Path:
        self.cleanup_stale_runtime_dirs(prefix=self.config.runtime_prefix)
        path = Path(tempfile.mkdtemp(prefix=self.config.runtime_prefix))
        return path

    @staticmethod
    def cleanup_stale_runtime_dirs(prefix: str) -> None:
        temp_root = Path(tempfile.gettempdir())
        for entry in temp_root.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                shutil.rmtree(entry, ignore_errors=True)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            details = dict(self.status)
            details["recent_errors"] = list(self._recent_errors)
            if self._index is None:
                details["document_count"] = len(self._documents)
                details["section_count"] = sum(len(document.sections) for document in self._documents.values())
                details["chunk_count"] = 0
                details["documents"] = []
                details["topics"] = []
                details["relationships"] = []
            else:
                details.update(self._index.snapshot())
            return details

    def stage_uploads(self, uploads: list[UploadFile]) -> dict[str, object]:
        if not uploads:
            raise ValueError("No files were uploaded.")
        if len(uploads) > self.config.max_files_per_upload:
            raise ValueError(f"Upload up to {self.config.max_files_per_upload} files at a time.")
        staged: list[Path] = []
        skipped: list[str] = []
        with self._lock:
            known_checksums = {document.checksum for document in self._documents.values()}
        for upload in uploads:
            path = self._save_upload(upload)
            checksum = digest_file(path)
            if checksum in known_checksums:
                skipped.append(upload.filename or path.name)
                path.unlink(missing_ok=True)
                continue
            staged.append(path)
            known_checksums.add(checksum)
        if staged:
            self._enqueue(staged)
        return {
            "accepted": [path.name for path in staged],
            "skipped": skipped,
            "status": self.snapshot(),
        }

    def ask(self, question: str) -> AnswerResult:
        with self._lock:
            index = self._index
        if index is None:
            return AnswerResult(
                answer="Upload at least one readable document before asking questions.",
                confidence=0.0,
                mode="no-index",
                template="empty",
                title="No corpus yet",
                lead="Upload at least one readable document before asking questions.",
                bullets=[],
                note="",
                quote="",
                quote_source="",
                steps=[],
                evidence=[],
                related_topics=[],
            )
        return index.answer(question)

    def reset(self) -> dict[str, object]:
        with self._lock:
            self._generation += 1
            self._queue.clear()
            self._documents.clear()
            self._index = None
            self._recent_errors.clear()
            runtime_dir = self.runtime_dir
            self.runtime_dir = self._new_runtime_dir()
            self.upload_dir = self.runtime_dir / "uploads"
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            self.status = self._base_status("Workspace cleared. Add files to build a new index.")
        shutil.rmtree(runtime_dir, ignore_errors=True)
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            self._generation += 1
            self._queue.clear()
            runtime_dir = self.runtime_dir
        shutil.rmtree(runtime_dir, ignore_errors=True)

    def _save_upload(self, upload: UploadFile) -> Path:
        filename = safe_filename(upload.filename or "upload")
        target = self.upload_dir / filename
        suffix = 1
        while target.exists():
            target = self.upload_dir / f"{target.stem}-{suffix}{target.suffix}"
            suffix += 1
        size = 0
        with target.open("wb") as handle:
            upload.file.seek(0)
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > self.config.max_upload_bytes:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise ValueError(
                        f"{upload.filename or filename} exceeds the {self.config.max_upload_bytes // (1024 * 1024)} MB limit."
                    )
                handle.write(chunk)
        return target

    def _enqueue(self, paths: list[Path]) -> None:
        with self._lock:
            generation = self._generation
            self._queue.append((generation, paths))
            self.status.update(
                {
                    "state": "queued",
                    "phase": "waiting",
                    "queued_files": sum(len(item[1]) for item in self._queue),
                    "updated_at": utc_now(),
                    "message": "Files queued. Building the index.",
                }
            )
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drain_queue, name="documetro-worker", daemon=True)
                self._worker.start()

    def _drain_queue(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self.status.update(
                        {
                            "state": "ready" if self._index else "idle",
                            "phase": "ready",
                            "queued_files": 0,
                            "progress": 1.0 if self._index else 0.0,
                            "updated_at": utc_now(),
                        }
                    )
                    return
                generation, paths = self._queue.popleft()
                self.status.update(
                    {
                        "state": "processing",
                        "phase": "extracting",
                        "queued_files": sum(len(item[1]) for item in self._queue),
                        "progress": 0.0,
                        "updated_at": utc_now(),
                    }
                )
            self._process_batch(generation, paths)

    def _process_batch(self, generation: int, paths: list[Path]) -> None:
        extracted: list[ExtractedDocument] = []
        for index, path in enumerate(paths, start=1):
            with self._lock:
                if generation != self._generation:
                    return
                self.status.update(
                    {
                        "message": f"Extracting {path.name}",
                        "phase": "extracting",
                        "progress": round((index - 1) / max(len(paths), 1), 3),
                        "updated_at": utc_now(),
                    }
                )
            try:
                extracted.append(self.extractor.extract(path))
            except (ExtractionError, ValueError, OSError) as exc:
                self._push_error(f"{path.name}: {exc}")
        with self._lock:
            if generation != self._generation:
                return
            merged = dict(self._documents)
            for document in extracted:
                merged[document.document_id] = document
            documents = list(merged.values())
            self.status.update(
                {
                    "phase": "indexing",
                    "progress": 0.82 if extracted else self.status.get("progress", 0.0),
                    "message": "Rebuilding the local index.",
                    "updated_at": utc_now(),
                }
            )
        index = CorpusEngine.build(
            self.config,
            documents,
            provider=self.provider,
            reasoning_provider=self.reasoning_provider,
        )
        with self._lock:
            if generation != self._generation:
                return
            self._documents = {document.document_id: document for document in documents}
            self._index = index
            self.status.update(
                {
                    "state": "ready",
                    "phase": "ready",
                    "progress": 1.0 if documents else 0.0,
                    "message": "Ready for questions." if documents else "No readable files indexed.",
                    "updated_at": utc_now(),
                }
            )

    def _push_error(self, message: str) -> None:
        with self._lock:
            self._recent_errors.append(message)
            self._recent_errors = self._recent_errors[-8:]
            self.status["updated_at"] = utc_now()
