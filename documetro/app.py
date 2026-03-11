from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import AppConfig
from .service import CorpusService


class QueryPayload(BaseModel):
    question: str


class ConfigPayload(BaseModel):
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embedding_model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2:free"
    openrouter_multimodal_embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    openrouter_multimodal_model: str = "openai/gpt-4o-mini"
    openrouter_generation_model: str = "liquid/lfm2-8b-a1b"
    nous_api_key: str = ""
    nous_base_url: str = "https://inference-api.nousresearch.com/v1"
    nous_reasoning_model: str = "Hermes-4-70B"


def create_app() -> FastAPI:
    config = AppConfig.from_env()
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent
    static_dir = package_dir / "static"
    assets_dir = project_root / "assets"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = CorpusService(config)
        app.state.config = config
        app.state.service = service
        yield
        service.shutdown()

    app = FastAPI(title=config.app_name, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health(request: Request) -> dict[str, object]:
        config = request.app.state.config
        return {
            "name": config.app_name,
            "host": config.host,
            "port": config.port,
            "status": "ok",
        }

    @app.get("/api/status")
    def status(request: Request) -> dict[str, object]:
        service: CorpusService = request.app.state.service
        return service.snapshot()

    @app.get("/api/config")
    def config_snapshot(request: Request) -> dict[str, object]:
        service: CorpusService = request.app.state.service
        return service.config_snapshot()

    @app.post("/api/config")
    def update_config(request: Request, payload: ConfigPayload) -> dict[str, object]:
        service: CorpusService = request.app.state.service
        return service.update_runtime_config(payload.model_dump())

    @app.post("/api/upload")
    def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> dict[str, object]:
        service: CorpusService = request.app.state.service
        return service.stage_uploads(files)

    @app.post("/api/query")
    def query_documents(request: Request, payload: QueryPayload) -> dict[str, object]:
        service: CorpusService = request.app.state.service
        config: AppConfig = request.app.state.config
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        if len(question) > config.max_question_chars:
            raise HTTPException(
                status_code=400,
                detail=f"Question exceeds the {config.max_question_chars} character limit.",
            )
        return asdict(service.ask(question))

    @app.post("/api/reset")
    def reset_corpus(request: Request) -> dict[str, object]:
        service: CorpusService = request.app.state.service
        return service.reset()

    return app
