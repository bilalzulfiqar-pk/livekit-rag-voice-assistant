import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api.router import api_router
from app.chat.reranker import build_chat_reranker
from app.core.config import settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import register_request_middleware
from app.core.readiness import readiness_state
from app.db.init_db import close_database_engine, initialize_database
from app.embeddings.provider import warm_configured_embedding_provider

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


def _log_warmup_task_result(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        logger.info("Local embedding warmup task was cancelled during shutdown.")
    except Exception:
        logger.exception("Local embedding warmup failed.")


@asynccontextmanager
async def lifespan(app: FastAPI | None):
    state_holder = app.state if app is not None else SimpleNamespace(chat_reranker=None)
    logger.info("Starting %s in %s mode", settings.app_name, settings.app_env)
    flashrank_warmup_required = settings.flashrank_enabled and settings.flashrank_warmup_enabled
    readiness_state.begin_startup(
        requires_local_embedding=settings.embedding_provider == "local" and settings.local_embedding_warmup_enabled,
        embedding_runtime=settings.local_embedding_runtime if settings.embedding_provider == "local" else None,
        requires_flashrank_warmup=flashrank_warmup_required,
        flashrank_model=settings.flashrank_model if flashrank_warmup_required else None,
    )
    await initialize_database()
    readiness_state.mark_database_ready()
    state_holder.chat_reranker = build_chat_reranker(
        enabled=settings.flashrank_enabled,
        model_name=settings.flashrank_model,
        cache_dir=settings.flashrank_cache_dir,
    )
    if flashrank_warmup_required:
        try:
            readiness_state.mark_flashrank_warming(model_name=settings.flashrank_model)
            await state_holder.chat_reranker.warmup()
            readiness_state.mark_flashrank_ready(model_name=settings.flashrank_model)
        except Exception:
            readiness_state.mark_flashrank_failed(
                "Service is ready, but FlashRank warmup failed. Reranking may fall back to fast mode.",
                model_name=settings.flashrank_model,
            )
            logger.exception("FlashRank warmup failed. The backend will continue and fall back to fast mode if needed.")
    else:
        readiness_state.mark_flashrank_skipped()
    warmup_task: asyncio.Task[None] | None = None
    if settings.embedding_provider == "local" and settings.local_embedding_warmup_enabled:
        if settings.local_embedding_warmup_mode == "blocking":
            readiness_state.mark_embedding_warming(runtime=settings.local_embedding_runtime)
            try:
                await warm_configured_embedding_provider()
            except Exception as exc:
                readiness_state.mark_embedding_failed(str(exc), runtime=settings.local_embedding_runtime)
                raise
            readiness_state.mark_embedding_ready(runtime=settings.local_embedding_runtime)
        else:
            readiness_state.mark_embedding_warming(runtime=settings.local_embedding_runtime)

            async def background_warmup() -> None:
                try:
                    await warm_configured_embedding_provider()
                except Exception as exc:
                    readiness_state.mark_embedding_failed(str(exc), runtime=settings.local_embedding_runtime)
                    raise
                readiness_state.mark_embedding_ready(runtime=settings.local_embedding_runtime)

            warmup_task = asyncio.create_task(background_warmup())
            warmup_task.add_done_callback(_log_warmup_task_result)
    else:
        readiness_state.mark_embedding_skipped()
    yield
    if warmup_task is not None and not warmup_task.done():
        warmup_task.cancel()
        with suppress(asyncio.CancelledError):
            await warmup_task
    readiness_state.mark_database_not_ready()
    await close_database_engine()
    logger.info("Shutting down %s", settings.app_name)


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        default_response_class=ORJSONResponse,
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        docs_url=f"{settings.api_v1_prefix}/docs",
        redoc_url=f"{settings.api_v1_prefix}/redoc",
    )
    app.state.chat_reranker = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_request_middleware(app)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_application()
