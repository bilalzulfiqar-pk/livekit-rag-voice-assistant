import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.retrieval.manager import RetrievalManager
from app.retrieval.schemas import (
    RetrievalContextRequest,
    RetrievalContextResponse,
    RetrievalContextExcerpt,
    RetrievalContextLatency,
    RetrievalContextRef,
    RetrievalRequest,
    RetrievalResponse,
)
from app.retrieval.service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
logger = logging.getLogger(__name__)


@router.post("/search", response_model=RetrievalResponse, status_code=status.HTTP_200_OK)
async def search_chunks(
    payload: RetrievalRequest,
    session: AsyncSession = Depends(get_db_session),
) -> RetrievalResponse:
    service = RetrievalService(session)

    try:
        return await service.search(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while searching chunks.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while retrieving chunks.",
        ) from exc


@router.post("/context", response_model=RetrievalContextResponse, status_code=status.HTTP_200_OK)
async def get_retrieval_context(
    payload: RetrievalContextRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> RetrievalContextResponse:
    manager = RetrievalManager(session, reranker=request.app.state.chat_reranker)

    try:
        prepared = await manager.prepare_context(
            user_question=payload.query,
            top_k=payload.top_k,
            document_id=payload.document_id,
            retrieval_mode=payload.retrieval_mode,
            rerank_strategy=payload.rerank_strategy,
            include_debug=payload.include_debug,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while preparing retrieval context.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while preparing retrieval context.",
        ) from exc

    has_sufficient_context = bool(prepared.context_chunks)
    if has_sufficient_context:
        message = "Context excerpts returned."
    else:
        message = "No sufficient context was found for this query."

    return RetrievalContextResponse(
        query=prepared.question,
        returned_count=len(prepared.context_chunks),
        message=message,
        retrieval_mode=manager._resolve_retrieval_mode(payload.retrieval_mode),
        rerank_strategy=manager._resolve_rerank_strategy(payload.rerank_strategy),
        has_sufficient_context=has_sufficient_context,
        context_excerpts=[
            RetrievalContextExcerpt(
                source_id=f"document:{chunk.document_id}",
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                similarity_score=chunk.similarity_score,
                section_anchor=str(prepared.retrieval_matches[index].metadata.get("section_anchor") or "") or None,
                chunk_text=chunk.chunk_text,
            )
            for index, chunk in enumerate(prepared.context_chunks)
        ],
        context_refs=[
            RetrievalContextRef(
                source_id=f"document:{chunk.document_id}",
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                similarity_score=chunk.similarity_score,
                section_anchor=str(prepared.retrieval_matches[index].metadata.get("section_anchor") or "") or None,
            )
            for index, chunk in enumerate(prepared.context_refs)
        ],
        latency=RetrievalContextLatency(
            retrieval=prepared.latency.retrieval,
            prompt_build_ms=prepared.latency.prompt_build_ms,
            preparation_ms=prepared.latency.total_ms,
            rerank_ms=prepared.latency.rerank_ms,
            support_retrieval_ms=prepared.latency.support_retrieval_ms,
            neighbor_retrieval_ms=prepared.latency.neighbor_retrieval_ms,
            candidate_fusion_ms=prepared.latency.candidate_fusion_ms,
        ),
        debug_trace=prepared.debug_trace.model_dump() if payload.include_debug and prepared.debug_trace else None,
    )
