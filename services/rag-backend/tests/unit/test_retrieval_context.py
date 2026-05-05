import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.api.routes.retrieval import get_retrieval_context
from app.chat.schemas import ChatRequest
from app.chat.service import ChatService
from app.chat.types import ChatPreparationLatency, PreparedChat
from app.chat.guardrails import route_query
from app.retrieval.schemas import (
    RetrievalContextRequest,
    RetrievalLatency,
    RetrievalMatch,
)


def _build_match() -> RetrievalMatch:
    return RetrievalMatch(
        chunk_id=11,
        document_id=1,
        filename="Guide To Benefits.pdf",
        chunk_index=0,
        chunk_text="Coverage includes damage or theft up to $75,000.",
        metadata={"section_anchor": "Auto Rental Coverage"},
        similarity_score=0.91,
    )


def _build_prepared_chat(*, has_context: bool) -> PreparedChat:
    query_route = route_query("What is covered under auto rental coverage?")
    retrieval_match = _build_match()
    context_chunks = []
    context_refs = []
    retrieval_matches = []
    fallback_answer = "I'm sorry, I don't have that information in my records."
    if has_context:
        retrieval_matches = [retrieval_match]
        context_chunks = [
            SimpleNamespace(
                chunk_id=retrieval_match.chunk_id,
                document_id=retrieval_match.document_id,
                filename=retrieval_match.filename,
                chunk_index=retrieval_match.chunk_index,
                chunk_text=retrieval_match.chunk_text,
                similarity_score=retrieval_match.similarity_score,
            )
        ]
        context_refs = [
            SimpleNamespace(
                chunk_id=retrieval_match.chunk_id,
                document_id=retrieval_match.document_id,
                filename=retrieval_match.filename,
                chunk_index=retrieval_match.chunk_index,
                similarity_score=retrieval_match.similarity_score,
            )
        ]
        fallback_answer = None

    return PreparedChat(
        question="What is covered under auto rental coverage?",
        query_route=query_route,
        system_prompt="system",
        prompt="prompt",
        provider="context-only",
        fallback_answer=fallback_answer,
        retrieval_matches=retrieval_matches,
        context_refs=context_refs,
        context_chunks=context_chunks,
        include_debug=False,
        debug_trace=None,
        latency=ChatPreparationLatency(
            retrieval=RetrievalLatency(
                document_lookup_ms=None,
                query_embedding_ms=9.0,
                vector_search_ms=18.0,
                total_ms=27.0,
                vector_hydration_ms=3.0,
                sparse_search_ms=4.0,
                lexical_rescue_ms=1.0,
                retrieval_fusion_ms=2.0,
            ),
            prompt_build_ms=6.0,
            rerank_ms=5.0,
            total_ms=40.0,
            support_retrieval_ms=4.0,
            neighbor_retrieval_ms=3.0,
            candidate_fusion_ms=2.0,
        ),
    )


class RetrievalContextRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_retrieval_context_returns_context_only_shape(self) -> None:
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chat_reranker=None)))
        payload = RetrievalContextRequest(
            query="What is covered under auto rental coverage?",
            top_k=3,
            retrieval_mode="ann_rerank",
            rerank_strategy="hybrid",
        )

        with patch(
            "app.api.routes.retrieval.RetrievalManager.prepare_context",
            new=AsyncMock(return_value=_build_prepared_chat(has_context=True)),
        ) as prepare_context:
            response = await get_retrieval_context(payload, request, session=Mock())

        self.assertTrue(response.has_sufficient_context)
        self.assertEqual(response.retrieval_mode, "ann_rerank")
        self.assertEqual(response.rerank_strategy, "hybrid")
        self.assertEqual(response.context_refs[0].source_id, "document:1")
        self.assertEqual(response.context_refs[0].section_anchor, "Auto Rental Coverage")
        prepare_context.assert_awaited_once()

    async def test_get_retrieval_context_reports_no_context(self) -> None:
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chat_reranker=None)))
        payload = RetrievalContextRequest(query="What card is this guide for?", top_k=3)

        with patch(
            "app.api.routes.retrieval.RetrievalManager.prepare_context",
            new=AsyncMock(return_value=_build_prepared_chat(has_context=False)),
        ):
            response = await get_retrieval_context(payload, request, session=Mock())

        self.assertFalse(response.has_sufficient_context)
        self.assertEqual(response.returned_count, 0)
        self.assertEqual(response.message, "No sufficient context was found for this query.")


class ChatServiceDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_chat_delegates_to_retrieval_manager(self) -> None:
        prepared_chat = _build_prepared_chat(has_context=True)
        provider = Mock(display_name="groq:llama-3.1-8b-instant")

        with patch("app.chat.service.get_chat_provider", return_value=provider):
            service = ChatService(session=Mock())

        with patch(
            "app.retrieval.manager.RetrievalManager.prepare_context",
            new=AsyncMock(return_value=prepared_chat),
        ) as prepare_context:
            result = await service.prepare_chat(
                ChatRequest(
                    question="What is covered under auto rental coverage?",
                    top_k=3,
                    retrieval_mode="ann_rerank",
                    rerank_strategy="hybrid",
                )
            )

        self.assertIs(result, prepared_chat)
        prepare_context.assert_awaited_once_with(
            user_question="What is covered under auto rental coverage?",
            top_k=3,
            document_id=None,
            retrieval_mode="ann_rerank",
            rerank_strategy="hybrid",
            include_debug=False,
            provider_display_name="groq:llama-3.1-8b-instant",
        )


if __name__ == "__main__":
    unittest.main()
