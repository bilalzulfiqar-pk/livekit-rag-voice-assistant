from __future__ import annotations

import logging
from time import perf_counter

import httpx
from livekit.agents.llm import Toolset, function_tool

from voice_agent.telemetry import VoiceAgentTelemetry
from voice_agent.tools.text_utils import sanitize_tool_text

logger = logging.getLogger("livekit-rag-voice-agent.rag-tool")

RAG_FAILURE_MESSAGE = "I couldn't access the knowledge base right now."


class KnowledgeBaseToolset(Toolset):
    def __init__(
        self,
        *,
        backend_url: str,
        chat_path: str,
        telemetry: VoiceAgentTelemetry | None = None,
        timeout_seconds: float = 6.0,
    ) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
        self._telemetry = telemetry
        self._timeout = httpx.Timeout(
            timeout_seconds,
            connect=min(2.0, timeout_seconds),
            read=timeout_seconds,
            write=timeout_seconds,
            pool=timeout_seconds,
        )
        self._client: httpx.AsyncClient | None = None
        super().__init__(id="knowledge_base")

    async def setup(self) -> KnowledgeBaseToolset:
        await self._ensure_client()
        await super().setup()
        return self

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await super().aclose()

    @property
    def endpoint_url(self) -> str:
        return f"{self._backend_url}{self._chat_path}"

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    @function_tool(
        description=(
            "Use this tool only for company, FAQ, policy, support, and "
            "uploaded-document questions. Use it for questions about the uploaded "
            "guide or PDF too, including phrases like 'this guide' or 'this document'. "
            "Pass the user's question as-is."
        )
    )
    async def ask_knowledge_base(self, question: str) -> str:
        """Fetch a grounded answer from the RAG backend.

        Args:
            question: The user's exact question about company, FAQ, policy, support, or uploaded documents.
        """

        cleaned_question = question.strip()
        if not cleaned_question:
            return RAG_FAILURE_MESSAGE

        if self._telemetry is not None:
            self._telemetry.publish_kb_querying()

        client = await self._ensure_client()
        start_time = perf_counter()
        try:
            response = await client.post(
                self.endpoint_url,
                json={
                    "question": cleaned_question,
                    "top_k": 3,
                    "include_debug": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("RAG backend request failed", exc_info=exc)
            if self._telemetry is not None:
                self._telemetry.publish_kb_result(
                    success=False,
                    latency_ms=round((perf_counter() - start_time) * 1000),
                    fallback=True,
                )
            return RAG_FAILURE_MESSAGE

        fallback_used = str(payload.get("answer_path", "")).lower() == "fallback"
        if self._telemetry is not None:
            self._telemetry.publish_kb_result(
                success=True,
                latency_ms=round((perf_counter() - start_time) * 1000),
                fallback=fallback_used,
            )

        answer = sanitize_tool_text(str(payload.get("answer", "")))
        return answer or RAG_FAILURE_MESSAGE
