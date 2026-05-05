from __future__ import annotations

import logging
import re
from time import perf_counter

import httpx
from livekit.agents.llm import Toolset, function_tool

from voice_agent.telemetry import VoiceAgentTelemetry
from voice_agent.tools.text_utils import sanitize_tool_text

logger = logging.getLogger("livekit-rag-voice-agent.rag-tool")

RAG_FALLBACK_MESSAGE = "I'm sorry, I don't have that information in my records."
LEGAL_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    "sixty|seventy|eighty|ninety|hundred|thousand|million|and"
)
LEGAL_NUMBER_PAIR_PATTERN = re.compile(
    rf"\b((?:{LEGAL_NUMBER_WORDS})(?:[-\s]+(?:{LEGAL_NUMBER_WORDS}))*)\s*\((\d[\d,]*)\)",
    re.IGNORECASE,
)


class KnowledgeBaseToolset(Toolset):
    def __init__(
        self,
        *,
        backend_url: str,
        context_path: str,
        telemetry: VoiceAgentTelemetry | None = None,
        timeout_seconds: float = 6.0,
    ) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._context_path = context_path if context_path.startswith("/") else f"/{context_path}"
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
        return f"{self._backend_url}{self._context_path}"

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    @function_tool(
        description=(
            "Use this tool only for company, FAQ, policy, support, and "
            "uploaded-document questions. Use it for questions about the uploaded "
            "guide or PDF too, including phrases like 'this guide' or 'this document'. "
            "Use it for document-based contact details too, such as website, phone number, "
            "support number, claim number, or contact information mentioned in the uploaded guide. "
            "This tool returns retrieved document excerpts, not a final spoken answer. "
            "If the user makes a short follow-up such as 'yes', 'more', or "
            "'tell me more' right after a document answer, rewrite it into a clear "
            "standalone question using the recent conversation topic before calling this tool."
        )
    )
    async def ask_knowledge_base(self, question: str) -> str:
        """Fetch a grounded answer from the RAG backend.

        Args:
            question: The user's exact question about company, FAQ, policy, support, or uploaded documents.
        """

        cleaned_question = question.strip()
        if not cleaned_question:
            return RAG_FALLBACK_MESSAGE

        if self._telemetry is not None:
            self._telemetry.publish_kb_querying()

        client = await self._ensure_client()
        start_time = perf_counter()
        try:
            response = await client.post(
                self.endpoint_url,
                json={
                    "query": cleaned_question,
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
                    context_refs=[],
                )
            return RAG_FALLBACK_MESSAGE

        raw_context_refs = payload.get("context_refs") or []
        context_refs = [
            self._format_context_ref(ref)
            for ref in raw_context_refs
            if isinstance(ref, dict)
        ]
        has_sufficient_context = bool(payload.get("has_sufficient_context")) and bool(
            payload.get("context_excerpts") or []
        )
        fallback_used = not has_sufficient_context
        if self._telemetry is not None:
            self._telemetry.publish_kb_result(
                success=True,
                latency_ms=round((perf_counter() - start_time) * 1000),
                fallback=fallback_used,
                context_refs=context_refs,
            )

        if not has_sufficient_context:
            return RAG_FALLBACK_MESSAGE

        return self._format_context_packet(
            question=cleaned_question,
            context_excerpts=payload.get("context_excerpts") or [],
        )

    @staticmethod
    def _format_context_ref(payload: dict[str, object]) -> dict[str, object]:
        return {
            "sourceId": str(payload.get("source_id", "")),
            "documentId": int(payload.get("document_id", 0) or 0),
            "filename": str(payload.get("filename", "")),
            "chunkId": int(payload.get("chunk_id", 0) or 0),
            "chunkIndex": int(payload.get("chunk_index", 0) or 0),
            "similarityScore": float(payload.get("similarity_score", 0.0) or 0.0),
            "sectionAnchor": str(payload.get("section_anchor", "") or ""),
        }

    @staticmethod
    def _format_context_packet(*, question: str, context_excerpts: list[object]) -> str:
        formatted_excerpts: list[str] = []
        for index, item in enumerate(context_excerpts[:5], start=1):
            if not isinstance(item, dict):
                continue
            excerpt_text = sanitize_tool_text(str(item.get("chunk_text", "")))
            excerpt_text = KnowledgeBaseToolset._normalize_for_voice(excerpt_text)
            if not excerpt_text:
                continue
            filename = sanitize_tool_text(str(item.get("filename", "") or ""))
            section_anchor = sanitize_tool_text(str(item.get("section_anchor", "") or ""))
            source_parts = [part for part in (filename, section_anchor) if part]
            source_label = " | ".join(source_parts) if source_parts else f"Excerpt {index}"
            formatted_excerpts.append(f"{index}. [{source_label}] {excerpt_text}")

        if not formatted_excerpts:
            return RAG_FALLBACK_MESSAGE

        context_block = "\n".join(formatted_excerpts)
        return (
            "Knowledge base retrieval result.\n"
            f"Topic hint: {question}\n"
            "Answer only from the excerpts below. Speak naturally and briefly. "
            "Rewrite document wording into a conversational voice-friendly answer. "
            "If the excerpts use legal duplicated number forms like 'twenty (20)', speak only the number once, like '20'. "
            "Do not sound like you are reading a document verbatim. "
            "If they do not contain the answer, "
            f"say exactly: {RAG_FALLBACK_MESSAGE}\n"
            f"{context_block}"
        )

    @staticmethod
    def _normalize_for_voice(text: str) -> str:
        return LEGAL_NUMBER_PAIR_PATTERN.sub(lambda match: match.group(2), text)
