from __future__ import annotations

import logging
import re
from asyncio import sleep
from time import perf_counter

import httpx
from livekit.agents.llm import Toolset, function_tool

from voice_agent.telemetry import VoiceAgentTelemetry
from voice_agent.tools.text_utils import sanitize_tool_text

logger = logging.getLogger("livekit-rag-voice-agent.rag-tool")

RAG_FALLBACK_MESSAGE = "I'm sorry, I don't have that information in my records."
RAG_NO_RECORDS_MARKER = "[KB_NO_RECORDS]"
LEGAL_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
    "sixty|seventy|eighty|ninety|hundred|thousand|million|and"
)
LEGAL_NUMBER_PAIR_PATTERN = re.compile(
    rf"\b((?:{LEGAL_NUMBER_WORDS})(?:[-\s]+(?:{LEGAL_NUMBER_WORDS}))*)\s*\((\d[\d,]*)\)",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"\b(?:https?://)?(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\s().-]{6,}\d)\b")
MONEY_PATTERN = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
DEADLINE_PATTERN = re.compile(
    r"\b(?:within|no later than)\s+\d[\d,]*\s+(?:business\s+)?days?\b|\bas soon as reasonably possible\b",
    re.IGNORECASE,
)
QUESTION_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "company",
    "do",
    "does",
    "for",
    "get",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "long",
    "must",
    "number",
    "of",
    "on",
    "or",
    "soon",
    "the",
    "their",
    "there",
    "this",
    "to",
    "what",
    "when",
    "which",
    "who",
    "website",
    "with",
}


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
            "Use it for document-grounded questions about coverage, exclusions, limits, deadlines, eligibility, "
            "required documents, claim steps, procedures, benefits, reimbursement, support details, or contact details. "
            "Use it for document-based contact details too, such as website, phone number, "
            "support number, claim number, or contact information mentioned in the uploaded guide. "
            "This tool returns retrieved document excerpts, not a final spoken answer. "
            "If the user makes a short follow-up such as 'yes', 'more', or "
            "'tell me more' right after a document answer, rewrite it into a clear "
            "standalone question using the recent conversation topic before calling this tool. "
            "If the user sends a short clarification or correction after a document answer, treat it as a follow-up "
            "to the same document topic. "
            "Never mention this tool name to the user. Call it silently."
        )
    )
    async def ask_knowledge_base(self, question: str) -> str:
        """Fetch a grounded answer from the RAG backend.

        Args:
            question: The user's exact question about company, FAQ, policy, support, or uploaded documents.
        """

        cleaned_question = question.strip()
        if not cleaned_question:
            return RAG_NO_RECORDS_MARKER

        if self._telemetry is not None:
            self._telemetry.publish_kb_querying()

        client = await self._ensure_client()
        start_time = perf_counter()
        try:
            payload = await self._fetch_context_payload(client, cleaned_question)
        except Exception as exc:
            logger.warning("RAG backend request failed", exc_info=exc)
            if self._telemetry is not None:
                self._telemetry.publish_kb_result(
                    success=False,
                    latency_ms=round((perf_counter() - start_time) * 1000),
                    fallback=True,
                    context_refs=[],
                )
            return RAG_NO_RECORDS_MARKER

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
            return RAG_NO_RECORDS_MARKER

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
        direct_facts = KnowledgeBaseToolset._extract_direct_facts(question, context_excerpts)
        preferred_answer = KnowledgeBaseToolset._build_preferred_answer(question, direct_facts)
        for index, item in enumerate(context_excerpts[:5], start=1):
            if not isinstance(item, dict):
                continue
            excerpt_text = sanitize_tool_text(str(item.get("chunk_text", "")))
            excerpt_text = KnowledgeBaseToolset._normalize_for_voice(excerpt_text)
            excerpt_text = KnowledgeBaseToolset._focus_excerpt_for_question(question, excerpt_text)
            if not excerpt_text:
                continue
            filename = sanitize_tool_text(str(item.get("filename", "") or ""))
            section_anchor = sanitize_tool_text(str(item.get("section_anchor", "") or ""))
            source_parts = [part for part in (filename, section_anchor) if part]
            source_label = " | ".join(source_parts) if source_parts else f"Excerpt {index}"
            formatted_excerpts.append(f"{index}. [{source_label}] {excerpt_text}")

        if not formatted_excerpts:
            return RAG_NO_RECORDS_MARKER

        context_block = "\n".join(formatted_excerpts)
        direct_facts_block = ""
        if direct_facts:
            direct_facts_block = "Direct answer facts:\n" + "\n".join(
                f"- {fact}" for fact in direct_facts
            ) + "\n"
        preferred_answer_block = ""
        if preferred_answer:
            preferred_answer_block = f"Preferred grounded answer: {preferred_answer}\n"
        return (
            "Knowledge base retrieval result.\n"
            f"Topic hint: {question}\n"
            "Answer only from the excerpts below. Speak naturally and briefly. "
            "The excerpts below are sufficient context for this turn unless the tool result is exactly [KB_NO_RECORDS]. "
            "Treat any exact website, phone number, deadline, amount, duration, limit, exclusion, or required document in the excerpts as authoritative. "
            "If a deadline or amount is stated, say that exact value directly instead of softening it into vague advice. "
            "Rewrite document wording into a conversational voice-friendly answer. "
            "If the excerpts use legal duplicated number forms like 'twenty (20)', speak only the number once, like '20'. "
            "Never mention tool names, retrieval, or tell the user to use a tool. "
            "Do not sound like you are reading a document verbatim. "
            f"If the tool result is exactly {RAG_NO_RECORDS_MARKER}, say exactly: {RAG_FALLBACK_MESSAGE}\n"
            f"{preferred_answer_block}"
            f"{direct_facts_block}"
            f"{context_block}"
        )

    @staticmethod
    def _normalize_for_voice(text: str) -> str:
        return LEGAL_NUMBER_PAIR_PATTERN.sub(lambda match: match.group(2), text)

    @staticmethod
    def _focus_excerpt_for_question(question: str, excerpt_text: str) -> str:
        text = excerpt_text.strip()
        if not text:
            return ""

        sentences = [segment.strip(" -") for segment in SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]
        if len(sentences) <= 1:
            return text

        keywords = KnowledgeBaseToolset._question_keywords(question)
        asks_contact = KnowledgeBaseToolset._question_mentions_contact(question)
        asks_deadline = KnowledgeBaseToolset._question_mentions_deadline(question)
        asks_documents = KnowledgeBaseToolset._question_mentions_documents(question)

        scored_sentences: list[tuple[int, int, str]] = []
        for index, sentence in enumerate(sentences):
            lowered = sentence.lower()
            score = 0
            if keywords:
                sentence_tokens = set(QUESTION_TOKEN_PATTERN.findall(lowered))
                score += len(keywords & sentence_tokens) * 2
            if asks_contact and (URL_PATTERN.search(sentence) or PHONE_PATTERN.search(sentence)):
                score += 10
            if asks_deadline and DEADLINE_PATTERN.search(sentence):
                score += 10
            if asks_documents and "document" in lowered:
                score += 6
            if "claim" in lowered and "claim" in keywords:
                score += 3
            if "baggage" in lowered and "baggage" in keywords:
                score += 3
            if "website" in keywords and URL_PATTERN.search(sentence):
                score += 5
            if ("phone" in keywords or "number" in keywords) and PHONE_PATTERN.search(sentence):
                score += 5
            if score > 0:
                scored_sentences.append((score, index, sentence))

        if not scored_sentences and (asks_contact or asks_deadline or asks_documents):
            return ""

        if not scored_sentences:
            return text

        scored_sentences.sort(key=lambda item: (-item[0], item[1]))
        selected_indexes = sorted({index for _, index, _ in scored_sentences[:2]})
        selected = [sentences[index] for index in selected_indexes]
        focused = " ".join(selected).strip()
        return focused or text

    @staticmethod
    def _extract_direct_facts(question: str, context_excerpts: list[object]) -> list[str]:
        asks_contact = KnowledgeBaseToolset._question_mentions_contact(question)
        asks_deadline = KnowledgeBaseToolset._question_mentions_deadline(question)
        asks_documents = KnowledgeBaseToolset._question_mentions_documents(question)
        asks_amount = KnowledgeBaseToolset._question_mentions_amount_or_max(question)

        urls: list[str] = []
        phones: list[str] = []
        deadlines: list[str] = []
        document_deadlines: list[str] = []
        benefit_amounts: list[str] = []

        for item in context_excerpts[:5]:
            if not isinstance(item, dict):
                continue
            excerpt_text = sanitize_tool_text(str(item.get("chunk_text", "")))
            excerpt_text = KnowledgeBaseToolset._normalize_for_voice(excerpt_text)
            if not excerpt_text:
                continue

            for match in URL_PATTERN.findall(excerpt_text):
                value = match.strip(".,;:)")
                if value not in urls:
                    urls.append(value)

            for match in PHONE_PATTERN.findall(excerpt_text):
                value = " ".join(match.split()).strip(".,;:")
                if value not in phones:
                    phones.append(value)

            for sentence in SENTENCE_SPLIT_PATTERN.split(excerpt_text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_lower = sentence.lower()
                if asks_amount:
                    for match in MONEY_PATTERN.findall(sentence):
                        value = match.strip()
                        if KnowledgeBaseToolset._sentence_is_amount_relevant(sentence_lower) and value not in benefit_amounts:
                            benefit_amounts.append(value)
                for match in DEADLINE_PATTERN.findall(sentence):
                    value = " ".join(match.split()).strip(".,;:")
                    if ("document" in sentence_lower or "submit" in sentence_lower) and value not in document_deadlines:
                        document_deadlines.append(value)
                    if ("claim" in sentence_lower or "report" in sentence_lower or "file" in sentence_lower) and value not in deadlines:
                        deadlines.append(value)
                    elif value not in deadlines and value not in document_deadlines:
                        deadlines.append(value)

        facts: list[str] = []
        if asks_contact:
            if urls:
                facts.append(f"Website: {urls[0]}")
            if phones:
                facts.append(f"Phone number: {phones[0]}")

        if asks_deadline:
            if deadlines:
                facts.append(f"Filing deadline: {KnowledgeBaseToolset._preferred_deadline(deadlines)}")
            if asks_documents and document_deadlines:
                facts.append(f"Document deadline: {KnowledgeBaseToolset._preferred_deadline(document_deadlines)}")

        if asks_documents and document_deadlines and not any(
            fact.startswith("Document deadline:") for fact in facts
        ):
            facts.append(f"Document deadline: {KnowledgeBaseToolset._preferred_deadline(document_deadlines)}")

        if asks_amount and benefit_amounts:
            facts.append(f"Benefit amount: {benefit_amounts[0]}")

        return facts

    @staticmethod
    def _question_keywords(question: str) -> set[str]:
        tokens = {
            token.lower()
            for token in QUESTION_TOKEN_PATTERN.findall(question.lower())
            if len(token) > 2 and token.lower() not in QUESTION_STOPWORDS
        }
        return tokens

    @staticmethod
    def _question_mentions_contact(question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in ("website", "phone", "number", "contact", "call"))

    @staticmethod
    def _question_mentions_deadline(question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in ("how soon", "how long", "when", "deadline", "filed", "submit", "report"))

    @staticmethod
    def _question_mentions_documents(question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in ("document", "documents", "paperwork", "submit"))

    @staticmethod
    def _question_mentions_amount_or_max(question: str) -> bool:
        lowered = question.lower()
        return any(
            keyword in lowered
            for keyword in ("maximum", "max", "amount", "limit", "benefit", "coverage amount")
        )

    @staticmethod
    def _preferred_deadline(values: list[str]) -> str:
        if not values:
            return ""
        numeric = [value for value in values if any(char.isdigit() for char in value)]
        if numeric:
            return numeric[0]
        return values[0]

    @staticmethod
    def _build_preferred_answer(question: str, direct_facts: list[str]) -> str:
        lower_question = question.lower()
        fact_map: dict[str, str] = {}
        for fact in direct_facts:
            if ":" not in fact:
                continue
            key, value = fact.split(":", 1)
            fact_map[key.strip()] = value.strip()

        website = fact_map.get("Website")
        phone = fact_map.get("Phone number")
        filing_deadline = fact_map.get("Filing deadline")
        document_deadline = fact_map.get("Document deadline")
        benefit_amount = fact_map.get("Benefit amount")

        if website and phone:
            return f"The company's website is {website} and the phone number is {phone}."
        if website:
            return f"The company's website is {website}."
        if phone:
            return f"The phone number is {phone}."
        if document_deadline and any(keyword in lower_question for keyword in ("document", "documents", "paperwork", "submit")):
            return f"The requested documents should be submitted {document_deadline}."
        if filing_deadline and document_deadline:
            return (
                f"The claim should be filed {filing_deadline}, and the requested documents "
                f"should be submitted {document_deadline}."
            )
        if filing_deadline:
            return f"The claim should be filed {filing_deadline}."
        if document_deadline:
            return f"The requested documents should be submitted {document_deadline}."
        if benefit_amount and any(keyword in lower_question for keyword in ("maximum", "max", "limit", "benefit amount")):
            return f"The maximum benefit is {benefit_amount}."
        if benefit_amount:
            return f"The benefit amount is {benefit_amount}."
        return ""

    async def _fetch_context_payload(self, client: httpx.AsyncClient, cleaned_question: str) -> dict[str, object]:
        last_exception: Exception | None = None
        for attempt in range(2):
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
                if not isinstance(payload, dict):
                    raise ValueError("RAG backend returned a non-object payload.")
                return payload
            except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
                last_exception = exc
                if attempt == 0:
                    await sleep(0.15)
                    continue
                raise

        assert last_exception is not None
        raise last_exception

    @staticmethod
    def _sentence_is_amount_relevant(sentence_lower: str) -> bool:
        return any(
            phrase in sentence_lower
            for phrase in (
                "limited to",
                "maximum benefit",
                "maximum of",
                "up to",
                "benefit is",
                "benefit provides",
                "covered medical expenses",
            )
        )
