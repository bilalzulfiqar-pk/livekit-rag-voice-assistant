from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RouteName = Literal["auto", "knowledge_base", "weather"]

_WEATHER_PATTERN = re.compile(
    r"\b(weather|temperature|forecast|rain|snow|humidity|wind|degrees?|celsius|fahrenheit)\b",
    re.IGNORECASE,
)
_DOC_REFERENCE_PATTERN = re.compile(
    r"\b(document|documents|doc|docs|guide|pdf|file|policy|faq|handbook|manual|uploaded|records?)\b",
    re.IGNORECASE,
)
_DOC_TOPIC_PATTERN = re.compile(
    r"\b(coverage|covered|benefit|benefits|claim|claims|deadline|deadlines|submit|submission|"
    r"required|required documents|forms?|eligibility|limits?|maximum|minimum|"
    r"procedure|procedures|process|steps?|reimbursement|support|contact|website|phone|number|"
    r"services|service|pricing|plan|plans|terms?|conditions?|refund|cancellation|policy|policies)\b",
    re.IGNORECASE,
)
_DOC_QUESTION_PATTERN = re.compile(
    r"\b(how soon|how long|when must|when do|what is covered|what's covered|"
    r"what do i need|how do i|where do i|who do i contact|what is the website|"
    r"what is the phone|what number|what documents|which documents)\b",
    re.IGNORECASE,
)
_DOC_COVERAGE_QUESTION_PATTERN = re.compile(
    r"\b(are|is|does|do)\b.*\b(covered|cover|coverage|excluded|allowed|eligible)\b",
    re.IGNORECASE,
)
_KB_FOLLOW_UP_PATTERN = re.compile(
    r"^\s*(yes|more|tell me more|what about that|and that|about that|"
    r"asking about|i mean|for that|for this|that one|this one|that|this|those|these)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    reason: str


def decide_route(message: str, *, last_answer_path: str = "unknown") -> RouteDecision:
    text = " ".join(message.strip().split())
    if not text:
        return RouteDecision(route="auto", reason="empty")

    lowered = text.lower()
    if _WEATHER_PATTERN.search(lowered):
        return RouteDecision(route="weather", reason="weather_keywords")

    if last_answer_path == "knowledge_base" and _KB_FOLLOW_UP_PATTERN.search(lowered):
        return RouteDecision(route="knowledge_base", reason="kb_follow_up")

    if last_answer_path == "knowledge_base" and _DOC_TOPIC_PATTERN.search(lowered):
        return RouteDecision(route="knowledge_base", reason="kb_topic_follow_up")

    doc_ref = bool(_DOC_REFERENCE_PATTERN.search(lowered))
    doc_topic_hits = len(_DOC_TOPIC_PATTERN.findall(lowered))
    doc_question = bool(_DOC_QUESTION_PATTERN.search(lowered))
    coverage_question = bool(_DOC_COVERAGE_QUESTION_PATTERN.search(lowered))

    if doc_ref and (doc_question or coverage_question or doc_topic_hits >= 1):
        return RouteDecision(route="knowledge_base", reason="doc_reference")

    if doc_topic_hits >= 2:
        return RouteDecision(route="knowledge_base", reason="doc_topic_density")

    if doc_topic_hits >= 1 and (doc_question or coverage_question):
        return RouteDecision(route="knowledge_base", reason="doc_topic_question")

    return RouteDecision(route="auto", reason="default")
