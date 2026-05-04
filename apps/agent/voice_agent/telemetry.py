from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("livekit-rag-voice-agent.telemetry")

TOOLING_STATUS_TOPIC = "app.tooling.status"
READY_PATH = "/ready"

ToolStatusValue = str
AnswerPathValue = str
RagBackendState = str


@dataclass(slots=True)
class ToolStatusSnapshot:
    status: ToolStatusValue = "idle"
    latency_ms: int | None = None
    fallback: bool | None = None

    def to_payload(self) -> dict[str, object | None]:
        return {
            "status": self.status,
            "latencyMs": self.latency_ms,
            "fallback": self.fallback,
        }


@dataclass(slots=True)
class ToolingSnapshot:
    session_id: str
    last_answer_path: AnswerPathValue = "unknown"
    last_fallback: bool | None = None
    rag_backend: RagBackendState = "unknown"
    knowledge_base: ToolStatusSnapshot = field(default_factory=ToolStatusSnapshot)
    weather: ToolStatusSnapshot = field(default_factory=ToolStatusSnapshot)

    def to_payload(self, *, sequence: int) -> dict[str, object]:
        return {
            "type": "tooling_snapshot",
            "version": 1,
            "sequence": sequence,
            "sessionId": self.session_id,
            "lastAnswerPath": self.last_answer_path,
            "lastFallback": self.last_fallback,
            "ragBackend": self.rag_backend,
            "knowledgeBase": self.knowledge_base.to_payload(),
            "weather": self.weather.to_payload(),
        }


class VoiceAgentTelemetry:
    def __init__(
        self,
        *,
        session_id: str,
        rag_backend_url: str,
        publisher: Callable[[str], None],
        readiness_probe: Callable[[str], Awaitable[bool]] | None = None,
        readiness_timeout_seconds: float = 4.0,
    ) -> None:
        self._publisher = publisher
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._readiness_url = f"{rag_backend_url.rstrip('/')}{READY_PATH}"
        self._readiness_probe = readiness_probe or self._probe_readiness
        self._sequence = 0
        self._snapshot = ToolingSnapshot(session_id=session_id)
        self._current_turn_has_tool = False
        self._current_turn_started = False

    @property
    def snapshot(self) -> ToolingSnapshot:
        return self._snapshot

    def publish_initial_state(self) -> None:
        self._snapshot.last_answer_path = "unknown"
        self._snapshot.last_fallback = None
        self._snapshot.rag_backend = "warming_up"
        self._snapshot.knowledge_base = ToolStatusSnapshot()
        self._snapshot.weather = ToolStatusSnapshot()
        self.publish_snapshot()

    def publish_snapshot(self) -> None:
        self._sequence += 1
        payload = json.dumps(self._snapshot.to_payload(sequence=self._sequence))
        try:
            publish_result = self._publisher(payload)
        except Exception as exc:
            logger.warning("Failed to publish tooling snapshot", exc_info=exc)
            return

        if inspect.isawaitable(publish_result):
            task = asyncio.create_task(publish_result)
            task.add_done_callback(self._handle_publish_task_done)

    async def publish_startup_ready_state(self) -> None:
        try:
            is_ready = await self._readiness_probe(self._readiness_url)
        except Exception as exc:
            logger.warning("RAG readiness probe failed", exc_info=exc)
            is_ready = False

        if self._snapshot.rag_backend != "warming_up":
            return

        self._snapshot.rag_backend = "ready" if is_ready else "degraded"
        self.publish_snapshot()

    def start_user_turn(self) -> None:
        self._current_turn_started = True
        self._current_turn_has_tool = False
        self._snapshot.last_answer_path = "unknown"
        self._snapshot.last_fallback = None
        self._snapshot.knowledge_base = ToolStatusSnapshot()
        self._snapshot.weather = ToolStatusSnapshot()
        self.publish_snapshot()

    def mark_tool_turn(self, function_name: str) -> None:
        answer_path = _tool_name_to_answer_path(function_name)
        if answer_path is None:
            return

        self._current_turn_started = False
        self._current_turn_has_tool = True
        self._snapshot.last_answer_path = answer_path
        self._snapshot.last_fallback = self._fallback_for_path(answer_path)
        self.publish_snapshot()

    def mark_normal_reply(self, source: str) -> None:
        if source != "generate_reply":
            return
        if not self._current_turn_started or self._current_turn_has_tool:
            self._current_turn_started = False
            return

        self._current_turn_started = False
        self._snapshot.last_answer_path = "normal"
        self._snapshot.last_fallback = False
        self.publish_snapshot()

    def publish_kb_querying(self) -> None:
        self._snapshot.knowledge_base.status = "querying"
        self._snapshot.knowledge_base.latency_ms = None
        self._snapshot.knowledge_base.fallback = None
        self.publish_snapshot()

    def publish_kb_result(self, *, success: bool, latency_ms: int, fallback: bool) -> None:
        self._snapshot.knowledge_base.status = "success" if success else "failed"
        self._snapshot.knowledge_base.latency_ms = latency_ms
        self._snapshot.knowledge_base.fallback = fallback
        self._snapshot.rag_backend = "ready" if success else "degraded"

        if self._snapshot.last_answer_path == "knowledge_base":
            self._snapshot.last_fallback = fallback

        self.publish_snapshot()

    def publish_weather_querying(self) -> None:
        self._snapshot.weather.status = "querying"
        self._snapshot.weather.latency_ms = None
        self._snapshot.weather.fallback = None
        self.publish_snapshot()

    def publish_weather_result(
        self,
        *,
        success: bool,
        latency_ms: int,
        fallback: bool,
    ) -> None:
        self._snapshot.weather.status = "success" if success else "failed"
        self._snapshot.weather.latency_ms = latency_ms
        self._snapshot.weather.fallback = fallback

        if self._snapshot.last_answer_path == "weather":
            self._snapshot.last_fallback = fallback

        self.publish_snapshot()

    async def _probe_readiness(self, readiness_url: str) -> bool:
        timeout = httpx.Timeout(
            self._readiness_timeout_seconds,
            connect=min(2.0, self._readiness_timeout_seconds),
            read=self._readiness_timeout_seconds,
            write=self._readiness_timeout_seconds,
            pool=self._readiness_timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(readiness_url)
            response.raise_for_status()
            payload = response.json()
        return str(payload.get("status", "")).lower() == "ready"

    def _fallback_for_path(self, answer_path: AnswerPathValue) -> bool | None:
        if answer_path == "knowledge_base":
            return self._snapshot.knowledge_base.fallback
        if answer_path == "weather":
            return self._snapshot.weather.fallback
        if answer_path == "normal":
            return False
        return None

    @staticmethod
    def _handle_publish_task_done(task: asyncio.Task[object]) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.warning("Failed to publish tooling snapshot", exc_info=exc)


def _tool_name_to_answer_path(function_name: str) -> AnswerPathValue | None:
    if function_name == "ask_knowledge_base":
        return "knowledge_base"
    if function_name == "get_current_weather":
        return "weather"
    return None
