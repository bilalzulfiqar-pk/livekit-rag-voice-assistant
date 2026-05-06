from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(slots=True)
class ReadinessSnapshot:
    status: str
    database_ready: bool
    embedding_ready: bool
    embedding_required: bool
    embedding_state: str
    embedding_runtime: str | None
    flashrank_ready: bool
    flashrank_required: bool
    flashrank_state: str
    flashrank_model: str | None
    message: str


class AppReadinessState:
    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._database_ready = False
            self._embedding_ready = True
            self._embedding_required = False
            self._embedding_state = "not_required"
            self._embedding_runtime: str | None = None
            self._flashrank_ready = True
            self._flashrank_required = False
            self._flashrank_state = "not_required"
            self._flashrank_model: str | None = None
            self._message = "Service is starting."

    def begin_startup(
        self,
        *,
        requires_local_embedding: bool,
        embedding_runtime: str | None,
        requires_flashrank_warmup: bool,
        flashrank_model: str | None,
    ) -> None:
        with self._lock:
            self._database_ready = False
            self._embedding_required = requires_local_embedding
            self._embedding_ready = not requires_local_embedding
            self._embedding_state = "warming" if requires_local_embedding else "not_required"
            self._embedding_runtime = embedding_runtime
            self._flashrank_required = requires_flashrank_warmup
            self._flashrank_ready = not requires_flashrank_warmup
            self._flashrank_state = "warming" if requires_flashrank_warmup else "not_required"
            self._flashrank_model = flashrank_model if requires_flashrank_warmup else None
            self._update_message_locked()

    def mark_database_ready(self) -> None:
        with self._lock:
            self._database_ready = True
            self._update_message_locked()

    def mark_database_not_ready(self) -> None:
        with self._lock:
            self._database_ready = False
            self._message = "Database is not ready."

    def mark_embedding_warming(self, *, runtime: str | None) -> None:
        with self._lock:
            self._embedding_required = True
            self._embedding_ready = False
            self._embedding_state = "warming"
            self._embedding_runtime = runtime
            self._update_message_locked()

    def mark_embedding_ready(self, *, runtime: str | None) -> None:
        with self._lock:
            self._embedding_required = runtime is not None or self._embedding_required
            self._embedding_ready = True
            self._embedding_state = "ready" if self._embedding_required else "not_required"
            self._embedding_runtime = runtime
            self._update_message_locked()

    def mark_embedding_skipped(self, message: str = "Local embedding warmup is not required.") -> None:
        with self._lock:
            self._embedding_required = False
            self._embedding_ready = True
            self._embedding_state = "not_required"
            self._embedding_runtime = None
            self._message = message if not self._database_ready else "Service is ready."
            self._update_message_locked()

    def mark_embedding_failed(self, message: str, *, runtime: str | None) -> None:
        with self._lock:
            self._embedding_required = True
            self._embedding_ready = False
            self._embedding_state = "failed"
            self._embedding_runtime = runtime
            self._message = message

    def mark_flashrank_warming(self, *, model_name: str | None) -> None:
        with self._lock:
            self._flashrank_required = True
            self._flashrank_ready = False
            self._flashrank_state = "warming"
            self._flashrank_model = model_name
            self._update_message_locked()

    def mark_flashrank_ready(self, *, model_name: str | None) -> None:
        with self._lock:
            self._flashrank_required = model_name is not None or self._flashrank_required
            self._flashrank_ready = True
            self._flashrank_state = "ready" if self._flashrank_required else "not_required"
            self._flashrank_model = model_name
            self._update_message_locked()

    def mark_flashrank_skipped(self, *, model_name: str | None = None) -> None:
        with self._lock:
            self._flashrank_required = False
            self._flashrank_ready = True
            self._flashrank_state = "not_required"
            self._flashrank_model = model_name
            self._update_message_locked()

    def mark_flashrank_failed(self, message: str, *, model_name: str | None) -> None:
        with self._lock:
            self._flashrank_required = True
            self._flashrank_ready = True
            self._flashrank_state = "failed"
            self._flashrank_model = model_name
            self._message = message

    def _update_message_locked(self) -> None:
        if not self._database_ready:
            self._message = "Service is starting."
        elif self._embedding_state == "warming":
            self._message = "Database is ready. Local embedding warmup is still in progress."
        elif self._flashrank_state == "warming":
            self._message = "Database and embeddings are ready. FlashRank warmup is still in progress."
        elif self._flashrank_state == "failed":
            self._message = "Service is ready, but FlashRank warmup failed. Reranking may fall back to fast mode."
        else:
            self._message = "Service is ready."

    def snapshot(self) -> ReadinessSnapshot:
        with self._lock:
            is_ready = self._database_ready and self._embedding_ready and self._flashrank_ready
            return ReadinessSnapshot(
                status="ready" if is_ready else "starting",
                database_ready=self._database_ready,
                embedding_ready=self._embedding_ready,
                embedding_required=self._embedding_required,
                embedding_state=self._embedding_state,
                embedding_runtime=self._embedding_runtime,
                flashrank_ready=self._flashrank_ready,
                flashrank_required=self._flashrank_required,
                flashrank_state=self._flashrank_state,
                flashrank_model=self._flashrank_model,
                message=self._message,
            )


readiness_state = AppReadinessState()
