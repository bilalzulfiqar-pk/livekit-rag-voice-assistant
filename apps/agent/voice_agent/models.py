from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class VoiceCapabilities:
    stt_model: str
    llm_model: str
    tts_model: str
    tts_voice: str
    vad: str = "silero"
    turn_detection: str = "livekit-multilingual"
    noise_cancellation: str = "bvc"
    interruptions: str = "adaptive"
    preemptive_generation: bool = True
    text_input: bool = True
    short_term_memory: bool = True

    def to_payload(self) -> dict[str, object]:
        return {
            "sttModel": self.stt_model,
            "llmModel": self.llm_model,
            "ttsModel": self.tts_model,
            "ttsVoice": self.tts_voice,
            "vad": self.vad,
            "turnDetection": self.turn_detection,
            "noiseCancellation": self.noise_cancellation,
            "interruptions": self.interruptions,
            "preemptiveGeneration": self.preemptive_generation,
            "textInput": self.text_input,
            "shortTermMemory": self.short_term_memory,
        }


@dataclass(slots=True)
class VoiceSessionInfo:
    session_id: str
    agent_name: str
    started_at: str = field(default_factory=_utc_now_iso)
    transport: str = "livekit"

    def to_payload(self) -> dict[str, str]:
        return {
            "sessionId": self.session_id,
            "agentName": self.agent_name,
            "startedAt": self.started_at,
            "transport": self.transport,
        }
