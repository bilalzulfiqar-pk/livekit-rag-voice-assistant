from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_AGENT_NAME = "livekit-rag-voice-agent"
DEFAULT_STT_MODEL = "deepgram/flux-general-en"
DEFAULT_LLM_MODEL = "openai/gpt-4.1-mini"
DEFAULT_TTS_MODEL = "cartesia/sonic-3"
DEFAULT_TTS_VOICE = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
DEFAULT_RAG_BACKEND_URL = "http://localhost:8000"
DEFAULT_RAG_CHAT_PATH = "/chat/ask"
DEFAULT_RAG_CONTEXT_PATH = "/retrieval/context"


@dataclass(slots=True)
class Settings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    livekit_agent_name: str = DEFAULT_AGENT_NAME
    stt_model: str = DEFAULT_STT_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    tts_voice: str = DEFAULT_TTS_VOICE
    rag_backend_url: str = DEFAULT_RAG_BACKEND_URL
    rag_chat_path: str = DEFAULT_RAG_CHAT_PATH
    rag_context_path: str = DEFAULT_RAG_CONTEXT_PATH

    @classmethod
    def from_env(cls) -> "Settings":
        livekit_url = os.getenv("LIVEKIT_URL", "").strip()
        livekit_api_key = os.getenv("LIVEKIT_API_KEY", "").strip()
        livekit_api_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()

        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise RuntimeError(
                "LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set."
            )

        rag_chat_path = (
            os.getenv("RAG_CHAT_PATH", DEFAULT_RAG_CHAT_PATH).strip()
            or DEFAULT_RAG_CHAT_PATH
        )
        if not rag_chat_path.startswith("/"):
            rag_chat_path = f"/{rag_chat_path}"
        rag_context_path = (
            os.getenv("RAG_CONTEXT_PATH", DEFAULT_RAG_CONTEXT_PATH).strip()
            or DEFAULT_RAG_CONTEXT_PATH
        )
        if not rag_context_path.startswith("/"):
            rag_context_path = f"/{rag_context_path}"

        return cls(
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            livekit_agent_name=(
                os.getenv("LIVEKIT_AGENT_NAME", DEFAULT_AGENT_NAME).strip()
                or DEFAULT_AGENT_NAME
            ),
            stt_model=(
                os.getenv("VOICE_AGENT_STT_MODEL", DEFAULT_STT_MODEL).strip()
                or DEFAULT_STT_MODEL
            ),
            llm_model=(
                os.getenv("VOICE_AGENT_LLM_MODEL", DEFAULT_LLM_MODEL).strip()
                or DEFAULT_LLM_MODEL
            ),
            tts_model=(
                os.getenv("VOICE_AGENT_TTS_MODEL", DEFAULT_TTS_MODEL).strip()
                or DEFAULT_TTS_MODEL
            ),
            tts_voice=(
                os.getenv("VOICE_AGENT_TTS_VOICE", DEFAULT_TTS_VOICE).strip()
                or DEFAULT_TTS_VOICE
            ),
            rag_backend_url=(
                os.getenv("RAG_BACKEND_URL", DEFAULT_RAG_BACKEND_URL).strip()
                or DEFAULT_RAG_BACKEND_URL
            ).rstrip("/"),
            rag_chat_path=rag_chat_path,
            rag_context_path=rag_context_path,
        )
