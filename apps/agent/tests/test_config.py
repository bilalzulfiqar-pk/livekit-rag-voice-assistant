from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from voice_agent.config import (
    DEFAULT_AGENT_NAME,
    DEFAULT_LLM_MODEL,
    DEFAULT_RAG_BACKEND_URL,
    DEFAULT_RAG_CHAT_PATH,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    Settings,
)


class SettingsTests(unittest.TestCase):
    def test_from_env_uses_defaults_for_optional_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LIVEKIT_URL": "wss://example.livekit.cloud",
                "LIVEKIT_API_KEY": "key",
                "LIVEKIT_API_SECRET": "secret",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.livekit_agent_name, DEFAULT_AGENT_NAME)
        self.assertEqual(settings.stt_model, DEFAULT_STT_MODEL)
        self.assertEqual(settings.llm_model, DEFAULT_LLM_MODEL)
        self.assertEqual(settings.tts_model, DEFAULT_TTS_MODEL)
        self.assertEqual(settings.tts_voice, DEFAULT_TTS_VOICE)
        self.assertEqual(settings.rag_backend_url, DEFAULT_RAG_BACKEND_URL)
        self.assertEqual(settings.rag_chat_path, DEFAULT_RAG_CHAT_PATH)

    def test_from_env_normalizes_rag_chat_path(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LIVEKIT_URL": "wss://example.livekit.cloud",
                "LIVEKIT_API_KEY": "key",
                "LIVEKIT_API_SECRET": "secret",
                "RAG_CHAT_PATH": "chat/ask",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.rag_chat_path, "/chat/ask")

    def test_from_env_requires_livekit_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
