from __future__ import annotations

import unittest

from voice_agent.models import VoiceCapabilities, VoiceSessionInfo


class VoiceModelsTests(unittest.TestCase):
    def test_voice_capabilities_payload_is_frontend_friendly(self) -> None:
        payload = VoiceCapabilities(
            stt_model="deepgram/flux-general-en",
            llm_model="openai/gpt-4.1-mini",
            tts_model="cartesia/sonic-3",
            tts_voice="voice-id",
        ).to_payload()

        self.assertEqual(payload["sttModel"], "deepgram/flux-general-en")
        self.assertEqual(payload["llmModel"], "openai/gpt-4.1-mini")
        self.assertEqual(payload["ttsVoice"], "voice-id")
        self.assertTrue(payload["preemptiveGeneration"])

    def test_voice_session_info_payload_contains_started_at(self) -> None:
        payload = VoiceSessionInfo(
            session_id="room-1",
            agent_name="auralis-agent",
        ).to_payload()

        self.assertEqual(payload["sessionId"], "room-1")
        self.assertEqual(payload["agentName"], "auralis-agent")
        self.assertTrue(payload["startedAt"].endswith("Z"))
        self.assertEqual(payload["transport"], "livekit")


if __name__ == "__main__":
    unittest.main()
