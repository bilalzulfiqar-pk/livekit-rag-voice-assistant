from __future__ import annotations

import importlib
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from livekit.agents.llm import ToolContext

from voice_agent.tools.rag_tool import (
    KnowledgeBaseToolset,
    RAG_FAILURE_MESSAGE,
)
from voice_agent.tools.weather_tool import (
    UNKNOWN_CITY_MESSAGE,
    WEATHER_FAILURE_MESSAGE,
    WeatherToolset,
)


class KnowledgeBaseToolsetTests(unittest.IsolatedAsyncioTestCase):
    async def test_ask_knowledge_base_returns_sanitized_answer(self) -> None:
        toolset = KnowledgeBaseToolset(
            backend_url="http://localhost:8000",
            chat_path="/chat/ask",
        )
        response = httpx.Response(
            200,
            json={"answer": "## Services\n- Voice agents\n- RAG assistants"},
            request=httpx.Request("POST", "http://localhost:8000/chat/ask"),
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=response)
        toolset._client = mock_client

        result = await toolset.ask_knowledge_base("What services do you offer?")

        self.assertEqual(result, "Services Voice agents RAG assistants")
        mock_client.post.assert_awaited_once()

    async def test_ask_knowledge_base_handles_request_failures(self) -> None:
        toolset = KnowledgeBaseToolset(
            backend_url="http://localhost:8000",
            chat_path="/chat/ask",
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("boom"))
        toolset._client = mock_client

        result = await toolset.ask_knowledge_base("What services do you offer?")

        self.assertEqual(result, RAG_FAILURE_MESSAGE)


class WeatherToolsetTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_current_weather_returns_short_summary(self) -> None:
        toolset = WeatherToolset()
        geocode_response = httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Lahore",
                        "admin1": "Punjab",
                        "country": "Pakistan",
                        "latitude": 31.55,
                        "longitude": 74.34,
                    }
                ]
            },
            request=httpx.Request("GET", "https://geocoding-api.open-meteo.com"),
        )
        weather_response = httpx.Response(
            200,
            json={"current": {"temperature_2m": 31.2, "weather_code": 1}},
            request=httpx.Request("GET", "https://api.open-meteo.com"),
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=[geocode_response, weather_response])
        toolset._client = mock_client

        result = await toolset.get_current_weather("Lahore")

        self.assertEqual(
            result,
            "Lahore, Punjab, Pakistan is 31 degrees Celsius and mostly clear right now.",
        )

    async def test_get_current_weather_handles_unknown_city(self) -> None:
        toolset = WeatherToolset()
        geocode_response = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("GET", "https://geocoding-api.open-meteo.com"),
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=geocode_response)
        toolset._client = mock_client

        result = await toolset.get_current_weather("Unknown City")

        self.assertEqual(result, UNKNOWN_CITY_MESSAGE)

    async def test_get_current_weather_handles_failures(self) -> None:
        toolset = WeatherToolset()
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
        toolset._client = mock_client

        result = await toolset.get_current_weather("Lahore")

        self.assertEqual(result, WEATHER_FAILURE_MESSAGE)


class AgentRegistrationTests(unittest.TestCase):
    def test_voice_agent_registers_rag_and_weather_tools(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LIVEKIT_URL": "wss://example.livekit.cloud",
                "LIVEKIT_API_KEY": "key",
                "LIVEKIT_API_SECRET": "secret",
            },
            clear=True,
        ):
            import voice_agent.agent_server as agent_server

            agent_server = importlib.reload(agent_server)
            agent = agent_server.AuralisVoiceAgent()

        function_tools = ToolContext(agent.tools).function_tools
        self.assertIn("ask_knowledge_base", function_tools)
        self.assertIn("get_current_weather", function_tools)
