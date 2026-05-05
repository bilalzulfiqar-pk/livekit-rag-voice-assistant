from __future__ import annotations

import asyncio
import importlib
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from livekit.agents.llm import ToolContext
from livekit.agents.voice.agent import ModelSettings

from voice_agent.telemetry import VoiceAgentTelemetry
from voice_agent.routing import decide_route
from voice_agent.tools.rag_tool import (
    KnowledgeBaseToolset,
    RAG_FALLBACK_MESSAGE,
    RAG_NO_RECORDS_MARKER,
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
            context_path="/retrieval/context",
        )
        response = httpx.Response(
            200,
            json={
                "has_sufficient_context": True,
                "context_excerpts": [
                    {
                        "source_id": "document:1",
                        "chunk_id": 11,
                        "document_id": 1,
                        "filename": "company-faq.txt",
                        "chunk_index": 0,
                        "similarity_score": 0.92,
                        "section_anchor": "Services",
                        "chunk_text": "## Services\n- Voice agents\n- RAG assistants",
                    }
                ],
                "context_refs": [
                    {
                        "source_id": "document:1",
                        "chunk_id": 11,
                        "document_id": 1,
                        "filename": "company-faq.txt",
                        "chunk_index": 0,
                        "similarity_score": 0.92,
                        "section_anchor": "Services",
                    }
                ],
            },
            request=httpx.Request("POST", "http://localhost:8000/retrieval/context"),
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=response)
        toolset._client = mock_client

        result = await toolset.ask_knowledge_base("What services do you offer?")

        self.assertIn("Knowledge base retrieval result.", result)
        self.assertIn("Services Voice agents RAG assistants", result)
        mock_client.post.assert_awaited_once()

    def test_format_context_packet_normalizes_legal_number_pairs_for_voice(self) -> None:
        result = KnowledgeBaseToolset._format_context_packet(
            question="How soon must a lost luggage claim be filed?",
            context_excerpts=[
                {
                    "filename": "Guide To Benefits.pdf",
                    "section_anchor": "Baggage Delay Insurance",
                    "chunk_text": (
                        "Visit chasecardbenefits.com or call 1-800-350-1697 within twenty (20) days. "
                        "Documents must be submitted within ninety (90) days."
                    ),
                }
            ],
        )
        excerpt_line = result.splitlines()[-1]

        self.assertIn("within 20 days", excerpt_line)
        self.assertIn("within 90 days", excerpt_line)
        self.assertNotIn("twenty (20)", excerpt_line)
        self.assertNotIn("ninety (90)", excerpt_line)
        self.assertIn("Never mention tool names", result)
        self.assertIn("Treat any exact website, phone number, deadline, amount, duration, limit, exclusion, or required document in the excerpts as authoritative", result)
        self.assertIn("Direct answer facts:", result)
        self.assertIn("Filing deadline: within 20 days", result)

    def test_format_context_packet_surfaces_contact_facts_for_contact_question(self) -> None:
        result = KnowledgeBaseToolset._format_context_packet(
            question="what is the company number and website",
            context_excerpts=[
                {
                    "filename": "Guide To Benefits.pdf",
                    "section_anchor": "How Do You File a Claim?",
                    "chunk_text": (
                        "It is Your responsibility to make every effort to protect the Rental Vehicle from damage or theft. "
                        "Visit chasecardbenefits.com or call 1-800-350-1697 to report the theft or damage, "
                        "regardless of who is at fault."
                    ),
                }
            ],
        )

        self.assertIn("Direct answer facts:", result)
        self.assertIn("Website: chasecardbenefits.com", result)
        self.assertIn("Phone number: 1-800-350-1697", result)
        self.assertIn(
            "Preferred grounded answer: The company's website is chasecardbenefits.com and the phone number is 1-800-350-1697.",
            result,
        )
        self.assertIn("Visit chasecardbenefits.com or call 1-800-350-1697", result)

    def test_format_context_packet_builds_deadline_preferred_answer(self) -> None:
        result = KnowledgeBaseToolset._format_context_packet(
            question="How soon must a baggage delay claim be filed?",
            context_excerpts=[
                {
                    "filename": "Guide To Benefits.pdf",
                    "section_anchor": "How Do You File a Claim?",
                    "chunk_text": (
                        "Visit chasecardbenefits.com or call 1-800-350-1697 within twenty (20) days "
                        "of the date the Checked Baggage was delayed or as soon as reasonably possible. "
                        "The requested documents must be submitted within ninety (90) days."
                    ),
                }
            ],
        )

        self.assertIn(
            "Preferred grounded answer: The claim should be filed within 20 days.",
            result,
        )

    def test_format_context_packet_prefers_document_deadline_for_document_question(self) -> None:
        result = KnowledgeBaseToolset._format_context_packet(
            question="How long does the traveler have to submit baggage delay documents?",
            context_excerpts=[
                {
                    "filename": "Guide To Benefits.pdf",
                    "section_anchor": "How Do You File a Claim?",
                    "chunk_text": (
                        "Visit chasecardbenefits.com or call 1-800-350-1697 within twenty (20) days of the date "
                        "the Checked Baggage was delayed or as soon as reasonably possible. "
                        "The requested documents must be submitted within ninety (90) days of the Checked Baggage "
                        "being delayed or as soon as reasonably possible."
                    ),
                }
            ],
        )

        self.assertIn(
            "Preferred grounded answer: The requested documents should be submitted within 90 days.",
            result,
        )

    def test_format_context_packet_builds_amount_preferred_answer(self) -> None:
        result = KnowledgeBaseToolset._format_context_packet(
            question="What is the maximum emergency medical and dental benefit?",
            context_excerpts=[
                {
                    "filename": "Guide To Benefits.pdf",
                    "section_anchor": "Emergency Medical and Dental",
                    "chunk_text": (
                        "Covered medical expenses are limited to $2,500 and are subject to a $50 deductible. "
                        "The Covered Traveler is eligible for an additional benefit of $75 per day for up to a maximum of five days."
                    ),
                }
            ],
        )

        self.assertIn("Benefit amount: $2,500", result)
        self.assertIn("Preferred grounded answer: The maximum benefit is $2,500.", result)

    async def test_ask_knowledge_base_handles_request_failures(self) -> None:
        toolset = KnowledgeBaseToolset(
            backend_url="http://localhost:8000",
            context_path="/retrieval/context",
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("boom"))
        toolset._client = mock_client

        result = await toolset.ask_knowledge_base("What services do you offer?")

        self.assertEqual(result, RAG_NO_RECORDS_MARKER)

    async def test_ask_knowledge_base_publishes_telemetry(self) -> None:
        telemetry = Mock()
        toolset = KnowledgeBaseToolset(
            backend_url="http://localhost:8000",
            context_path="/retrieval/context",
            telemetry=telemetry,
        )
        response = httpx.Response(
            200,
            json={
                "has_sufficient_context": True,
                "context_excerpts": [
                    {
                        "source_id": "document:1",
                        "chunk_id": 1,
                        "document_id": 1,
                        "filename": "company-faq.txt",
                        "chunk_index": 0,
                        "similarity_score": 0.88,
                        "section_anchor": "Support",
                        "chunk_text": "We offer support.",
                    }
                ],
                "context_refs": [
                    {
                        "source_id": "document:1",
                        "chunk_id": 1,
                        "document_id": 1,
                        "filename": "company-faq.txt",
                        "chunk_index": 0,
                        "similarity_score": 0.88,
                        "section_anchor": "Support",
                    }
                ],
            },
            request=httpx.Request("POST", "http://localhost:8000/retrieval/context"),
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=response)
        toolset._client = mock_client

        with patch("voice_agent.tools.rag_tool.perf_counter", side_effect=[10.0, 10.4]):
            await toolset.ask_knowledge_base("What services do you offer?")

        telemetry.publish_kb_querying.assert_called_once_with()
        telemetry.publish_kb_result.assert_called_once_with(
            success=True,
            latency_ms=400,
            fallback=False,
            context_refs=[
                {
                    "sourceId": "document:1",
                    "documentId": 1,
                    "filename": "company-faq.txt",
                    "chunkId": 1,
                    "chunkIndex": 0,
                    "similarityScore": 0.88,
                    "sectionAnchor": "Support",
                }
            ],
        )

    async def test_ask_knowledge_base_publishes_failure_telemetry(self) -> None:
        telemetry = Mock()
        toolset = KnowledgeBaseToolset(
            backend_url="http://localhost:8000",
            context_path="/retrieval/context",
            telemetry=telemetry,
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("boom"))
        toolset._client = mock_client

        with patch("voice_agent.tools.rag_tool.perf_counter", side_effect=[5.0, 5.25]):
            await toolset.ask_knowledge_base("What services do you offer?")

        telemetry.publish_kb_querying.assert_called_once_with()
        telemetry.publish_kb_result.assert_called_once_with(
            success=False,
            latency_ms=250,
            fallback=True,
            context_refs=[],
        )

    async def test_ask_knowledge_base_marks_insufficient_context_in_telemetry(self) -> None:
        telemetry = Mock()
        toolset = KnowledgeBaseToolset(
            backend_url="http://localhost:8000",
            context_path="/retrieval/context",
            telemetry=telemetry,
        )
        response = httpx.Response(
            200,
            json={
                "has_sufficient_context": False,
                "context_excerpts": [],
                "context_refs": [],
            },
            request=httpx.Request("POST", "http://localhost:8000/retrieval/context"),
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=response)
        toolset._client = mock_client

        with patch("voice_agent.tools.rag_tool.perf_counter", side_effect=[8.0, 8.15]):
            result = await toolset.ask_knowledge_base("What services do you offer?")

        self.assertEqual(result, RAG_NO_RECORDS_MARKER)
        mock_client.post.assert_awaited_once()

        telemetry.publish_kb_result.assert_called_once_with(
            success=True,
            latency_ms=150,
            fallback=True,
            context_refs=[],
        )


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

    async def test_get_current_weather_publishes_telemetry(self) -> None:
        telemetry = Mock()
        toolset = WeatherToolset(telemetry=telemetry)
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

        with patch("voice_agent.tools.weather_tool.perf_counter", side_effect=[20.0, 20.3]):
            await toolset.get_current_weather("Lahore")

        telemetry.publish_weather_querying.assert_called_once_with()
        telemetry.publish_weather_result.assert_called_once_with(
            success=True,
            latency_ms=300,
            fallback=False,
        )

    async def test_get_current_weather_unknown_city_publishes_failure_telemetry(self) -> None:
        telemetry = Mock()
        toolset = WeatherToolset(telemetry=telemetry)
        geocode_response = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("GET", "https://geocoding-api.open-meteo.com"),
        )
        mock_client = Mock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=geocode_response)
        toolset._client = mock_client

        with patch("voice_agent.tools.weather_tool.perf_counter", side_effect=[20.0, 20.2]):
            await toolset.get_current_weather("Unknown City")

        telemetry.publish_weather_querying.assert_called_once_with()
        telemetry.publish_weather_result.assert_called_once_with(
            success=False,
            latency_ms=200,
            fallback=True,
        )


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
            agent = agent_server.LiveKitRagVoiceAgent()

        function_tools = ToolContext(agent.tools).function_tools
        self.assertIn("ask_knowledge_base", function_tools)
        self.assertIn("get_current_weather", function_tools)

    def test_system_instructions_cover_document_routing_and_hide_tool_names(self) -> None:
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

        self.assertIn("document-grounded questions about coverage", agent_server.SYSTEM_INSTRUCTIONS)
        self.assertIn("Never say tool names", agent_server.SYSTEM_INSTRUCTIONS)
        self.assertIn("authoritative records for that turn", agent_server.SYSTEM_INSTRUCTIONS)

    def test_resolve_turn_tool_choice_forces_kb_for_document_question(self) -> None:
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

        telemetry = Mock()
        telemetry.current_turn_has_tool = False
        telemetry.routing_last_answer_path = "unknown"
        chat_ctx = agent_server.llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="How soon must a Purchase Protection claim be filed?")

        tool_choice, reason, message = agent_server._resolve_turn_tool_choice(
            chat_ctx,
            telemetry,
            ModelSettings(),
        )

        self.assertEqual(
            tool_choice,
            {"type": "function", "function": {"name": "ask_knowledge_base"}},
        )
        self.assertEqual(reason, "doc_topic_question")
        self.assertIn("Purchase Protection claim", message)

    def test_resolve_turn_tool_choice_keeps_general_question_on_auto(self) -> None:
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

        telemetry = Mock()
        telemetry.current_turn_has_tool = False
        telemetry.routing_last_answer_path = "unknown"
        chat_ctx = agent_server.llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="Explain what FastAPI is.")

        tool_choice, reason, _ = agent_server._resolve_turn_tool_choice(
            chat_ctx,
            telemetry,
            ModelSettings(),
        )

        self.assertIsNone(tool_choice)
        self.assertEqual(reason, "default")

    def test_resolve_turn_tool_choice_stops_forcing_after_tool_runs(self) -> None:
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

        telemetry = Mock()
        telemetry.current_turn_has_tool = True
        telemetry.routing_last_answer_path = "knowledge_base"
        chat_ctx = agent_server.llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="What is the company website and number?")

        tool_choice, reason, message = agent_server._resolve_turn_tool_choice(
            chat_ctx,
            telemetry,
            ModelSettings(),
        )

        self.assertIsNone(tool_choice)
        self.assertEqual(reason, "tool_already_used_this_turn")
        self.assertIn("company website and number", message.lower())


class RoutingTests(unittest.TestCase):
    def test_decide_route_detects_document_question(self) -> None:
        decision = decide_route("How long does the user have to submit documents?")

        self.assertEqual(decision.route, "knowledge_base")

    def test_decide_route_detects_kb_follow_up(self) -> None:
        decision = decide_route("asking about that coverage", last_answer_path="knowledge_base")

        self.assertEqual(decision.route, "knowledge_base")

    def test_decide_route_detects_kb_topic_follow_up(self) -> None:
        decision = decide_route(
            "Are cameras and other electronic equipment covered?",
            last_answer_path="knowledge_base",
        )

        self.assertEqual(decision.route, "knowledge_base")

    def test_decide_route_detects_weather(self) -> None:
        decision = decide_route("What's the weather in Lahore?")

        self.assertEqual(decision.route, "weather")

    def test_decide_route_detects_coverage_question_without_explicit_doc_reference(self) -> None:
        decision = decide_route("Are cameras and other electronic equipment covered?")

        self.assertEqual(decision.route, "knowledge_base")

    def test_decide_route_leaves_general_question_on_auto(self) -> None:
        decision = decide_route("Explain what FastAPI is.")

        self.assertEqual(decision.route, "auto")


class VoiceAgentTelemetryTests(unittest.IsolatedAsyncioTestCase):
    def test_publish_initial_state_uses_idle_defaults(self) -> None:
        messages: list[dict[str, object]] = []
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: messages.append(json.loads(payload)),
        )

        telemetry.publish_initial_state()

        self.assertEqual(messages[-1]["ragBackend"], "warming_up")
        self.assertEqual(messages[-1]["lastAnswerPath"], "unknown")
        self.assertEqual(messages[-1]["lastFallback"], None)
        self.assertEqual(
            messages[-1]["knowledgeBase"],
            {"status": "idle", "latencyMs": None, "fallback": None, "contextRefs": []},
        )
        self.assertEqual(
            messages[-1]["weather"],
            {"status": "idle", "latencyMs": None, "fallback": None, "contextRefs": []},
        )
        self.assertEqual(
            messages[-1]["pipeline"],
            {"sttLatencyMs": None, "llmLatencyMs": None, "ttsLatencyMs": None, "inputMode": None},
        )

    async def test_startup_ready_probe_publishes_ready(self) -> None:
        probe = AsyncMock(return_value=True)
        messages: list[dict[str, object]] = []
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: messages.append(json.loads(payload)),
            readiness_probe=probe,
        )
        telemetry.publish_initial_state()

        await telemetry.publish_startup_ready_state()

        self.assertEqual(messages[-1]["ragBackend"], "ready")
        probe.assert_awaited_once_with("http://localhost:8000/ready")

    async def test_publish_initial_state_supports_async_publishers(self) -> None:
        messages: list[dict[str, object]] = []

        async def publisher(payload: str) -> None:
            messages.append(json.loads(payload))

        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=publisher,
        )

        telemetry.publish_initial_state()
        await asyncio.sleep(0)

        self.assertEqual(messages[-1]["ragBackend"], "warming_up")

    async def test_startup_ready_probe_publishes_degraded(self) -> None:
        messages: list[dict[str, object]] = []
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: messages.append(json.loads(payload)),
            readiness_probe=AsyncMock(return_value=False),
        )
        telemetry.publish_initial_state()

        await telemetry.publish_startup_ready_state()

        self.assertEqual(messages[-1]["ragBackend"], "degraded")

    def test_mark_normal_reply_requires_generate_reply(self) -> None:
        messages: list[dict[str, object]] = []
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: messages.append(json.loads(payload)),
        )

        telemetry.start_user_turn()
        telemetry.mark_normal_reply("say")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[-1]["lastAnswerPath"], "unknown")

        telemetry.start_user_turn()
        telemetry.mark_normal_reply("generate_reply")
        self.assertEqual(messages[-1]["lastAnswerPath"], "normal")
        self.assertEqual(messages[-1]["lastFallback"], False)

    def test_multi_tool_summary_follows_last_executed_tool(self) -> None:
        messages: list[dict[str, object]] = []
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: messages.append(json.loads(payload)),
        )

        telemetry.start_user_turn()
        telemetry.publish_kb_result(success=True, latency_ms=180, fallback=False, context_refs=[])
        telemetry.mark_tool_turn("ask_knowledge_base")
        telemetry.publish_weather_result(success=False, latency_ms=240, fallback=True)
        telemetry.mark_tool_turn("get_current_weather")

        self.assertEqual(messages[-1]["lastAnswerPath"], "weather")
        self.assertEqual(messages[-1]["lastFallback"], True)

    def test_kb_success_restores_rag_backend_ready(self) -> None:
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: None,
        )

        telemetry.publish_kb_result(success=False, latency_ms=210, fallback=True, context_refs=[])
        self.assertEqual(telemetry.snapshot.rag_backend, "degraded")

        telemetry.publish_kb_result(success=True, latency_ms=190, fallback=False, context_refs=[])
        self.assertEqual(telemetry.snapshot.rag_backend, "ready")

    def test_kb_result_preserves_context_refs(self) -> None:
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: None,
        )

        telemetry.publish_kb_result(
            success=True,
            latency_ms=190,
            fallback=False,
            context_refs=[{"sourceId": "document:12", "filename": "Guide To Benefits.pdf"}],
        )

        self.assertEqual(
            telemetry.snapshot.knowledge_base.context_refs,
            [{"sourceId": "document:12", "filename": "Guide To Benefits.pdf"}],
        )

    def test_start_user_turn_resets_tool_states_but_keeps_backend_state(self) -> None:
        messages: list[dict[str, object]] = []
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: messages.append(json.loads(payload)),
        )

        telemetry.publish_kb_result(
            success=True,
            latency_ms=427,
            fallback=False,
            context_refs=[{"sourceId": "document:1"}],
        )
        telemetry.publish_weather_result(success=True, latency_ms=1366, fallback=False)
        telemetry.snapshot.rag_backend = "ready"
        telemetry.start_user_turn()

        self.assertEqual(telemetry.snapshot.last_answer_path, "unknown")
        self.assertEqual(telemetry.snapshot.last_fallback, None)
        self.assertEqual(
            telemetry.snapshot.knowledge_base.to_payload(),
            {"status": "idle", "latencyMs": None, "fallback": None, "contextRefs": []},
        )
        self.assertEqual(
            telemetry.snapshot.weather.to_payload(),
            {"status": "idle", "latencyMs": None, "fallback": None, "contextRefs": []},
        )
        self.assertEqual(telemetry.snapshot.rag_backend, "ready")
        self.assertEqual(messages[-1]["knowledgeBase"]["status"], "idle")
        self.assertEqual(messages[-1]["weather"]["status"], "idle")

    def test_pipeline_latency_updates_are_published(self) -> None:
        messages: list[dict[str, object]] = []
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: messages.append(json.loads(payload)),
        )

        telemetry.publish_stt_latency(190)
        telemetry.publish_llm_latency(620)
        telemetry.publish_tts_latency(280)

        self.assertEqual(
            telemetry.snapshot.pipeline.to_payload(),
            {"sttLatencyMs": 190, "llmLatencyMs": 620, "ttsLatencyMs": 280, "inputMode": None},
        )
        self.assertEqual(messages[-1]["pipeline"]["sttLatencyMs"], 190)
        self.assertEqual(messages[-1]["pipeline"]["llmLatencyMs"], 620)
        self.assertEqual(messages[-1]["pipeline"]["ttsLatencyMs"], 280)

    def test_start_user_turn_keeps_latest_pipeline_latencies(self) -> None:
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: None,
        )

        telemetry.publish_stt_latency(210)
        telemetry.publish_llm_latency(540)
        telemetry.publish_tts_latency(310)
        telemetry.start_user_turn(input_mode="text")

        self.assertEqual(
            telemetry.snapshot.pipeline.to_payload(),
            {"sttLatencyMs": None, "llmLatencyMs": None, "ttsLatencyMs": None, "inputMode": "text"},
        )

    def test_routing_last_answer_path_survives_new_turn_reset(self) -> None:
        telemetry = VoiceAgentTelemetry(
            session_id="room-123",
            rag_backend_url="http://localhost:8000",
            publisher=lambda payload: None,
        )

        telemetry.mark_tool_turn("ask_knowledge_base")
        telemetry.start_user_turn()

        self.assertEqual(telemetry.snapshot.last_answer_path, "unknown")
        self.assertEqual(telemetry.routing_last_answer_path, "knowledge_base")
