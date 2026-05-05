from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from livekit.agents import llm
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    WorkerPermissions,
    inference,
    room_io,
)
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from voice_agent.config import Settings
from voice_agent.models import VoiceCapabilities, VoiceSessionInfo
from voice_agent.routing import decide_route
from voice_agent.telemetry import TOOLING_STATUS_TOPIC, VoiceAgentTelemetry
from voice_agent.tools import KnowledgeBaseToolset, WeatherToolset

load_dotenv()
settings = Settings.from_env()
logger = logging.getLogger("livekit-rag-voice-agent")

AGENT_DISPLAY_NAME = "LiveKit RAG Voice Assistant"

SYSTEM_INSTRUCTIONS = (
    "You are the LiveKit RAG Voice Assistant, a helpful realtime voice agent. "
    "Speak briefly, clearly, and naturally. Keep final spoken replies to one "
    "to three short sentences. "
    "Use ask_knowledge_base for company, FAQ, policy, support, and uploaded-document questions. "
    "That tool returns retrieved document excerpts, and you must synthesize the final grounded answer yourself. "
    "If the user asks about the uploaded guide or document, including phrases like "
    "'this guide' or 'this document', use ask_knowledge_base. "
    "For document-grounded questions about coverage, exclusions, limits, deadlines, eligibility, required documents, "
    "claim steps, procedures, benefits, reimbursement, support details, or contact details, "
    "use ask_knowledge_base instead of answering from general memory. "
    "If the user asks for a website, phone number, support number, claim number, or contact information that should "
    "come from the uploaded documents, use ask_knowledge_base. "
    "If the user gives a short follow-up like 'yes', 'more', 'tell me more', "
    "'what about that', or 'and that' after a knowledge-base answer, treat it as a follow-up "
    "to the same document topic and use ask_knowledge_base again with a clear standalone question. "
    "If the user sends a short clarification or correction after a knowledge-base answer, treat it as a document "
    "follow-up and use ask_knowledge_base again with a clearer standalone question. "
    "Use get_current_weather only for current weather questions about a named city. "
    "For weather, only current conditions are supported. Never offer forecasts, weekly outlooks, "
    "historical weather, alerts, or unsupported weather features. Do not ask the user if they want a forecast. "
    "Answer general questions normally without tools. "
    "When the knowledge base contains the answer, answer directly like a helpful company assistant. "
    "When ask_knowledge_base returns retrieved excerpts, answer only from those excerpts. "
    "If ask_knowledge_base provides a Preferred grounded answer, use that answer unless the user asked for more detail. "
    "If ask_knowledge_base returns retrieved excerpts, treat them as the authoritative records for that turn. "
    "Do not say the records lack the answer if excerpts were returned. "
    "If the excerpts include an exact website, phone number, deadline, amount, duration, limit, exclusion, or required document, "
    "state that exact fact directly instead of replacing it with vague advice. "
    "Do not weaken exact deadlines into phrases like 'there is no specific deadline' when a deadline is stated in the excerpts. "
    "Only if ask_knowledge_base returns the exact marker [KB_NO_RECORDS], say exactly: "
    "\"I'm sorry, I don't have that information in my records.\" Never use that sentence unless the KB tool just returned that marker. "
    "Do not guess or fill in missing facts. "
    "Never say tool names like ask_knowledge_base or get_current_weather to the user. Call tools silently. "
    "Make grounded spoken answers sound natural rather than document-like. "
    "If the excerpts use legal duplicate number formats like 'twenty (20)' or 'ninety (90)', say the number only once in a clean spoken form. "
    "Do not tell the user to check, refer to, read, upload, or review the documents or policy unless "
    "the knowledge base does not contain enough information or the tool fails. "
    "Do not add generic disclaimers such as 'typically', 'please review your policy', or "
    "'contact support' when the knowledge-base answer is already sufficient. "
    "Do not claim live web browsing, email, calendar, reminders, music, file access, "
    "or unrelated external actions. "
    "If a tool fails, say so plainly and continue helpfully. "
    "Do not mention internal retrieval steps, tool calls, or backend details unless the user asks. "
    "Keep responses conversational, allow interruptions, and ask follow-up "
    "questions only when they improve the conversation. After a short factual answer, stop instead of "
    "offering to repeat it. Do not say 'again' unless the user explicitly asks for repetition. "
    "Do not volunteer offers like 'Would you like details?' or 'Would you like the full coverage details?' "
    "after a grounded answer unless the user explicitly asks for more. "
    "If you are unsure about a fact, say so plainly instead of guessing."
)

OPENING_MESSAGE = (
    "Hello, I'm the LiveKit RAG Voice Assistant. Ask about your documents, the weather, or anything general."
)

KB_TOOL_CHOICE: llm.ToolChoice = {"type": "function", "function": {"name": "ask_knowledge_base"}}
WEATHER_TOOL_CHOICE: llm.ToolChoice = {
    "type": "function",
    "function": {"name": "get_current_weather"},
}


def _seconds_to_ms(value: float | int | None) -> int | None:
    if value is None:
        return None
    return max(round(float(value) * 1000), 0)


def _resolve_turn_tool_choice(
    chat_ctx: llm.ChatContext,
    telemetry: VoiceAgentTelemetry,
    model_settings: ModelSettings,
) -> tuple[llm.ToolChoice | None, str, str]:
    if model_settings.tool_choice == "none":
        return None, "preserve_none", ""

    latest_user_message = _latest_user_message(chat_ctx)
    if telemetry.current_turn_has_tool:
        return None, "tool_already_used_this_turn", latest_user_message

    if not latest_user_message:
        return None, "no_user_message", ""

    decision = decide_route(
        latest_user_message,
        last_answer_path=telemetry.routing_last_answer_path,
    )
    if decision.route == "knowledge_base":
        return KB_TOOL_CHOICE, decision.reason, latest_user_message
    if decision.route == "weather":
        return WEATHER_TOOL_CHOICE, decision.reason, latest_user_message
    return None, decision.reason, latest_user_message


def _latest_user_message(chat_ctx: llm.ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if getattr(item, "role", "") != "user":
            continue
        text_content = getattr(item, "text_content", "") or ""
        normalized = " ".join(str(text_content).split())
        if normalized:
            return normalized
    return ""

server = AgentServer(
    permissions=WorkerPermissions(
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        can_update_metadata=True,
        hidden=False,
    )
)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


class LiveKitRagVoiceAgent(Agent):
    def __init__(self, *, telemetry: VoiceAgentTelemetry | None = None) -> None:
        self._telemetry = telemetry
        super().__init__(
            instructions=SYSTEM_INSTRUCTIONS,
            tools=[
                KnowledgeBaseToolset(
                    backend_url=settings.rag_backend_url,
                    context_path=settings.rag_context_path,
                    telemetry=telemetry,
                ),
                WeatherToolset(telemetry=telemetry),
            ],
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[llm.ChatChunk | str, None]:
        telemetry = self._telemetry
        forced_tool_choice: llm.ToolChoice | None = None
        route_reason = "no_telemetry"
        route_message = ""
        if telemetry is not None:
            forced_tool_choice, route_reason, route_message = _resolve_turn_tool_choice(
                chat_ctx,
                telemetry,
                model_settings,
            )

        routed_model_settings = (
            model_settings
            if forced_tool_choice is None
            else ModelSettings(tool_choice=forced_tool_choice)
        )
        logger.info(
            "llm turn route decided",
            extra={
                "tool_choice": forced_tool_choice if forced_tool_choice is not None else "auto",
                "reason": route_reason,
                "user_message": route_message,
            },
        )

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, routed_model_settings):
            yield chunk


async def _publish_metadata(ctx: JobContext) -> None:
    capabilities = VoiceCapabilities(
        stt_model=settings.stt_model,
        llm_model=settings.llm_model,
        tts_model=settings.tts_model,
        tts_voice=settings.tts_voice,
    )
    session_info = VoiceSessionInfo(
        session_id=ctx.room.name,
        agent_name=AGENT_DISPLAY_NAME,
    )
    await ctx.room.local_participant.set_attributes(
        {
            "app.voice.capabilities": json.dumps(capabilities.to_payload()),
            "app.voice.session": json.dumps(session_info.to_payload()),
        }
    )


@server.rtc_session(agent_name=settings.livekit_agent_name)
async def entrypoint(ctx: JobContext) -> None:
    turn_detector = MultilingualModel()
    telemetry = VoiceAgentTelemetry(
        session_id=ctx.room.name,
        rag_backend_url=settings.rag_backend_url,
        publisher=lambda payload: ctx.room.local_participant.publish_data(
            payload,
            reliable=True,
            topic=TOOLING_STATUS_TOPIC,
        ),
    )

    session = AgentSession(
        stt=inference.STT(model=settings.stt_model),
        llm=inference.LLM(model=settings.llm_model),
        tts=inference.TTS(model=settings.tts_model, voice=settings.tts_voice),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection=turn_detector,
            preemptive_generation={"enabled": True},
        ),
    )

    @session.on("agent_state_changed")
    def on_agent_state_changed(event) -> None:
        logger.info(
            "agent state changed",
            extra={
                "old_state": event.old_state,
                "new_state": event.new_state,
            },
        )

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event) -> None:
        if not event.is_final:
            return
        transcript = event.transcript.strip()
        telemetry.start_user_turn(input_mode="voice")
        logger.info(
            "user input transcribed",
            extra={"transcript": transcript},
        )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event) -> None:
        item = getattr(event, "item", None)
        if getattr(item, "role", "") == "user":
            telemetry.start_user_turn(input_mode="text")

    @session.on("function_tools_executed")
    def on_function_tools_executed(event) -> None:
        for function_call in getattr(event, "function_calls", []):
            function_name = getattr(function_call, "name", "")
            telemetry.mark_tool_turn(function_name)

    @session.on("metrics_collected")
    def on_metrics_collected(event) -> None:
        metrics = getattr(event, "metrics", None)
        metrics_type = getattr(metrics, "type", "")
        if metrics_type == "eou_metrics":
            latency_ms = _seconds_to_ms(getattr(metrics, "transcription_delay", None))
            if latency_ms is not None:
                telemetry.publish_stt_latency(latency_ms)
            return

        if metrics_type in {"llm_metrics", "realtime_model_metrics"}:
            latency_ms = _seconds_to_ms(getattr(metrics, "ttft", None))
            if latency_ms is not None:
                telemetry.publish_llm_latency(latency_ms)
            return

        if metrics_type == "tts_metrics":
            latency_ms = _seconds_to_ms(getattr(metrics, "ttfb", None))
            if latency_ms is not None:
                telemetry.publish_tts_latency(latency_ms)

    @session.on("speech_created")
    def on_speech_created(event) -> None:
        telemetry.mark_normal_reply(getattr(event, "source", ""))

    await session.start(
        room=ctx.room,
        agent=LiveKitRagVoiceAgent(telemetry=telemetry),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
            text_input=True,
        ),
    )
    await ctx.connect()
    await _publish_metadata(ctx)
    telemetry.publish_initial_state()
    asyncio.create_task(telemetry.publish_startup_ready_state())
    session.say(OPENING_MESSAGE)
