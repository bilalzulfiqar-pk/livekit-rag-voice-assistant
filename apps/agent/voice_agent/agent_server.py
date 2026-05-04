from __future__ import annotations

import asyncio
import json
import logging

from dotenv import load_dotenv
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
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from voice_agent.config import Settings
from voice_agent.models import VoiceCapabilities, VoiceSessionInfo
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
    "If the user asks about the uploaded guide or document, including phrases like "
    "'this guide', 'this document', or 'this benefits guide', use ask_knowledge_base. "
    "Use get_current_weather only for current weather questions about a named city. "
    "For weather, only current conditions are supported. Never offer forecasts, weekly outlooks, "
    "historical weather, alerts, or unsupported weather features. Do not ask the user if they want a forecast. "
    "Answer general questions normally without tools. "
    "Do not claim live web browsing, email, calendar, reminders, music, file access, "
    "or unrelated external actions. "
    "If a tool fails, say so plainly and continue helpfully. "
    "Do not mention internal retrieval steps, tool calls, or backend details unless the user asks. "
    "Keep responses conversational, allow interruptions, and ask follow-up "
    "questions only when they improve the conversation. After a short factual answer, stop instead of "
    "offering to repeat it. Do not say 'again' unless the user explicitly asks for repetition. "
    "If you are unsure about a fact, say so plainly instead of guessing."
)

OPENING_MESSAGE = (
    "Hello, I'm the LiveKit RAG Voice Assistant. Ask about your documents, the weather, or anything general."
)

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


class AuralisVoiceAgent(Agent):
    def __init__(self, *, telemetry: VoiceAgentTelemetry | None = None) -> None:
        super().__init__(
            instructions=SYSTEM_INSTRUCTIONS,
            tools=[
                KnowledgeBaseToolset(
                    backend_url=settings.rag_backend_url,
                    chat_path=settings.rag_chat_path,
                    telemetry=telemetry,
                ),
                WeatherToolset(telemetry=telemetry),
            ],
        )


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
        telemetry.start_user_turn()
        logger.info(
            "user input transcribed",
            extra={"transcript": event.transcript.strip()},
        )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event) -> None:
        item = getattr(event, "item", None)
        if getattr(item, "role", "") == "user":
            telemetry.start_user_turn()

    @session.on("function_tools_executed")
    def on_function_tools_executed(event) -> None:
        for function_call in getattr(event, "function_calls", []):
            function_name = getattr(function_call, "name", "")
            telemetry.mark_tool_turn(function_name)

    @session.on("speech_created")
    def on_speech_created(event) -> None:
        telemetry.mark_normal_reply(getattr(event, "source", ""))

    await session.start(
        room=ctx.room,
        agent=AuralisVoiceAgent(telemetry=telemetry),
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
