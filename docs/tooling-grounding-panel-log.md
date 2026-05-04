# Tooling / Grounding Panel Log

## Summary
- Added app-level tool telemetry for the LiveKit RAG Voice Assistant without using LiveKit observability.
- Introduced a lightweight snapshot topic, `app.tooling.status`, published by the Python agent and consumed by the web app.
- Extended the existing system status panel with a compact Tooling / Grounding area for KB status, weather status, answer path, fallback state, and tool latency.

## Agent Changes
- Added `voice_agent.telemetry.VoiceAgentTelemetry` to own the latest tooling snapshot and publish fire-and-forget JSON payloads over LiveKit data messages.
- Added one-time startup RAG readiness probing through `GET /ready`.
- Added turn tracking from:
  - `user_input_transcribed`
  - `conversation_item_added` for typed user turns
  - `function_tools_executed`
  - `speech_created`
- The KB tool now reports:
  - querying
  - success or failed
  - latency
  - fallback usage from backend `answer_path`
- The weather tool now reports:
  - querying
  - success or failed
  - latency
  - fallback usage for unknown-city and upstream failures

## Frontend Changes
- Added a typed parser for tooling snapshots in `apps/web/src/lib/voice.ts`.
- Added stale-sequence protection so older packets do not overwrite newer UI state.
- Added a `RoomEvent.DataReceived` listener in `apps/web/src/components/voice-agent-app.tsx`.
- Added a compact Tooling / Grounding row under the existing pipeline cards.

## Snapshot Contract
```json
{
  "type": "tooling_snapshot",
  "version": 1,
  "sequence": 4,
  "sessionId": "livekit-rag-session-123",
  "lastAnswerPath": "knowledge_base",
  "lastFallback": false,
  "ragBackend": "ready",
  "knowledgeBase": {
    "status": "success",
    "latencyMs": 412,
    "fallback": false
  },
  "weather": {
    "status": "idle",
    "latencyMs": null,
    "fallback": null
  }
}
```

## Notes
- `speech_created(source="generate_reply")` was verified against the installed LiveKit agents SDK before using it for normal-answer detection.
- `ragBackend` is restored to `ready` after a successful KB request following a previous degraded state.
- `updatedAt` was intentionally omitted from milestone 1.5 because it is not rendered in the current UI.
