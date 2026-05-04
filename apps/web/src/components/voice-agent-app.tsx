"use client";

import {
  useCallback,
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  RoomAudioRenderer,
  SessionProvider,
  useAgent,
  useParticipantAttributes,
  useSession,
  useSessionMessages,
  type ReceivedMessage,
} from "@livekit/components-react";
import {
  AlertTriangle,
  ChevronDown,
  Cpu,
  Keyboard,
  LoaderCircle,
  MessageSquareText,
  Mic,
  MicOff,
  PhoneOff,
  Radio,
  Send,
  Settings2,
  ShieldCheck,
  Trash2,
  Volume2,
  Waves,
} from "lucide-react";
import { ConnectionState, Room, RoomEvent, TokenSource } from "livekit-client";

import { SessionTranscript } from "@/components/session-transcript";
import { StatusChip } from "@/components/status-chip";
import { VoiceOrb } from "@/components/voice-orb";
import {
  createDefaultToolingSnapshot,
  parseToolingSnapshot,
  parseVoiceCapabilities,
  parseVoiceSessionInfo,
  TOOLING_STATUS_TOPIC,
  type TranscriptEntry,
  type ToolingSnapshot,
  type VoiceCapabilities,
  type VoiceControlMode,
} from "@/lib/voice";

const TOKEN_SOURCE = TokenSource.endpoint("/api/livekit/token");
const AGENT_NAME =
  process.env.NEXT_PUBLIC_LIVEKIT_AGENT_NAME ?? "livekit-rag-voice-agent";
const HAS_PUBLIC_URL = Boolean(process.env.NEXT_PUBLIC_LIVEKIT_URL);

const SAMPLE_PROMPTS = [
  "What services do you offer?",
  "What is the weather in Lahore?",
  "Explain what FastAPI is.",
];
const LIVE_TRANSCRIPT_SETTLE_MS = 1500;

const STATE_COPY = {
  idle: {
    title: "Talk to your RAG voice assistant",
    description:
      "Natural conversation with realtime voice, grounded document answers, and tool calling.",
  },
  connecting: {
    title: "Initializing voice session...",
    description: "Preparing the room, microphone, and realtime audio pipeline.",
  },
  listening: {
    title: "I'm listening.",
    description:
      "Speak naturally and the assistant will follow your turn in real time.",
  },
  thinking: {
    title: "Let me think about that.",
    description: "Composing a short response with pre-response generation enabled.",
  },
  speaking: {
    title: "Replying now.",
    description: "Audio is streaming back into the room and remains interruptible.",
  },
  reconnecting: {
    title: "Connection lost. Rejoining session...",
    description: "The session is trying to recover without losing the conversation.",
  },
  failed: {
    title: "The session needs attention.",
    description: "Check your microphone, LiveKit credentials, or current network state.",
  },
} as const;

function createRoomName() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `livekit-rag-session-${crypto.randomUUID()}`;
  }
  return `livekit-rag-session-${Date.now()}`;
}

function formatConnectionState(state: ConnectionState) {
  switch (state) {
    case ConnectionState.Connected:
      return "Active";
    case ConnectionState.Connecting:
      return "Connecting";
    case ConnectionState.Reconnecting:
    case ConnectionState.SignalReconnecting:
      return "Reconnecting";
    default:
      return "Offline";
  }
}

function connectionTone(
  state: ConnectionState,
): "neutral" | "live" | "success" | "warn" {
  switch (state) {
    case ConnectionState.Connected:
      return "success";
    case ConnectionState.Connecting:
      return "live";
    case ConnectionState.Reconnecting:
    case ConnectionState.SignalReconnecting:
      return "warn";
    default:
      return "neutral";
  }
}

function agentStateLabel(state: string, connectionState: ConnectionState) {
  if (
    connectionState === ConnectionState.Reconnecting ||
    connectionState === ConnectionState.SignalReconnecting
  ) {
    return "Reconnecting";
  }

  switch (state) {
    case "pre-connect-buffering":
      return "Buffering";
    case "connecting":
    case "initializing":
      return "Connecting";
    case "idle":
      return "Ready";
    case "listening":
      return "Listening";
    case "thinking":
      return "Thinking";
    case "speaking":
      return "Speaking";
    case "failed":
      return "Failed";
    default:
      return "Standby";
  }
}

function agentTone(
  state: string,
  connectionState: ConnectionState,
): "neutral" | "live" | "success" | "warn" | "accent" {
  if (
    connectionState === ConnectionState.Reconnecting ||
    connectionState === ConnectionState.SignalReconnecting
  ) {
    return "warn";
  }

  if (state === "listening" || state === "speaking") {
    return "live";
  }
  if (state === "thinking") {
    return "accent";
  }
  if (state === "failed") {
    return "warn";
  }
  if (state === "idle") {
    return "success";
  }
  return "neutral";
}

function heroState(
  state: string,
  connectionState: ConnectionState,
  hasError: boolean,
) {
  if (hasError || state === "failed") {
    return "failed";
  }
  if (
    connectionState === ConnectionState.Reconnecting ||
    connectionState === ConnectionState.SignalReconnecting
  ) {
    return "reconnecting";
  }
  if (
    connectionState === ConnectionState.Connecting ||
    state === "connecting" ||
    state === "initializing" ||
    state === "pre-connect-buffering"
  ) {
    return "connecting";
  }
  if (state === "listening" || state === "thinking" || state === "speaking") {
    return state;
  }
  return "idle";
}

function formatMicLabel(
  isConnected: boolean,
  isMuted: boolean,
  controlMode: VoiceControlMode,
  isPressingToTalk: boolean,
) {
  if (!isConnected) {
    return "Mic offline";
  }
  if (isMuted) {
    return "Mic muted";
  }
  if (controlMode === "push-to-talk") {
    return isPressingToTalk ? "Push-to-talk live" : "Push-to-talk armed";
  }
  return "Mic live";
}

function getMessageText(message: ReceivedMessage) {
  if ("message" in message && typeof message.message === "string") {
    return message.message;
  }
  return "";
}

function getMessageRole(message: ReceivedMessage, localIdentity: string) {
  if (message.type === "userTranscript") {
    return "user";
  }
  if (message.type === "agentTranscript") {
    return "assistant";
  }
  return message.from?.identity === localIdentity ? "user" : "assistant";
}

function toneForPipeline(value: string): "neutral" | "live" | "success" | "accent" {
  if (
    value === "Streaming" ||
    value === "Reasoning" ||
    value === "Replying" ||
    value === "Active"
  ) {
    return "live";
  }
  if (value === "Enabled" || value === "Ready" || value === "Adaptive") {
    return "success";
  }
  if (value === "Preparing") {
    return "accent";
  }
  return "neutral";
}

function pipelineCardToneClass(value: string) {
  switch (toneForPipeline(value)) {
    case "live":
      return "pipeline-card-live";
    case "success":
      return "pipeline-card-success";
    case "accent":
      return "pipeline-card-accent";
    default:
      return "pipeline-card-neutral";
  }
}

function toolingStatusLabel(status: ToolingSnapshot["knowledgeBase"]["status"]) {
  switch (status) {
    case "querying":
      return "Querying";
    case "success":
      return "Success";
    case "failed":
      return "Failed";
    default:
      return "Idle";
  }
}

function toolingStatusTone(
  status: ToolingSnapshot["knowledgeBase"]["status"],
): "neutral" | "live" | "success" | "warn" {
  switch (status) {
    case "querying":
      return "live";
    case "success":
      return "success";
    case "failed":
      return "warn";
    default:
      return "neutral";
  }
}

function answerPathLabel(answerPath: ToolingSnapshot["lastAnswerPath"]) {
  switch (answerPath) {
    case "knowledge_base":
      return "Knowledge Base";
    case "weather":
      return "Weather";
    case "normal":
      return "Normal";
    default:
      return "Waiting";
  }
}

function ragBackendLabel(ragBackend: ToolingSnapshot["ragBackend"]) {
  switch (ragBackend) {
    case "warming_up":
      return "Warming up";
    case "ready":
      return "Ready";
    case "degraded":
      return "Degraded";
    default:
      return "Unknown";
  }
}

function ragBackendTone(
  ragBackend: ToolingSnapshot["ragBackend"],
): "neutral" | "live" | "success" | "warn" {
  switch (ragBackend) {
    case "warming_up":
      return "live";
    case "ready":
      return "success";
    case "degraded":
      return "warn";
    default:
      return "neutral";
  }
}

function fallbackLabel(fallback: boolean | null) {
  if (fallback == null) {
    return "Waiting";
  }
  return fallback ? "Yes" : "No";
}

function formatLatency(latencyMs: number | null) {
  return latencyMs == null ? "Awaiting" : `${latencyMs} ms`;
}

function AuralisLogoMark() {
  return (
    <div className="auralis-logo-mark" aria-hidden="true">
      <span className="auralis-logo-ring auralis-logo-ring-outer" />
      <span className="auralis-logo-ring auralis-logo-ring-inner" />
      <span className="auralis-logo-core" />
      <span className="auralis-logo-bars">
        <span />
        <span />
        <span />
      </span>
    </div>
  );
}

type VoiceAgentShellProps = {
  onSessionReset: () => void;
};

function VoiceAgentShell({ onSessionReset }: VoiceAgentShellProps) {
  const [sessionSeed, setSessionSeed] = useState(0);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [textInputValue, setTextInputValue] = useState("");
  const [textInputError, setTextInputError] = useState<string | null>(null);
  const [controlMode, setControlMode] = useState<VoiceControlMode>("auto");
  const [isMuted, setIsMuted] = useState(false);
  const [isPressingToTalk, setIsPressingToTalk] = useState(false);
  const [isDebugOpen, setIsDebugOpen] = useState(false);
  const [isSystemStatusOpen, setIsSystemStatusOpen] = useState(false);
  const [toolingSnapshot, setToolingSnapshot] = useState<ToolingSnapshot | null>(null);
  const [interruptedIds, setInterruptedIds] = useState<string[]>([]);
  const [liveTranscriptIds, setLiveTranscriptIds] = useState<string[]>([]);
  const [needsFreshSession, setNeedsFreshSession] = useState(false);
  const [pendingSessionStart, setPendingSessionStart] = useState(false);
  const [hasEnteredConsole, setHasEnteredConsole] = useState(false);
  const [isActivatingConsole, setIsActivatingConsole] = useState(false);
  const [isResettingShell, setIsResettingShell] = useState(false);
  const previousAgentStateRef = useRef<string>("disconnected");
  const transcriptTimersRef = useRef<Map<string, number>>(new Map());
  const previousTranscriptTextsRef = useRef<Map<string, string>>(new Map());
  const recentSpeakingAssistantIdRef = useRef<string | null>(null);
  const recentSpeakingTimestampRef = useRef(0);
  const roomName = useMemo(() => {
    // sessionSeed intentionally forces a fresh room name after a reset.
    void sessionSeed;
    return createRoomName();
  }, [sessionSeed]);

  const room = useMemo(
    () => {
      // sessionSeed intentionally forces a fresh room instance after a reset.
      void sessionSeed;
      return new Room({
        audioCaptureDefaults: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
          voiceIsolation: true,
          channelCount: 1,
        },
      });
    },
    [sessionSeed],
  );

  const session = useSession(TOKEN_SOURCE, {
    room,
    roomName,
    agentName: AGENT_NAME,
    participantName: "LiveKit RAG Visitor",
  });
  const agent = useAgent(session);
  const { attributes: agentAttributes } = useParticipantAttributes({
    participant: agent.internal.agentParticipant ?? undefined,
  });
  const sessionMessages = useSessionMessages(session);

  const capabilities = useMemo(
    () => parseVoiceCapabilities(agentAttributes?.["app.voice.capabilities"]),
    [agentAttributes],
  );
  const sessionInfo = useMemo(
    () => parseVoiceSessionInfo(agentAttributes?.["app.voice.session"]),
    [agentAttributes],
  );
  const currentToolingSnapshot = useMemo(
    () => toolingSnapshot ?? createDefaultToolingSnapshot(sessionInfo?.sessionId ?? roomName),
    [roomName, sessionInfo?.sessionId, toolingSnapshot],
  );

  function resetTranscriptState() {
    for (const timerId of transcriptTimersRef.current.values()) {
      window.clearTimeout(timerId);
    }
    transcriptTimersRef.current.clear();
    previousTranscriptTextsRef.current.clear();
    setInterruptedIds([]);
    setLiveTranscriptIds([]);
    setToolingSnapshot(null);
    previousAgentStateRef.current = "disconnected";
    recentSpeakingAssistantIdRef.current = null;
    recentSpeakingTimestampRef.current = 0;
  }

  const baseTranscriptEntries = useMemo(() => {
    const localIdentity = session.room.localParticipant.identity;

    return sessionMessages.messages
      .map<TranscriptEntry>((message) => {
        const normalizedType = message.type ?? "chatMessage";

        return {
          id: message.id,
          role: getMessageRole(message, localIdentity),
          text: getMessageText(message),
          timestamp: message.timestamp,
          isFinal: normalizedType === "chatMessage",
          source: normalizedType === "chatMessage" ? "chat" : "transcript",
        };
      })
      .filter((entry) => entry.text.trim().length > 0)
      .sort((left, right) => left.timestamp - right.timestamp)
      .slice(-100);
  }, [session.room.localParticipant.identity, sessionMessages.messages]);

  useEffect(() => {
    const activeTranscriptIds = new Set(
      baseTranscriptEntries
        .filter((entry) => entry.source === "transcript")
        .map((entry) => entry.id),
    );

    for (const [id, timerId] of transcriptTimersRef.current) {
      if (activeTranscriptIds.has(id)) {
        continue;
      }

      window.clearTimeout(timerId);
      transcriptTimersRef.current.delete(id);
      previousTranscriptTextsRef.current.delete(id);
      setLiveTranscriptIds((currentIds) =>
        currentIds.includes(id)
          ? currentIds.filter((currentId) => currentId !== id)
          : currentIds,
      );
    }

    for (const entry of baseTranscriptEntries) {
      if (entry.source !== "transcript") {
        continue;
      }

      const previousText = previousTranscriptTextsRef.current.get(entry.id);
      if (previousText === entry.text) {
        continue;
      }

      previousTranscriptTextsRef.current.set(entry.id, entry.text);
      const existingTimer = transcriptTimersRef.current.get(entry.id);
      if (existingTimer) {
        window.clearTimeout(existingTimer);
      }

      setLiveTranscriptIds((currentIds) =>
        currentIds.includes(entry.id) ? currentIds : [...currentIds, entry.id],
      );

      const timerId = window.setTimeout(() => {
        transcriptTimersRef.current.delete(entry.id);
        setLiveTranscriptIds((currentIds) =>
          currentIds.filter((currentId) => currentId !== entry.id),
        );
      }, LIVE_TRANSCRIPT_SETTLE_MS);

      transcriptTimersRef.current.set(entry.id, timerId);
    }
  }, [baseTranscriptEntries]);

  useEffect(() => {
    const transcriptTimers = transcriptTimersRef.current;
    return () => {
      for (const timerId of transcriptTimers.values()) {
        window.clearTimeout(timerId);
      }
      transcriptTimers.clear();
    };
  }, []);

  const activeLiveTranscriptId = useMemo(
    () =>
      [...baseTranscriptEntries]
        .reverse()
        .find(
          (entry) =>
            entry.source === "transcript" && liveTranscriptIds.includes(entry.id),
        )?.id ?? null,
    [baseTranscriptEntries, liveTranscriptIds],
  );

  const transcriptEntries = useMemo(
    () =>
      baseTranscriptEntries.map((entry) => {
        const isInterrupted = interruptedIds.includes(entry.id);
        const isActiveLiveTranscript =
          entry.source === "transcript" && entry.id === activeLiveTranscriptId;

        return {
          ...entry,
          isInterrupted,
          isFinal: entry.source === "chat" || (!isActiveLiveTranscript && !isInterrupted),
        };
      }),
    [activeLiveTranscriptId, baseTranscriptEntries, interruptedIds],
  );

  const deferredEntries = useDeferredValue(transcriptEntries);

  useEffect(() => {
    const previousState = previousAgentStateRef.current;
    previousAgentStateRef.current = agent.state;

    const activeLiveEntry = activeLiveTranscriptId
      ? transcriptEntries.find((entry) => entry.id === activeLiveTranscriptId) ?? null
      : null;

    if (agent.state === "speaking" && activeLiveEntry?.role === "assistant") {
      recentSpeakingAssistantIdRef.current = activeLiveEntry.id;
      recentSpeakingTimestampRef.current = Date.now();
      return;
    }

    if (previousState === "speaking") {
      recentSpeakingTimestampRef.current = Date.now();
    }

    if (
      activeLiveEntry?.role === "user" &&
      recentSpeakingAssistantIdRef.current &&
      Date.now() - recentSpeakingTimestampRef.current < LIVE_TRANSCRIPT_SETTLE_MS
    ) {
      const interruptedId = recentSpeakingAssistantIdRef.current;
      setInterruptedIds((currentIds) =>
        currentIds.includes(interruptedId)
          ? currentIds
          : [...currentIds, interruptedId],
      );
      recentSpeakingAssistantIdRef.current = null;
    }
  }, [activeLiveTranscriptId, agent.state, transcriptEntries]);

  const activeInterruptedIds = useMemo(
    () =>
      interruptedIds.filter((id) => {
        const matchingEntry = transcriptEntries.find((entry) => entry.id === id);
        return Boolean(matchingEntry);
      }),
    [interruptedIds, transcriptEntries],
  );

  const decoratedEntries = useMemo(
    () => deferredEntries,
    [deferredEntries],
  );

  const livePartialEntry = useMemo(
    () =>
      activeLiveTranscriptId
        ? transcriptEntries.find((entry) => entry.id === activeLiveTranscriptId) ?? null
        : null,
    [activeLiveTranscriptId, transcriptEntries],
  );
  const activeState = heroState(
    agent.state,
    session.connectionState,
    Boolean(sessionError),
  );
  const desiredMicEnabled =
    session.isConnected &&
    !isMuted &&
    (controlMode === "auto" || isPressingToTalk);

  useEffect(() => {
    if (!session.isConnected) {
      return;
    }

    let cancelled = false;

    async function syncMicrophone() {
      try {
        await session.room.localParticipant.setMicrophoneEnabled(desiredMicEnabled);
      } catch (error) {
        if (!cancelled) {
          const message =
            error instanceof Error && error.message
              ? error.message
              : "The microphone could not be updated.";
          setSessionError(message);
        }
      }
    }

    void syncMicrophone();

    return () => {
      cancelled = true;
    };
  }, [desiredMicEnabled, session.isConnected, session.room.localParticipant]);

  useEffect(() => {
    const room = session.room;
    const agentIdentity = agent.internal.agentParticipant?.identity;
    const decoder = new TextDecoder();

    function handleDataReceived(
      payload: Uint8Array,
      participant?: { identity?: string },
      _kind?: unknown,
      topic?: string,
    ) {
      if (topic !== TOOLING_STATUS_TOPIC) {
        return;
      }
      if (agentIdentity && participant?.identity && participant.identity !== agentIdentity) {
        return;
      }

      const rawPayload = decoder.decode(payload);
      setToolingSnapshot((currentValue) => {
        const nextValue = parseToolingSnapshot(
          rawPayload,
          currentValue?.sequence ?? -1,
        );
        return nextValue ?? currentValue;
      });
    }

    room.on(RoomEvent.DataReceived, handleDataReceived);
    return () => {
      room.off(RoomEvent.DataReceived, handleDataReceived);
    };
  }, [agent.internal.agentParticipant?.identity, session.room]);

  const pipelineItems = useMemo(() => {
    const currentCapabilities: VoiceCapabilities = capabilities ?? {
      sttModel: "deepgram/flux-general-en",
      llmModel: "openai/gpt-4.1-mini",
      ttsModel: "cartesia/sonic-3",
      ttsVoice: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
      vad: "silero",
      turnDetection: "livekit-multilingual",
      noiseCancellation: "bvc",
      interruptions: "adaptive",
      preemptiveGeneration: true,
      textInput: true,
      shortTermMemory: true,
    };

    return [
      {
        label: "Turn detection",
        value: session.isConnected ? "Active" : "Standby",
      },
      {
        label: "VAD",
        value: session.isConnected ? "Active" : "Standby",
      },
      {
        label: "STT",
        value: livePartialEntry?.role === "user" ? "Streaming" : session.isConnected ? "Ready" : "Standby",
      },
      {
        label: "LLM",
        value:
          agent.state === "thinking"
            ? "Reasoning"
            : livePartialEntry?.role === "user" &&
                currentCapabilities.preemptiveGeneration &&
                session.isConnected
              ? "Preparing"
              : session.isConnected
                ? "Ready"
                : "Standby",
      },
      {
        label: "TTS",
        value: agent.state === "speaking" ? "Replying" : session.isConnected ? "Ready" : "Standby",
      },
      {
        label: "Noise filter",
        value:
          currentCapabilities.noiseCancellation.toUpperCase() === "BVC"
            ? "Enabled"
            : "Standby",
      },
      {
        label: "Interruptions",
        value:
          currentCapabilities.interruptions === "adaptive" ? "Adaptive" : "Ready",
      },
    ];
  }, [agent.state, capabilities, livePartialEntry, session.isConnected]);

  const startCurrentSession = useCallback(async () => {
    try {
      setHasEnteredConsole(true);
      setIsActivatingConsole(true);
      setSessionError(null);
      setTextInputError(null);
      await session.start({
        tracks: {
          microphone: {
            enabled: controlMode === "auto" && !isMuted,
          },
        },
      });
      window.setTimeout(() => {
        setIsActivatingConsole(false);
      }, 420);
    } catch (error) {
      setIsActivatingConsole(false);
      const message =
        error instanceof Error && error.message
          ? error.message
          : "The session could not be started.";

      if (
        error instanceof DOMException &&
        (error.name === "NotAllowedError" || error.name === "SecurityError")
      ) {
        setSessionError(
          "Microphone access was blocked. Please allow microphone permission and try again.",
        );
        return;
      }

      setSessionError(message);
    }
  }, [controlMode, isMuted, session]);

  useEffect(() => {
    if (!pendingSessionStart) {
      return;
    }

    let cancelled = false;

    async function startPendingSession() {
      await startCurrentSession();
      if (!cancelled) {
        setPendingSessionStart(false);
      }
    }

    void startPendingSession();

    return () => {
      cancelled = true;
    };
  }, [pendingSessionStart, startCurrentSession]);

  async function handleStartSession() {
    if (pendingSessionStart) {
      return;
    }

    if (needsFreshSession) {
      setPendingSessionStart(true);
      setNeedsFreshSession(false);
      setSessionSeed((currentSeed) => currentSeed + 1);
      return;
    }

    await startCurrentSession();
  }

  async function handleEndSession() {
    setSessionError(null);
    setTextInputError(null);
    setIsPressingToTalk(false);
    resetTranscriptState();
    setPendingSessionStart(false);
    if (session.isConnected) {
      await session.end();
    }
    setNeedsFreshSession(true);
  }

  async function handleClearSession() {
    setTextInputError(null);
    setSessionError(null);
    setIsPressingToTalk(false);
    resetTranscriptState();
    setIsResettingShell(true);
    if (session.isConnected) {
      await session.end();
    }
    window.setTimeout(() => {
      onSessionReset();
    }, 280);
  }

  async function handleSendTextInput() {
    const message = textInputValue.trim();
    if (!message) {
      return;
    }

    try {
      setTextInputError(null);
      await sessionMessages.send(message);
      setTextInputValue("");
    } catch (error) {
      const message =
        error instanceof Error && error.message
          ? error.message
          : "The text message could not be sent.";
      setTextInputError(message);
    }
  }

  const headerCopy = STATE_COPY[activeState];
  const isConnectingOrRecovering =
    session.connectionState === ConnectionState.Connecting ||
    session.connectionState === ConnectionState.Reconnecting ||
    session.connectionState === ConnectionState.SignalReconnecting;
  const showActiveConsole =
    hasEnteredConsole ||
    session.isConnected ||
    isConnectingOrRecovering ||
    isActivatingConsole ||
    isResettingShell;
  const activeConsoleTransitionClass = isResettingShell
    ? "screen-view-fade-out"
    : isActivatingConsole
      ? "screen-view-fade-in"
      : "screen-view-steady";
  const idleTransitionClass = isActivatingConsole
    ? "screen-view-fade-out"
    : "screen-view-fade-in";

  return (
    <SessionProvider session={session}>
      <RoomAudioRenderer room={session.room} />

      <main className="relative min-h-screen overflow-hidden px-4 py-4 sm:px-6 lg:px-8">
        <div className="aurora-grid pointer-events-none absolute inset-0 opacity-80" />

        <div className="relative mx-auto flex min-h-[calc(100vh-2rem)] w-full max-w-[1380px] flex-col gap-5 rounded-[2rem] border border-white/10 bg-[rgba(7,17,31,0.78)] p-4 shadow-[0_30px_120px_rgba(2,8,18,0.58)] backdrop-blur-xl sm:p-6">
          <header className="auralis-panel flex flex-col gap-4 rounded-[1.6rem] px-5 py-4 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-[linear-gradient(135deg,rgba(124,92,255,0.22),rgba(25,211,255,0.14))] text-[color:var(--color-text-primary)] shadow-[0_0_32px_rgba(124,92,255,0.18)]">
                  <AuralisLogoMark />
                </div>
                <div>
                  <p className="font-heading text-2xl font-semibold tracking-[-0.03em] text-[color:var(--color-text-primary)]">
                    LiveKit RAG Voice Assistant
                  </p>
                  <p className="text-sm text-[color:var(--color-text-secondary)]">
                    LiveKit Voice + RAG + Tool Calling
                  </p>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <StatusChip
                label={`Session ${formatConnectionState(session.connectionState)}`}
                tone={connectionTone(session.connectionState)}
              />
              <StatusChip
                label={agentStateLabel(agent.state, session.connectionState)}
                tone={agentTone(agent.state, session.connectionState)}
              />
              <StatusChip
                label={formatMicLabel(
                  session.isConnected,
                  isMuted,
                  controlMode,
                  isPressingToTalk,
                )}
                tone={isMuted ? "warn" : session.isConnected ? "success" : "neutral"}
              />
            </div>
          </header>

          {!HAS_PUBLIC_URL && (
            <section className="auralis-panel flex items-start gap-3 rounded-[1.4rem] border border-[color:var(--color-warning)]/20 bg-[color:var(--color-warning)]/8 px-4 py-3 text-sm text-[color:var(--color-warning)]">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <p>
                `NEXT_PUBLIC_LIVEKIT_URL` is not configured yet. The shell can
                still render locally, but a real browser session needs your
                LiveKit project URL.
              </p>
            </section>
          )}

          {!showActiveConsole ? (
            <section
              className={`screen-view ${idleTransitionClass} flex flex-1 flex-col items-center justify-center gap-8 px-3 py-6 text-center`}
            >
              <div className="max-w-3xl">
                <p className="mb-3 text-xs font-semibold uppercase tracking-[0.28em] text-[color:var(--color-cyan)]">
                  Grounded Voice Demo
                </p>
                <h1 className="font-heading text-4xl font-semibold tracking-[-0.05em] text-[color:var(--color-text-primary)] sm:text-6xl">
                  {headerCopy.title}
                </h1>
                <p className="mx-auto mt-4 max-w-2xl text-sm leading-7 text-[color:var(--color-text-secondary)] sm:text-base">
                  {headerCopy.description}
                </p>
              </div>

              <div className="flex w-full max-w-3xl flex-col items-center gap-6 rounded-[2rem] border border-white/10 bg-[linear-gradient(180deg,rgba(19,34,56,0.8),rgba(11,19,33,0.74))] px-6 py-8 shadow-[0_0_60px_rgba(25,211,255,0.08)]">
                <VoiceOrb state={activeState} />

                <div className="flex flex-wrap items-center justify-center gap-2">
                  <StatusChip label="TURN" tone="accent" />
                  <StatusChip label="VAD" tone="live" />
                  <StatusChip label="STT" tone="success" />
                  <StatusChip label="TTS" tone="success" />
                  <StatusChip label="NC ON" tone="live" />
                </div>

                <button
                  type="button"
                  onClick={() => void handleStartSession()}
                  disabled={pendingSessionStart}
                  className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-full border border-[color:var(--color-primary)]/35 bg-[linear-gradient(135deg,rgba(124,92,255,0.9),rgba(25,211,255,0.48))] px-6 py-3 text-sm font-semibold text-white shadow-[0_0_38px_rgba(124,92,255,0.28)] transition hover:brightness-110 disabled:cursor-wait disabled:opacity-70"
                >
                  {pendingSessionStart ? (
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                  ) : (
                    <Mic className="h-4 w-4" />
                  )}
                  {pendingSessionStart ? "Preparing session..." : "Start conversation"}
                </button>

                <p className="text-xs uppercase tracking-[0.2em] text-[color:var(--color-text-secondary)]">
                  Microphone permission is required for voice mode.
                </p>
              </div>

              <div className="grid w-full max-w-4xl gap-3 sm:grid-cols-3">
                {SAMPLE_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => setTextInputValue(prompt)}
                    className="auralis-panel rounded-[1.35rem] px-4 py-4 text-left text-sm leading-6 text-[color:var(--color-text-secondary)] transition hover:border-[color:var(--color-primary)]/25 hover:bg-white/[0.06] hover:text-[color:var(--color-text-primary)]"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <>
            <section
              className={`screen-view ${activeConsoleTransitionClass} grid gap-5 xl:grid-cols-[0.9fr_1.1fr]`}
            >
                <article className="auralis-panel auralis-panel-elevated flex flex-col gap-5 rounded-[1.8rem] px-5 py-5">
                  <div className="text-center">
                    <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[color:var(--color-cyan)]">
                      Presence
                    </p>
                    <h2 className="mt-3 font-heading text-3xl font-semibold tracking-[-0.04em] text-[color:var(--color-text-primary)]">
                      {headerCopy.title}
                    </h2>
                    <p className="mx-auto mt-3 max-w-xl text-sm leading-7 text-[color:var(--color-text-secondary)]">
                      {headerCopy.description}
                    </p>
                  </div>

                  <div className="rounded-[1.8rem] border border-white/10 bg-[radial-gradient(circle_at_top,rgba(124,92,255,0.18),transparent_55%),radial-gradient(circle_at_bottom,rgba(25,211,255,0.16),transparent_50%),rgba(11,19,33,0.72)] px-4 py-2">
                    <VoiceOrb
                      state={activeState}
                      isInterrupted={activeInterruptedIds.length > 0}
                    />
                  </div>

                  <div className="flex flex-wrap items-center justify-center gap-2">
                    <StatusChip
                      label={agentStateLabel(agent.state, session.connectionState)}
                      tone={agentTone(agent.state, session.connectionState)}
                    />
                    <StatusChip
                      label={controlMode === "auto" ? "Auto mode" : "Push to talk"}
                      tone={controlMode === "auto" ? "accent" : "live"}
                    />
                    <StatusChip
                      label={isMuted ? "Muted" : "Noise reduction on"}
                      tone={isMuted ? "warn" : "success"}
                    />
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    {session.isConnected ? (
                      <button
                        type="button"
                        onClick={() => void handleEndSession()}
                        className="control-button control-button-danger"
                      >
                        <PhoneOff className="h-4 w-4" />
                        End session
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => void handleStartSession()}
                        disabled={pendingSessionStart}
                        className="control-button disabled:cursor-wait disabled:opacity-70"
                      >
                        {pendingSessionStart ? (
                          <LoaderCircle className="h-4 w-4 animate-spin" />
                        ) : (
                          <Mic className="h-4 w-4" />
                        )}
                        {pendingSessionStart ? "Preparing..." : "Start session"}
                      </button>
                    )}

                    <button
                      type="button"
                      onClick={() => {
                        setIsMuted((currentValue) => !currentValue);
                        setIsPressingToTalk(false);
                      }}
                      className="control-button"
                    >
                      {isMuted ? (
                        <>
                          <Mic className="h-4 w-4" />
                          Unmute mic
                        </>
                      ) : (
                        <>
                          <MicOff className="h-4 w-4" />
                          Mute mic
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setControlMode("auto");
                        setIsPressingToTalk(false);
                      }}
                      className={`control-button ${
                        controlMode === "auto" ? "control-button-active" : ""
                      }`}
                    >
                      <Radio className="h-4 w-4" />
                      Auto mode
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        setControlMode("push-to-talk");
                        setIsPressingToTalk(false);
                      }}
                      className={`control-button ${
                        controlMode === "push-to-talk" ? "control-button-active" : ""
                      }`}
                    >
                      <Waves className="h-4 w-4" />
                      Push-to-talk
                    </button>
                  </div>

                  <button
                    type="button"
                    onPointerDown={() => {
                      if (controlMode === "push-to-talk" && !isMuted) {
                        setIsPressingToTalk(true);
                      }
                    }}
                    onPointerUp={() => setIsPressingToTalk(false)}
                    onPointerLeave={() => setIsPressingToTalk(false)}
                    onPointerCancel={() => setIsPressingToTalk(false)}
                    disabled={
                      controlMode !== "push-to-talk" || !session.isConnected || isMuted
                    }
                    className="inline-flex items-center justify-center gap-2 rounded-[1.2rem] border border-[color:var(--color-cyan)]/25 bg-[color:var(--color-cyan)]/8 px-4 py-4 text-sm font-semibold text-[color:var(--color-text-primary)] transition hover:border-[color:var(--color-cyan)]/45 hover:bg-[color:var(--color-cyan)]/12 disabled:cursor-not-allowed disabled:opacity-45"
                  >
                    <Volume2 className="h-4 w-4" />
                    {isPressingToTalk ? "Transmitting..." : "Hold to talk"}
                  </button>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <button
                      type="button"
                      onClick={() => setIsDebugOpen((open) => !open)}
                      className="control-button"
                    >
                      <Settings2 className="h-4 w-4" />
                      Debug drawer
                      <ChevronDown
                        className={`h-4 w-4 transition ${
                          isDebugOpen ? "rotate-180" : ""
                        }`}
                      />
                    </button>

                    <button
                      type="button"
                      onClick={() => void handleClearSession()}
                      className="control-button"
                    >
                      <Trash2 className="h-4 w-4" />
                      Clear session
                    </button>
                  </div>

                  {sessionError && (
                    <div className="rounded-[1.2rem] border border-[color:var(--color-error)]/25 bg-[color:var(--color-error)]/10 px-4 py-3 text-sm text-[color:var(--color-error)]">
                      {sessionError}
                    </div>
                  )}

                  {isDebugOpen && (
                    <div className="rounded-[1.35rem] border border-white/10 bg-white/[0.04] p-4 text-sm text-[color:var(--color-text-secondary)]">
                      <div className="grid gap-3 sm:grid-cols-2">
                        <div>
                          <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-white/65">
                            Session ID
                          </p>
                          <p className="mt-1 break-all text-[color:var(--color-text-primary)]">
                            {sessionInfo?.sessionId ?? roomName}
                          </p>
                        </div>
                        <div>
                          <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-white/65">
                            Agent
                          </p>
                          <p className="mt-1 text-[color:var(--color-text-primary)]">
                            {sessionInfo?.agentName ?? AGENT_NAME}
                          </p>
                        </div>
                        <div>
                          <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-white/65">
                            Current state
                          </p>
                          <p className="mt-1 text-[color:var(--color-text-primary)]">
                            {agentStateLabel(agent.state, session.connectionState)}
                          </p>
                        </div>
                        <div>
                          <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-white/65">
                            Latest partial
                          </p>
                          <p className="mt-1 text-[color:var(--color-text-primary)]">
                            {livePartialEntry?.text ?? "None"}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </article>

                <article className="auralis-panel auralis-panel-elevated flex flex-col gap-5 rounded-[1.8rem] px-5 py-5">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[color:var(--color-cyan)]">
                        Live transcript
                      </p>
                      <h2 className="mt-3 font-heading text-2xl font-semibold tracking-[-0.03em] text-[color:var(--color-text-primary)]">
                        Conversation flow
                      </h2>
                      <p className="mt-2 text-sm leading-7 text-[color:var(--color-text-secondary)]">
                        Speech and typed messages share one timeline so the whole
                        session stays readable.
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <StatusChip
                        label={`${decoratedEntries.length} turns`}
                        tone="neutral"
                      />
                    </div>
                  </div>

                  <SessionTranscript
                    entries={decoratedEntries}
                    emptyMessage="Connect to the LiveKit RAG Voice Assistant and your live transcript will begin filling here."
                  />

                  <div className="rounded-[1.45rem] border border-white/10 bg-white/[0.04] p-4">
                    <div className="mb-3 flex items-center gap-2">
                      <Keyboard className="h-4 w-4 text-[color:var(--color-cyan)]" />
                      <p className="text-sm font-semibold text-[color:var(--color-text-primary)]">
                        Text fallback
                      </p>
                    </div>
                    <div className="flex flex-col gap-3 lg:flex-row">
                      <input
                        type="text"
                        value={textInputValue}
                        onChange={(event) => setTextInputValue(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            void handleSendTextInput();
                          }
                        }}
                        placeholder="Prefer typing? Send a message here."
                        disabled={!session.isConnected || sessionMessages.isSending}
                        className="min-w-0 flex-1 rounded-[1rem] border border-white/10 bg-black/15 px-4 py-3 text-sm text-[color:var(--color-text-primary)] outline-none transition placeholder:text-white/35 focus:border-[color:var(--color-primary)]/45 focus:ring-2 focus:ring-[color:var(--color-primary)]/15 disabled:cursor-not-allowed disabled:opacity-60"
                      />
                      <button
                        type="button"
                        onClick={() => void handleSendTextInput()}
                        disabled={
                          !session.isConnected ||
                          sessionMessages.isSending ||
                          !textInputValue.trim()
                        }
                        className="inline-flex items-center justify-center gap-2 rounded-[1rem] border border-[color:var(--color-primary)]/25 bg-[color:var(--color-primary)]/12 px-4 py-3 text-sm font-semibold text-[color:var(--color-text-primary)] transition hover:border-[color:var(--color-primary)]/45 hover:bg-[color:var(--color-primary)]/18 disabled:cursor-not-allowed disabled:opacity-55"
                      >
                        {sessionMessages.isSending ? (
                          <LoaderCircle className="h-4 w-4 animate-spin" />
                        ) : (
                          <Send className="h-4 w-4" />
                        )}
                        Send
                      </button>
                    </div>
                    {textInputError && (
                      <p className="mt-3 text-sm text-[color:var(--color-error)]">
                        {textInputError}
                      </p>
                    )}
                  </div>
                </article>
            </section>

              <section
                className={`auralis-panel rounded-[1.8rem] px-5 py-5 ${activeConsoleTransitionClass}`}
              >
                <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-[color:var(--color-cyan)]">
                      System status
                    </p>
                    <h2 className="mt-2 font-heading text-2xl font-semibold tracking-[-0.03em] text-[color:var(--color-text-primary)]">
                      Live pipeline rail
                    </h2>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <StatusChip label="Voice assistant active" tone="success" />
                    <StatusChip
                      label={capabilities?.textInput ? "Text enabled" : "Voice only"}
                      tone={capabilities?.textInput ? "accent" : "neutral"}
                    />
                    <button
                      type="button"
                      onClick={() => setIsSystemStatusOpen((open) => !open)}
                      className="control-button !rounded-full !px-3.5 !py-2 text-[0.75rem]"
                    >
                      <Settings2 className="h-4 w-4" />
                      {isSystemStatusOpen ? "Hide details" : "Show details"}
                      <ChevronDown
                        className={`h-4 w-4 transition ${
                          isSystemStatusOpen ? "rotate-180" : ""
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {isSystemStatusOpen && (
                  <div className="system-status-content animate-[panel-fade-up_360ms_cubic-bezier(0.22,1,0.36,1)]">
                    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                      {pipelineItems.map((item) => (
                        <div
                          key={item.label}
                          className={`pipeline-card ${pipelineCardToneClass(item.value)}`}
                        >
                          <div className="flex items-start justify-between gap-4">
                            <p className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-white/65">
                              {item.label}
                            </p>
                            <span className="pipeline-card-orbit" aria-hidden="true" />
                          </div>
                          <div className="mt-4">
                            <StatusChip
                              label={item.value}
                              tone={toneForPipeline(item.value)}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="mt-5 grid gap-3 lg:grid-cols-3">
                      <div className="rounded-[1.2rem] border border-white/10 bg-white/[0.04] px-4 py-4">
                        <div className="flex items-center gap-2 text-[color:var(--color-primary)]">
                          <Settings2 className="h-4 w-4" />
                          <p className="text-sm font-semibold text-[color:var(--color-text-primary)]">
                            Tooling / Grounding
                          </p>
                        </div>
                        <div className="mt-4 flex flex-wrap gap-2">
                          <StatusChip
                            label={answerPathLabel(currentToolingSnapshot.lastAnswerPath)}
                            tone={
                              currentToolingSnapshot.lastAnswerPath === "normal"
                                ? "accent"
                                : currentToolingSnapshot.lastAnswerPath === "unknown"
                                  ? "neutral"
                                  : "success"
                            }
                          />
                          <StatusChip
                            label={`Fallback ${fallbackLabel(currentToolingSnapshot.lastFallback)}`}
                            tone={
                              currentToolingSnapshot.lastFallback == null
                                ? "neutral"
                                : currentToolingSnapshot.lastFallback
                                  ? "warn"
                                  : "success"
                            }
                          />
                          <StatusChip
                            label={`RAG ${ragBackendLabel(currentToolingSnapshot.ragBackend)}`}
                            tone={ragBackendTone(currentToolingSnapshot.ragBackend)}
                          />
                        </div>
                      </div>

                      <div className="rounded-[1.2rem] border border-white/10 bg-white/[0.04] px-4 py-4">
                        <div className="flex items-center gap-2 text-[color:var(--color-cyan)]">
                          <ShieldCheck className="h-4 w-4" />
                          <p className="text-sm font-semibold text-[color:var(--color-text-primary)]">
                            Knowledge base
                          </p>
                        </div>
                        <div className="mt-4">
                          <StatusChip
                            label={toolingStatusLabel(currentToolingSnapshot.knowledgeBase.status)}
                            tone={toolingStatusTone(currentToolingSnapshot.knowledgeBase.status)}
                          />
                        </div>
                        <p className="mt-3 text-sm leading-7 text-[color:var(--color-text-secondary)]">
                          Last KB latency: {formatLatency(currentToolingSnapshot.knowledgeBase.latencyMs)}
                        </p>
                      </div>

                      <div className="rounded-[1.2rem] border border-white/10 bg-white/[0.04] px-4 py-4">
                        <div className="flex items-center gap-2 text-[color:var(--color-accent)]">
                          <Waves className="h-4 w-4" />
                          <p className="text-sm font-semibold text-[color:var(--color-text-primary)]">
                            Weather tool
                          </p>
                        </div>
                        <div className="mt-4">
                          <StatusChip
                            label={toolingStatusLabel(currentToolingSnapshot.weather.status)}
                            tone={toolingStatusTone(currentToolingSnapshot.weather.status)}
                          />
                        </div>
                        <p className="mt-3 text-sm leading-7 text-[color:var(--color-text-secondary)]">
                          Last weather latency: {formatLatency(currentToolingSnapshot.weather.latencyMs)}
                        </p>
                      </div>
                    </div>

                    <div className="mt-5 grid gap-3 md:grid-cols-3">
                      <div className="rounded-[1.2rem] border border-white/10 bg-white/[0.04] px-4 py-4">
                        <div className="flex items-center gap-2 text-[color:var(--color-cyan)]">
                          <Cpu className="h-4 w-4" />
                          <p className="text-sm font-semibold text-[color:var(--color-text-primary)]">
                            Models
                          </p>
                        </div>
                        <p className="mt-3 text-sm leading-7 text-[color:var(--color-text-secondary)]">
                          STT: {capabilities?.sttModel ?? "deepgram/flux-general-en"}
                          <br />
                          LLM: {capabilities?.llmModel ?? "openai/gpt-4.1-mini"}
                          <br />
                          TTS: {capabilities?.ttsModel ?? "cartesia/sonic-3"}
                        </p>
                      </div>

                      <div className="rounded-[1.2rem] border border-white/10 bg-white/[0.04] px-4 py-4">
                        <div className="flex items-center gap-2 text-[color:var(--color-primary)]">
                          <ShieldCheck className="h-4 w-4" />
                          <p className="text-sm font-semibold text-[color:var(--color-text-primary)]">
                            Realtime behavior
                          </p>
                        </div>
                        <p className="mt-3 text-sm leading-7 text-[color:var(--color-text-secondary)]">
                          Turn detection, short in-session context, adaptive interruption
                          handling, and inbound BVC noise cancellation stay active during
                          the call.
                        </p>
                      </div>

                      <div className="rounded-[1.2rem] border border-white/10 bg-white/[0.04] px-4 py-4">
                        <div className="flex items-center gap-2 text-[color:var(--color-accent)]">
                          <MessageSquareText className="h-4 w-4" />
                          <p className="text-sm font-semibold text-[color:var(--color-text-primary)]">
                            Session memory
                          </p>
                        </div>
                        <p className="mt-3 text-sm leading-7 text-[color:var(--color-text-secondary)]">
                          The assistant keeps the current conversation context in
                          memory for the live session only.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </section>
            </>
          )}
        </div>
      </main>
    </SessionProvider>
  );
}

export function VoiceAgentApp() {
  const [sessionKey, setSessionKey] = useState(0);

  return (
    <VoiceAgentShell
      key={sessionKey}
      onSessionReset={() => {
        startTransition(() => {
          setSessionKey((currentKey) => currentKey + 1);
        });
      }}
    />
  );
}
