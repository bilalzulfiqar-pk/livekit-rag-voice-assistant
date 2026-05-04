import { z } from "zod";

export type TranscriptEntry = {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
  isFinal: boolean;
  isInterrupted?: boolean;
  source: "transcript" | "chat";
};

export type VoiceCapabilities = {
  sttModel: string;
  llmModel: string;
  ttsModel: string;
  ttsVoice: string;
  vad: string;
  turnDetection: string;
  noiseCancellation: string;
  interruptions: string;
  preemptiveGeneration: boolean;
  textInput: boolean;
  shortTermMemory: boolean;
};

export type VoiceSessionInfo = {
  sessionId: string;
  agentName: string;
  startedAt: string;
  transport: string;
};

export type VoiceControlMode = "auto" | "push-to-talk";
export type ToolingAnswerPath =
  | "unknown"
  | "normal"
  | "knowledge_base"
  | "weather";
export type ToolingStatusValue = "idle" | "querying" | "success" | "failed";
export type RagBackendState = "unknown" | "warming_up" | "ready" | "degraded";

export type ToolingToolState = {
  status: ToolingStatusValue;
  latencyMs: number | null;
  fallback: boolean | null;
};

export type PipelineLatencySnapshot = {
  sttLatencyMs: number | null;
  llmLatencyMs: number | null;
  ttsLatencyMs: number | null;
};

export type ToolingSnapshot = {
  type: "tooling_snapshot";
  version: 1;
  sequence: number;
  sessionId: string;
  lastAnswerPath: ToolingAnswerPath;
  lastFallback: boolean | null;
  ragBackend: RagBackendState;
  knowledgeBase: ToolingToolState;
  weather: ToolingToolState;
  pipeline: PipelineLatencySnapshot;
};

export const TOOLING_STATUS_TOPIC = "app.tooling.status";

const voiceCapabilitiesSchema = z.object({
  sttModel: z.string(),
  llmModel: z.string(),
  ttsModel: z.string(),
  ttsVoice: z.string(),
  vad: z.string().default("silero"),
  turnDetection: z.string().default("livekit-multilingual"),
  noiseCancellation: z.string().default("bvc"),
  interruptions: z.string().default("adaptive"),
  preemptiveGeneration: z.boolean().default(true),
  textInput: z.boolean().default(true),
  shortTermMemory: z.boolean().default(true),
});

const voiceSessionInfoSchema = z.object({
  sessionId: z.string(),
  agentName: z.string(),
  startedAt: z.string(),
  transport: z.string().default("livekit"),
});

const toolingToolStateSchema = z.object({
  status: z.enum(["idle", "querying", "success", "failed"]),
  latencyMs: z.number().int().nonnegative().nullable(),
  fallback: z.boolean().nullable(),
});

const pipelineLatencySchema = z.object({
  sttLatencyMs: z.number().int().nonnegative().nullable(),
  llmLatencyMs: z.number().int().nonnegative().nullable(),
  ttsLatencyMs: z.number().int().nonnegative().nullable(),
});

const toolingSnapshotSchema = z.object({
  type: z.literal("tooling_snapshot"),
  version: z.literal(1),
  sequence: z.number().int().nonnegative(),
  sessionId: z.string(),
  lastAnswerPath: z.enum(["unknown", "normal", "knowledge_base", "weather"]),
  lastFallback: z.boolean().nullable(),
  ragBackend: z.enum(["unknown", "warming_up", "ready", "degraded"]),
  knowledgeBase: toolingToolStateSchema,
  weather: toolingToolStateSchema,
  pipeline: pipelineLatencySchema,
});

export const TRANSCRIPTION_FINAL_ATTRIBUTE = "lk.transcription_final";
export const TRANSCRIPTION_SEGMENT_ID_ATTRIBUTE = "lk.segment_id";

function parseJsonString(rawValue?: string | null): unknown {
  if (!rawValue) {
    return null;
  }

  try {
    return JSON.parse(rawValue);
  } catch {
    return null;
  }
}

export function parseVoiceCapabilities(
  rawValue?: string | null,
): VoiceCapabilities | null {
  const parsed = voiceCapabilitiesSchema.safeParse(parseJsonString(rawValue));
  return parsed.success ? parsed.data : null;
}

export function parseVoiceSessionInfo(
  rawValue?: string | null,
): VoiceSessionInfo | null {
  const parsed = voiceSessionInfoSchema.safeParse(parseJsonString(rawValue));
  return parsed.success ? parsed.data : null;
}

export function parseToolingSnapshot(
  rawValue: string,
  currentSequence = -1,
): ToolingSnapshot | null {
  const parsed = toolingSnapshotSchema.safeParse(parseJsonString(rawValue));
  if (!parsed.success) {
    return null;
  }

  return parsed.data.sequence > currentSequence ? parsed.data : null;
}

export function createDefaultToolingSnapshot(
  sessionId = "",
): ToolingSnapshot {
  return {
    type: "tooling_snapshot",
    version: 1,
    sequence: 0,
    sessionId,
    lastAnswerPath: "unknown",
    lastFallback: null,
    ragBackend: "unknown",
    knowledgeBase: {
      status: "idle",
      latencyMs: null,
      fallback: null,
    },
    weather: {
      status: "idle",
      latencyMs: null,
      fallback: null,
    },
    pipeline: {
      sttLatencyMs: null,
      llmLatencyMs: null,
      ttsLatencyMs: null,
    },
  };
}
