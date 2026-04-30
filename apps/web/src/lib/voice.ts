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
