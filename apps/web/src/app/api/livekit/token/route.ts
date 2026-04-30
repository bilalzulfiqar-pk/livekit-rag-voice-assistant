import { NextRequest, NextResponse } from "next/server";

import { AccessToken, type RoomConfiguration } from "livekit-server-sdk";
import { z } from "zod";

export const runtime = "nodejs";

const tokenRequestSchema = z.object({
  room_name: z.string().min(1).optional(),
  participant_name: z.string().min(1).optional(),
  participant_identity: z.string().min(1).optional(),
  participant_metadata: z.string().optional(),
  participant_attributes: z.record(z.string(), z.string()).optional(),
  room_config: z.custom<RoomConfiguration>().optional(),
});

export async function POST(request: NextRequest) {
  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;
  const serverUrl = process.env.LIVEKIT_URL;

  if (!apiKey || !apiSecret || !serverUrl) {
    return NextResponse.json(
      {
        error:
          "LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL must be configured.",
      },
      { status: 500 },
    );
  }

  const body = await request.json();
  const parsedRequest = tokenRequestSchema.safeParse(body);
  if (!parsedRequest.success) {
    return NextResponse.json(
      {
        error: "Invalid token request payload.",
        details: parsedRequest.error.flatten(),
      },
      { status: 400 },
    );
  }

  const roomName =
    parsedRequest.data.room_name ??
    `livekit-rag-session-${typeof crypto !== "undefined" ? crypto.randomUUID() : Date.now()}`;
  const participantIdentity =
    parsedRequest.data.participant_identity ??
    `web-${typeof crypto !== "undefined" ? crypto.randomUUID() : Date.now()}`;

  const accessToken = new AccessToken(apiKey, apiSecret, {
    identity: participantIdentity,
    name: parsedRequest.data.participant_name ?? "LiveKit RAG Visitor",
    metadata: parsedRequest.data.participant_metadata,
    attributes: parsedRequest.data.participant_attributes,
    ttl: "15m",
  });

  if (parsedRequest.data.room_config) {
    accessToken.roomConfig = parsedRequest.data.room_config;
  }

  accessToken.addGrant({
    roomJoin: true,
    room: roomName,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true,
  });

  const participantToken = await accessToken.toJwt();

  return NextResponse.json(
    {
      server_url: serverUrl,
      participant_token: participantToken,
    },
    { status: 201 },
  );
}
