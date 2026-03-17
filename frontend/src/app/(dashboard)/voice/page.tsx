"use client";

import React from "react";
import { VoiceSession } from "@/components/voice/voice-session";

export default function VoicePage() {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Voice Agent</h1>
          <p className="text-muted-foreground text-sm">
            Real-time conversational AI powered by Nova 2 Sonic
          </p>
        </div>
      </div>
      <VoiceSession />
    </div>
  );
}
