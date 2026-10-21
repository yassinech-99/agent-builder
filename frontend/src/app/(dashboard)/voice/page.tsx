"use client";

import React from "react";
import { VoiceSession } from "@/components/voice/voice-session";

export default function VoicePage() {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Voice Studio</h1>
          <p className="text-muted-foreground text-sm">
            Real-time streaming voice &amp; conversational sessions powered by NVIDIA Riva Speech NIM
          </p>
        </div>
      </div>
      <VoiceSession />
    </div>
  );
}
