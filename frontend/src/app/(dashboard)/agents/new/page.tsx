"use client";

import React from "react";
import { AgentConfigProvider } from "@/providers/AgentConfig";
import { AgentBuilder } from "@/components/agent-builder";

export default function NewAgentPage(): React.ReactNode {
  return (
    <React.Suspense
      fallback={
        <div className="flex h-full items-center justify-center">Loading...</div>
      }
    >
      <AgentConfigProvider>
        <AgentBuilder />
      </AgentConfigProvider>
    </React.Suspense>
  );
}
