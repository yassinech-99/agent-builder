"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { AgentConfigProvider, type AgentConfig } from "@/providers/AgentConfig";
import { AgentBuilder } from "@/components/agent-builder";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024";

export default function EditAgentPage() {
  const params = useParams();
  const router = useRouter();
  const assistantId = params.id as string;
  const [initialConfig, setInitialConfig] = useState<AgentConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const client = createClient(API_URL, getApiKey() ?? undefined);
        const assistant = await client.assistants.get(assistantId);
        const cfg = (assistant.config?.configurable ?? {}) as any;
        setInitialConfig({
          agent_name: assistant.name || cfg.agent_name || "",
          system_prompt: cfg.system_prompt || "",
          enabled_prebuilt_tools: cfg.enabled_prebuilt_tools || [],
          tool_configs: cfg.tool_configs || {},
          mcp_servers: cfg.mcp_servers || [],
          model_id: cfg.model_id || "",
          rag_source_ids: cfg.rag_source_ids || [],
          local_rag_source_ids: cfg.local_rag_source_ids || [],
          lexical_rag_source_ids: cfg.lexical_rag_source_ids || [],
          voice_enabled: !!cfg.voice_enabled,
        });
      } catch (e) {
        console.error("Failed to load agent:", e);
        router.push("/agents");
      } finally {
        setLoading(false);
      }
    })();
  }, [assistantId, router]);

  if (loading || !initialConfig) {
    return (
      <div className="flex h-full items-center justify-center">Loading agent...</div>
    );
  }

  return (
    <React.Suspense
      fallback={
        <div className="flex h-full items-center justify-center">Loading...</div>
      }
    >
      <AgentConfigProvider initialConfig={initialConfig}>
        <AgentBuilder editId={assistantId} />
      </AgentConfigProvider>
    </React.Suspense>
  );
}
