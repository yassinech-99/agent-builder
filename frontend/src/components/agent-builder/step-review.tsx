"use client";

import { useEffect, useState } from "react";
import { useAgentConfig } from "@/providers/AgentConfig";

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

interface ModelSummary {
  id: string;
  name: string;
  provider: string;
  model_name: string;
}

export function StepReview() {
  const { config } = useAgentConfig();
  const [modelMeta, setModelMeta] = useState<ModelSummary | null>(null);

  const totalMcpTools = config.mcp_servers.reduce(
    (sum, s) => sum + s.selected_tools.length,
    0,
  );

  const prebuiltCount = config.enabled_prebuilt_tools.length;

  useEffect(() => {
    if (!config.model_id) {
      setModelMeta(null);
      return;
    }
    const loadModel = async () => {
      try {
        const res = await fetch(`${BUILDER_API}/api/models`);
        if (!res.ok) return;
        const data = (await res.json()) as ModelSummary[];
        setModelMeta(data.find((m) => m.id === config.model_id) || null);
      } catch {
        setModelMeta(null);
      }
    };
    loadModel();
  }, [config.model_id]);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
          Agent Name
        </span>
        <span className="text-sm font-medium">
          {config.agent_name || "Unnamed Agent"}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
          System Prompt
        </span>
        <span className="whitespace-pre-wrap text-sm">
          {config.system_prompt || "(default prompt)"}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
          Prebuilt Tools ({prebuiltCount})
        </span>
        {prebuiltCount > 0 ? (
          <div className="flex flex-wrap gap-2">
            {config.enabled_prebuilt_tools.map((id) => (
              <span
                key={id}
                className="bg-primary/10 text-primary rounded-full px-3 py-1 text-xs font-medium"
              >
                {id}
              </span>
            ))}
          </div>
        ) : (
          <span className="text-muted-foreground text-sm">None selected</span>
        )}
      </div>

      {config.model_id && (
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
            Model
          </span>
          <span className="text-sm font-medium">
            {modelMeta
              ? modelMeta.name
              : config.model_id}
          </span>
          {modelMeta && (
            <span className="text-muted-foreground text-xs">
              {modelMeta.provider} / {modelMeta.model_name}
            </span>
          )}
        </div>
      )}

      {config.rag_source_ids.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
            Bedrock Knowledge Bases ({config.rag_source_ids.length})
          </span>
          <div className="flex flex-wrap gap-2">
            {config.rag_source_ids.map((id) => (
              <span
                key={id}
                className="bg-primary/10 text-primary rounded-full px-3 py-1 text-xs font-medium"
              >
                {id.slice(0, 8)}…
              </span>
            ))}
          </div>
        </div>
      )}

      {config.local_rag_source_ids.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
            Local RAG Sources ({config.local_rag_source_ids.length})
          </span>
          <div className="flex flex-wrap gap-2">
            {config.local_rag_source_ids.map((id) => (
              <span
                key={id}
                className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300"
              >
                {id.slice(0, 8)}…
              </span>
            ))}
          </div>
        </div>
      )}

      {config.lexical_rag_source_ids.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
            Lexical RAG Sources ({config.lexical_rag_source_ids.length})
          </span>
          <div className="flex flex-wrap gap-2">
            {config.lexical_rag_source_ids.map((id) => (
              <span
                key={id}
                className="bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300 rounded-full px-3 py-1 text-xs font-medium"
              >
                {id.slice(0, 8)}…
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="flex flex-col gap-1">
        <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
          MCP Servers ({config.mcp_servers.length})
        </span>
        {config.mcp_servers.length > 0 ? (
          <div className="flex flex-col gap-2">
            {config.mcp_servers.map((srv, i) => (
              <div key={i} className="rounded-md border p-3">
                <p className="truncate text-sm font-medium">{srv.url}</p>
                <p className="text-muted-foreground text-xs">
                  {srv.selected_tools.join(", ")}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <span className="text-muted-foreground text-sm">None added</span>
        )}
      </div>

      <div className="flex flex-col gap-1">
        <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
          Voice Mode
        </span>
        <span className="text-sm font-medium">
          {config.voice_enabled ? "Enabled (Nova 2 Sonic)" : "Disabled"}
        </span>
      </div>

      <div className="bg-muted/50 mt-2 rounded-lg p-4">
        <p className="text-sm">
          Total tools:{" "}
          <span className="font-semibold">
            {prebuiltCount + totalMcpTools}
          </span>
        </p>
      </div>
    </div>
  );
}
