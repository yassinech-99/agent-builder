"use client";

import React from "react";
import { useAgentConfig } from "@/providers/AgentConfig";

export function StepReview() {
  const { config } = useAgentConfig();
  const totalRag =
    (config.rag_source_ids?.length ?? 0) +
    (config.local_rag_source_ids?.length ?? 0) +
    (config.lexical_rag_source_ids?.length ?? 0);

  return (
    <div className="space-y-5">
      <p className="text-muted-foreground text-sm">
        Verify your nimAgents configuration before deploying.
      </p>

      <div className="bg-card divide-y rounded-xl border shadow-xs">
        <div className="p-4">
          <p className="text-muted-foreground text-xs font-medium uppercase">
            Agent Name
          </p>
          <p className="mt-1 font-medium">{config.agent_name || "(unnamed)"}</p>
        </div>

        <div className="p-4">
          <p className="text-muted-foreground text-xs font-medium uppercase">
            Model Endpoint
          </p>
          <p className="mt-1 font-mono text-sm">
            {config.model_id || "Default Workspace Model"}
          </p>
        </div>

        <div className="p-4">
          <p className="text-muted-foreground text-xs font-medium uppercase">
            System Prompt
          </p>
          <p className="mt-1 whitespace-pre-wrap text-sm">
            {config.system_prompt || "(default system prompt)"}
          </p>
        </div>

        <div className="p-4">
          <p className="text-muted-foreground text-xs font-medium uppercase">
            Prebuilt Tools ({config.enabled_prebuilt_tools.length})
          </p>
          <p className="mt-1 text-sm">
            {config.enabled_prebuilt_tools.length > 0
              ? config.enabled_prebuilt_tools.join(", ")
              : "None"}
          </p>
        </div>

        <div className="p-4">
          <p className="text-muted-foreground text-xs font-medium uppercase">
            MCP Servers ({config.mcp_servers.length})
          </p>
          {config.mcp_servers.length === 0 ? (
            <p className="text-muted-foreground mt-1 text-sm">None</p>
          ) : (
            <ul className="mt-1 space-y-1">
              {config.mcp_servers.map((s, i) => (
                <li key={i} className="text-sm">
                  <span className="font-medium">{s.url}</span>
                  <span className="text-muted-foreground ml-2 text-xs">
                    ({s.selected_tools.length} tools)
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="p-4">
          <p className="text-muted-foreground text-xs font-medium uppercase">
            RAG Knowledge Sources ({totalRag})
          </p>
          <p className="mt-1 text-sm">
            {totalRag === 0
              ? "None"
              : `${config.local_rag_source_ids.length} vector · ${config.lexical_rag_source_ids.length} lexical · ${config.rag_source_ids.length} NIM RAG`}
          </p>
        </div>
      </div>
    </div>
  );
}
