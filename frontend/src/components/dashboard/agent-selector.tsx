"use client";

import { useEffect, useState, useCallback } from "react";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";
import { Bot } from "lucide-react";
import type { Assistant } from "@langchain/langgraph-sdk";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024";

interface AgentSelectorProps {
  value: string;
  onChange: (assistantId: string) => void;
}

export function AgentSelector({ value, onChange }: AgentSelectorProps) {
  const [agents, setAgents] = useState<Assistant[]>([]);

  const fetchAgents = useCallback(async () => {
    try {
      const client = createClient(API_URL, getApiKey() ?? undefined);
      const results = await client.assistants.search({
        graphId: "agent",
        limit: 100,
      });
      setAgents(results);
    } catch (e) {
      console.error("Failed to fetch agents:", e);
    }
  }, []);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  return (
    <div className="flex items-center gap-2">
      <Bot className="text-muted-foreground h-4 w-4" />
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border-input bg-background flex h-8 rounded-md border px-2 text-sm"
      >
        <option value="agent">Default Agent</option>
        {agents.map((a) => (
          <option key={a.assistant_id} value={a.assistant_id}>
            {a.name || a.assistant_id}
          </option>
        ))}
      </select>
      <span className="text-muted-foreground text-xs">
        {agents.length} agent{agents.length !== 1 ? "s" : ""} available
      </span>
    </div>
  );
}
