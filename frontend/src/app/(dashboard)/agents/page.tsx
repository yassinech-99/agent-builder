"use client";

import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Plus, Trash2, Pencil, Bot, Copy } from "lucide-react";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";
import type { Assistant } from "@langchain/langgraph-sdk";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024";

export default function AgentsPage() {
  const [agents, setAgents] = useState<Assistant[]>([]);
  const [loading, setLoading] = useState(true);

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
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this agent and all its threads?")) return;
    try {
      const client = createClient(API_URL, getApiKey() ?? undefined);
      await client.assistants.delete(id, { deleteThreads: true });
      setAgents((prev) => prev.filter((a) => a.assistant_id !== id));
    } catch (e) {
      console.error("Failed to delete agent:", e);
    }
  };

  const handleDuplicate = async (agent: Assistant) => {
    try {
      const client = createClient(API_URL, getApiKey() ?? undefined);
      const created = await client.assistants.create({
        graphId: "agent",
        name: `${agent.name || "Agent"} (copy)`,
        config: agent.config,
        metadata: agent.metadata,
      });
      setAgents((prev) => [created, ...prev]);
    } catch (e) {
      console.error("Failed to duplicate agent:", e);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Agents</h1>
          <p className="text-muted-foreground text-sm">
            Create and manage your AI agents
          </p>
        </div>
        <Link href="/agents/new">
          <Button>
            <Plus className="mr-1.5 h-4 w-4" />
            New Agent
          </Button>
        </Link>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {loading ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="bg-muted h-40 animate-pulse rounded-lg" />
            ))}
          </div>
        ) : agents.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20">
            <Bot className="text-muted-foreground mb-4 h-16 w-16" />
            <h2 className="mb-2 text-lg font-semibold">No agents yet</h2>
            <p className="text-muted-foreground mb-6 text-sm">
              Create your first agent to get started
            </p>
            <Link href="/agents/new">
              <Button>
                <Plus className="mr-1.5 h-4 w-4" />
                Create Agent
              </Button>
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {agents.map((agent) => {
              const config = (agent.config?.configurable ?? {}) as Record<
                string,
                any
              >;
              const toolCount =
                (config.enabled_prebuilt_tools?.length ?? 0) +
                (config.mcp_servers?.length ?? 0);
              return (
                <div
                  key={agent.assistant_id}
                  className="bg-card flex flex-col rounded-lg border p-5 shadow-sm transition-shadow hover:shadow-md"
                >
                  <div className="mb-3 flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <div className="bg-primary/10 text-primary flex h-9 w-9 items-center justify-center rounded-lg">
                        <Bot className="h-5 w-5" />
                      </div>
                      <div>
                        <h3 className="font-semibold">
                          {agent.name || "Unnamed Agent"}
                        </h3>
                        <p className="text-muted-foreground text-xs">
                          {toolCount} tool{toolCount !== 1 ? "s" : ""}
                        </p>
                      </div>
                    </div>
                  </div>

                  {config.system_prompt ? (
                    <p className="text-muted-foreground mb-4 line-clamp-2 flex-1 text-xs">
                      {config.system_prompt}
                    </p>
                  ) : (
                    <div className="flex-1" />
                  )}

                  <div className="flex items-center gap-1 border-t pt-3">
                    <Link
                      href={`/agents/${agent.assistant_id}`}
                      className="flex-1"
                    >
                      <Button variant="ghost" size="sm" className="w-full">
                        <Pencil className="mr-1 h-3.5 w-3.5" />
                        Edit
                      </Button>
                    </Link>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDuplicate(agent)}
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-destructive hover:text-destructive"
                      onClick={() => handleDelete(agent.assistant_id)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
