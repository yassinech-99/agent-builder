"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useAgentConfig, MCPServerConfig } from "@/providers/AgentConfig";
import { Loader2, Plus, Trash2 } from "lucide-react";

interface MCPToolInfo {
  name: string;
  description: string | null;
  inputSchema: Record<string, unknown> | null;
}

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

export function StepMCPServers() {
  const { config, setConfig } = useAgentConfig();
  const [newUrl, setNewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [discoveredTools, setDiscoveredTools] = useState<MCPToolInfo[]>([]);
  const [discoverUrl, setDiscoverUrl] = useState<string | null>(null);

  const fetchTools = async () => {
    if (!newUrl.trim()) return;
    setLoading(true);
    setError(null);
    setDiscoveredTools([]);
    setDiscoverUrl(newUrl.trim());

    try {
      const res = await fetch(`${BUILDER_API}/api/mcp/discover`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: newUrl.trim() }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setDiscoveredTools(data.tools || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect");
    } finally {
      setLoading(false);
    }
  };

  const addServer = (selectedTools: string[]) => {
    if (!discoverUrl) return;
    const server: MCPServerConfig = {
      url: discoverUrl,
      selected_tools: selectedTools,
    };
    setConfig((prev) => ({
      ...prev,
      mcp_servers: [...prev.mcp_servers, server],
    }));
    setNewUrl("");
    setDiscoveredTools([]);
    setDiscoverUrl(null);
  };

  const removeServer = (idx: number) => {
    setConfig((prev) => ({
      ...prev,
      mcp_servers: prev.mcp_servers.filter((_, i) => i !== idx),
    }));
  };

  return (
    <div className="flex flex-col gap-6">
      <p className="text-muted-foreground text-sm">
        Connect to MCP servers to add external tools. Enter the server URL and
        fetch available tools.
      </p>

      {config.mcp_servers.length > 0 && (
        <div className="flex flex-col gap-3">
          <Label>Added MCP Servers</Label>
          {config.mcp_servers.map((srv, idx) => (
            <div
              key={idx}
              className="flex items-center gap-3 rounded-lg border p-3"
            >
              <div className="flex-1 overflow-hidden">
                <p className="truncate text-sm font-medium">{srv.url}</p>
                <p className="text-muted-foreground text-xs">
                  {srv.selected_tools.length} tool
                  {srv.selected_tools.length !== 1 ? "s" : ""} selected
                </p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removeServer(idx)}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>
      )}

      <div className="flex flex-col gap-3">
        <Label>Add MCP Server</Label>
        <div className="flex gap-2">
          <Input
            value={newUrl}
            onChange={(e) => setNewUrl(e.target.value)}
            placeholder="http://localhost:8000/mcp"
            className="bg-background flex-1"
            onKeyDown={(e) => e.key === "Enter" && fetchTools()}
          />
          <Button onClick={fetchTools} disabled={loading || !newUrl.trim()}>
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <>
                <Plus className="mr-1 h-4 w-4" /> Fetch Tools
              </>
            )}
          </Button>
        </div>
        {error && <p className="text-sm text-red-500">{error}</p>}
      </div>

      {discoveredTools.length > 0 && (
        <ToolSelector tools={discoveredTools} onConfirm={addServer} />
      )}
    </div>
  );
}

function ToolSelector({
  tools,
  onConfirm,
}: {
  tools: MCPToolInfo[];
  onConfirm: (selected: string[]) => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(tools.map((t) => t.name)),
  );

  const toggle = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  return (
    <div className="flex flex-col gap-3 rounded-lg border p-4">
      <Label>Select tools to enable</Label>
      <div className="flex flex-col gap-2">
        {tools.map((tool) => (
          <label
            key={tool.name}
            className="hover:bg-accent/50 flex cursor-pointer items-start gap-3 rounded-md border p-3 transition-colors"
          >
            <input
              type="checkbox"
              checked={selected.has(tool.name)}
              onChange={() => toggle(tool.name)}
              className="mt-0.5 h-4 w-4 shrink-0"
            />
            <div className="flex flex-col gap-0.5">
              <span className="text-sm font-medium">{tool.name}</span>
              {tool.description && (
                <span className="text-muted-foreground text-xs">
                  {tool.description}
                </span>
              )}
            </div>
          </label>
        ))}
      </div>
      <Button
        onClick={() => onConfirm(Array.from(selected))}
        disabled={selected.size === 0}
        className="self-end"
      >
        Add {selected.size} tool{selected.size !== 1 ? "s" : ""}
      </Button>
    </div>
  );
}
