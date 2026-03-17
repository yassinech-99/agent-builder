"use client";

import { useCallback, useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useAgentConfig } from "@/providers/AgentConfig";

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

interface ModelOption {
  id: string;
  name: string;
  provider: string;
  model_name: string;
  is_default: boolean;
}

export function StepBasics() {
  const { config, setConfig } = useAgentConfig();
  const [models, setModels] = useState<ModelOption[]>([]);

  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch(`${BUILDER_API}/api/models`);
      if (!res.ok) return;
      const data = (await res.json()) as ModelOption[];
      setModels(data);
    } catch {
      setModels([]);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const selectedModel = models.find((m) => m.id === config.model_id);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <Label htmlFor="agentName">
          Agent Name<span className="text-rose-500">*</span>
        </Label>
        <Input
          id="agentName"
          value={config.agent_name}
          onChange={(e) =>
            setConfig((prev) => ({ ...prev, agent_name: e.target.value }))
          }
          placeholder="My Agent"
          className="bg-background"
        />
      </div>

      <div className="flex flex-col gap-2">
        <Label htmlFor="systemPrompt">System Prompt</Label>
        <p className="text-muted-foreground text-sm">
          Instructions that define how the agent behaves. Leave empty for
          default.
        </p>
        <textarea
          id="systemPrompt"
          value={config.system_prompt}
          onChange={(e) =>
            setConfig((prev) => ({ ...prev, system_prompt: e.target.value }))
          }
          placeholder="You are a helpful AI assistant..."
          rows={6}
          className="border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:ring-ring flex w-full rounded-md border px-3 py-2 text-sm focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50"
        />
      </div>

      <div className="flex flex-col gap-2">
        <Label htmlFor="modelId">Model</Label>
        <p className="text-muted-foreground text-sm">
          Select a chat or reasoning model for this agent.
        </p>
        <select
          id="modelId"
          value={config.model_id}
          onChange={(e) =>
            setConfig((prev) => ({ ...prev, model_id: e.target.value }))
          }
          className="border-input bg-background flex h-10 rounded-md border px-3 py-1 text-sm"
        >
          <option value="">Use default runtime model</option>
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}{model.is_default ? " • default" : ""}
            </option>
          ))}
        </select>
        {selectedModel && (
          <p className="text-muted-foreground text-xs">
            Selected: {selectedModel.name} ({selectedModel.provider} /{" "}
            {selectedModel.model_name})
          </p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <Label>Voice Mode</Label>
        <p className="text-muted-foreground text-sm">
          Enable real-time voice interaction using Nova 2 Sonic
        </p>
        <label className="flex items-center gap-3 rounded-lg border p-3">
          <input
            type="checkbox"
            checked={config.voice_enabled}
            onChange={(e) =>
              setConfig((prev) => ({ ...prev, voice_enabled: e.target.checked }))
            }
            className="h-4 w-4"
          />
          <span className="text-sm font-medium">
            {config.voice_enabled ? "Voice mode enabled" : "Voice mode disabled"}
          </span>
        </label>
      </div>
    </div>
  );
}
