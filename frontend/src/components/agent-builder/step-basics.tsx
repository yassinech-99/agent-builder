"use client";

import React, { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAgentConfig } from "@/providers/AgentConfig";
import { getBuilderApiUrl } from "@/lib/runtime-urls";

interface ModelOption {
  id: string;
  name: string;
  provider: string;
  model_name: string;
}

export function StepBasics() {
  const { config, setConfig } = useAgentConfig();
  const [models, setModels] = useState<ModelOption[]>([]);

  useEffect(() => {
    fetch(`${getBuilderApiUrl()}/api/models`)
      .then((r) => (r.ok ? r.json() : []))
      .then(setModels)
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="agent-name">Agent Name</Label>
        <Input
          id="agent-name"
          value={config.agent_name}
          onChange={(e) =>
            setConfig((prev) => ({ ...prev, agent_name: e.target.value }))
          }
          placeholder="Deep Research Assistant"
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="agent-model">Model Endpoint</Label>
        <select
          id="agent-model"
          value={config.model_id}
          onChange={(e) =>
            setConfig((prev) => ({ ...prev, model_id: e.target.value }))
          }
          className="border-input bg-background flex h-9 w-full rounded-md border px-3 py-1 text-sm shadow-xs focus-visible:outline-none focus-visible:ring-1"
        >
          <option value="">Default Workspace Model (NVIDIA NIM / Env)</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} ({m.provider} · {m.model_name})
            </option>
          ))}
        </select>
        <p className="text-muted-foreground text-xs">
          Configure NVIDIA NIM, Dynamo, OpenAI, Anthropic, or local vLLM/Ollama
          endpoints in{" "}
          <a href="/models" className="text-primary underline">
            Models
          </a>
          .
        </p>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="agent-prompt">System Prompt</Label>
        <textarea
          id="agent-prompt"
          rows={5}
          value={config.system_prompt}
          onChange={(e) =>
            setConfig((prev) => ({ ...prev, system_prompt: e.target.value }))
          }
          placeholder="You are a precise, tool-using assistant..."
          className="border-input bg-background w-full rounded-md border px-3 py-2 text-sm shadow-xs focus-visible:outline-none focus-visible:ring-1"
        />
      </div>

      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={config.voice_enabled}
          onChange={(e) =>
            setConfig((prev) => ({ ...prev, voice_enabled: e.target.checked }))
          }
        />
        Enable real-time Voice Studio toggle in chat
      </label>
    </div>
  );
}
