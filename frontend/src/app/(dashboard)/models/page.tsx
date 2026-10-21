"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getBuilderApiUrl } from "@/lib/runtime-urls";
import {
  Plus,
  Trash2,
  Pencil,
  Cpu,
  Star,
  X,
  RefreshCw,
  Activity,
  CheckCircle2,
  XCircle,
  Loader2,
  Sparkles,
} from "lucide-react";

const PROVIDERS = [
  {
    value: "nvidia_nim",
    label: "NVIDIA NIM",
    defaultBaseUrl: "https://integrate.api.nvidia.com/v1",
    keyPlaceholder: "nvapi-... (falls back to NVIDIA_API_KEY)",
  },
  {
    value: "nvidia_dynamo",
    label: "NVIDIA Dynamo (KV Router)",
    defaultBaseUrl: "http://localhost:8000/v1",
    keyPlaceholder: "Optional cluster token / NVIDIA_API_KEY",
  },
  {
    value: "openai",
    label: "OpenAI",
    defaultBaseUrl: "https://api.openai.com/v1",
    keyPlaceholder: "sk-... (falls back to OPENAI_API_KEY)",
  },
  {
    value: "anthropic",
    label: "Anthropic",
    defaultBaseUrl: "https://api.anthropic.com",
    keyPlaceholder: "sk-ant-... (falls back to ANTHROPIC_API_KEY)",
  },
  {
    value: "openrouter",
    label: "OpenRouter",
    defaultBaseUrl: "https://openrouter.ai/api/v1",
    keyPlaceholder: "sk-or-... (falls back to OPENROUTER_API_KEY)",
  },
  {
    value: "vllm",
    label: "vLLM (Self-Hosted)",
    defaultBaseUrl: "http://localhost:8000/v1",
    keyPlaceholder: "Optional vLLM bearer token",
  },
  {
    value: "sglang",
    label: "SGLang (Self-Hosted)",
    defaultBaseUrl: "http://localhost:30000/v1",
    keyPlaceholder: "Optional SGLang token",
  },
  {
    value: "ollama",
    label: "Ollama (Local / LAN)",
    defaultBaseUrl: "http://localhost:11434/v1",
    keyPlaceholder: "Optional (not required for local Ollama)",
  },
  {
    value: "openai_compatible",
    label: "OpenAI-Compatible Gateway",
    defaultBaseUrl: "http://localhost:8080/v1",
    keyPlaceholder: "Optional API key",
  },
] as const;

const NIM_PRESETS = [
  {
    value: "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    label: "Llama 3.3 Nemotron Super 49B v1.5",
    contextWindow: 131072,
    supportsThinking: true,
    role: "primary_chat",
  },
  {
    value: "nvidia/nemotron-3-ultra-550b-a55b",
    label: "Nemotron 3 Ultra 550B (MoE)",
    contextWindow: 131072,
    supportsThinking: true,
    role: "primary_chat",
  },
  {
    value: "nvidia/nemotron-3-super-120b-a12b",
    label: "Nemotron 3 Super 120B (1M Context)",
    contextWindow: 1048576,
    supportsThinking: true,
    role: "compactor",
  },
  {
    value: "nvidia/llama-3.1-nemotron-nano-8b-v1",
    label: "Llama 3.1 Nemotron Nano 8B",
    contextWindow: 131072,
    supportsThinking: false,
    role: "fast_router",
  },
  {
    value: "deepseek-ai/deepseek-r1",
    label: "DeepSeek R1 (Reasoning)",
    contextWindow: 131072,
    supportsThinking: true,
    role: "primary_chat",
  },
  {
    value: "qwen/qwen3-235b-a22b",
    label: "Qwen3 235B A22B (MoE)",
    contextWindow: 131072,
    supportsThinking: true,
    role: "primary_chat",
  },
  {
    value: "meta/llama-3.3-70b-instruct",
    label: "Llama 3.3 70B Instruct",
    contextWindow: 131072,
    supportsThinking: false,
    role: "primary_chat",
  },
] as const;

const ROLE_OPTIONS = [
  { value: "primary_chat", label: "Primary Chat & Reasoning" },
  { value: "fast_router", label: "Fast Router / Utility" },
  { value: "compactor", label: "Context Compactor" },
  { value: "contextualizer", label: "RAG Chunk Contextualizer" },
] as const;

interface ModelConfig {
  id: string;
  name: string;
  provider: string;
  base_url: string;
  api_key: string;
  api_key_masked: string;
  has_api_key: boolean;
  model_name: string;
  temperature: number;
  max_tokens: number | null;
  is_default: boolean;
  role: string;
  fallback_model_id: string;
  context_window: number;
  supports_thinking: boolean;
  created_at: string;
}

interface PingStatus {
  ok: boolean;
  latency_ms: number;
  message: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingKeyMask, setEditingKeyMask] = useState<string>("");
  const [discoveredModels, setDiscoveredModels] = useState<string[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [discoverError, setDiscoverError] = useState<string | null>(null);
  const [formPing, setFormPing] = useState<PingStatus | "loading" | null>(null);
  const [cardPings, setCardPings] = useState<Record<string, PingStatus | "loading">>({});

  const [form, setForm] = useState({
    name: "",
    provider: "nvidia_nim",
    base_url: "https://integrate.api.nvidia.com/v1",
    api_key: "",
    model_name: NIM_PRESETS[0].value as string,
    temperature: 0.7,
    max_tokens: null as number | null,
    is_default: false,
    role: "primary_chat",
    fallback_model_id: "",
    context_window: 131072,
    supports_thinking: true,
  });

  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/models`);
      if (res.ok) setModels(await res.json());
    } catch (e) {
      console.error("Failed to fetch models:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const resetForm = () => {
    setForm({
      name: "",
      provider: "nvidia_nim",
      base_url: "https://integrate.api.nvidia.com/v1",
      api_key: "",
      model_name: NIM_PRESETS[0].value,
      temperature: 0.7,
      max_tokens: null,
      is_default: false,
      role: "primary_chat",
      fallback_model_id: "",
      context_window: 131072,
      supports_thinking: true,
    });
    setEditingId(null);
    setEditingKeyMask("");
    setDiscoveredModels([]);
    setDiscoverError(null);
    setFormPing(null);
    setShowForm(false);
  };

  const handleProviderChange = (provider: string) => {
    const meta = PROVIDERS.find((p) => p.value === provider);
    setDiscoveredModels([]);
    setDiscoverError(null);
    setForm((prev) => ({
      ...prev,
      provider,
      base_url: meta?.defaultBaseUrl || "",
      model_name:
        provider === "nvidia_nim" || provider === "nvidia_dynamo"
          ? NIM_PRESETS[0].value
          : provider === "openai"
            ? "gpt-4o"
            : provider === "anthropic"
              ? "claude-sonnet-4-5"
              : provider === "ollama"
                ? "qwen3:8b"
                : prev.model_name,
    }));
  };

  const handleApplyPreset = (presetValue: string) => {
    const preset = NIM_PRESETS.find((p) => p.value === presetValue);
    if (!preset) return;
    setForm((prev) => ({
      ...prev,
      name: prev.name || preset.label,
      model_name: preset.value,
      context_window: preset.contextWindow,
      supports_thinking: preset.supportsThinking,
      role: preset.role,
    }));
  };

  const handleDiscoverModels = async () => {
    setDiscovering(true);
    setDiscoverError(null);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/models/discover`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: form.provider,
          base_url: form.base_url,
          api_key: form.api_key,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setDiscoverError(data.detail || `Discovery failed (HTTP ${res.status})`);
        return;
      }
      const ids = Array.isArray(data) ? data.map((m: { id: string }) => m.id) : [];
      setDiscoveredModels(ids);
      if (ids.length > 0 && !ids.includes(form.model_name)) {
        setForm((prev) => ({ ...prev, model_name: ids[0] }));
      }
    } catch (e: any) {
      setDiscoverError(e?.message || "Network error during discovery");
    } finally {
      setDiscovering(false);
    }
  };

  const handleTestFormModel = async () => {
    setFormPing("loading");
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/models/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: form.provider,
          base_url: form.base_url,
          api_key: form.api_key,
          model_name: form.model_name,
          temperature: form.temperature,
        }),
      });
      const data = await res.json();
      setFormPing({
        ok: !!data.ok,
        latency_ms: data.latency_ms ?? 0,
        message: data.ok
          ? `Connected (${data.latency_ms} ms)`
          : data.error || "Connection check failed",
      });
    } catch {
      setFormPing({ ok: false, latency_ms: 0, message: "Network error" });
    }
  };

  const handleTestCardModel = async (id: string) => {
    setCardPings((prev) => ({ ...prev, [id]: "loading" }));
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/models/${id}/test`, {
        method: "POST",
      });
      const data = await res.json();
      setCardPings((prev) => ({
        ...prev,
        [id]: {
          ok: !!data.ok,
          latency_ms: data.latency_ms ?? 0,
          message: data.ok
            ? `${data.latency_ms} ms · Ready`
            : data.error || "Failed",
        },
      }));
    } catch {
      setCardPings((prev) => ({
        ...prev,
        [id]: { ok: false, latency_ms: 0, message: "Network error" },
      }));
    }
  };

  const handleSave = async () => {
    const method = editingId ? "PUT" : "POST";
    const url = editingId
      ? `${getBuilderApiUrl()}/api/models/${editingId}`
      : `${getBuilderApiUrl()}/api/models`;
    try {
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (res.ok) {
        resetForm();
        fetchModels();
      }
    } catch (e) {
      console.error("Failed to save model:", e);
    }
  };

  const handleEdit = (model: ModelConfig) => {
    setEditingId(model.id);
    setEditingKeyMask(model.api_key_masked || "");
    setForm({
      name: model.name,
      provider: model.provider || "nvidia_nim",
      base_url: model.base_url || "",
      api_key: "",
      model_name: model.model_name,
      temperature: model.temperature,
      max_tokens: model.max_tokens,
      is_default: model.is_default,
      role: model.role || "primary_chat",
      fallback_model_id: model.fallback_model_id || "",
      context_window: model.context_window || 131072,
      supports_thinking: model.supports_thinking ?? false,
    });
    setShowForm(true);
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this model configuration?")) return;
    try {
      await fetch(`${getBuilderApiUrl()}/api/models/${id}`, { method: "DELETE" });
      fetchModels();
    } catch (e) {
      console.error("Failed to delete model:", e);
    }
  };

  const providerLabel = (p: string) =>
    PROVIDERS.find((item) => item.value === p)?.label ?? p;

  const currentProviderMeta =
    PROVIDERS.find((p) => p.value === form.provider) ?? PROVIDERS[0];

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Models</h1>
          <p className="text-muted-foreground text-sm">
            NVIDIA NIM, NVIDIA Dynamo &amp; multi-provider LLM registry
          </p>
        </div>
        <Button
          onClick={() => {
            resetForm();
            setShowForm(true);
          }}
        >
          <Plus className="mr-1.5 h-4 w-4" />
          Add Model
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {showForm && (
          <div className="bg-card mb-6 rounded-xl border p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">
                  {editingId ? "Edit Model Endpoint" : "Register Model Endpoint"}
                </h2>
                <p className="text-muted-foreground text-xs">
                  Configure Hosted NVIDIA NIM, self-hosted NIM/Dynamo, or any OpenAI/Anthropic provider
                </p>
              </div>
              <Button variant="ghost" size="icon" onClick={resetForm}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            {(form.provider === "nvidia_nim" || form.provider === "nvidia_dynamo") && (
              <div className="mb-5">
                <Label className="text-muted-foreground mb-2 block text-xs uppercase tracking-wider">
                  Quick NVIDIA NIM Presets
                </Label>
                <div className="flex flex-wrap gap-1.5">
                  {NIM_PRESETS.map((preset) => {
                    const active = form.model_name === preset.value;
                    return (
                      <button
                        key={preset.value}
                        type="button"
                        onClick={() => handleApplyPreset(preset.value)}
                        className={`inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-medium transition-colors ${
                          active
                            ? "border-primary bg-primary text-primary-foreground"
                            : "hover:border-primary/50 bg-background text-foreground"
                        }`}
                      >
                        {preset.supportsThinking && <Sparkles className="h-3 w-3" />}
                        {preset.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <Label>Display Name</Label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="Nemotron Super 49B (Production)"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>Provider</Label>
                <select
                  value={form.provider}
                  onChange={(e) => handleProviderChange(e.target.value)}
                  className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
                >
                  {PROVIDERS.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>Base URL</Label>
                <div className="flex gap-2">
                  <Input
                    value={form.base_url}
                    onChange={(e) => setForm({ ...form, base_url: e.target.value })}
                    placeholder={currentProviderMeta.defaultBaseUrl}
                    className="flex-1 font-mono text-xs"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleDiscoverModels}
                    disabled={discovering}
                    title="Query live /v1/models from endpoint"
                  >
                    {discovering ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <RefreshCw className="mr-1 h-3.5 w-3.5" />
                    )}
                    Discover
                  </Button>
                </div>
                {discoverError && (
                  <p className="text-xs text-red-500">{discoverError}</p>
                )}
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>
                  API Key{" "}
                  {editingKeyMask && (
                    <span className="text-muted-foreground font-mono text-xs font-normal">
                      (saved: {editingKeyMask} — leave blank to keep)
                    </span>
                  )}
                </Label>
                <Input
                  type="password"
                  value={form.api_key}
                  onChange={(e) => setForm({ ...form, api_key: e.target.value })}
                  placeholder={currentProviderMeta.keyPlaceholder}
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>Model ID</Label>
                {discoveredModels.length > 0 ? (
                  <select
                    value={form.model_name}
                    onChange={(e) => setForm({ ...form, model_name: e.target.value })}
                    className="border-input bg-background flex h-9 rounded-md border px-3 py-1 font-mono text-xs"
                  >
                    {discoveredModels.map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))}
                  </select>
                ) : (
                  <Input
                    value={form.model_name}
                    onChange={(e) => setForm({ ...form, model_name: e.target.value })}
                    placeholder="nvidia/llama-3.3-nemotron-super-49b-v1.5"
                    className="font-mono text-xs"
                  />
                )}
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>Routing Role</Label>
                <select
                  value={form.role}
                  onChange={(e) => setForm({ ...form, role: e.target.value })}
                  className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
                >
                  {ROLE_OPTIONS.map((r) => (
                    <option key={r.value} value={r.value}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>Context Window (tokens)</Label>
                <Input
                  type="number"
                  value={form.context_window}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      context_window: Number(e.target.value) || 131072,
                    })
                  }
                  min={1024}
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>Fallback Model (optional)</Label>
                <select
                  value={form.fallback_model_id}
                  onChange={(e) =>
                    setForm({ ...form, fallback_model_id: e.target.value })
                  }
                  className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
                >
                  <option value="">None</option>
                  {models
                    .filter((m) => m.id !== editingId)
                    .map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} ({m.provider})
                      </option>
                    ))}
                </select>
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>Temperature ({form.temperature})</Label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={form.temperature}
                  onChange={(e) =>
                    setForm({ ...form, temperature: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <Label>Max Output Tokens (optional)</Label>
                <Input
                  type="number"
                  value={form.max_tokens ?? ""}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      max_tokens: e.target.value ? parseInt(e.target.value) : null,
                    })
                  }
                  placeholder="8192"
                />
              </div>

              <div className="flex flex-wrap items-center gap-6 md:col-span-2">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={form.supports_thinking}
                    onChange={(e) =>
                      setForm({ ...form, supports_thinking: e.target.checked })
                    }
                  />
                  Supports <code>&lt;think&gt;</code> reasoning traces
                </label>

                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={form.is_default}
                    onChange={(e) =>
                      setForm({ ...form, is_default: e.target.checked })
                    }
                  />
                  Set as workspace default model
                </label>
              </div>
            </div>

            {formPing && formPing !== "loading" && (
              <div
                className={`mt-4 flex items-center gap-2 rounded-md px-3 py-2 text-xs ${
                  formPing.ok
                    ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-400"
                    : "bg-red-50 text-red-700 dark:bg-red-950/30 dark:text-red-400"
                }`}
              >
                {formPing.ok ? (
                  <CheckCircle2 className="h-4 w-4 shrink-0" />
                ) : (
                  <XCircle className="h-4 w-4 shrink-0" />
                )}
                <span className="truncate">{formPing.message}</span>
              </div>
            )}

            <div className="mt-5 flex flex-wrap justify-between gap-2 border-t pt-4">
              <Button
                type="button"
                variant="outline"
                onClick={handleTestFormModel}
                disabled={!form.model_name || formPing === "loading"}
              >
                {formPing === "loading" ? (
                  <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Activity className="mr-1.5 h-3.5 w-3.5" />
                )}
                Test Connection
              </Button>
              <div className="flex gap-2">
                <Button variant="outline" onClick={resetForm}>
                  Cancel
                </Button>
                <Button onClick={handleSave} disabled={!form.name || !form.model_name}>
                  {editingId ? "Update Model" : "Save Model"}
                </Button>
              </div>
            </div>
          </div>
        )}

        {loading ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="bg-muted h-36 animate-pulse rounded-lg" />
            ))}
          </div>
        ) : models.length === 0 && !showForm ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <Cpu className="text-muted-foreground mb-4 h-16 w-16" />
            <h2 className="mb-2 text-lg font-semibold">No models registered yet</h2>
            <p className="text-muted-foreground mb-6 max-w-md text-sm">
              Connect NVIDIA NIM, NVIDIA Dynamo, OpenAI, Anthropic, or a local vLLM /
              Ollama endpoint to power your agents.
            </p>
            <Button onClick={() => setShowForm(true)}>
              <Plus className="mr-1.5 h-4 w-4" />
              Add NVIDIA NIM Model
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {models.map((model) => {
              const ping = cardPings[model.id];
              return (
                <div
                  key={model.id}
                  className="bg-card flex flex-col rounded-xl border p-5 shadow-sm"
                >
                  <div className="mb-3 flex items-start justify-between">
                    <div className="flex items-center gap-2.5">
                      <div className="bg-primary/10 text-primary flex h-9 w-9 shrink-0 items-center justify-center rounded-lg">
                        <Cpu className="h-5 w-5" />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <h3 className="truncate font-semibold">{model.name}</h3>
                          {model.is_default && (
                            <Star className="h-3.5 w-3.5 shrink-0 fill-amber-400 text-amber-400" />
                          )}
                        </div>
                        <p className="text-muted-foreground truncate font-mono text-xs">
                          {model.model_name}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mb-3 flex flex-wrap gap-1.5">
                    <span className="bg-primary/10 text-primary rounded px-2 py-0.5 text-[11px] font-medium">
                      {providerLabel(model.provider)}
                    </span>
                    <span className="bg-muted text-muted-foreground rounded px-2 py-0.5 text-[11px]">
                      {Math.round((model.context_window || 131072) / 1024)}K ctx
                    </span>
                    {model.supports_thinking && (
                      <span className="rounded bg-purple-100 px-2 py-0.5 text-[11px] font-medium text-purple-700 dark:bg-purple-950/40 dark:text-purple-300">
                        Thinking
                      </span>
                    )}
                  </div>

                  <p className="text-muted-foreground mb-3 flex-1 text-xs">
                    Role: <span className="font-medium">{model.role}</span> · Temp:{" "}
                    {model.temperature}
                    {model.has_api_key ? ` · Key: ${model.api_key_masked}` : " · Env key"}
                  </p>

                  {ping && ping !== "loading" && (
                    <div
                      className={`mb-3 flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs ${
                        ping.ok
                          ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-400"
                          : "bg-red-50 text-red-700 dark:bg-red-950/30 dark:text-red-400"
                      }`}
                    >
                      {ping.ok ? (
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 shrink-0" />
                      )}
                      <span className="truncate">{ping.message}</span>
                    </div>
                  )}

                  <div className="flex items-center gap-1 border-t pt-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-1"
                      onClick={() => handleTestCardModel(model.id)}
                      disabled={ping === "loading"}
                    >
                      {ping === "loading" ? (
                        <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Activity className="mr-1 h-3.5 w-3.5" />
                      )}
                      Ping
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-1"
                      onClick={() => handleEdit(model)}
                    >
                      <Pencil className="mr-1 h-3.5 w-3.5" />
                      Edit
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-destructive hover:text-destructive"
                      onClick={() => handleDelete(model.id)}
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
