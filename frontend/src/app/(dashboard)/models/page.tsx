"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plus, Trash2, Pencil, Cpu, Star, X } from "lucide-react";

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

const NOVA_MODELS = [
  { value: "amazon.nova-micro-v1:0", label: "Nova Micro" },
  { value: "amazon.nova-lite-v1:0", label: "Nova Lite" },
  { value: "eu.amazon.nova-2-lite-v1:0", label: "Nova 2 Lite (eu cross-region)" },
  { value: "amazon.nova-pro-v1:0", label: "Nova Pro" },
  { value: "amazon.nova-premier-v1:0", label: "Nova Premier" },
] as const;

interface ModelConfig {
  id: string;
  name: string;
  provider: string;
  base_url: string;
  api_key: string;
  model_name: string;
  temperature: number;
  max_tokens: number | null;
  is_default: boolean;
  aws_region: string;
  aws_secret_key: string;
  created_at: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<{
    name: string;
    provider: string;
    base_url: string;
    api_key: string;
    model_name: string;
    temperature: number;
    max_tokens: number | null;
    is_default: boolean;
    aws_region: string;
    aws_secret_key: string;
  }>({
    name: "",
    provider: "bedrock",
    base_url: "",
    api_key: "",
    model_name: NOVA_MODELS[0].value,
    temperature: 0.7,
    max_tokens: null,
    is_default: false,
    aws_region: "",
    aws_secret_key: "",
  });

  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch(`${BUILDER_API}/api/models`);
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
      provider: "bedrock",
      base_url: "",
      api_key: "",
      model_name: NOVA_MODELS[0].value,
      temperature: 0.7,
      max_tokens: null,
      is_default: false,
      aws_region: "",
      aws_secret_key: "",
    });
    setEditingId(null);
    setShowForm(false);
  };

  const handleSave = async () => {
    const method = editingId ? "PUT" : "POST";
    const url = editingId
      ? `${BUILDER_API}/api/models/${editingId}`
      : `${BUILDER_API}/api/models`;
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
    setForm({
      name: model.name,
      provider: "bedrock",
      base_url: "",
      api_key: model.api_key,
      model_name: model.model_name,
      temperature: model.temperature,
      max_tokens: model.max_tokens,
      is_default: model.is_default,
      aws_region: model.aws_region || "",
      aws_secret_key: model.aws_secret_key || "",
    });
    setShowForm(true);
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this model?")) return;
    try {
      await fetch(`${BUILDER_API}/api/models/${id}`, { method: "DELETE" });
      fetchModels();
    } catch (e) {
      console.error("Failed to delete model:", e);
    }
  };

  const novaLabel = (modelName: string) =>
    NOVA_MODELS.find((m) => m.value === modelName)?.label ?? modelName;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Models</h1>
          <p className="text-muted-foreground text-sm">
            Amazon Nova models on Bedrock
          </p>
        </div>
        <Button onClick={() => { resetForm(); setShowForm(true); }}>
          <Plus className="mr-1.5 h-4 w-4" />
          Add Model
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {showForm && (
          <div className="bg-card mb-6 rounded-lg border p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                {editingId ? "Edit Model" : "Add Nova Model"}
              </h2>
              <Button variant="ghost" size="icon" onClick={resetForm}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <Label>Display Name</Label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="My Nova Lite"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Nova Model</Label>
                <select
                  value={form.model_name}
                  onChange={(e) => setForm({ ...form, model_name: e.target.value })}
                  className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
                >
                  {NOVA_MODELS.map((m) => (
                    <option key={m.value} value={m.value}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>AWS Region</Label>
                <Input
                  value={form.aws_region}
                  onChange={(e) => setForm({ ...form, aws_region: e.target.value })}
                  placeholder="us-east-1"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>AWS Access Key ID (optional)</Label>
                <Input
                  type="password"
                  value={form.api_key}
                  onChange={(e) => setForm({ ...form, api_key: e.target.value })}
                  placeholder="AKIA... (falls back to env)"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>AWS Secret Access Key (optional)</Label>
                <Input
                  type="password"
                  value={form.aws_secret_key}
                  onChange={(e) => setForm({ ...form, aws_secret_key: e.target.value })}
                  placeholder="wJalr... (falls back to env)"
                />
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
                <Label>Max Tokens (optional)</Label>
                <Input
                  type="number"
                  value={form.max_tokens ?? ""}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      max_tokens: e.target.value ? parseInt(e.target.value) : null,
                    })
                  }
                  placeholder="4096"
                />
              </div>
              <div className="flex items-end gap-2">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={form.is_default}
                    onChange={(e) =>
                      setForm({ ...form, is_default: e.target.checked })
                    }
                  />
                  Set as default model
                </label>
              </div>
            </div>
            <div className="mt-4 flex justify-end gap-2">
              <Button variant="outline" onClick={resetForm}>
                Cancel
              </Button>
              <Button onClick={handleSave} disabled={!form.name || !form.model_name}>
                {editingId ? "Update" : "Create"}
              </Button>
            </div>
          </div>
        )}

        {loading ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="bg-muted h-32 animate-pulse rounded-lg" />
            ))}
          </div>
        ) : models.length === 0 && !showForm ? (
          <div className="flex flex-col items-center justify-center py-20">
            <Cpu className="text-muted-foreground mb-4 h-16 w-16" />
            <h2 className="mb-2 text-lg font-semibold">No models configured</h2>
            <p className="text-muted-foreground mb-6 text-sm">
              Add an Amazon Nova model to use with your agents
            </p>
            <Button onClick={() => setShowForm(true)}>
              <Plus className="mr-1.5 h-4 w-4" />
              Add Model
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {models.map((model) => (
              <div
                key={model.id}
                className="bg-card flex flex-col rounded-lg border p-5 shadow-sm"
              >
                <div className="mb-3 flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <div className="bg-primary/10 text-primary flex h-9 w-9 items-center justify-center rounded-lg">
                      <Cpu className="h-5 w-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <h3 className="font-semibold">{model.name}</h3>
                        {model.is_default && (
                          <Star className="h-3.5 w-3.5 fill-yellow-400 text-yellow-400" />
                        )}
                      </div>
                      <p className="text-muted-foreground text-xs">
                        {novaLabel(model.model_name)}
                      </p>
                    </div>
                  </div>
                </div>
                <p className="text-muted-foreground mb-4 flex-1 text-xs">
                  Temp: {model.temperature}
                  {model.max_tokens ? ` · Max: ${model.max_tokens}` : ""}
                  {model.aws_region ? ` · ${model.aws_region}` : ""}
                </p>
                <div className="flex items-center gap-1 border-t pt-3">
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
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
