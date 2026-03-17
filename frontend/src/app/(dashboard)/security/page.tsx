"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Shield,
  Plus,
  Trash2,
  Pencil,
  X,
  AlertTriangle,
  Activity,
} from "lucide-react";

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

interface GuardrailRule {
  id: string;
  name: string;
  category: string;
  pattern: string;
  action: string;
  enabled: boolean;
  created_at: string;
}

interface GuardrailEvent {
  id: string;
  rule_id: string | null;
  rule_name: string | null;
  category: string | null;
  action: string | null;
  matched_text: string | null;
  thread_id: string;
  assistant_id: string;
  created_at: string;
}

type Tab = "rules" | "monitor";

const CATEGORIES = [
  { value: "pii", label: "PII" },
  { value: "secrets", label: "Secrets" },
  { value: "harmful", label: "Harmful Content" },
  { value: "custom", label: "Custom" },
];

const ACTIONS = [
  { value: "warn", label: "Warn" },
  { value: "block", label: "Block" },
  { value: "redact", label: "Redact" },
];

export default function SecurityPage() {
  const [tab, setTab] = useState<Tab>("rules");

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Security</h1>
          <p className="text-muted-foreground text-sm">
            LangChain Guardrails — content filtering and monitoring
          </p>
        </div>
      </div>

      <div className="flex gap-1 border-b px-6 pt-2">
        <button
          onClick={() => setTab("rules")}
          className={`flex items-center gap-1.5 rounded-t-md px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "rules"
              ? "border-primary text-primary border-b-2 bg-transparent"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Shield className="h-4 w-4" />
          Rules
        </button>
        <button
          onClick={() => setTab("monitor")}
          className={`flex items-center gap-1.5 rounded-t-md px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "monitor"
              ? "border-primary text-primary border-b-2 bg-transparent"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Activity className="h-4 w-4" />
          Live Monitor
        </button>
      </div>

      <div className="flex-1 overflow-auto">
        {tab === "rules" ? <RulesTab /> : <MonitorTab />}
      </div>
    </div>
  );
}

function actionColor(action: string) {
  switch (action) {
    case "block":
      return "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400";
    case "redact":
      return "bg-amber-100 text-amber-700 dark:bg-amber-950/30 dark:text-amber-400";
    default:
      return "bg-yellow-100 text-yellow-700 dark:bg-yellow-950/30 dark:text-yellow-400";
  }
}

function categoryColor(category: string) {
  switch (category) {
    case "pii":
      return "bg-blue-100 text-blue-700 dark:bg-blue-950/30 dark:text-blue-400";
    case "secrets":
      return "bg-purple-100 text-purple-700 dark:bg-purple-950/30 dark:text-purple-400";
    case "harmful":
      return "bg-red-100 text-red-700 dark:bg-red-950/30 dark:text-red-400";
    default:
      return "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400";
  }
}

function RulesTab() {
  const [rules, setRules] = useState<GuardrailRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState({
    name: "",
    category: "pii",
    pattern: "",
    action: "warn",
    enabled: true,
  });

  const fetchRules = useCallback(async () => {
    try {
      const res = await fetch(`${BUILDER_API}/api/guardrails/rules`);
      if (res.ok) setRules(await res.json());
    } catch (e) {
      console.error("Failed to fetch rules:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRules();
  }, [fetchRules]);

  const resetForm = () => {
    setForm({ name: "", category: "pii", pattern: "", action: "warn", enabled: true });
    setEditingId(null);
    setShowForm(false);
  };

  const handleSave = async () => {
    const method = editingId ? "PUT" : "POST";
    const url = editingId
      ? `${BUILDER_API}/api/guardrails/rules/${editingId}`
      : `${BUILDER_API}/api/guardrails/rules`;
    try {
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (res.ok) {
        resetForm();
        fetchRules();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "Save failed");
      }
    } catch (e) {
      console.error("Failed to save rule:", e);
    }
  };

  const handleEdit = (rule: GuardrailRule) => {
    setEditingId(rule.id);
    setForm({
      name: rule.name,
      category: rule.category,
      pattern: rule.pattern,
      action: rule.action,
      enabled: rule.enabled,
    });
    setShowForm(true);
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this guardrail rule?")) return;
    try {
      await fetch(`${BUILDER_API}/api/guardrails/rules/${id}`, { method: "DELETE" });
      fetchRules();
    } catch (e) {
      console.error("Failed to delete rule:", e);
    }
  };

  const handleToggle = async (rule: GuardrailRule) => {
    try {
      await fetch(`${BUILDER_API}/api/guardrails/rules/${rule.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...rule, enabled: !rule.enabled }),
      });
      fetchRules();
    } catch (e) {
      console.error("Failed to toggle rule:", e);
    }
  };

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-muted-foreground text-sm">
          Define content filtering rules with regex patterns
        </p>
        <Button size="sm" onClick={() => { resetForm(); setShowForm(true); }}>
          <Plus className="mr-1.5 h-4 w-4" />
          Add Rule
        </Button>
      </div>

      {showForm && (
        <div className="bg-card mb-6 rounded-lg border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              {editingId ? "Edit Rule" : "New Guardrail Rule"}
            </h2>
            <Button variant="ghost" size="icon" onClick={resetForm}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label>Rule Name</Label>
              <Input
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="Email Filter"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Category</Label>
              <select
                value={form.category}
                onChange={(e) => setForm({ ...form, category: e.target.value })}
                className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
              >
                {CATEGORIES.map((c) => (
                  <option key={c.value} value={c.value}>{c.label}</option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1.5 md:col-span-2">
              <Label>Regex Pattern</Label>
              <Input
                value={form.pattern}
                onChange={(e) => setForm({ ...form, pattern: e.target.value })}
                placeholder="[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
                className="font-mono text-sm"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Action</Label>
              <select
                value={form.action}
                onChange={(e) => setForm({ ...form, action: e.target.value })}
                className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
              >
                {ACTIONS.map((a) => (
                  <option key={a.value} value={a.value}>{a.label}</option>
                ))}
              </select>
            </div>
            <div className="flex items-end gap-2">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={form.enabled}
                  onChange={(e) => setForm({ ...form, enabled: e.target.checked })}
                />
                Enabled
              </label>
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="outline" onClick={resetForm}>Cancel</Button>
            <Button onClick={handleSave} disabled={!form.name || !form.pattern}>
              {editingId ? "Update" : "Create"}
            </Button>
          </div>
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-muted h-24 animate-pulse rounded-lg" />
          ))}
        </div>
      ) : rules.length === 0 && !showForm ? (
        <div className="flex flex-col items-center justify-center py-20">
          <Shield className="text-muted-foreground mb-4 h-16 w-16" />
          <h2 className="mb-2 text-lg font-semibold">No guardrail rules</h2>
          <p className="text-muted-foreground mb-6 text-sm">
            Create rules to filter sensitive content
          </p>
          <Button onClick={() => setShowForm(true)}>
            <Plus className="mr-1.5 h-4 w-4" />
            Add Rule
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {rules.map((rule) => (
            <div key={rule.id} className="bg-card flex flex-col rounded-lg border p-5 shadow-sm">
              <div className="mb-2 flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold">{rule.name}</h3>
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${categoryColor(rule.category)}`}>
                      {rule.category}
                    </span>
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${actionColor(rule.action)}`}>
                      {rule.action}
                    </span>
                  </div>
                  <p className="text-muted-foreground mt-1 truncate font-mono text-xs">
                    {rule.pattern || "No pattern"}
                  </p>
                </div>
              </div>
              <div className="mt-auto flex items-center gap-1 border-t pt-3">
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={rule.enabled}
                    onChange={() => handleToggle(rule)}
                  />
                  <span className="text-muted-foreground text-xs">
                    {rule.enabled ? "Active" : "Disabled"}
                  </span>
                </label>
                <div className="ml-auto flex gap-1">
                  <Button variant="ghost" size="sm" onClick={() => handleEdit(rule)}>
                    <Pencil className="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-destructive hover:text-destructive"
                    onClick={() => handleDelete(rule.id)}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MonitorTab() {
  const [events, setEvents] = useState<GuardrailEvent[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchEvents = useCallback(async () => {
    try {
      const res = await fetch(`${BUILDER_API}/api/guardrails/events?limit=100`);
      if (res.ok) setEvents(await res.json());
    } catch (e) {
      console.error("Failed to fetch events:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEvents();
    const interval = setInterval(fetchEvents, 5000);
    return () => clearInterval(interval);
  }, [fetchEvents]);

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-muted-foreground text-sm">
          Real-time guardrail activity feed (auto-refreshes every 5s)
        </p>
        <Button size="sm" variant="outline" onClick={fetchEvents}>
          Refresh
        </Button>
      </div>

      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="bg-muted h-16 animate-pulse rounded-lg" />
          ))}
        </div>
      ) : events.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20">
          <Activity className="text-muted-foreground mb-4 h-16 w-16" />
          <h2 className="mb-2 text-lg font-semibold">No events yet</h2>
          <p className="text-muted-foreground text-sm">
            Guardrail events will appear here as they are triggered
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {events.map((ev) => (
            <div
              key={ev.id}
              className="bg-card flex items-start gap-3 rounded-lg border p-4 shadow-sm"
            >
              <AlertTriangle className={`mt-0.5 h-4 w-4 shrink-0 ${
                ev.action === "block" ? "text-red-500" : ev.action === "redact" ? "text-amber-500" : "text-yellow-500"
              }`} />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{ev.rule_name}</span>
                  {ev.category && (
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${categoryColor(ev.category)}`}>
                      {ev.category}
                    </span>
                  )}
                  {ev.action && (
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${actionColor(ev.action)}`}>
                      {ev.action}
                    </span>
                  )}
                </div>
                {ev.matched_text && (
                  <p className="text-muted-foreground mt-1 truncate font-mono text-xs">
                    Matched: {ev.matched_text}
                  </p>
                )}
                <p className="text-muted-foreground mt-1 text-xs">
                  {new Date(ev.created_at).toLocaleString()}
                  {ev.thread_id ? ` · Thread: ${ev.thread_id.slice(0, 8)}...` : ""}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
