"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { getBuilderApiUrl, getTenantHeaders } from "@/lib/runtime-urls";
import {
  BookOpenCheck,
  Plus,
  Trash2,
  Sparkles,
  Layers,
  Loader2,
  Play,
} from "lucide-react";
import { toast } from "sonner";

interface SkillItem {
  id: string;
  slug: string;
  name: string;
  description: string;
  category: string;
  version: string;
  tags: string[];
  allowed_tools: string[];
  body_markdown: string;
  resources: Record<string, string>;
  enabled: boolean;
  tier1_index_tokens: number;
  tier2_playbook_tokens: number;
  tier3_resources_tokens: number;
  progressive_savings_pct: number;
  raw_skill_md: string;
}

const SAMPLE_SKILL_MD = [
  "---",
  "name: Kubernetes GPU Cluster Debugger",
  "slug: k8s-gpu-debugger",
  "description: Diagnose NVIDIA device plugin, NCCL timeout, XID hardware errors, and Dynamo pod scheduling bottlenecks.",
  "category: operations",
  "version: 1.0.0",
  "tags: [kubernetes, nvidia, nccl, gpu]",
  "allowed_tools: [orchestrate_tools, read_artifact]",
  "---",
  "",
  "## GPU Cluster Triage Protocol",
  "1. Check DCGM XID telemetry (`XID 79` fallen off bus vs `XID 48` double-bit ECC) before restarting pods.",
  "2. Verify `NCCL_IB_DISABLE` / `UCX_TLS` RDMA transport health across multi-node prefill/decode pools.",
  '3. Offload `kubectl logs` > 3K tokens automatically via `read_artifact(..., grep="WARN|ERROR|NCCL")`.',
].join("\n");

export default function SkillsAndContextPage() {
  const [skills, setSkills] = useState<SkillItem[]>([]);
  const [selected, setSelected] = useState<SkillItem | null>(null);
  const [loading, setLoading] = useState(true);

  const [showCreate, setShowCreate] = useState(false);
  const [draftSkillMd, setDraftSkillMd] = useState(SAMPLE_SKILL_MD);
  const [saving, setSaving] = useState(false);

  const [simResult, setSimResult] = useState<Record<string, unknown> | null>(null);
  const [simRunning, setSimRunning] = useState(false);

  const loadSkills = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/skills`, {
        headers: getTenantHeaders(),
        cache: "no-store",
      });
      if (res.ok) {
        const items = (await res.json()) as SkillItem[];
        setSkills(items);
        if (!selected && items.length > 0) setSelected(items[0]);
      }
    } catch {
      toast.error("Failed to load SKILL.md registry");
    } finally {
      setLoading(false);
    }
  }, [selected]);

  useEffect(() => {
    void loadSkills();
  }, [loadSkills]);

  const handleImportSkill = async () => {
    setSaving(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/skills`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getTenantHeaders() },
        body: JSON.stringify({ raw_skill_md: draftSkillMd }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.detail || "Invalid SKILL.md");
        return;
      }
      toast.success(`Registered progressive skill '${data.slug}'`);
      setShowCreate(false);
      await loadSkills();
    } finally {
      setSaving(false);
    }
  };

  const handleToggleSkill = async (skill: SkillItem, nextVal: boolean) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/skills/${skill.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({ enabled: nextVal }),
    });
    if (res.ok) {
      await loadSkills();
    }
  };

  const handleDeleteSkill = async (skillId: string) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/skills/${skillId}`, {
      method: "DELETE",
      headers: getTenantHeaders(),
    });
    if (res.ok) {
      toast.success("Skill removed");
      if (selected?.id === skillId) setSelected(null);
      await loadSkills();
    }
  };

  const runFiveStageSimulation = async () => {
    setSimRunning(true);
    try {
      const hugeJson = JSON.stringify(
        Array.from({ length: 180 }, (_, idx) => ({
          pod: `nim-decode-${idx}`,
          kv_utilization: 0.87,
          p99_ttft_ms: 42 + idx,
        })),
        null,
        2,
      );
      const res = await fetch(`${getBuilderApiUrl()}/api/context/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getTenantHeaders() },
        body: JSON.stringify({
          context_window: 8192,
          offload_threshold_tokens: 600,
          clear_at_least_tokens: 400,
          force_compact: true,
          messages: [
            { type: "human", content: "Project target: keep p99 TTFT under 75ms on 16 GPUs." },
            {
              type: "ai",
              content:
                "<think>Let me carefully inspect all 180 NIM pod telemetry records first...</think>Inspecting cluster metrics now.",
              tool_calls: [{ name: "cluster_metrics", args: {}, id: "c1" }],
            },
            { type: "tool", name: "cluster_metrics", tool_call_id: "c1", content: hugeJson },
            {
              type: "ai",
              content: "<think>Analyzing tail pods...</think>Pods 34+ exceed target.",
            },
            { type: "human", content: "Summarize recommended Dynamo scaling config." },
          ],
        }),
      });
      if (res.ok) {
        setSimResult(await res.json());
      }
    } finally {
      setSimRunning(false);
    }
  };

  const totalTier1 = skills.reduce((acc, s) => acc + s.tier1_index_tokens, 0);
  const totalFull = skills.reduce(
    (acc, s) => acc + s.tier1_index_tokens + s.tier2_playbook_tokens + s.tier3_resources_tokens,
    0,
  );
  const overallSavedPct =
    totalFull > 0 ? (((totalFull - totalTier1) / totalFull) * 100).toFixed(1) : "0.0";

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Agent Skills (<code>SKILL.md</code>) &amp; 5-Stage Context Engine
          </h1>
          <p className="text-muted-foreground text-sm">
            3-Tier progressive playbooks, cross-user KV prefix alignment &amp; automatic artifact offloading
          </p>
        </div>
        <Button size="sm" onClick={() => setShowCreate((v) => !v)}>
          <Plus className="mr-1.5 h-4 w-4" />
          Import / Author SKILL.md
        </Button>
      </div>

      <div className="flex-1 space-y-6 overflow-auto p-6">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="bg-card rounded-xl border p-4 shadow-xs">
            <p className="text-muted-foreground text-xs font-medium">Installed SKILL.md Packs</p>
            <p className="mt-1 font-mono text-xl font-bold">{skills.length}</p>
            <p className="text-muted-foreground text-xs">Tenant &amp; workspace isolated</p>
          </div>
          <div className="bg-card rounded-xl border p-4 shadow-xs">
            <p className="text-muted-foreground text-xs font-medium">
              Tier-1 Block 1 Startup Cost
            </p>
            <p className="mt-1 font-mono text-xl font-bold text-emerald-600 dark:text-emerald-400">
              {totalTier1} tok
            </p>
            <p className="text-muted-foreground text-xs">
              vs {totalFull.toLocaleString()} tok if all playbooks injected upfront
            </p>
          </div>
          <div className="bg-card rounded-xl border p-4 shadow-xs">
            <p className="text-muted-foreground text-xs font-medium">Progressive Skill Savings</p>
            <p className="mt-1 flex items-center gap-1.5 font-mono text-xl font-bold text-emerald-600 dark:text-emerald-400">
              <Sparkles className="h-4 w-4" />
              {overallSavedPct}%
            </p>
            <p className="text-muted-foreground text-xs">
              Full instructions load JIT via <code>activate_skill</code>
            </p>
          </div>
        </div>

        {showCreate && (
          <div className="bg-card space-y-3 rounded-xl border p-5 shadow-xs">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold">Import Portable SKILL.md (YAML Frontmatter + Playbook)</h2>
              <Button size="sm" onClick={handleImportSkill} disabled={saving}>
                {saving && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
                Save Skill to Workspace
              </Button>
            </div>
            <Textarea
              rows={11}
              value={draftSkillMd}
              onChange={(e) => setDraftSkillMd(e.target.value)}
              className="font-mono text-xs"
            />
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-2.5 lg:col-span-2">
            {loading ? (
              <div className="text-muted-foreground rounded-xl border p-8 text-center text-sm">
                Loading progressive skills...
              </div>
            ) : (
              skills.map((s) => (
                <div
                  key={s.id}
                  onClick={() => setSelected(s)}
                  className={`cursor-pointer rounded-xl border p-4 transition-all ${
                    selected?.id === s.id
                      ? "border-primary bg-primary/5 shadow-xs"
                      : "hover:border-primary/40"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <BookOpenCheck className="text-primary h-4 w-4" />
                        <span className="font-semibold">{s.name}</span>
                        <span className="bg-muted rounded px-1.5 py-0.5 font-mono text-[10px]">
                          v{s.version}
                        </span>
                      </div>
                      <p className="text-muted-foreground font-mono text-xs">{s.slug}</p>
                    </div>
                    <div
                      className="flex items-center gap-2"
                      onClick={(ev) => ev.stopPropagation()}
                    >
                      <Switch
                        checked={s.enabled}
                        onCheckedChange={(val) => void handleToggleSkill(s, val)}
                      />
                      <button
                        type="button"
                        onClick={() => void handleDeleteSkill(s.id)}
                        className="text-destructive hover:opacity-80"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  <p className="text-muted-foreground mt-2 text-xs">{s.description}</p>
                  <div className="mt-2.5 flex flex-wrap gap-2 font-mono text-[11px]">
                    <span className="rounded bg-emerald-500/15 px-2 py-0.5 text-emerald-700 dark:text-emerald-400">
                      Tier 1: {s.tier1_index_tokens} tok
                    </span>
                    <span className="bg-muted rounded px-2 py-0.5">
                      Tier 2 Playbook: {s.tier2_playbook_tokens} tok
                    </span>
                    <span className="bg-muted rounded px-2 py-0.5">
                      Tier 3 Assets: {Object.keys(s.resources || {}).length} ({s.tier3_resources_tokens} tok)
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="bg-card h-fit space-y-3 rounded-xl border p-5 shadow-xs">
            {selected ? (
              <>
                <div className="text-muted-foreground font-mono text-xs">{selected.slug}</div>
                <h3 className="text-lg font-semibold">{selected.name}</h3>
                <pre className="bg-muted max-h-80 overflow-auto rounded-lg p-3 font-mono text-xs whitespace-pre-wrap">
                  {selected.raw_skill_md}
                </pre>
              </>
            ) : (
              <p className="text-muted-foreground text-xs">Select a skill to inspect its SKILL.md source.</p>
            )}
          </div>
        </div>

        {/* Interactive 5-Stage Context Pipeline Simulator */}
        <div className="bg-card space-y-3 rounded-xl border p-5 shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Layers className="text-primary h-5 w-5" />
              <div>
                <h2 className="text-base font-semibold">
                  5-Stage Graduated Context Shaper Verification Bench
                </h2>
                <p className="text-muted-foreground text-xs">
                  Test Stage 1 MinIO/DB artifact offload + Stage 2 &lt;think&gt; stripping + Stage 4 fact flush &amp; compaction live.
                </p>
              </div>
            </div>
            <Button size="sm" onClick={runFiveStageSimulation} disabled={simRunning}>
              {simRunning ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="mr-1.5 h-3.5 w-3.5" />
              )}
              Simulate 5-Stage Shaper Run
            </Button>
          </div>
          {simResult && (
            <pre className="bg-muted max-h-64 overflow-auto rounded-lg p-3 font-mono text-xs whitespace-pre-wrap">
              {JSON.stringify(simResult, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
