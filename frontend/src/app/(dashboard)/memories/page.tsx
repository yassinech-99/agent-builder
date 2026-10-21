"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { getBuilderApiUrl, getTenantHeaders } from "@/lib/runtime-urls";
import {
  BrainCircuit,
  CheckCircle2,
  Clock,
  History,
  Plus,
  RotateCcw,
  Search,
  Sparkles,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

interface MemorySummary {
  core_blocks_count: number;
  pinned_core_tokens: number;
  max_core_tokens: number;
  active_semantic_facts: number;
  superseded_semantic_facts: number;
  episodic_memories_count: number;
  procedural_active_count: number;
  procedural_proposed_count: number;
}

interface CoreBlock {
  id: string;
  block_type: string;
  title: string;
  content: string;
  token_count: number;
  pinned: boolean;
  updated_at: string;
}

interface SemanticFact {
  id: string;
  subject: string;
  predicate: string;
  object: string;
  statement: string;
  category: string;
  confidence: number;
  importance: number;
  access_count: number;
  valid_from: string;
  invalidated_at: string | null;
  superseded_by_id: string;
}

interface EpisodicItem {
  id: string;
  summary: string;
  problem: string;
  approach: string;
  key_tool_steps: string[];
  user_corrections: string;
  outcome: string;
  importance: number;
  created_at: string;
}

interface ProceduralItem {
  id: string;
  title: string;
  rule_prompt: string;
  rationale: string;
  status: string;
  version: number;
  history: Array<{
    version: number;
    title: string;
    rule_prompt: string;
    status: string;
    timestamp: string;
  }>;
}

type ActiveTab = "blocks" | "facts" | "episodes" | "procedural";

export default function CognitiveMemoriesPage() {
  const [tab, setTab] = useState<ActiveTab>("facts");
  const [summary, setSummary] = useState<MemorySummary | null>(null);
  const [blocks, setBlocks] = useState<CoreBlock[]>([]);
  const [facts, setFacts] = useState<SemanticFact[]>([]);
  const [episodes, setEpisodes] = useState<EpisodicItem[]>([]);
  const [procedural, setProcedural] = useState<ProceduralItem[]>([]);

  const [includeHistorical, setIncludeHistorical] = useState(true);
  const [asOfFilter, setAsOfFilter] = useState("");

  // Hybrid recall debugger state
  const [recallQuery, setRecallQuery] = useState("");
  const [recallLatencyMs, setRecallLatencyMs] = useState<number | null>(null);
  const [recallHits, setRecallHits] = useState<Array<Record<string, unknown>>>([]);

  // Creation forms
  const [blockType, setBlockType] = useState("user_profile");
  const [blockTitle, setBlockTitle] = useState("User Profile");
  const [blockContent, setBlockContent] = useState("");

  const [subj, setSubj] = useState("");
  const [pred, setPred] = useState("");
  const [objVal, setObjVal] = useState("");

  const [ruleTitle, setRuleTitle] = useState("");
  const [rulePrompt, setRulePrompt] = useState("");

  const refreshAll = useCallback(async () => {
    try {
      const headers = getTenantHeaders();
      const base = getBuilderApiUrl();
      const factParams = new URLSearchParams();
      if (includeHistorical) factParams.set("include_historical", "true");
      if (asOfFilter.trim()) factParams.set("as_of", asOfFilter.trim());

      const [sumR, blkR, factR, epR, procR] = await Promise.all([
        fetch(`${base}/api/memories/summary`, { headers, cache: "no-store" }),
        fetch(`${base}/api/memories/blocks`, { headers, cache: "no-store" }),
        fetch(`${base}/api/memories/facts?${factParams.toString()}`, { headers, cache: "no-store" }),
        fetch(`${base}/api/memories/episodes`, { headers, cache: "no-store" }),
        fetch(`${base}/api/memories/procedural`, { headers, cache: "no-store" }),
      ]);

      if (sumR.ok) setSummary((await sumR.json()) as MemorySummary);
      if (blkR.ok) setBlocks((await blkR.json()) as CoreBlock[]);
      if (factR.ok) setFacts((await factR.json()) as SemanticFact[]);
      if (epR.ok) setEpisodes((await epR.json()) as EpisodicItem[]);
      if (procR.ok) setProcedural((await procR.json()) as ProceduralItem[]);
    } catch {
      toast.error("Failed to load 4-tier cognitive memories");
    }
  }, [includeHistorical, asOfFilter]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  const handleHybridRecall = async () => {
    if (!recallQuery.trim()) return;
    const res = await fetch(`${getBuilderApiUrl()}/api/memories/recall`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({ query: recallQuery.trim(), limit: 6 }),
    });
    if (res.ok) {
      const data = await res.json();
      setRecallLatencyMs(data.latency_ms);
      setRecallHits(data.results || []);
    }
  };

  const handleSaveBlock = async () => {
    if (!blockContent.trim()) return;
    const res = await fetch(`${getBuilderApiUrl()}/api/memories/blocks`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({
        block_type: blockType,
        title: blockTitle,
        content: blockContent,
        pinned: true,
      }),
    });
    if (res.ok) {
      setBlockContent("");
      toast.success("Pinned Tier-1 Core Memory Block updated");
      await refreshAll();
    } else {
      const err = await res.json();
      toast.error(err.detail || "Failed to save memory block");
    }
  };

  const handleSaveFact = async () => {
    if (!subj.trim() || !pred.trim() || !objVal.trim()) return;
    const res = await fetch(`${getBuilderApiUrl()}/api/memories/facts`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({
        subject: subj.trim(),
        predicate: pred.trim(),
        object: objVal.trim(),
      }),
    });
    if (res.ok) {
      const out = await res.json();
      setSubj("");
      setPred("");
      setObjVal("");
      toast.success(
        out.superseded_ids?.length
          ? `Saved fact & bi-temporally superseded ${out.superseded_ids.length} conflicting fact(s)`
          : "Saved bi-temporal semantic fact",
      );
      await refreshAll();
    }
  };

  const handleCreateProcedural = async () => {
    if (!ruleTitle.trim() || !rulePrompt.trim()) return;
    const res = await fetch(`${getBuilderApiUrl()}/api/memories/procedural`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({
        title: ruleTitle.trim(),
        rule_prompt: rulePrompt.trim(),
        status: "proposed",
      }),
    });
    if (res.ok) {
      setRuleTitle("");
      setRulePrompt("");
      toast.success("Proposed Tier-4 procedural prompt rule");
      await refreshAll();
    }
  };

  const handleProceduralStatus = async (procId: string, status: string) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/memories/procedural/${procId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({ status }),
    });
    if (res.ok) {
      toast.success(`Updated rule status → ${status}`);
      await refreshAll();
    }
  };

  const handleRollback = async (procId: string, targetVersion: number) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/memories/procedural/${procId}/rollback`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({ target_version: targetVersion }),
    });
    if (res.ok) {
      toast.success(`Rolled back rule cleanly to snapshot v${targetVersion}`);
      await refreshAll();
    }
  };

  const handleRunDemoConsolidation = async () => {
    const res = await fetch(`${getBuilderApiUrl()}/api/memories/consolidate`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({
        background: false,
        session_id: "studio-sleep-pass",
        messages: [
          {
            role: "user",
            content:
              "We migrated our primary inference cluster to NVIDIA H200 Dynamo in Algiers. Always include p95 TTFT metrics in capacity plans.",
          },
          {
            role: "assistant",
            content:
              "Understood — recorded the NVIDIA H200 Dynamo cluster migration and standing p95 TTFT reporting requirement.",
          },
        ],
      }),
    });
    if (res.ok) {
      toast.success("Completed Lane-P1 Sleep-Time Consolidation pass");
      await refreshAll();
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-4">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
            <BrainCircuit className="text-primary h-6 w-6" />
            4-Tier Cognitive Memory &amp; Bi-Temporal Lineage Studio
          </h1>
          <p className="text-muted-foreground text-sm">
            KV-aligned Block 2 Core Memory, Bi-Temporal Semantic Fact Graph, Episodic Trajectories, Procedural Rollbacks &amp; Lane-P1 Sleep Consolidation
          </p>
        </div>

        <Button size="sm" variant="outline" onClick={() => void handleRunDemoConsolidation()}>
          <Sparkles className="mr-1.5 h-4 w-4" />
          Run Sleep-Time Consolidation Demo
        </Button>
      </div>

      <div className="flex-1 space-y-6 overflow-auto p-6">
        {summary && (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="bg-card rounded-xl border p-4 shadow-xs">
              <p className="text-muted-foreground text-xs font-medium uppercase">Tier 1 · Core Memory</p>
              <p className="mt-1 font-mono text-xl font-bold">
                {summary.pinned_core_tokens} / {summary.max_core_tokens} tok
              </p>
              <p className="text-muted-foreground text-xs">{summary.core_blocks_count} pinned Block-2 block(s)</p>
            </div>
            <div className="bg-card rounded-xl border p-4 shadow-xs">
              <p className="text-muted-foreground text-xs font-medium uppercase">Tier 2 · Bi-Temporal Facts</p>
              <p className="mt-1 font-mono text-xl font-bold">{summary.active_semantic_facts} active</p>
              <p className="text-muted-foreground text-xs">
                {summary.superseded_semantic_facts} superseded historical fact(s)
              </p>
            </div>
            <div className="bg-card rounded-xl border p-4 shadow-xs">
              <p className="text-muted-foreground text-xs font-medium uppercase">Tier 3 · Episodic Traces</p>
              <p className="mt-1 font-mono text-xl font-bold">{summary.episodic_memories_count}</p>
              <p className="text-muted-foreground text-xs">Distilled problem → outcome trajectories</p>
            </div>
            <div className="bg-card rounded-xl border p-4 shadow-xs">
              <p className="text-muted-foreground text-xs font-medium uppercase">Tier 4 · Procedural Rules</p>
              <p className="mt-1 font-mono text-xl font-bold">{summary.procedural_active_count} approved</p>
              <p className="text-muted-foreground text-xs">
                {summary.procedural_proposed_count} awaiting human approval
              </p>
            </div>
          </div>
        )}

        {/* Sub-35ms Hybrid Recall Debugger */}
        <div className="bg-card space-y-3 rounded-xl border p-4 shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider">
              Hybrid Qdrant (`is_tenant=True`) Dense + BM25 RRF Recall Debugger
            </span>
            {recallLatencyMs !== null && (
              <span className="rounded bg-emerald-500/15 px-2.5 py-0.5 font-mono text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                Recall Latency: {recallLatencyMs} ms
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" />
              <Input
                value={recallQuery}
                onChange={(e) => setRecallQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void handleHybridRecall()}
                placeholder="Test live <35ms hybrid memory recall (e.g., 'primary inference cluster H200')..."
                className="pl-9 text-sm"
              />
            </div>
            <Button size="sm" onClick={() => void handleHybridRecall()}>
              Recall
            </Button>
          </div>
          {recallHits.length > 0 && (
            <div className="grid gap-2 pt-1">
              {recallHits.map((hit, i) => (
                <div
                  key={String(hit.id || i)}
                  className="bg-muted/40 flex flex-wrap items-center justify-between gap-2 rounded-lg border px-3 py-2 text-xs"
                >
                  <div>
                    <span className="mr-2 rounded bg-primary/10 px-1.5 py-0.5 font-mono uppercase">
                      {String(hit.tier)}
                    </span>
                    <span className="font-medium">{String(hit.text)}</span>
                  </div>
                  <span className="text-muted-foreground font-mono text-[11px]">
                    score={String(hit.score)} · dense=#{String(hit.dense_rank ?? "—")} · bm25=#
                    {String(hit.lexical_rank ?? "—")}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Tier Selector Tabs */}
        <div className="flex flex-wrap gap-2 border-b pb-2">
          {(
            [
              ["facts", "Tier 2 · Bi-Temporal Facts"],
              ["blocks", "Tier 1 · Core Memory Blocks"],
              ["episodes", "Tier 3 · Episodic Trajectories"],
              ["procedural", "Tier 4 · Procedural Rules & Rollback"],
            ] as const
          ).map(([id, label]) => (
            <Button
              key={id}
              size="sm"
              variant={tab === id ? "default" : "ghost"}
              onClick={() => setTab(id)}
            >
              {label}
            </Button>
          ))}
        </div>

        {tab === "facts" && (
          <div className="space-y-4">
            <div className="bg-card grid grid-cols-1 gap-2 rounded-xl border p-4 md:grid-cols-4">
              <Input
                placeholder="Subject (e.g., workspace)"
                value={subj}
                onChange={(e) => setSubj(e.target.value)}
              />
              <Input
                placeholder="Predicate (e.g., primary inference cluster)"
                value={pred}
                onChange={(e) => setPred(e.target.value)}
              />
              <Input
                placeholder="Object (e.g., NVIDIA H200 Dynamo)"
                value={objVal}
                onChange={(e) => setObjVal(e.target.value)}
              />
              <Button onClick={() => void handleSaveFact()}>
                <Plus className="mr-1.5 h-4 w-4" />
                Upsert Bi-Temporal Fact
              </Button>
            </div>

            <div className="flex flex-wrap items-center gap-3 text-xs">
              <Button
                size="sm"
                variant={includeHistorical ? "secondary" : "outline"}
                onClick={() => setIncludeHistorical((v) => !v)}
              >
                <History className="mr-1.5 h-3.5 w-3.5" />
                {includeHistorical ? "Showing Superseded Lineage" : "Active Facts Only"}
              </Button>
              <div className="flex items-center gap-2">
                <Clock className="text-muted-foreground h-3.5 w-3.5" />
                <Input
                  value={asOfFilter}
                  onChange={(e) => setAsOfFilter(e.target.value)}
                  placeholder="Optional Point-in-Time ISO timestamp (`as_of`)..."
                  className="h-8 w-72 font-mono text-xs"
                />
              </div>
            </div>

            <div className="space-y-2">
              {facts.map((f) => {
                const isSuperseded = Boolean(f.invalidated_at);
                return (
                  <div
                    key={f.id}
                    className={`flex flex-wrap items-center justify-between gap-3 rounded-xl border p-3.5 text-xs ${
                      isSuperseded ? "bg-muted/30 opacity-65" : "bg-card"
                    }`}
                  >
                    <div className="space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={`rounded px-2 py-0.5 font-mono font-semibold uppercase ${
                            isSuperseded
                              ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                              : "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
                          }`}
                        >
                          {isSuperseded ? "Superseded" : "Active"}
                        </span>
                        <span className="font-mono font-semibold">{f.subject}</span>
                        <span className="text-muted-foreground">→ {f.predicate} →</span>
                        <span className="font-semibold">{f.object}</span>
                      </div>
                      <p className="text-muted-foreground font-mono text-[11px]">
                        id={f.id} · valid_from={f.valid_from}
                        {isSuperseded
                          ? ` · invalidated_at=${f.invalidated_at} · superseded_by=${f.superseded_by_id}`
                          : ""}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab === "blocks" && (
          <div className="space-y-4">
            <div className="bg-card space-y-3 rounded-xl border p-4">
              <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                <select
                  value={blockType}
                  onChange={(e) => setBlockType(e.target.value)}
                  className="bg-background rounded-md border px-3 py-2 text-xs"
                >
                  <option value="user_profile">user_profile</option>
                  <option value="workspace_context">workspace_context</option>
                  <option value="agent_notes">agent_notes</option>
                </select>
                <Input
                  value={blockTitle}
                  onChange={(e) => setBlockTitle(e.target.value)}
                  placeholder="Block Title"
                />
              </div>
              <Textarea
                rows={4}
                value={blockContent}
                onChange={(e) => setBlockContent(e.target.value)}
                placeholder="Markdown pinned into Block 2 (`session_prefix`, <= 1,500 tokens total)..."
              />
              <Button size="sm" onClick={() => void handleSaveBlock()}>
                Save Pinned Core Block
              </Button>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              {blocks.map((b) => (
                <div key={b.id} className="bg-card space-y-2 rounded-xl border p-4 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-semibold">
                      [{b.block_type}] {b.title} ({b.token_count} tok)
                    </span>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="text-destructive h-7 w-7"
                      onClick={async () => {
                        await fetch(`${getBuilderApiUrl()}/api/memories/blocks/${b.id}`, {
                          method: "DELETE",
                          headers: getTenantHeaders(),
                        });
                        await refreshAll();
                      }}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                  <pre className="bg-muted rounded-lg p-3 whitespace-pre-wrap font-mono text-[11px]">
                    {b.content}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "episodes" && (
          <div className="space-y-3">
            {episodes.map((ep) => (
              <div key={ep.id} className="bg-card space-y-1.5 rounded-xl border p-4 text-xs">
                <p className="font-semibold">{ep.summary}</p>
                <p className="text-muted-foreground">
                  <strong>Approach:</strong> {ep.approach} · <strong>Outcome:</strong> {ep.outcome}
                </p>
                {ep.user_corrections && (
                  <p className="text-amber-600 dark:text-amber-400">
                    <strong>User Standing Instruction:</strong> {ep.user_corrections}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}

        {tab === "procedural" && (
          <div className="space-y-4">
            <div className="bg-card space-y-3 rounded-xl border p-4">
              <Input
                value={ruleTitle}
                onChange={(e) => setRuleTitle(e.target.value)}
                placeholder="Procedural Rule Title (e.g., Include Citation IDs)"
              />
              <Textarea
                rows={3}
                value={rulePrompt}
                onChange={(e) => setRulePrompt(e.target.value)}
                placeholder="Behavioral instruction proposed for human review..."
              />
              <Button size="sm" onClick={() => void handleCreateProcedural()}>
                Propose Procedural Rule
              </Button>
            </div>

            <div className="space-y-3">
              {procedural.map((pm) => (
                <div key={pm.id} className="bg-card space-y-2.5 rounded-xl border p-4 text-xs">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-primary/10 px-2 py-0.5 font-mono font-semibold uppercase">
                        {pm.status} · v{pm.version}
                      </span>
                      <span className="text-sm font-semibold">{pm.title}</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {pm.status !== "active" && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => void handleProceduralStatus(pm.id, "active")}
                        >
                          <CheckCircle2 className="mr-1 h-3.5 w-3.5" />
                          Approve
                        </Button>
                      )}
                      {pm.status !== "rejected" && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => void handleProceduralStatus(pm.id, "rejected")}
                        >
                          Reject
                        </Button>
                      )}
                    </div>
                  </div>
                  <p className="font-mono text-[11px]">{pm.rule_prompt}</p>
                  {pm.history.length > 1 && (
                    <div className="border-t pt-2">
                      <p className="text-muted-foreground mb-1 font-mono text-[10px] uppercase">
                        1-Click Version Rollback Snapshots
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {pm.history.slice(0, -1).map((snap) => (
                          <Button
                            key={snap.version}
                            size="sm"
                            variant="ghost"
                            onClick={() => void handleRollback(pm.id, snap.version)}
                          >
                            <RotateCcw className="mr-1 h-3 w-3" />
                            Rollback → v{snap.version} ({snap.status})
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
