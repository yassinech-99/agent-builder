"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { getBuilderApiUrl } from "@/lib/runtime-urls";
import {
  CheckCircle2,
  ClipboardCopy,
  FlaskConical,
  Play,
  Plus,
  Swords,
  Trophy,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

interface EvalCase {
  id: string;
  name: string;
  category: string;
  input: string;
  expected_blocked: boolean;
  expected_substrings: string[];
  forbidden_substrings: string[];
  reference_answer: string;
}

interface EvalDataset {
  id: string;
  name: string;
  description: string;
  category: string;
  pass_threshold: number;
  cases: EvalCase[];
}

interface EvalRun {
  id: string;
  dataset_id: string;
  dataset_name: string;
  run_mode: string;
  model_a: string;
  model_b: string;
  pass_rate: number;
  avg_score_a: number;
  avg_score_b: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  total_tokens: number;
  estimated_usd: number;
  gate_passed: boolean;
  created_at: string;
  results: Array<{
    case_id?: string;
    name?: string;
    category?: string;
    passed?: boolean;
    score?: number;
    latency_ms?: number;
    output?: string;
    checks?: string[];
  }>;
}

interface ArenaBattleResult {
  id: string;
  winner: "model_a" | "model_b";
  speedup_ratio: number;
  arm_a: {
    model_name: string;
    output: string;
    ttft_ms: number;
    latency_ms: number;
    prompt_tokens: number;
    completion_tokens: number;
    estimated_usd: number;
    quality_score: number;
  };
  arm_b: {
    model_name: string;
    output: string;
    ttft_ms: number;
    latency_ms: number;
    prompt_tokens: number;
    completion_tokens: number;
    estimated_usd: number;
    quality_score: number;
  };
}

const NIM_CANDIDATES = [
  "nvidia/nemotron-3-super-120b-a12b",
  "nvidia/llama-3.3-nemotron-super-49b-v1.5",
  "nvidia/llama-3.1-nemotron-nano-8b-v1",
  "deepseek-ai/deepseek-r1",
  "qwen/qwen3-235b-a22b",
  "meta/llama-3.3-70b-instruct",
];

type Tab = "regression" | "arena" | "history";

export default function EvalsStudioPage() {
  const [tab, setTab] = useState<Tab>("regression");
  const [datasets, setDatasets] = useState<EvalDataset[]>([]);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [selectedModel, setSelectedModel] = useState(NIM_CANDIDATES[0]);
  const [runningDatasetId, setRunningDatasetId] = useState<string | null>(null);
  const [latestRun, setLatestRun] = useState<EvalRun | null>(null);
  const [ciMarkdown, setCiMarkdown] = useState<string>("");

  // Arena state
  const [arenaPrompt, setArenaPrompt] = useState(
    "Design a zero-downtime multi-tenant RAG & KV-cache routing strategy for 500 concurrent enterprise users on NVIDIA NIM.",
  );
  const [modelA, setModelA] = useState("nvidia/nemotron-3-super-120b-a12b");
  const [modelB, setModelB] = useState("nvidia/llama-3.1-nemotron-nano-8b-v1");
  const [arenaRunning, setArenaRunning] = useState(false);
  const [arenaResult, setArenaResult] = useState<ArenaBattleResult | null>(null);

  // Quick dataset creator state
  const [showCreate, setShowCreate] = useState(false);
  const [newSuiteName, setNewSuiteName] = useState("");
  const [newCasePrompt, setNewCasePrompt] = useState("");
  const [newCaseExpected, setNewCaseExpected] = useState("");

  const fetchAll = useCallback(async () => {
    const base = getBuilderApiUrl();
    try {
      const [dsRes, runsRes] = await Promise.all([
        fetch(`${base}/api/evals/datasets`),
        fetch(`${base}/api/evals/runs`),
      ]);
      if (dsRes.ok) setDatasets(await dsRes.json());
      if (runsRes.ok) {
        const loadedRuns: EvalRun[] = await runsRes.json();
        setRuns(loadedRuns);
        const firstReg = loadedRuns.find((r) => r.run_mode === "regression");
        if (firstReg) setLatestRun((prev) => prev || firstReg);
      }
    } catch (err) {
      console.error("Failed to fetch evaluation suites:", err);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const handleRunSuite = async (datasetId: string) => {
    setRunningDatasetId(datasetId);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/evals/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          model_a: selectedModel,
        }),
      });
      if (res.ok) {
        const run: EvalRun = await res.json();
        setLatestRun(run);
        const ciRes = await fetch(
          `${getBuilderApiUrl()}/api/evals/ci-report/${run.id}`,
        );
        if (ciRes.ok) {
          const ciData = await ciRes.json();
          setCiMarkdown(ciData.markdown || "");
        }
        toast.success(
          `Suite complete: ${(run.pass_rate * 100).toFixed(0)}% pass rate (${
            run.gate_passed ? "GATE PASSED" : "GATE FAILED"
          })`,
        );
        fetchAll();
      }
    } finally {
      setRunningDatasetId(null);
    }
  };

  const handleCreateSuite = async () => {
    if (!newSuiteName.trim() || !newCasePrompt.trim()) return;
    const expectedList = newCaseExpected
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const res = await fetch(`${getBuilderApiUrl()}/api/evals/datasets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: newSuiteName.trim(),
        description: "Custom regression suite authored in Evaluation Studio",
        pass_threshold: 0.85,
        cases: [
          {
            name: "Custom Golden Assertion #1",
            category: "custom",
            input: newCasePrompt.trim(),
            expected_blocked: false,
            expected_substrings: expectedList,
            forbidden_substrings: [],
            reference_answer: expectedList.join(" · "),
          },
        ],
      }),
    });
    if (res.ok) {
      toast.success("Evaluation dataset created");
      setNewSuiteName("");
      setNewCasePrompt("");
      setNewCaseExpected("");
      setShowCreate(false);
      fetchAll();
    }
  };

  const handleRunArena = async () => {
    if (!arenaPrompt.trim()) return;
    setArenaRunning(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/evals/arena`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: arenaPrompt,
          model_a: modelA,
          model_b: modelB,
        }),
      });
      if (res.ok) {
        setArenaResult(await res.json());
        fetchAll();
      } else {
        const err = await res.json().catch(() => ({}));
        toast.error(err.detail || "Arena run rejected");
      }
    } finally {
      setArenaRunning(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Evaluation Studio &amp; Model A/B Arena
          </h1>
          <p className="text-muted-foreground text-sm">
            Automated CI quality gates, NemoGuard red-team verification, and parallel NVIDIA NIM cost/latency A/B battles
          </p>
        </div>
      </div>

      <div className="flex gap-1 border-b px-6 pt-2">
        <button
          onClick={() => setTab("regression")}
          className={`flex items-center gap-1.5 rounded-t-md px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "regression"
              ? "border-primary text-primary border-b-2"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <FlaskConical className="h-4 w-4" />
          CI Regression Suites
        </button>
        <button
          onClick={() => setTab("arena")}
          className={`flex items-center gap-1.5 rounded-t-md px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "arena"
              ? "border-primary text-primary border-b-2"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Swords className="h-4 w-4" />
          NIM Model A/B Arena
        </button>
        <button
          onClick={() => setTab("history")}
          className={`flex items-center gap-1.5 rounded-t-md px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "history"
              ? "border-primary text-primary border-b-2"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Trophy className="h-4 w-4" />
          Run History ({runs.length})
        </button>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {tab === "regression" && (
          <div className="mx-auto max-w-6xl space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Label className="text-xs">Target Candidate Model:</Label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="border-input bg-background h-9 rounded-md border px-3 font-mono text-xs"
                >
                  {NIM_CANDIDATES.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <Button size="sm" onClick={() => setShowCreate((s) => !s)}>
                <Plus className="mr-1.5 h-3.5 w-3.5" />
                New Golden Dataset
              </Button>
            </div>

            {showCreate && (
              <div className="bg-card space-y-3 rounded-xl border p-5 shadow-xs">
                <h3 className="text-sm font-semibold">Create Custom Evaluation Suite</h3>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  <div>
                    <Label className="text-xs">Suite Name</Label>
                    <Input
                      value={newSuiteName}
                      onChange={(e) => setNewSuiteName(e.target.value)}
                      placeholder="Billing Support RAG & PII Suite"
                    />
                  </div>
                  <div>
                    <Label className="text-xs">
                      Expected Substrings (comma-separated)
                    </Label>
                    <Input
                      value={newCaseExpected}
                      onChange={(e) => setNewCaseExpected(e.target.value)}
                      placeholder="is_tenant, Row-Level Security"
                    />
                  </div>
                  <div className="md:col-span-2">
                    <Label className="text-xs">Test Case Prompt</Label>
                    <Textarea
                      rows={2}
                      value={newCasePrompt}
                      onChange={(e) => setNewCasePrompt(e.target.value)}
                      placeholder="Enter prompt to evaluate..."
                    />
                  </div>
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline" size="sm" onClick={() => setShowCreate(false)}>
                    Cancel
                  </Button>
                  <Button size="sm" onClick={handleCreateSuite}>
                    Save Dataset
                  </Button>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {datasets.map((ds) => (
                <div
                  key={ds.id}
                  className="bg-card flex flex-col justify-between rounded-xl border p-5 shadow-xs"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="font-semibold">{ds.name}</h3>
                      <span className="bg-muted rounded px-2 py-0.5 font-mono text-xs">
                        {ds.cases.length} cases · gate {(ds.pass_threshold * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-muted-foreground mt-1 text-xs">
                      {ds.description}
                    </p>
                  </div>
                  <div className="mt-4 flex items-center justify-end border-t pt-3">
                    <Button
                      size="sm"
                      onClick={() => handleRunSuite(ds.id)}
                      disabled={runningDatasetId === ds.id}
                    >
                      <Play className="mr-1.5 h-3.5 w-3.5" />
                      {runningDatasetId === ds.id
                        ? "Running Suite..."
                        : "Run CI Regression Gate"}
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            {latestRun && (
              <div className="bg-card space-y-4 rounded-xl border p-6 shadow-xs">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3">
                  <div className="flex items-center gap-2">
                    {latestRun.gate_passed ? (
                      <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-500" />
                    )}
                    <h3 className="font-semibold">
                      {latestRun.dataset_name} —{" "}
                      {(latestRun.pass_rate * 100).toFixed(0)}% Pass Rate
                    </h3>
                  </div>
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span>Model: {latestRun.model_a.split("/").pop()}</span>
                    <span>· p95: {latestRun.p95_latency_ms} ms</span>
                    {ciMarkdown && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          navigator.clipboard.writeText(ciMarkdown);
                          toast.success("Copied GitHub Actions CI Markdown report");
                        }}
                      >
                        <ClipboardCopy className="mr-1.5 h-3.5 w-3.5" />
                        Copy CI Report
                      </Button>
                    )}
                  </div>
                </div>

                <div className="space-y-2.5">
                  {latestRun.results.map((r, i) => (
                    <div
                      key={r.case_id || i}
                      className="bg-muted/40 rounded-lg border p-3.5 text-xs"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span
                            className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase ${
                              r.passed
                                ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                                : "bg-red-500/15 text-red-600 dark:text-red-400"
                            }`}
                          >
                            {r.passed ? "PASS" : "FAIL"}
                          </span>
                          <span className="font-semibold">{r.name}</span>
                          <span className="text-muted-foreground font-mono">
                            [{r.category}]
                          </span>
                        </div>
                        <span className="text-muted-foreground font-mono">
                          Score: {r.score} · {r.latency_ms} ms
                        </span>
                      </div>
                      <p className="text-muted-foreground mt-1.5 font-mono">
                        {r.output}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {tab === "arena" && (
          <div className="mx-auto max-w-6xl space-y-6">
            <div className="bg-card space-y-4 rounded-xl border p-5 shadow-xs">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="space-y-1.5">
                  <Label>Arm A (Flagship / Reasoning Model)</Label>
                  <select
                    value={modelA}
                    onChange={(e) => setModelA(e.target.value)}
                    className="border-input bg-background flex h-9 w-full rounded-md border px-3 font-mono text-xs"
                  >
                    {NIM_CANDIDATES.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-1.5">
                  <Label>Arm B (Fast / Cost-Optimized Model)</Label>
                  <select
                    value={modelB}
                    onChange={(e) => setModelB(e.target.value)}
                    className="border-input bg-background flex h-9 w-full rounded-md border px-3 font-mono text-xs"
                  >
                    {NIM_CANDIDATES.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="space-y-1.5">
                <Label>Head-to-Head Evaluation Prompt</Label>
                <Textarea
                  rows={3}
                  value={arenaPrompt}
                  onChange={(e) => setArenaPrompt(e.target.value)}
                />
              </div>

              <div className="flex justify-end">
                <Button onClick={handleRunArena} disabled={arenaRunning}>
                  <Swords className="mr-1.5 h-4 w-4" />
                  {arenaRunning
                    ? "Running Parallel NIM Battle..."
                    : "Run Side-by-Side A/B Battle"}
                </Button>
              </div>
            </div>

            {arenaResult && (
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                {(
                  [
                    ["model_a", arenaResult.arm_a],
                    ["model_b", arenaResult.arm_b],
                  ] as const
                ).map(([key, arm]) => {
                  const isWinner = arenaResult.winner === key;
                  return (
                    <div
                      key={key}
                      className={`bg-card space-y-3 rounded-xl border p-5 shadow-xs ${
                        isWinner ? "border-emerald-500 ring-1 ring-emerald-500/30" : ""
                      }`}
                    >
                      <div className="flex items-center justify-between border-b pb-2.5">
                        <span className="font-mono text-xs font-semibold">
                          {arm.model_name}
                        </span>
                        {isWinner && (
                          <span className="rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                            Quality Winner
                          </span>
                        )}
                      </div>
                      <div className="text-muted-foreground flex flex-wrap gap-3 font-mono text-xs">
                        <span>TTFT: {arm.ttft_ms} ms</span>
                        <span>Total: {arm.latency_ms} ms</span>
                        <span>Score: {arm.quality_score}</span>
                        <span>Cost: ${arm.estimated_usd.toFixed(6)}</span>
                      </div>
                      <pre className="bg-muted overflow-x-auto rounded-lg p-3.5 text-xs whitespace-pre-wrap">
                        {arm.output}
                      </pre>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {tab === "history" && (
          <div className="mx-auto max-w-5xl space-y-2.5">
            {runs.map((r) => (
              <div
                key={r.id}
                className="bg-card flex flex-wrap items-center justify-between gap-3 rounded-xl border p-4 text-xs"
              >
                <div>
                  <span className="font-semibold">{r.dataset_name}</span>
                  <span className="text-muted-foreground ml-2 font-mono">
                    ({r.run_mode} · {r.model_a.split("/").pop()}
                    {r.model_b ? ` vs ${r.model_b.split("/").pop()}` : ""})
                  </span>
                </div>
                <div className="flex items-center gap-3 font-mono">
                  <span>Pass: {(r.pass_rate * 100).toFixed(0)}%</span>
                  <span>p95: {r.p95_latency_ms} ms</span>
                  <span>${r.estimated_usd.toFixed(6)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
