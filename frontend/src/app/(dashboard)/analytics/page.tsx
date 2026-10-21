"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { getBuilderApiUrl } from "@/lib/runtime-urls";
import {
  Activity,
  Coins,
  Cpu,
  Gauge,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import { toast } from "sonner";

interface AnalyticsOverview {
  tenant: {
    tenant_id: string;
    workspace_id: string;
    plan_tier: string;
  };
  window_hours: number;
  kpis: {
    total_calls: number;
    total_tokens: number;
    prompt_tokens: number;
    completion_tokens: number;
    avg_tpm: number;
    peak_tpm: number;
    total_usd: number;
    daily_budget_usd: number;
    budget_utilization_pct: number;
    projected_monthly_usd: number;
    kv_cache_hit_pct: number;
    semantic_cache_hit_pct: number;
    semantic_cache_saved_usd: number;
    p50_ttft_ms: number;
    p95_ttft_ms: number;
    p99_ttft_ms: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    p50_queue_wait_ms: number;
    p95_queue_wait_ms: number;
  };
  safety: {
    blocked: number;
    redacted: number;
    warned: number;
    total: number;
  };
  latest_eval: {
    pass_rate: number;
    gate_passed: boolean;
    dataset_name: string;
    created_at: string;
  } | null;
  timeseries: Array<{
    bucket: string;
    tpm: number;
    tokens: number;
    kv_cached_tokens: number;
    usd: number;
    saved_usd: number;
    kv_hit_pct: number;
    p50_ttft_ms: number;
    p95_ttft_ms: number;
    p99_ttft_ms: number;
    queue_wait_ms: number;
  }>;
  by_model: Array<{
    model_name: string;
    short_name: string;
    provider: string;
    calls: number;
    tokens: number;
    usd: number;
    avg_latency_ms: number;
    cache_hits: number;
  }>;
  by_agent: Array<{
    agent_id: string;
    calls: number;
    tokens: number;
    tpm: number;
    usd: number;
    p95_ttft_ms: number;
  }>;
}

export default function AnalyticsPage() {
  const [hours, setHours] = useState(24);
  const [data, setData] = useState<AnalyticsOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [seeding, setSeeding] = useState(false);

  const loadAnalytics = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(
        `${getBuilderApiUrl()}/api/analytics/overview?hours=${hours}`,
      );
      if (res.ok) {
        setData(await res.json());
      }
    } catch (err) {
      console.error("Failed to fetch FinOps analytics:", err);
    } finally {
      setLoading(false);
    }
  }, [hours]);

  useEffect(() => {
    loadAnalytics();
  }, [loadAnalytics]);

  const handleSeedDemo = async () => {
    setSeeding(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/analytics/seed-demo`, {
        method: "POST",
      });
      if (res.ok) {
        const out = await res.json();
        toast.success(`Populated ${out.seeded_spans} multi-NIM telemetry spans`);
        loadAnalytics();
      }
    } finally {
      setSeeding(false);
    }
  };

  const k = data?.kpis;

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            FinOps Cost &amp; Latency Observatory
          </h1>
          <p className="text-muted-foreground text-sm">
            Real-time per-tenant &amp; per-agent TPM, NVIDIA Dynamo KV-cache reuse, Semantic Cache savings, and p50/p95/p99 TTFT
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {([6, 24, 168] as const).map((h) => (
            <Button
              key={h}
              size="sm"
              variant={hours === h ? "default" : "outline"}
              onClick={() => setHours(h)}
            >
              {h === 168 ? "7d" : `${h}h`}
            </Button>
          ))}
          <Button
            size="sm"
            variant="outline"
            onClick={handleSeedDemo}
            disabled={seeding}
          >
            <Sparkles className="mr-1.5 h-3.5 w-3.5" />
            {seeding ? "Seeding..." : "Seed Sample Traffic"}
          </Button>
          <Button size="sm" variant="outline" onClick={loadAnalytics}>
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
            Refresh
          </Button>
        </div>
      </div>

      <div className="flex-1 space-y-6 overflow-auto p-6">
        {loading && !data ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="bg-muted h-28 animate-pulse rounded-xl" />
            ))}
          </div>
        ) : k ? (
          <>
            {/* Top FinOps & Latency KPI Cards */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <div className="bg-card rounded-xl border p-5 shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground text-xs font-medium">
                    Tenant Window Spend
                  </span>
                  <Coins className="text-primary h-4 w-4" />
                </div>
                <div className="mt-2 text-2xl font-bold font-mono">
                  ${k.total_usd.toFixed(4)}
                </div>
                <p className="text-muted-foreground mt-1 text-xs">
                  {k.budget_utilization_pct}% of ${k.daily_budget_usd} cap ·{" "}
                  <span className="text-emerald-600 dark:text-emerald-400">
                    ${k.semantic_cache_saved_usd.toFixed(4)} saved
                  </span>
                </p>
              </div>

              <div className="bg-card rounded-xl border p-5 shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground text-xs font-medium">
                    Throughput &amp; Token Volume
                  </span>
                  <Zap className="text-primary h-4 w-4" />
                </div>
                <div className="mt-2 text-2xl font-bold font-mono">
                  {k.total_tokens.toLocaleString()} tok
                </div>
                <p className="text-muted-foreground mt-1 text-xs">
                  Peak: {k.peak_tpm.toLocaleString()} TPM · {k.total_calls} calls
                </p>
              </div>

              <div className="bg-card rounded-xl border p-5 shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground text-xs font-medium">
                    GPU KV-Cache &amp; L1/L2 Cache
                  </span>
                  <Cpu className="text-primary h-4 w-4" />
                </div>
                <div className="mt-2 text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
                  {k.kv_cache_hit_pct.toFixed(1)}% KV Hit
                </div>
                <p className="text-muted-foreground mt-1 text-xs">
                  L1/L2 Semantic Hit: {k.semantic_cache_hit_pct.toFixed(1)}%
                </p>
              </div>

              <div className="bg-card rounded-xl border p-5 shadow-xs">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground text-xs font-medium">
                    TTFT Percentiles &amp; Queue Wait
                  </span>
                  <Gauge className="text-primary h-4 w-4" />
                </div>
                <div className="mt-2 text-xl font-bold font-mono">
                  {k.p50_ttft_ms} / {k.p95_ttft_ms} / {k.p99_ttft_ms} ms
                </div>
                <p className="text-muted-foreground mt-1 text-xs">
                  p50 / p95 / p99 TTFT · Queue p95: {k.p95_queue_wait_ms} ms
                </p>
              </div>
            </div>

            {data.timeseries.length === 0 ? (
              <div className="bg-card flex flex-col items-center justify-center rounded-xl border py-16 text-center">
                <Activity className="text-muted-foreground mb-3 h-12 w-12" />
                <h2 className="text-base font-semibold">
                  No usage spans recorded in this window yet
                </h2>
                <p className="text-muted-foreground mt-1 mb-4 max-w-md text-xs">
                  Chat with any agent, run an Evaluation Suite, or click Seed Sample Traffic to populate multi-model charts immediately.
                </p>
                <Button size="sm" onClick={handleSeedDemo} disabled={seeding}>
                  <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                  Seed Sample NVIDIA NIM Traffic
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                {/* Chart 1: TPM & Cached Token Volume */}
                <div className="bg-card rounded-xl border p-5 shadow-xs">
                  <h3 className="mb-1 text-sm font-semibold">
                    Token Volume vs Prefix KV-Cached Tokens
                  </h3>
                  <p className="text-muted-foreground mb-4 text-xs">
                    Byte-stable Block 1 prefixes keep most prompt tokens served directly from GPU HBM KV cache
                  </p>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={data.timeseries}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                        <XAxis dataKey="bucket" fontSize={11} />
                        <YAxis fontSize={11} />
                        <Tooltip />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="tokens"
                          name="Total Tokens"
                          stroke="#10b981"
                          fill="#10b981"
                          fillOpacity={0.2}
                        />
                        <Area
                          type="monotone"
                          dataKey="kv_cached_tokens"
                          name="KV-Cached Tokens"
                          stroke="#3b82f6"
                          fill="#3b82f6"
                          fillOpacity={0.28}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Chart 2: p50 / p95 / p99 TTFT & Queue Wait */}
                <div className="bg-card rounded-xl border p-5 shadow-xs">
                  <h3 className="mb-1 text-sm font-semibold">
                    TTFT Percentiles (p50 / p95 / p99) &amp; Lane Queue Wait (ms)
                  </h3>
                  <p className="text-muted-foreground mb-4 text-xs">
                    Interactive P0 chat lane head-of-line queue wait vs Time-To-First-Token
                  </p>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data.timeseries}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                        <XAxis dataKey="bucket" fontSize={11} />
                        <YAxis fontSize={11} />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="p50_ttft_ms"
                          name="p50 TTFT (ms)"
                          stroke="#10b981"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="p95_ttft_ms"
                          name="p95 TTFT (ms)"
                          stroke="#f59e0b"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="p99_ttft_ms"
                          name="p99 TTFT (ms)"
                          stroke="#ef4444"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="queue_wait_ms"
                          name="p95 Queue Wait (ms)"
                          stroke="#8b5cf6"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Chart 3: Per-Model Cost & Token Share */}
                <div className="bg-card rounded-xl border p-5 shadow-xs">
                  <h3 className="mb-1 text-sm font-semibold">
                    Per-Model Token &amp; Latency Distribution
                  </h3>
                  <p className="text-muted-foreground mb-4 text-xs">
                    Cascading utility work to Nemotron-Nano-8B preserves flagship capacity
                  </p>
                  <div className="h-60 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.by_model}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                        <XAxis dataKey="short_name" fontSize={10} />
                        <YAxis fontSize={11} />
                        <Tooltip />
                        <Legend />
                        <Bar
                          dataKey="tokens"
                          name="Tokens"
                          fill="#10b981"
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Per-Agent Cost & Safety Governance Table */}
                <div className="bg-card flex flex-col justify-between rounded-xl border p-5 shadow-xs">
                  <div>
                    <h3 className="mb-1 text-sm font-semibold">
                      Per-Agent FinOps Ledger &amp; Governance Posture
                    </h3>
                    <p className="text-muted-foreground mb-3 text-xs">
                      Multi-tenant cost attribution strictly isolated under PostgreSQL RLS
                    </p>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="text-muted-foreground border-b">
                          <tr>
                            <th className="py-2 font-medium">Agent</th>
                            <th className="py-2 font-medium">Calls</th>
                            <th className="py-2 font-medium">TPM</th>
                            <th className="py-2 font-medium">p95 TTFT</th>
                            <th className="py-2 text-right font-medium">USD</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y font-mono">
                          {data.by_agent.map((ag) => (
                            <tr key={ag.agent_id}>
                              <td className="py-2 font-sans font-medium">
                                {ag.agent_id}
                              </td>
                              <td className="py-2">{ag.calls}</td>
                              <td className="py-2">{ag.tpm}</td>
                              <td className="py-2">{ag.p95_ttft_ms} ms</td>
                              <td className="py-2 text-right">
                                ${ag.usd.toFixed(5)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="bg-muted/40 mt-4 flex flex-wrap items-center justify-between gap-2 rounded-lg border p-3 text-xs">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="h-4 w-4 text-emerald-500" />
                      <span>
                        NemoGuard Shield: <strong>{data.safety.blocked}</strong> blocked ·{" "}
                        <strong>{data.safety.redacted}</strong> redacted
                      </span>
                    </div>
                    {data.latest_eval && (
                      <span className="font-mono">
                        CI Eval Gate:{" "}
                        <strong
                          className={
                            data.latest_eval.gate_passed
                              ? "text-emerald-600 dark:text-emerald-400"
                              : "text-red-500"
                          }
                        >
                          {(data.latest_eval.pass_rate * 100).toFixed(0)}%
                        </strong>
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
