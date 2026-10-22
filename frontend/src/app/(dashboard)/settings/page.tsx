"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  getActiveTenantScope,
  getBuilderApiUrl,
  getRuntimeApiUrl,
  getSavedBuilderOverride,
  getSavedRuntimeOverride,
  getTenantHeaders,
  NIMAGENTS_STORAGE,
} from "@/lib/runtime-urls";
import {
  CheckCircle2,
  XCircle,
  Loader2,
  RotateCcw,
  Activity,
  ShieldCheck,
  Cpu,
  Layers,
  RefreshCw,
} from "lucide-react";
import { toast } from "sonner";

interface ScaleStatus {
  tenant: {
    tenant_id: string;
    workspace_id: string;
    plan_tier: string;
    storage_prefix: string;
  };
  database: {
    engine: string;
    pool_budget: {
      total_pool_connections: number;
      safe_ceiling: number;
      within_budget: boolean;
    };
  };
  event_loop: {
    p50_lag_ms: number;
    p95_lag_ms: number;
    max_lag_ms: number;
    stall_events: number;
  };
  admission: {
    backend: string;
    policy: {
      max_concurrent_runs: number;
      max_rpm: number;
      max_tpm: number;
      max_daily_usd: number;
    };
    usage: {
      active_concurrent: number;
      rpm_used: number;
      tpm_used: number;
      usd_spent: number;
    };
    total_admitted: number;
    total_rejected: number;
  };
  workers: {
    p0_interactive: { workers: number; active: number; completed: number; p95_wait_ms: number };
    p1_background: { workers: number; active: number; completed: number; p95_wait_ms: number };
  };
  vector_store: {
    mode: string;
    collections: string[];
    total_upserts: number;
    total_queries: number;
    quantization: string;
  };
  inference_gateway: {
    prefix_router: { kv_affinity_rate: number; affinity_hits: number; spillovers: number };
    semantic_cache: { l1_hits: number; l2_semantic_hits: number };
    gpu_micro_batcher: {
      window_ms: number;
      logical_requests: number;
      gpu_batches_sent: number;
      avg_coalescing_ratio: number;
      l1_embedding_cache_hits: number;
    };
  };
}

export default function SettingsPage() {
  const [runtimeUrl, setRuntimeUrl] = useState("");
  const [builderUrl, setBuilderUrl] = useState("");
  const [tenantId, setTenantId] = useState("default");
  const [workspaceId, setWorkspaceId] = useState("default");
  const [planTier, setPlanTier] = useState("pro");
  const [testing, setTesting] = useState(false);
  const [loadingScale, setLoadingScale] = useState(false);
  const [scaleStatus, setScaleStatus] = useState<ScaleStatus | null>(null);
  const [healthResult, setHealthResult] = useState<{
    runtimeOk: boolean;
    builderOk: boolean;
  } | null>(null);

  const refreshScaleTelemetry = useCallback(async () => {
    setLoadingScale(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/scale/status`, {
        headers: getTenantHeaders(),
        cache: "no-store",
      });
      if (res.ok) {
        const data = (await res.json()) as ScaleStatus;
        setScaleStatus(data);
      }
    } catch {
      // Non-fatal in offline build/preview
    } finally {
      setLoadingScale(false);
    }
  }, []);

  useEffect(() => {
    setRuntimeUrl(getSavedRuntimeOverride());
    setBuilderUrl(getSavedBuilderOverride());
    const scope = getActiveTenantScope();
    setTenantId(scope.tenantId);
    setWorkspaceId(scope.workspaceId);
    setPlanTier(scope.planTier);
    void refreshScaleTelemetry();
  }, [refreshScaleTelemetry]);

  const handleSave = () => {
    if (runtimeUrl.trim()) {
      localStorage.setItem(NIMAGENTS_STORAGE.runtimeApiUrl, runtimeUrl.trim());
    } else {
      localStorage.removeItem(NIMAGENTS_STORAGE.runtimeApiUrl);
    }

    if (builderUrl.trim()) {
      localStorage.setItem(NIMAGENTS_STORAGE.builderApiUrl, builderUrl.trim());
    } else {
      localStorage.removeItem(NIMAGENTS_STORAGE.builderApiUrl);
    }

    localStorage.setItem(NIMAGENTS_STORAGE.tenantId, tenantId.trim() || "default");
    localStorage.setItem(NIMAGENTS_STORAGE.workspaceId, workspaceId.trim() || "default");
    localStorage.setItem(NIMAGENTS_STORAGE.planTier, planTier.trim() || "pro");

    toast.success("nimAgents tenant & endpoint settings saved");
    void refreshScaleTelemetry();
  };

  const handleReset = () => {
    localStorage.removeItem(NIMAGENTS_STORAGE.runtimeApiUrl);
    localStorage.removeItem(NIMAGENTS_STORAGE.builderApiUrl);
    localStorage.removeItem(NIMAGENTS_STORAGE.tenantId);
    localStorage.removeItem(NIMAGENTS_STORAGE.workspaceId);
    localStorage.removeItem(NIMAGENTS_STORAGE.planTier);
    setRuntimeUrl("");
    setBuilderUrl("");
    setTenantId("default");
    setWorkspaceId("default");
    setPlanTier("pro");
    setHealthResult(null);
    toast.info("Reverted to built-in multi-tenant & proxy defaults");
    void refreshScaleTelemetry();
  };

  const handleTestConnectivity = async () => {
    handleSave();
    setTesting(true);
    try {
      const [rRes, bRes] = await Promise.all([
        fetch(`${getRuntimeApiUrl()}/info`, {
          headers: getTenantHeaders(),
          cache: "no-store",
        }).catch(() => null),
        fetch(`${getBuilderApiUrl()}/health`, {
          headers: getTenantHeaders(),
          cache: "no-store",
        }).catch(() => null),
      ]);
      setHealthResult({
        runtimeOk: !!rRes?.ok,
        builderOk: !!bRes?.ok,
      });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b px-6 py-4">
        <h1 className="text-2xl font-bold tracking-tight">Settings &amp; Multi-Tenant Scale</h1>
        <p className="text-muted-foreground text-sm">
          Configure tenant workspace isolation, runtime routing, quotas, and GPU inference telemetry
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl space-y-6">
          {/* Active Tenant & Workspace Isolation Card */}
          <div className="bg-card rounded-xl border p-6 shadow-sm">
            <div className="mb-4 flex items-center gap-2">
              <ShieldCheck className="h-5 w-5 text-emerald-500" />
              <div>
                <h2 className="text-lg font-semibold">Tenant &amp; Workspace Isolation Scope</h2>
                <p className="text-muted-foreground text-xs">
                  Propagates <code>X-Tenant-ID</code> &amp; <code>X-Workspace-ID</code> across Postgres RLS, Qdrant{" "}
                  <code>is_tenant</code> shards, Meilisearch filters, and RustFS prefixes.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="flex flex-col gap-1.5">
                <Label>Tenant ID</Label>
                <Input
                  value={tenantId}
                  onChange={(e) => setTenantId(e.target.value)}
                  placeholder="default"
                  className="font-mono text-xs"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Workspace ID</Label>
                <Input
                  value={workspaceId}
                  onChange={(e) => setWorkspaceId(e.target.value)}
                  placeholder="default"
                  className="font-mono text-xs"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Quota Plan Tier</Label>
                <select
                  value={planTier}
                  onChange={(e) => setPlanTier(e.target.value)}
                  className="border-input bg-background flex h-9 w-full rounded-md border px-3 py-1 text-xs shadow-sm"
                >
                  <option value="free">Free (3 conc · 30 RPM · 40K TPM)</option>
                  <option value="pro">Pro (12 conc · 240 RPM · 400K TPM)</option>
                  <option value="enterprise">Enterprise (64 conc · 1.2K RPM · 2.5M TPM)</option>
                </select>
              </div>
            </div>
            <p className="text-muted-foreground mt-3 text-xs">
              Enforced Object Prefix:{" "}
              <code>
                tenants/{tenantId || "default"}/workspaces/{workspaceId || "default"}/
              </code>
            </p>
          </div>

          {/* Live Scale & AI Inference Gateway Observatory */}
          <div className="bg-card rounded-xl border p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="h-5 w-5 text-emerald-500" />
                <div>
                  <h2 className="text-lg font-semibold">
                    Async Data Plane &amp; GPU Inference Observatory
                  </h2>
                  <p className="text-muted-foreground text-xs">
                    Live telemetry from <code>/api/scale/status</code> — Event-loop jitter, admission quotas, P0/P1 fair queues &amp; NIM KV routing
                  </p>
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void refreshScaleTelemetry()}
                disabled={loadingScale}
              >
                <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${loadingScale ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>

            {scaleStatus ? (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <div className="rounded-lg border p-3 text-xs">
                  <div className="text-muted-foreground font-medium">Event Loop p95 Lag</div>
                  <div className="mt-1 font-mono text-base font-bold">
                    {scaleStatus.event_loop.p95_lag_ms} ms
                  </div>
                  <div className="text-muted-foreground mt-1">
                    Stalls: {scaleStatus.event_loop.stall_events} · DB: {scaleStatus.database.engine}
                  </div>
                </div>

                <div className="rounded-lg border p-3 text-xs">
                  <div className="text-muted-foreground font-medium">Admission Gate ({scaleStatus.admission.backend})</div>
                  <div className="mt-1 font-mono text-base font-bold">
                    {scaleStatus.admission.usage.rpm_used}/{scaleStatus.admission.policy.max_rpm} RPM
                  </div>
                  <div className="text-muted-foreground mt-1">
                    Admitted: {scaleStatus.admission.total_admitted} · 429 Shed: {scaleStatus.admission.total_rejected}
                  </div>
                </div>

                <div className="rounded-lg border p-3 text-xs">
                  <div className="text-muted-foreground font-medium">Two-Lane Fair Pool</div>
                  <div className="mt-1 font-mono text-base font-bold">
                    P0: {scaleStatus.workers.p0_interactive.completed} · P1: {scaleStatus.workers.p1_background.completed}
                  </div>
                  <div className="text-muted-foreground mt-1">
                    Qdrant ({scaleStatus.vector_store.quantization}): {scaleStatus.vector_store.total_upserts} chunks
                  </div>
                </div>

                <div className="rounded-lg border p-3 text-xs">
                  <div className="text-muted-foreground font-medium">GPU Micro-Batcher &amp; KV Router</div>
                  <div className="mt-1 font-mono text-base font-bold">
                    {Math.round(scaleStatus.inference_gateway.prefix_router.kv_affinity_rate * 100)}% KV Affinity
                  </div>
                  <div className="text-muted-foreground mt-1">
                    Coalescing: {scaleStatus.inference_gateway.gpu_micro_batcher.avg_coalescing_ratio}x · L2 Hits:{" "}
                    {scaleStatus.inference_gateway.semantic_cache.l2_semantic_hits}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-muted-foreground flex items-center gap-2 rounded-lg border border-dashed p-4 text-xs">
                <Layers className="h-4 w-4" />
                Connect to Control Plane API to view real-time multi-tenant &amp; GPU telemetry.
              </div>
            )}
          </div>

          {/* Service Endpoints Card */}
          <div className="bg-card rounded-xl border p-6 shadow-sm">
            <h2 className="mb-1 text-lg font-semibold">Service Endpoints</h2>
            <p className="text-muted-foreground mb-5 text-xs">
              Leave blank to use same-origin Next.js reverse-proxy routing
              (<code>/api</code> &amp; <code>/api/builder</code>) automatically.
            </p>
            <div className="space-y-4">
              <div className="flex flex-col gap-1.5">
                <Label>Agent Graph Runtime URL Override</Label>
                <Input
                  value={runtimeUrl}
                  onChange={(e) => setRuntimeUrl(e.target.value)}
                  placeholder="Default: same-origin /api proxy (-> :2024)"
                  className="font-mono text-xs"
                />
                <p className="text-muted-foreground text-xs">
                  Active URL: <code>{getRuntimeApiUrl()}</code>
                </p>
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Control Plane API URL Override</Label>
                <Input
                  value={builderUrl}
                  onChange={(e) => setBuilderUrl(e.target.value)}
                  placeholder="Default: same-origin /api/builder proxy (-> :8100)"
                  className="font-mono text-xs"
                />
                <p className="text-muted-foreground text-xs">
                  Active URL: <code>{getBuilderApiUrl()}</code>
                </p>
              </div>
            </div>

            {healthResult && (
              <div className="mt-5 grid grid-cols-1 gap-2 sm:grid-cols-2">
                <div
                  className={`flex items-center gap-2 rounded-md border px-3 py-2 text-xs ${
                    healthResult.runtimeOk
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/30 dark:bg-emerald-400"
                      : "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:bg-red-400"
                  }`}
                >
                  {healthResult.runtimeOk ? (
                    <CheckCircle2 className="h-4 w-4 shrink-0" />
                  ) : (
                    <XCircle className="h-4 w-4 shrink-0" />
                  )}
                  Graph Runtime: {healthResult.runtimeOk ? "Healthy" : "Unreachable"}
                </div>
                <div
                  className={`flex items-center gap-2 rounded-md border px-3 py-2 text-xs ${
                    healthResult.builderOk
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/30 dark:bg-emerald-400"
                      : "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:bg-red-400"
                  }`}
                >
                  {healthResult.builderOk ? (
                    <CheckCircle2 className="h-4 w-4 shrink-0" />
                  ) : (
                    <XCircle className="h-4 w-4 shrink-0" />
                  )}
                  Control Plane API: {healthResult.builderOk ? "Healthy" : "Unreachable"}
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-wrap justify-between gap-2">
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleReset}>
                <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
                Reset Defaults
              </Button>
              <Button
                variant="outline"
                onClick={handleTestConnectivity}
                disabled={testing}
              >
                {testing ? (
                  <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Activity className="mr-1.5 h-3.5 w-3.5" />
                )}
                Test Connectivity
              </Button>
            </div>
            <Button onClick={handleSave}>Save Settings</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
