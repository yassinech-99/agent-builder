"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  getBuilderApiUrl,
  getRuntimeApiUrl,
  getSavedBuilderOverride,
  getSavedRuntimeOverride,
  NIMAGENTS_STORAGE,
} from "@/lib/runtime-urls";
import { CheckCircle2, XCircle, Loader2, RotateCcw, Activity } from "lucide-react";
import { toast } from "sonner";

export default function SettingsPage() {
  const [runtimeUrl, setRuntimeUrl] = useState("");
  const [builderUrl, setBuilderUrl] = useState("");
  const [testing, setTesting] = useState(false);
  const [healthResult, setHealthResult] = useState<{
    runtimeOk: boolean;
    builderOk: boolean;
  } | null>(null);

  useEffect(() => {
    setRuntimeUrl(getSavedRuntimeOverride());
    setBuilderUrl(getSavedBuilderOverride());
  }, []);

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

    toast.success("nimAgents endpoint settings saved");
  };

  const handleReset = () => {
    localStorage.removeItem(NIMAGENTS_STORAGE.runtimeApiUrl);
    localStorage.removeItem(NIMAGENTS_STORAGE.builderApiUrl);
    setRuntimeUrl("");
    setBuilderUrl("");
    setHealthResult(null);
    toast.info("Reverted to built-in Next.js reverse-proxy defaults");
  };

  const handleTestConnectivity = async () => {
    handleSave();
    setTesting(true);
    try {
      const [rRes, bRes] = await Promise.all([
        fetch(`${getRuntimeApiUrl()}/info`, { cache: "no-store" }).catch(() => null),
        fetch(`${getBuilderApiUrl()}/health`, { cache: "no-store" }).catch(() => null),
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
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground text-sm">
          Configure nimAgents runtime routing and control-plane endpoints
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-xl space-y-6">
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
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-400"
                      : "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-400"
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
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-400"
                      : "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-400"
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
