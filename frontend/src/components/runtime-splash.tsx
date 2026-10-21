"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { NimAgentsLogo } from "@/components/icons/nim-agents";
import {
  getBuilderApiUrl,
  getRuntimeApiUrl,
  NIMAGENTS_STORAGE,
} from "@/lib/runtime-urls";

type StepStatus = "pending" | "running" | "done" | "warn" | "error";

interface Step {
  label: string;
  status: StepStatus;
  detail?: string;
}

async function checkEndpoint(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

function StatusIcon({ status }: { status: StepStatus }) {
  switch (status) {
    case "done":
      return (
        <svg className="h-5 w-5 text-emerald-500" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
            clipRule="evenodd"
          />
        </svg>
      );
    case "running":
      return (
        <svg className="text-primary h-5 w-5 animate-spin" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      );
    case "warn":
      return (
        <svg className="h-5 w-5 text-amber-500" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        </svg>
      );
    case "error":
      return (
        <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
      );
    default:
      return <div className="bg-muted h-5 w-5 rounded-full" />;
  }
}

export function RuntimeSplash() {
  const router = useRouter();
  const [steps, setSteps] = useState<Step[]>([
    { label: "Connecting to nimAgents graph runtime", status: "pending" },
    { label: "Checking Control Plane API", status: "pending" },
    { label: "Loading tool & NIM model registry", status: "pending" },
    { label: "Preparing studio workspace", status: "pending" },
  ]);
  const [progress, setProgress] = useState(0);
  const [ready, setReady] = useState(false);
  const [hasError, setHasError] = useState(false);
  const retries = useRef(0);
  const maxRetries = 12;

  const updateStep = useCallback(
    (idx: number, update: Partial<Step>) =>
      setSteps((prev) =>
        prev.map((s, i) => (i === idx ? { ...s, ...update } : s)),
      ),
    [],
  );

  const runHealthChecks = useCallback(async () => {
    const runtimeUrl = getRuntimeApiUrl();
    const builderUrl = getBuilderApiUrl();

    setSteps([
      { label: "Connecting to nimAgents graph runtime", status: "pending" },
      { label: "Checking Control Plane API", status: "pending" },
      { label: "Loading tool & NIM model registry", status: "pending" },
      { label: "Preparing studio workspace", status: "pending" },
    ]);
    setProgress(0);
    setReady(false);
    setHasError(false);
    retries.current = 0;

    updateStep(0, { status: "running" });
    let runtimeOk = false;
    while (!runtimeOk && retries.current < maxRetries) {
      runtimeOk = await checkEndpoint(`${runtimeUrl}/info`);
      if (!runtimeOk) {
        retries.current += 1;
        setProgress(Math.min(25, Math.round((retries.current / maxRetries) * 25)));
        updateStep(0, {
          detail: `Waiting for runtime… attempt ${retries.current}/${maxRetries}`,
        });
        await new Promise((r) => setTimeout(r, 1000));
      }
    }

    if (!runtimeOk) {
      updateStep(0, {
        status: "error",
        detail: `Graph runtime at ${runtimeUrl} did not respond`,
      });
      setHasError(true);
      return;
    }
    updateStep(0, { status: "done", detail: "Graph runtime online" });
    setProgress(35);

    updateStep(1, { status: "running" });
    const builderOk = await checkEndpoint(`${builderUrl}/health`);
    if (builderOk) {
      updateStep(1, { status: "done", detail: "Control Plane API online" });
    } else {
      updateStep(1, { status: "warn", detail: "Control Plane API offline" });
    }
    setProgress(65);

    updateStep(2, { status: "running" });
    let toolCount = 0;
    try {
      const res = await fetch(`${builderUrl}/api/tools/catalog`);
      if (res.ok) {
        const data = await res.json();
        toolCount = Array.isArray(data) ? data.length : 0;
      }
    } catch {
      // non-critical
    }
    if (toolCount > 0) {
      updateStep(2, { status: "done", detail: `${toolCount} tools & NIM presets ready` });
    } else {
      updateStep(2, { status: "warn", detail: "Catalog unavailable" });
    }
    setProgress(85);

    updateStep(3, { status: "running" });
    await new Promise((r) => setTimeout(r, 250));
    updateStep(3, { status: "done", detail: "Workspace ready" });
    setProgress(100);
    setReady(true);
  }, [updateStep]);

  useEffect(() => {
    const alreadyLoaded = sessionStorage.getItem(NIMAGENTS_STORAGE.loaded);
    if (alreadyLoaded === "true") {
      router.replace("/chat");
      return;
    }
    runHealthChecks();
  }, [runHealthChecks, router]);

  const handleEnter = () => {
    sessionStorage.setItem(NIMAGENTS_STORAGE.loaded, "true");
    router.push("/chat");
  };

  const handleLogout = () => {
    sessionStorage.removeItem(NIMAGENTS_STORAGE.auth);
    sessionStorage.removeItem(NIMAGENTS_STORAGE.loaded);
    router.push("/");
  };

  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-gradient-to-br from-zinc-50 via-white to-emerald-50/40 dark:from-zinc-950 dark:via-zinc-950 dark:to-emerald-950/20">
      <div className="flex w-full max-w-lg flex-col items-center gap-8 p-8">
        <div className="flex w-full justify-end">
          <button
            type="button"
            onClick={handleLogout}
            className="text-muted-foreground hover:text-foreground rounded-md border px-3 py-1.5 text-xs transition-colors"
          >
            Logout
          </button>
        </div>
        <div className="flex flex-col items-center gap-3">
          <div className={ready ? "" : "animate-pulse"}>
            <NimAgentsLogo width={56} height={56} className="text-primary" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">nimAgents</h1>
          <p className="text-muted-foreground text-sm">
            NVIDIA NIM &amp; Multi-Provider Agent Runtime
          </p>
        </div>

        <div className="w-full">
          <div className="bg-muted h-1.5 w-full overflow-hidden rounded-full">
            <div
              className="bg-primary h-full rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div className="w-full space-y-3">
          {steps.map((step, i) => (
            <div
              key={i}
              className={`flex items-center gap-3 rounded-lg border px-4 py-3 transition-all duration-300 ${
                step.status === "done"
                  ? "border-emerald-200 bg-emerald-50/50 dark:border-emerald-900 dark:bg-emerald-950/20"
                  : step.status === "error"
                    ? "border-red-200 bg-red-50/50 dark:border-red-900 dark:bg-red-950/20"
                    : step.status === "warn"
                      ? "border-amber-200 bg-amber-50/50 dark:border-amber-900 dark:bg-amber-950/20"
                      : step.status === "running"
                        ? "border-primary/40 bg-primary/5"
                        : "border-border bg-card"
              }`}
            >
              <StatusIcon status={step.status} />
              <div className="flex flex-1 flex-col">
                <span className="text-sm font-medium">{step.label}</span>
                {step.detail && (
                  <span className="text-muted-foreground text-xs">
                    {step.detail}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

        {ready && (
          <button
            onClick={handleEnter}
            className="bg-primary text-primary-foreground hover:bg-primary/90 w-full rounded-lg px-8 py-3 text-sm font-medium shadow-md transition-all"
          >
            Enter nimAgents Studio
          </button>
        )}

        {hasError && (
          <div className="flex w-full gap-3">
            <button
              onClick={() => {
                retries.current = 0;
                runHealthChecks();
              }}
              className="bg-primary text-primary-foreground hover:bg-primary/90 flex-1 rounded-lg px-6 py-3 text-sm font-medium shadow transition-all"
            >
              Retry Checks
            </button>
            <button
              onClick={handleEnter}
              className="border-border text-foreground hover:bg-muted flex-1 rounded-lg border px-6 py-3 text-sm font-medium transition-all"
            >
              Open Studio Anyway
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
