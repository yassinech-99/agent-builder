"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Message } from "@langchain/langgraph-sdk";
import { getBuilderApiUrl, getTenantHeaders } from "@/lib/runtime-urls";
import { Button } from "@/components/ui/button";
import { Layers, Sparkles, Minimize2, Loader2, RefreshCw } from "lucide-react";
import { toast } from "sonner";

interface ThreadContextTelemetry {
  thread_id: string;
  static_prefix_tokens: number;
  session_prefix_tokens: number;
  dynamic_tail_tokens: number;
  total_tokens: number;
  context_window: number;
  utilization_pct: number;
  kv_cache_hit_pct: number;
  deferred_tools_count: number;
  active_skills: string[];
  offloaded_artifacts_count: number;
  cleared_thinking_tokens: number;
  cleared_tool_tokens: number;
  compaction_count: number;
  pre_compact_facts: string[];
}

function formatK(tokens: number): string {
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}K`;
  return `${tokens}`;
}

export function ContextMeter({
  threadId,
  messages,
}: {
  threadId: string | null;
  messages: Message[];
}) {
  const [data, setData] = useState<ThreadContextTelemetry | null>(null);
  const [open, setOpen] = useState(false);
  const [compacting, setCompacting] = useState(false);

  const refresh = useCallback(async () => {
    const targetId = threadId || "preview";
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/context/threads/${targetId}`, {
        headers: getTenantHeaders(),
        cache: "no-store",
      });
      if (res.ok) {
        setData((await res.json()) as ThreadContextTelemetry);
      }
    } catch {
      /* non-fatal if control plane is restarting */
    }
  }, [threadId]);

  useEffect(() => {
    void refresh();
  }, [refresh, messages.length]);

  const handleManualCompact = async () => {
    if (!threadId) return;
    setCompacting(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/context/threads/${threadId}/compact`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getTenantHeaders() },
        body: JSON.stringify({
          messages: messages.map((m) => ({
            type: m.type,
            content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
          })),
        }),
      });
      if (res.ok) {
        const resData = await res.json();
        toast.success(
          `Compacted session context to ${formatK(resData.total_tokens)} tokens (${resData.pre_compact_facts?.length ?? 0} facts preserved)`,
        );
        await refresh();
      }
    } catch {
      toast.error("Manual compaction failed");
    } finally {
      setCompacting(false);
    }
  };

  if (!data) return null;

  const barColor =
    data.utilization_pct >= 80
      ? "bg-rose-500"
      : data.utilization_pct >= 60
        ? "bg-amber-500"
        : "bg-emerald-500";

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="hover:bg-muted/80 bg-muted/40 flex items-center gap-2 rounded-full border px-3 py-1 text-xs transition-colors"
        title="Inspect 3-Block Context Projection & GPU KV-Cache Alignment"
      >
        <Layers className="text-primary h-3.5 w-3.5 shrink-0" />
        <span className="font-mono">
          {formatK(data.total_tokens)} / {formatK(data.context_window)}
        </span>
        <div className="bg-muted hidden h-1.5 w-14 overflow-hidden rounded-full sm:block">
          <div
            className={`h-full transition-all ${barColor}`}
            style={{ width: `${Math.min(100, Math.max(3, data.utilization_pct))}%` }}
          />
        </div>
        <span className="hidden items-center gap-0.5 font-mono text-[11px] text-emerald-600 md:inline-flex dark:text-emerald-400">
          <Sparkles className="h-3 w-3" />
          KV {data.kv_cache_hit_pct}%
        </span>
        <span className="text-muted-foreground hidden font-mono text-[11px] xl:inline">
          · {data.deferred_tools_count} Deferred
        </span>
      </button>

      {open && (
        <div className="bg-popover text-popover-foreground absolute right-0 top-10 z-50 w-80 space-y-3 rounded-xl border p-4 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold">5-Stage Context &amp; KV-Cache Meter</span>
            <button
              type="button"
              onClick={() => void refresh()}
              className="text-muted-foreground hover:text-foreground"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="space-y-1.5 rounded-lg border p-2.5 font-mono text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Block 1 · Static Prefix:</span>
              <span>{formatK(data.static_prefix_tokens)} tok</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Block 2 · Skills &amp; Memory:</span>
              <span>{formatK(data.session_prefix_tokens)} tok</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Block 3 · Verbatim Tail:</span>
              <span>{formatK(data.dynamic_tail_tokens)} tok</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div className="rounded border p-2">
              <div className="text-muted-foreground">Artifacts Offloaded</div>
              <div className="font-mono font-bold">{data.offloaded_artifacts_count}</div>
            </div>
            <div className="rounded border p-2">
              <div className="text-muted-foreground">&lt;think&gt; Cleared</div>
              <div className="font-mono font-bold">
                {formatK(data.cleared_thinking_tokens)} tok
              </div>
            </div>
            <div className="rounded border p-2">
              <div className="text-muted-foreground">Batch Tool Evict</div>
              <div className="font-mono font-bold">{formatK(data.cleared_tool_tokens)} tok</div>
            </div>
            <div className="rounded border p-2">
              <div className="text-muted-foreground">Compactions</div>
              <div className="font-mono font-bold">{data.compaction_count}</div>
            </div>
          </div>

          {data.active_skills.length > 0 && (
            <div className="text-xs">
              <span className="text-muted-foreground">Active Skills: </span>
              <span className="font-mono">{data.active_skills.join(", ")}</span>
            </div>
          )}

          {threadId && messages.length > 2 && (
            <Button
              size="sm"
              variant="secondary"
              className="w-full text-xs"
              disabled={compacting}
              onClick={handleManualCompact}
            >
              {compacting ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Minimize2 className="mr-1.5 h-3.5 w-3.5" />
              )}
              Compact Session Now
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
