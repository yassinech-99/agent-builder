"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getBuilderApiUrl, getTenantHeaders } from "@/lib/runtime-urls";
import {
  GitFork,
  Pin,
  Download,
  Plus,
  Search,
  Trash2,
  RotateCcw,
  StopCircle,
} from "lucide-react";
import { toast } from "sonner";

interface DurableSessionItem {
  id: string;
  title: string;
  status: string;
  pinned: boolean;
  archived: boolean;
  tags: string[];
  forked_from_session_id: string;
  forked_from_seq: number;
  message_count: number;
  updated_at: string;
}

interface SessionEventItem {
  id: string;
  seq_id: number;
  event_type: string;
  role: string;
  checkpoint_id: string;
  payload: Record<string, unknown>;
  created_at: string;
}

interface SessionDetail extends DurableSessionItem {
  events: SessionEventItem[];
  subagents: Record<string, unknown>[];
}

export default function DurableSessionsPage() {
  const [sessions, setSessions] = useState<DurableSessionItem[]>([]);
  const [selected, setSelected] = useState<SessionDetail | null>(null);
  const [search, setSearch] = useState("");
  const [newTitle, setNewTitle] = useState("");

  const loadSessions = useCallback(async (q = search) => {
    try {
      const url = `${getBuilderApiUrl()}/api/sessions?search=${encodeURIComponent(q)}`;
      const res = await fetch(url, { headers: getTenantHeaders(), cache: "no-store" });
      if (res.ok) {
        const list = (await res.json()) as DurableSessionItem[];
        setSessions(list);
      }
    } catch {
      toast.error("Failed to fetch durable sessions");
    }
  }, [search]);

  const loadDetail = useCallback(async (id: string) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/sessions/${id}`, {
      headers: getTenantHeaders(),
      cache: "no-store",
    });
    if (res.ok) {
      setSelected((await res.json()) as SessionDetail);
    }
  }, []);

  useEffect(() => {
    void loadSessions("");
  }, [loadSessions]);

  const handleCreate = async () => {
    const res = await fetch(`${getBuilderApiUrl()}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({
        title: newTitle.trim() || "Durable Investigation Session",
        tags: ["durable", "phase5"],
      }),
    });
    if (res.ok) {
      const created = (await res.json()) as DurableSessionItem;
      setNewTitle("");
      await loadSessions();
      await loadDetail(created.id);
      toast.success("Created durable session");
    }
  };

  const handleFork = async (sessionId: string, upToSeq?: number) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/sessions/${sessionId}/fork`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify(upToSeq ? { up_to_seq: upToSeq } : {}),
    });
    if (res.ok) {
      const forked = (await res.json()) as DurableSessionItem;
      toast.success(`Forked branch '${forked.title}'`);
      await loadSessions();
      await loadDetail(forked.id);
    }
  };

  const handleTogglePin = async (s: DurableSessionItem) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/sessions/${s.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", ...getTenantHeaders() },
      body: JSON.stringify({ pinned: !s.pinned }),
    });
    if (res.ok) {
      await loadSessions();
    }
  };

  const handleExport = async (sessionId: string, fmt: "markdown" | "jsonl") => {
    const res = await fetch(
      `${getBuilderApiUrl()}/api/sessions/${sessionId}/export?format=${fmt}`,
      { headers: getTenantHeaders() },
    );
    if (!res.ok) return;
    const text = await res.text();
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const href = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = href;
    a.download = `${sessionId}.${fmt === "jsonl" ? "jsonl" : "md"}`;
    a.click();
    URL.revokeObjectURL(href);
  };

  const handleWake = async (sessionId: string) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/sessions/${sessionId}/wake`, {
      method: "POST",
      headers: getTenantHeaders(),
    });
    if (res.ok) {
      toast.success("Recovered session via stateless wake() with new fencing token");
      await loadDetail(sessionId);
    } else {
      const err = await res.json();
      toast.error(err.detail || "Lease currently active on healthy worker");
    }
  };

  const handleCancel = async (sessionId: string) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/sessions/${sessionId}/cancel`, {
      method: "POST",
      headers: getTenantHeaders(),
    });
    if (res.ok) {
      toast.success("Sent <5ms cancellation signal");
      await loadDetail(sessionId);
      await loadSessions();
    }
  };

  const handleDelete = async (sessionId: string) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/sessions/${sessionId}`, {
      method: "DELETE",
      headers: getTenantHeaders(),
    });
    if (res.ok) {
      if (selected?.id === sessionId) setSelected(null);
      await loadSessions();
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Durable Sessions, Time-Travel Forking &amp; Stream Replay
          </h1>
          <p className="text-muted-foreground text-sm">
            Stateless <code>wake(session_id)</code> fencing recovery, <code>Last-Event-ID</code> SSE replay &amp; sub-agent traces
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Input
            placeholder="New session title..."
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            className="h-9 w-52 text-xs"
          />
          <Button size="sm" onClick={handleCreate}>
            <Plus className="mr-1.5 h-4 w-4" />
            New Durable Session
          </Button>
        </div>
      </div>

      <div className="flex-1 space-y-5 overflow-auto p-6">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" />
            <Input
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                void loadSessions(e.target.value);
              }}
              placeholder="Search across all durable session titles, tags & append-only event logs..."
              className="pl-9 text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-2">
            {sessions.length === 0 ? (
              <div className="text-muted-foreground rounded-xl border p-6 text-center text-xs">
                No durable sessions yet — click <strong>New Durable Session</strong> above.
              </div>
            ) : (
              sessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => void loadDetail(s.id)}
                  className={`cursor-pointer rounded-xl border p-3.5 transition-all ${
                    selected?.id === s.id
                      ? "border-primary bg-primary/5 shadow-xs"
                      : "hover:border-primary/40"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-semibold">{s.title}</p>
                      <p className="text-muted-foreground font-mono text-[11px]">
                        {s.id} · {s.status}
                      </p>
                    </div>
                    <div
                      className="flex items-center gap-1"
                      onClick={(ev) => ev.stopPropagation()}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => void handleTogglePin(s)}
                        title="Pin/Unpin"
                      >
                        <Pin className={`h-3.5 w-3.5 ${s.pinned ? "text-amber-500" : ""}`} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => void handleFork(s.id)}
                        title="Fork session"
                      >
                        <GitFork className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-destructive h-7 w-7"
                        onClick={() => void handleDelete(s.id)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  {s.forked_from_session_id && (
                    <p className="text-muted-foreground mt-1 font-mono text-[10px]">
                      Forked from {s.forked_from_session_id} @ seq #{s.forked_from_seq}
                    </p>
                  )}
                </div>
              ))
            )}
          </div>

          <div className="bg-card space-y-4 rounded-xl border p-5 shadow-xs lg:col-span-2">
            {selected ? (
              <>
                <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3">
                  <div>
                    <h2 className="text-base font-semibold">{selected.title}</h2>
                    <p className="text-muted-foreground font-mono text-xs">
                      {selected.id} · Events: {selected.events.length} · Sub-Agents:{" "}
                      {selected.subagents.length}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => void handleWake(selected.id)}
                    >
                      <RotateCcw className="mr-1 h-3.5 w-3.5" />
                      wake()
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => void handleCancel(selected.id)}
                    >
                      <StopCircle className="mr-1 h-3.5 w-3.5" />
                      Cancel
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => void handleExport(selected.id, "markdown")}
                    >
                      <Download className="mr-1 h-3.5 w-3.5" />
                      .md
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => void handleExport(selected.id, "jsonl")}
                    >
                      <Download className="mr-1 h-3.5 w-3.5" />
                      .jsonl
                    </Button>
                  </div>
                </div>

                <div className="max-h-96 space-y-2 overflow-auto">
                  {selected.events.length === 0 ? (
                    <p className="text-muted-foreground text-xs">
                      Append-only event log is empty for this session.
                    </p>
                  ) : (
                    selected.events.map((ev) => (
                      <div
                        key={ev.id}
                        className="bg-muted/40 flex items-start justify-between gap-3 rounded-lg border p-3 text-xs"
                      >
                        <div className="space-y-1">
                          <div className="font-mono font-semibold">
                            seq #{ev.seq_id} · {ev.event_type} ({ev.checkpoint_id})
                          </div>
                          <pre className="font-mono text-[11px] whitespace-pre-wrap">
                            {JSON.stringify(ev.payload)}
                          </pre>
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => void handleFork(selected.id, ev.seq_id)}
                          title={`Fork time-travel branch @ seq #${ev.seq_id}`}
                        >
                          <GitFork className="mr-1 h-3 w-3" />
                          Fork @{ev.seq_id}
                        </Button>
                      </div>
                    ))
                  )}
                </div>
              </>
            ) : (
              <p className="text-muted-foreground text-xs">
                Select a durable session to inspect its append-only event log, time-travel fork from any turn, or export traces.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
