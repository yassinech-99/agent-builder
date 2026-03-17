"use client";

import { useEffect, useMemo, useState } from "react";
import { SessionSummary } from "@/providers/Thread";
import { Button } from "@/components/ui/button";
import { Trash2, X } from "lucide-react";

interface SessionChooserProps {
  open: boolean;
  sessions: SessionSummary[];
  targetAgentLabel: string;
  onContinue: (threadId: string) => void;
  onStartNew: () => void;
  onCancel: () => void;
  onDelete?: (threadId: string) => void;
}

function formatTime(ts: string): string {
  if (!ts) return "Unknown";
  const parsed = Date.parse(ts);
  if (Number.isNaN(parsed)) return "Unknown";
  return new Date(parsed).toLocaleString();
}

function getDateGroup(ts: string): string {
  if (!ts) return "Older";
  const parsed = Date.parse(ts);
  if (Number.isNaN(parsed)) return "Older";
  const date = new Date(parsed);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 86400000);
  const weekAgo = new Date(today.getTime() - 6 * 86400000);

  if (date >= today) return "Today";
  if (date >= yesterday) return "Yesterday";
  if (date >= weekAgo) return "This Week";
  return "Older";
}

const GROUP_ORDER = ["Today", "Yesterday", "This Week", "Older"];

export function SessionChooser({
  open,
  sessions,
  targetAgentLabel,
  onContinue,
  onStartNew,
  onCancel,
  onDelete,
}: SessionChooserProps) {
  const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);
  const orderedSessions = useMemo(() => sessions.slice(0, 24), [sessions]);

  const grouped = useMemo(() => {
    const groups: Record<string, SessionSummary[]> = {};
    for (const s of orderedSessions) {
      const g = getDateGroup(s.updated_at);
      if (!groups[g]) groups[g] = [];
      groups[g].push(s);
    }
    return GROUP_ORDER.filter((g) => groups[g]?.length).map((g) => ({
      label: g,
      items: groups[g],
    }));
  }, [orderedSessions]);

  useEffect(() => {
    if (!open) {
      setSelectedThreadId(null);
      return;
    }
    setSelectedThreadId(sessions[0]?.thread_id ?? null);
  }, [open, sessions]);

  if (!open) return null;

  const currentSelection =
    selectedThreadId || orderedSessions[0]?.thread_id || null;

  return (
    <div className="bg-background/70 absolute inset-0 z-50 flex items-center justify-center backdrop-blur-[1px]">
      <div className="bg-background w-full max-w-2xl rounded-xl border shadow-xl">
        <div className="flex items-start justify-between border-b px-5 py-4">
          <div>
            <h2 className="text-base font-semibold">Continue previous session?</h2>
            <p className="text-muted-foreground mt-0.5 text-sm">
              Pick a recent session for <span className="font-medium">{targetAgentLabel}</span> or start a new one.
            </p>
          </div>
          <Button type="button" variant="ghost" size="icon" onClick={onCancel}>
            <X className="size-4" />
          </Button>
        </div>

        <div className="max-h-[420px] space-y-1 overflow-auto px-5 py-4">
          {grouped.map((group) => (
            <div key={group.label}>
              <p className="text-muted-foreground mb-1 mt-2 px-1 text-xs font-semibold uppercase tracking-wider first:mt-0">
                {group.label}
              </p>
              <div className="space-y-1.5">
                {group.items.map((session) => {
                  const active = currentSelection === session.thread_id;
                  return (
                    <div
                      key={session.thread_id}
                      className={`group flex items-center gap-2 rounded-lg border px-3 py-3 transition-colors ${
                        active
                          ? "border-primary bg-primary/5"
                          : "hover:bg-muted/60 border-border"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => setSelectedThreadId(session.thread_id)}
                        className="min-w-0 flex-1 text-left"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <p className="truncate text-sm font-medium">
                            {session.title || "Untitled Session"}
                          </p>
                          <span className="text-muted-foreground shrink-0 text-xs">
                            {formatTime(session.updated_at)}
                          </span>
                        </div>
                        <div className="text-muted-foreground mt-1 flex gap-4 text-xs">
                          <span>
                            Messages:{" "}
                            {typeof session.message_count === "number"
                              ? session.message_count
                              : "\u2014"}
                          </span>
                          <span>KB: {session.rag_count}</span>
                          <span>Local: {session.local_rag_count}</span>
                          <span>Lexical: {session.lexical_count}</span>
                        </div>
                      </button>
                      {onDelete && (
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="text-destructive hover:text-destructive h-7 w-7 shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDelete(session.thread_id);
                          }}
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="flex justify-end gap-2 border-t px-5 py-3">
          <Button type="button" variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="button" variant="outline" onClick={onStartNew}>
            Start New Session
          </Button>
          <Button
            type="button"
            disabled={!currentSelection}
            onClick={() => {
              if (!currentSelection) return;
              onContinue(currentSelection);
            }}
          >
            Continue
          </Button>
        </div>
      </div>
    </div>
  );
}
