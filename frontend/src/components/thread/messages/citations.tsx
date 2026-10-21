"use client";

import React, { useMemo, useState } from "react";
import { Message } from "@langchain/langgraph-sdk";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { BookOpenCheck, Sparkles } from "lucide-react";

export interface GroundedCitation {
  citation_id: number;
  child_id: string;
  parent_id: string;
  source_id: string;
  doc_id: string;
  doc_name: string;
  heading_path: string;
  snippet: string;
  context_prefix: string;
  full_parent_markdown: string;
  dense_rank: number | null;
  dense_score: number;
  bm25_rank: number | null;
  bm25_score: number;
  rrf_score: number;
  rerank_score: number;
  final_score: number;
}

/**
 * Scan preceding tool results belonging to the current assistant turn to extract
 * structured `rag_citations_result` payloads emitted by `contextual_rag_search`.
 */
export function extractTurnCitations(
  messages: Message[],
  currentAiMessageId?: string,
): GroundedCitation[] {
  const aiIdx = currentAiMessageId
    ? messages.findIndex((m) => m.id === currentAiMessageId)
    : messages.length - 1;
  if (aiIdx <= 0) return [];

  const collected: GroundedCitation[] = [];
  for (let i = aiIdx - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.type === "human") break;
    if (m.type !== "tool") continue;
    const raw = typeof m.content === "string" ? m.content : "";
    if (!raw.includes("rag_citations_result")) continue;
    try {
      const parsed = JSON.parse(raw);
      if (
        parsed?.type === "rag_citations_result" &&
        Array.isArray(parsed.citations)
      ) {
        for (const c of parsed.citations) {
          collected.push(c as GroundedCitation);
        }
      }
    } catch {
      // ignore non-JSON tool payloads
    }
  }
  return collected;
}

export function GroundedCitationsDrawerBar({
  messages,
  messageId,
}: {
  messages: Message[];
  messageId?: string;
}) {
  const citations = useMemo(
    () => extractTurnCitations(messages, messageId),
    [messages, messageId],
  );
  const [selected, setSelected] = useState<GroundedCitation | null>(null);

  if (citations.length === 0) return null;

  return (
    <>
      <div className="flex flex-wrap items-center gap-1.5 pt-1">
        <span className="text-muted-foreground mr-1 inline-flex items-center gap-1 font-mono text-[11px] font-medium">
          <BookOpenCheck className="text-primary h-3.5 w-3.5" />
          Sources:
        </span>
        {citations.map((cit) => (
          <button
            key={`${cit.parent_id}-${cit.citation_id}`}
            type="button"
            onClick={() => setSelected(cit)}
            className="bg-muted/70 hover:bg-primary/10 hover:border-primary/40 inline-flex max-w-xs items-center gap-1.5 truncate rounded-full border px-2.5 py-0.5 text-xs transition-colors"
            title={`${cit.heading_path} (Click to inspect full parent evidence)`}
          >
            <span className="bg-primary/15 text-primary rounded px-1 font-mono text-[10px] font-bold">
              [{cit.citation_id}]
            </span>
            <span className="truncate font-medium">{cit.doc_name}</span>
          </button>
        ))}
      </div>

      <Sheet
        open={selected !== null}
        onOpenChange={(open) => {
          if (!open) setSelected(null);
        }}
      >
        <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
          {selected && (
            <div className="space-y-4">
              <SheetHeader>
                <SheetTitle className="flex items-center gap-2 text-base">
                  <Sparkles className="text-primary h-4 w-4 shrink-0" />
                  <span>
                    [{selected.citation_id}] {selected.doc_name}
                  </span>
                </SheetTitle>
                <p className="text-muted-foreground font-mono text-xs">
                  {selected.heading_path}
                </p>
              </SheetHeader>

              <div className="bg-muted/50 flex flex-wrap gap-2 rounded-xl border p-3 font-mono text-[11px]">
                <span>Dense #{selected.dense_rank ?? "—"}</span>
                <span>·</span>
                <span>BM25 #{selected.bm25_rank ?? "—"}</span>
                <span>·</span>
                <span>RRF {selected.rrf_score}</span>
                <span>·</span>
                <span>Rerank {selected.rerank_score}</span>
                <span>·</span>
                <span className="text-primary font-semibold">
                  Final {selected.final_score}
                </span>
              </div>

              <div className="space-y-1.5">
                <p className="text-muted-foreground text-[11px] font-semibold uppercase tracking-wider">
                  Matched Contextual Child Evidence
                </p>
                <div className="border-primary/40 bg-primary/5 rounded-xl border p-3.5 text-xs leading-relaxed whitespace-pre-wrap">
                  {selected.snippet}
                </div>
              </div>

              <div className="space-y-1.5">
                <p className="text-muted-foreground text-[11px] font-semibold uppercase tracking-wider">
                  Full Parent Section Context (`{selected.parent_id}`)
                </p>
                <pre className="bg-muted max-h-96 overflow-auto rounded-xl border p-3.5 font-mono text-xs leading-relaxed whitespace-pre-wrap">
                  {selected.full_parent_markdown}
                </pre>
              </div>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}
