"use client";

import React, { useEffect, useState } from "react";
import { Database, BookOpen, TextSearch, HardDrive } from "lucide-react";
import { useAgentConfig } from "@/providers/AgentConfig";
import { getBuilderApiUrl } from "@/lib/runtime-urls";

interface NimSource {
  id: string;
  name: string;
  description: string;
  collection_name: string;
  status: string;
}

interface LocalSource {
  id: string;
  name: string;
  status: string;
  doc_count: number;
  embedding_model: string;
}

interface LexicalSource {
  id: string;
  name: string;
  status: string;
  doc_count: number;
  meili_index: string;
}

export function StepRag() {
  const { config, setConfig } = useAgentConfig();
  const [nimSources, setNimSources] = useState<NimSource[]>([]);
  const [localSources, setLocalSources] = useState<LocalSource[]>([]);
  const [lexicalSources, setLexicalSources] = useState<LexicalSource[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const base = getBuilderApiUrl();
    Promise.all([
      fetch(`${base}/api/rag`)
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => []),
      fetch(`${base}/api/local-rag`)
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => []),
      fetch(`${base}/api/lexical-rag`)
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => []),
    ])
      .then(([nim, local, lexical]) => {
        setNimSources(nim.filter((s: NimSource) => s.status === "ready"));
        setLocalSources(local.filter((s: LocalSource) => s.status === "ready"));
        setLexicalSources(
          lexical.filter((s: LexicalSource) => s.status === "ready"),
        );
      })
      .finally(() => setLoading(false));
  }, []);

  const toggleNim = (id: string) => {
    setConfig((prev) => {
      const current = prev.rag_source_ids ?? [];
      const next = current.includes(id)
        ? current.filter((x) => x !== id)
        : [...current, id];
      return { ...prev, rag_source_ids: next };
    });
  };

  const toggleLocal = (id: string) => {
    setConfig((prev) => {
      const current = prev.local_rag_source_ids ?? [];
      const next = current.includes(id)
        ? current.filter((x) => x !== id)
        : [...current, id];
      return { ...prev, local_rag_source_ids: next };
    });
  };

  const toggleLexical = (id: string) => {
    setConfig((prev) => {
      const current = prev.lexical_rag_source_ids ?? [];
      const next = current.includes(id)
        ? current.filter((x) => x !== id)
        : [...current, id];
      return { ...prev, lexical_rag_source_ids: next };
    });
  };

  const isEmpty =
    nimSources.length === 0 &&
    localSources.length === 0 &&
    lexicalSources.length === 0;

  return (
    <div className="space-y-5">
      <p className="text-muted-foreground text-sm">
        Attach vector embeddings, lexical BM25 indexes, or NVIDIA NIM RAG
        collections to your agent.
      </p>

      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="bg-muted h-12 animate-pulse rounded-lg" />
          ))}
        </div>
      ) : isEmpty ? (
        <div className="rounded-lg border border-dashed p-8 text-center">
          <Database className="text-muted-foreground mx-auto mb-2 h-8 w-8" />
          <p className="text-muted-foreground text-sm">
            No ready RAG sources yet. Create sources in{" "}
            <a href="/rag" className="text-primary underline">
              RAG &amp; Knowledge
            </a>{" "}
            first.
          </p>
        </div>
      ) : (
        <div className="max-h-[340px] space-y-5 overflow-y-auto pr-1">
          {localSources.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-muted-foreground flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider">
                <HardDrive className="h-3.5 w-3.5" />
                Vector RAG (NVIDIA Nemotron / Local)
              </h3>
              {localSources.map((src) => {
                const selected = (config.local_rag_source_ids ?? []).includes(
                  src.id,
                );
                return (
                  <label
                    key={src.id}
                    className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3.5 transition-colors ${
                      selected
                        ? "border-primary bg-primary/5"
                        : "hover:border-muted-foreground/30"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleLocal(src.id)}
                    />
                    <Database className="text-primary h-4 w-4 shrink-0" />
                    <div>
                      <p className="text-sm font-medium">{src.name}</p>
                      <p className="text-muted-foreground text-xs">
                        {src.doc_count} chunks · {src.embedding_model}
                      </p>
                    </div>
                  </label>
                );
              })}
            </div>
          )}

          {lexicalSources.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-muted-foreground flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider">
                <TextSearch className="h-3.5 w-3.5" />
                Lexical BM25 (Meilisearch)
              </h3>
              {lexicalSources.map((src) => {
                const selected = (config.lexical_rag_source_ids ?? []).includes(
                  src.id,
                );
                return (
                  <label
                    key={src.id}
                    className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3.5 transition-colors ${
                      selected
                        ? "border-primary bg-primary/5"
                        : "hover:border-muted-foreground/30"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleLexical(src.id)}
                    />
                    <Database className="text-primary h-4 w-4 shrink-0" />
                    <div>
                      <p className="text-sm font-medium">{src.name}</p>
                      <p className="text-muted-foreground text-xs">
                        {src.doc_count} chunks · {src.meili_index}
                      </p>
                    </div>
                  </label>
                );
              })}
            </div>
          )}

          {nimSources.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-muted-foreground flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider">
                <BookOpen className="h-3.5 w-3.5" />
                NVIDIA NIM RAG Collections
              </h3>
              {nimSources.map((src) => {
                const selected = (config.rag_source_ids ?? []).includes(src.id);
                return (
                  <label
                    key={src.id}
                    className={`flex cursor-pointer items-center gap-3 rounded-lg border p-3.5 transition-colors ${
                      selected
                        ? "border-primary bg-primary/5"
                        : "hover:border-muted-foreground/30"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleNim(src.id)}
                    />
                    <Database className="text-primary h-4 w-4 shrink-0" />
                    <div>
                      <p className="text-sm font-medium">{src.name}</p>
                      <p className="text-muted-foreground text-xs">
                        Collection: {src.collection_name}
                      </p>
                    </div>
                  </label>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
