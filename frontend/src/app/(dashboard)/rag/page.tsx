"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getBuilderApiUrl } from "@/lib/runtime-urls";
import {
  Trash2,
  Database,
  Upload,
  FileText,
  Search,
  X,
  BookOpen,
  TextSearch,
  Plus,
  CloudUpload,
  HardDrive,
  Sparkles,
} from "lucide-react";

const EMBEDDING_MODELS = [
  {
    value: "nvidia/llama-nemotron-embed-1b-v2",
    label: "NVIDIA Llama Nemotron Embed 1B v2 (NIM)",
  },
  {
    value: "nvidia/nv-embedqa-e5-v5",
    label: "NVIDIA NV-EmbedQA E5 v5 (NIM)",
  },
  {
    value: "local/all-MiniLM-L6-v2",
    label: "Local SentenceTransformers (all-MiniLM-L6-v2 · Offline)",
  },
  {
    value: "text-embedding-3-small",
    label: "OpenAI text-embedding-3-small",
  },
] as const;

interface NimRagSource {
  id: string;
  name: string;
  description: string;
  endpoint_url: string;
  collection_name: string;
  ranking_model: string;
  storage_bucket: string;
  storage_prefix: string;
  status: string;
  created_at: string;
}

interface LocalRagSource {
  id: string;
  name: string;
  source_type: string;
  status: string;
  doc_count: number;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  storage_bucket: string;
  storage_prefix: string;
  created_at: string;
}

interface LexicalRagSource {
  id: string;
  name: string;
  source_type: string;
  meili_index: string;
  status: string;
  doc_count: number;
  chunk_size: number;
  chunk_overlap: number;
  created_at: string;
}

type Tab = "contextual" | "nim" | "local" | "lexical";

export default function RagPage() {
  const [tab, setTab] = useState<Tab>("contextual");

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">RAG &amp; Knowledge</h1>
          <p className="text-muted-foreground text-sm">
            Contextual Parent-Child Hybrid RAG, `NVIDIARerank` Debugger, Nemotron Embeddings &amp; Meilisearch BM25
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-1 border-b px-6 pt-2">
        {(
          [
            {
              key: "contextual",
              icon: Sparkles,
              label: "Contextual Hybrid RAG & Rank Debugger",
            },
            { key: "local", icon: HardDrive, label: "Vector RAG (NIM / Local)" },
            { key: "lexical", icon: TextSearch, label: "Lexical BM25 (Meilisearch)" },
            { key: "nim", icon: BookOpen, label: "NIM RAG Endpoints" },
          ] as const
        ).map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex items-center gap-1.5 rounded-t-md px-4 py-2.5 text-sm font-medium transition-colors ${
              tab === t.key
                ? "border-primary text-primary border-b-2 bg-transparent"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <t.icon className="h-4 w-4" />
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto">
        {tab === "contextual" && <ContextualHybridRagTab />}
        {tab === "nim" && <NimRagTab />}
        {tab === "local" && <LocalRagTab />}
        {tab === "lexical" && <LexicalRagTab />}
      </div>
    </div>
  );
}

function ContextualHybridRagTab() {
  const [collections, setCollections] = useState<any[]>([]);
  const [colName, setColName] = useState("");
  const [activeProgress, setActiveProgress] = useState<{
    progress: number;
    detail: string;
    status: string;
  } | null>(null);
  const [debugQuery, setDebugQuery] = useState("");
  const [debugResult, setDebugResult] = useState<any | null>(null);

  const loadCollections = useCallback(async () => {
    const res = await fetch(`${getBuilderApiUrl()}/api/rag-pipeline/collections`, {
      cache: "no-store",
    });
    if (res.ok) setCollections(await res.json());
  }, []);

  useEffect(() => {
    void loadCollections();
  }, [loadCollections]);

  const handleCreateCollection = async () => {
    if (!colName.trim()) return;
    const res = await fetch(`${getBuilderApiUrl()}/api/rag-pipeline/collections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: colName.trim() }),
    });
    if (res.ok) {
      setColName("");
      await loadCollections();
    }
  };

  const handleUploadAsync = async (sourceId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    form.append("contextual_enrich", "true");
    form.append("async_mode", "true");

    const res = await fetch(
      `${getBuilderApiUrl()}/api/rag-pipeline/collections/${sourceId}/ingest`,
      { method: "POST", body: form },
    );
    if (!res.ok) return;
    const queued = await res.json();
    setActiveProgress({
      progress: 5,
      detail: `Queued ${file.name} (${queued.job_id})`,
      status: "queued",
    });

    const es = new EventSource(
      `${getBuilderApiUrl()}/api/rag-pipeline/jobs/${queued.job_id}/stream`,
    );
    es.addEventListener("progress", (ev: MessageEvent) => {
      try {
        const snap = JSON.parse(ev.data);
        setActiveProgress(snap);
        if (snap.status === "completed" || snap.status === "error") {
          es.close();
          void loadCollections();
        }
      } catch {
        es.close();
      }
    });
    es.onerror = () => es.close();
  };

  const handleRunRankDebugger = async () => {
    if (!debugQuery.trim()) return;
    const res = await fetch(`${getBuilderApiUrl()}/api/rag-pipeline/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: debugQuery.trim(),
        top_k: 5,
        rerank: true,
        enable_query_rewrite: true,
      }),
    });
    if (res.ok) {
      setDebugResult(await res.json());
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="bg-card flex flex-wrap items-center gap-2 rounded-xl border p-4 shadow-xs">
        <Input
          placeholder="New Contextual Hybrid Collection name (e.g., H200 Architecture Runbooks)..."
          value={colName}
          onChange={(e) => setColName(e.target.value)}
          className="max-w-md text-sm"
        />
        <Button size="sm" onClick={() => void handleCreateCollection()}>
          <Plus className="mr-1.5 h-4 w-4" />
          Create Contextual Collection
        </Button>
      </div>

      {activeProgress && (
        <div className="bg-card space-y-2 rounded-xl border p-4 text-xs shadow-xs">
          <div className="flex items-center justify-between font-mono">
            <span>{activeProgress.detail}</span>
            <span className="font-bold">{activeProgress.progress}%</span>
          </div>
          <div className="bg-muted h-2 w-full overflow-hidden rounded-full">
            <div
              className="bg-primary h-full transition-all duration-300"
              style={{ width: `${activeProgress.progress}%` }}
            />
          </div>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {collections.map((c) => (
          <div key={c.id} className="bg-card space-y-3 rounded-xl border p-4 shadow-xs">
            <div className="flex items-start justify-between gap-2">
              <div>
                <h3 className="text-sm font-semibold">{c.name}</h3>
                <p className="text-muted-foreground font-mono text-xs">
                  {c.id} · {c.doc_count} indexed contextual chunks
                </p>
              </div>
              <label className="bg-primary text-primary-foreground hover:bg-primary/90 inline-flex cursor-pointer items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium">
                <Upload className="h-3.5 w-3.5" />
                Stream Upload (`202` SSE)
                <input
                  type="file"
                  className="hidden"
                  accept=".md,.markdown,.txt,.pdf,.docx,.csv,.json"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void handleUploadAsync(c.id, f);
                  }}
                />
              </label>
            </div>
            {Array.isArray(c.documents) && c.documents.length > 0 && (
              <div className="space-y-1 border-t pt-2 text-xs">
                {c.documents.map((d: any) => (
                  <div
                    key={d.id}
                    className="text-muted-foreground flex items-center justify-between font-mono text-[11px]"
                  >
                    <span>
                      {d.filename} ({d.parent_count}p / {d.child_count}c)
                    </span>
                    <span>sha256:{String(d.sha256_hash).slice(0, 10)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Interactive 6-Stage Retrieval Rank Debugger */}
      <div className="bg-card space-y-4 rounded-xl border p-5 shadow-xs">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold">
              6-Stage Hybrid Retrieval &amp; `NVIDIARerank` Score Debugger
            </h2>
            <p className="text-muted-foreground text-xs">
              Inspect Dense Rank + Contextual BM25 Rank → RRF ($k=60$) → Cross-Encoder Rerank → Deduped Parent Evidence
            </p>
          </div>
          {debugResult && (
            <span className="rounded bg-emerald-500/15 px-2.5 py-1 font-mono text-xs font-semibold text-emerald-700 dark:text-emerald-400">
              {debugResult.latency_ms}ms · Grade={debugResult.relevance_grade}
              {debugResult.rewritten_query
                ? ` · Rewritten="${debugResult.rewritten_query}"`
                : ""}
            </span>
          )}
        </div>

        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" />
            <Input
              value={debugQuery}
              onChange={(e) => setDebugQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void handleRunRankDebugger()}
              placeholder="Run hybrid query & inspect Dense / BM25 / RRF / Rerank scores..."
              className="pl-9 text-sm"
            />
          </div>
          <Button onClick={() => void handleRunRankDebugger()}>
            Debug Ranks
          </Button>
        </div>

        {debugResult?.citations?.length > 0 && (
          <div className="space-y-3">
            {debugResult.citations.map((cit: any) => (
              <div
                key={cit.citation_id}
                className="bg-muted/30 space-y-2 rounded-xl border p-4 text-xs"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="font-semibold">
                    <span className="text-primary mr-1.5 font-mono">
                      [{cit.citation_id}]
                    </span>
                    {cit.heading_path}
                  </div>
                  <div className="text-muted-foreground flex flex-wrap gap-2 font-mono text-[11px]">
                    <span>Dense #{cit.dense_rank ?? "—"}</span>
                    <span>·</span>
                    <span>BM25 #{cit.bm25_rank ?? "—"}</span>
                    <span>·</span>
                    <span>RRF {cit.rrf_score}</span>
                    <span>·</span>
                    <span>Rerank {cit.rerank_score}</span>
                    <span>·</span>
                    <span className="text-foreground font-bold">
                      Final {cit.final_score}
                    </span>
                  </div>
                </div>
                <p className="text-muted-foreground font-mono text-[11px]">
                  {cit.context_prefix}
                </p>
                <pre className="bg-background overflow-auto rounded-lg border p-3 font-mono text-[11px] whitespace-pre-wrap">
                  {cit.full_parent_markdown}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function statusColor(status: string) {
  switch (status) {
    case "ready":
      return "text-emerald-600 bg-emerald-50 dark:bg-emerald-950/30 dark:text-emerald-400";
    case "indexing":
      return "text-amber-600 bg-amber-50 dark:bg-amber-950/30 dark:text-amber-400";
    case "error":
      return "text-red-600 bg-red-50 dark:bg-red-950/30 dark:text-red-400";
    default:
      return "text-gray-600 bg-gray-50 dark:bg-gray-800 dark:text-gray-400";
  }
}

/* ─── NVIDIA NIM RAG Endpoints Tab ──────────────────────────────────── */

function NimRagTab() {
  const [sources, setSources] = useState<NimRagSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [showRegister, setShowRegister] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [regName, setRegName] = useState("");
  const [regDescription, setRegDescription] = useState("");
  const [regEndpoint, setRegEndpoint] = useState("http://localhost:8081/v1");
  const [regCollection, setRegCollection] = useState("");
  const [regApiKey, setRegApiKey] = useState("");
  const [regReranker, setRegReranker] = useState(
    "nvidia/llama-nemotron-rerank-1b-v2",
  );
  const [regBucket, setRegBucket] = useState("");
  const [regPrefix, setRegPrefix] = useState("");
  const [testQuery, setTestQuery] = useState("");
  const [testResults, setTestResults] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [addDataSourceId, setAddDataSourceId] = useState<string | null>(null);

  const fetchSources = useCallback(async () => {
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/rag`);
      if (res.ok) setSources(await res.json());
    } catch (e) {
      console.error("Failed to fetch NIM RAG sources:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSources();
  }, [fetchSources]);

  const handleRegister = async () => {
    if (!regName || !regCollection) return;
    setRegistering(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/rag/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: regName,
          description: regDescription,
          endpoint_url: regEndpoint,
          collection_name: regCollection,
          api_key: regApiKey,
          ranking_model: regReranker,
          storage_bucket: regBucket,
          storage_prefix: regPrefix,
        }),
      });
      if (res.ok) {
        setShowRegister(false);
        setRegName("");
        setRegDescription("");
        setRegCollection("");
        setRegApiKey("");
        setRegBucket("");
        setRegPrefix("");
        fetchSources();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "Registration failed");
      }
    } catch (e) {
      console.error("Failed to register:", e);
    } finally {
      setRegistering(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Remove this NIM RAG endpoint?")) return;
    try {
      await fetch(`${getBuilderApiUrl()}/api/rag/${id}`, { method: "DELETE" });
      fetchSources();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const handleTestQuery = async (sourceId: string) => {
    if (!testQuery.trim()) return;
    setTestResults(null);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/rag/${sourceId}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: testQuery, k: 5 }),
      });
      if (res.ok) {
        const data = await res.json();
        setTestResults(
          data.results?.map((r: any) => r.page_content).join("\n---\n") ||
            "No results",
        );
      } else {
        const err = await res.json().catch(() => ({}));
        setTestResults(`Error: ${err.detail || "Query failed"}`);
      }
    } catch (e) {
      console.error("Query failed:", e);
    }
  };

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-muted-foreground text-sm">
          Connect external NVIDIA NIM RAG Blueprint services with NVIDIARerank
        </p>
        <Button onClick={() => setShowRegister(true)} size="sm">
          <Plus className="mr-1.5 h-4 w-4" />
          Register NIM RAG Endpoint
        </Button>
      </div>

      {showRegister && (
        <div className="bg-card mb-6 rounded-xl border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Register NVIDIA NIM RAG Collection
            </h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowRegister(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label>Name</Label>
              <Input
                value={regName}
                onChange={(e) => setRegName(e.target.value)}
                placeholder="Engineering Architecture Docs"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Collection Name</Label>
              <Input
                value={regCollection}
                onChange={(e) => setRegCollection(e.target.value)}
                placeholder="eng_handbook"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>NIM RAG Endpoint URL</Label>
              <Input
                value={regEndpoint}
                onChange={(e) => setRegEndpoint(e.target.value)}
                placeholder="http://localhost:8081/v1"
                className="font-mono text-xs"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>NVIDIA Reranker Model</Label>
              <Input
                value={regReranker}
                onChange={(e) => setRegReranker(e.target.value)}
                placeholder="nvidia/llama-nemotron-rerank-1b-v2"
                className="font-mono text-xs"
              />
            </div>
            <div className="flex flex-col gap-1.5 md:col-span-2">
              <Label>Description</Label>
              <Input
                value={regDescription}
                onChange={(e) => setRegDescription(e.target.value)}
                placeholder="Contains internal specs, runbooks, and API references"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>
                Object Bucket{" "}
                <span className="text-muted-foreground text-xs font-normal">
                  (optional MinIO / S3 bucket for uploads)
                </span>
              </Label>
              <Input
                value={regBucket}
                onChange={(e) => setRegBucket(e.target.value)}
                placeholder="nimagents-kb"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>
                Object Prefix{" "}
                <span className="text-muted-foreground text-xs font-normal">
                  (optional folder)
                </span>
              </Label>
              <Input
                value={regPrefix}
                onChange={(e) => setRegPrefix(e.target.value)}
                placeholder="documents/"
              />
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="outline" onClick={() => setShowRegister(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleRegister}
              disabled={!regName || !regCollection || registering}
            >
              {registering ? "Registering..." : "Register Endpoint"}
            </Button>
          </div>
        </div>
      )}

      {addDataSourceId && (
        <AddDataModal
          sourceId={addDataSourceId}
          source={sources.find((s) => s.id === addDataSourceId)!}
          onClose={() => setAddDataSourceId(null)}
        />
      )}

      {loading ? (
        <LoadingSkeleton />
      ) : sources.length === 0 && !showRegister ? (
        <EmptyState
          icon={Database}
          title="No NIM RAG endpoints registered"
          subtitle="Connect an NVIDIA NIM RAG Retriever microservice for remote multi-collection search"
          actionLabel="Register NIM RAG Endpoint"
          onAction={() => setShowRegister(true)}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {sources.map((source) => (
            <KBSourceCard
              key={source.id}
              source={source}
              onDelete={() => handleDelete(source.id)}
              onAddData={() => setAddDataSourceId(source.id)}
              testingId={testingId}
              setTestingId={setTestingId}
              testQuery={testQuery}
              setTestQuery={setTestQuery}
              testResults={testResults}
              setTestResults={setTestResults}
              onTestQuery={() => handleTestQuery(source.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function AddDataModal({
  sourceId,
  source,
  onClose,
}: {
  sourceId: string;
  source: NimRagSource;
  onClose: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [metadataJson, setMetadataJson] = useState("");
  const [description, setDescription] = useState("");
  const [uploading, setUploading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!description.trim()) return;
    setGenerating(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/rag/generate-metadata`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          description,
          filename: file?.name || "",
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setMetadataJson(
          typeof data.metadata_json === "string"
            ? data.metadata_json
            : JSON.stringify(data.metadata_json, null, 2),
        );
      }
    } catch (e) {
      console.error("Metadata generation failed:", e);
    } finally {
      setGenerating(false);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      if (metadataJson.trim()) formData.append("metadata_json", metadataJson);
      if (description.trim()) formData.append("description", description);
      const res = await fetch(
        `${getBuilderApiUrl()}/api/rag/${sourceId}/upload-data`,
        { method: "POST", body: formData },
      );
      const data = await res.json();
      if (res.ok) {
        setResult(
          `Uploaded to ${data.storage_uri}${data.metadata_uploaded ? " (with metadata sidecar)" : ""}`,
        );
        setFile(null);
        setMetadataJson("");
        setDescription("");
      } else {
        setResult(`Error: ${data.detail || "Upload failed"}`);
      }
    } catch (e) {
      console.error("Upload failed:", e);
      setResult("Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-card mb-6 rounded-xl border p-6 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Upload Document to {source.name}</h2>
          <p className="text-muted-foreground text-xs">
            Destination: s3://{source.storage_bucket}/{source.storage_prefix}
          </p>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {!source.storage_bucket ? (
        <p className="text-sm text-red-500">
          No object bucket configured for this collection.
        </p>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-col gap-1.5">
            <Label>File</Label>
            <Input
              type="file"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <Label>Describe content (optional AI metadata enrichment)</Label>
            <div className="flex gap-2">
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Q3 2026 platform architecture specifications..."
                className="flex-1"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={handleGenerate}
                disabled={!description.trim() || generating}
              >
                <Sparkles className="mr-1 h-3.5 w-3.5" />
                {generating ? "Generating..." : "AI Metadata"}
              </Button>
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label>Metadata JSON</Label>
            <textarea
              value={metadataJson}
              onChange={(e) => setMetadataJson(e.target.value)}
              placeholder='{"metadataAttributes": {"category": "...", "topic": "..."}}'
              rows={4}
              className="border-input bg-background w-full rounded-md border px-3 py-2 font-mono text-xs"
            />
          </div>

          {result && (
            <p
              className={`text-sm ${
                result.startsWith("Error") ? "text-red-500" : "text-emerald-600"
              }`}
            >
              {result}
            </p>
          )}

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
            <Button onClick={handleUpload} disabled={!file || uploading}>
              <Upload className="mr-1.5 h-4 w-4" />
              {uploading ? "Uploading..." : "Upload to Bucket"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function KBSourceCard({
  source,
  onDelete,
  onAddData,
  testingId,
  setTestingId,
  testQuery,
  setTestQuery,
  testResults,
  setTestResults,
  onTestQuery,
}: {
  source: NimRagSource;
  onDelete: () => void;
  onAddData: () => void;
  testingId: string | null;
  setTestingId: (v: string | null) => void;
  testQuery: string;
  setTestQuery: (v: string) => void;
  testResults: string | null;
  setTestResults: (v: string | null) => void;
  onTestQuery: () => void;
}) {
  return (
    <div className="bg-card flex flex-col rounded-xl border p-5 shadow-sm">
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-2">
          <div className="bg-primary/10 text-primary flex h-9 w-9 items-center justify-center rounded-lg">
            <Database className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-semibold">{source.name}</h3>
            <p className="text-muted-foreground text-xs">
              <span
                className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${statusColor(source.status)}`}
              >
                {source.status}
              </span>
              <span className="bg-primary/10 text-primary ml-1.5 inline-block rounded px-1.5 py-0.5 font-mono text-[10px] font-medium">
                {source.collection_name}
              </span>
            </p>
            {source.description && (
              <p className="text-muted-foreground mt-1 text-xs">
                {source.description}
              </p>
            )}
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="text-destructive hover:text-destructive"
          onClick={onDelete}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>

      {testingId === source.id ? (
        <div className="mt-2 flex flex-col gap-2">
          <div className="flex gap-2">
            <Input
              placeholder="Test semantic retrieval..."
              value={testQuery}
              onChange={(e) => setTestQuery(e.target.value)}
              className="flex-1"
              onKeyDown={(e) => e.key === "Enter" && onTestQuery()}
            />
            <Button size="sm" onClick={onTestQuery}>
              <Search className="h-3.5 w-3.5" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setTestingId(null);
                setTestResults(null);
              }}
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
          {testResults && (
            <pre className="bg-muted max-h-40 overflow-auto rounded p-2 text-xs whitespace-pre-wrap">
              {testResults}
            </pre>
          )}
        </div>
      ) : (
        <div className="mt-2 flex gap-2 border-t pt-3">
          {source.storage_bucket && (
            <Button
              variant="ghost"
              size="sm"
              className="flex-1"
              onClick={onAddData}
            >
              <Upload className="mr-1 h-3.5 w-3.5" />
              Add Data
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="flex-1"
            onClick={() => {
              setTestingId(source.id);
              setTestQuery("");
              setTestResults(null);
            }}
          >
            <Search className="mr-1 h-3.5 w-3.5" />
            Test Query
          </Button>
        </div>
      )}
    </div>
  );
}

/* ─── Vector RAG Tab ─────────────────────────────────────────────────── */

function LocalRagTab() {
  const [sources, setSources] = useState<LocalRagSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadName, setUploadName] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [embeddingModel, setEmbeddingModel] = useState<string>(
    EMBEDDING_MODELS[0].value,
  );
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);

  const [showBucketImport, setShowBucketImport] = useState(false);
  const [importing, setImporting] = useState(false);
  const [bucketSourceName, setBucketSourceName] = useState("");
  const [bucketName, setBucketName] = useState("");
  const [bucketPrefix, setBucketPrefix] = useState("");
  const [bucketChunkSize, setBucketChunkSize] = useState(1000);
  const [bucketChunkOverlap, setBucketChunkOverlap] = useState(200);
  const [bucketEmbeddingModel, setBucketEmbeddingModel] = useState<string>(
    EMBEDDING_MODELS[0].value,
  );
  const [storageConnections, setStorageConnections] = useState<
    { id: string; name: string; bucket_name: string }[]
  >([]);

  const [testQuery, setTestQuery] = useState("");
  const [testResults, setTestResults] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${getBuilderApiUrl()}/api/storage`)
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => setStorageConnections(data))
      .catch(() => {});
  }, []);

  const fetchSources = useCallback(async () => {
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/local-rag`);
      if (res.ok) setSources(await res.json());
    } catch (e) {
      console.error("Failed to fetch vector RAG sources:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSources();
  }, [fetchSources]);

  const handleUpload = async () => {
    if (!uploadFile || !uploadName) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      formData.append("name", uploadName);
      formData.append("embedding_model", embeddingModel);
      formData.append("chunk_size", String(chunkSize));
      formData.append("chunk_overlap", String(chunkOverlap));
      const res = await fetch(`${getBuilderApiUrl()}/api/local-rag/upload`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        setShowUpload(false);
        setUploadName("");
        setUploadFile(null);
        fetchSources();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "Upload failed");
      }
    } catch (e) {
      console.error("Failed to upload:", e);
    } finally {
      setUploading(false);
    }
  };

  const handleBucketImport = async () => {
    if (!bucketSourceName || !bucketName) return;
    setImporting(true);
    try {
      const res = await fetch(
        `${getBuilderApiUrl()}/api/local-rag/bucket-import`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: bucketSourceName,
            bucket_name: bucketName,
            prefix: bucketPrefix,
            chunk_size: bucketChunkSize,
            chunk_overlap: bucketChunkOverlap,
            embedding_model: bucketEmbeddingModel,
          }),
        },
      );
      if (res.ok) {
        setShowBucketImport(false);
        setBucketSourceName("");
        setBucketName("");
        setBucketPrefix("");
        fetchSources();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "Bucket import failed");
      }
    } catch (e) {
      console.error("Bucket import failed:", e);
    } finally {
      setImporting(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this vector RAG source?")) return;
    try {
      await fetch(`${getBuilderApiUrl()}/api/local-rag/${id}`, {
        method: "DELETE",
      });
      fetchSources();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const handleTestQuery = async (sourceId: string) => {
    if (!testQuery.trim()) return;
    setTestResults(null);
    try {
      const res = await fetch(
        `${getBuilderApiUrl()}/api/local-rag/${sourceId}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: testQuery, k: 4 }),
        },
      );
      if (res.ok) {
        const data = await res.json();
        setTestResults(
          data.results?.map((r: any) => r.page_content).join("\n---\n") ||
            "No results",
        );
      }
    } catch (e) {
      console.error("Query failed:", e);
    }
  };

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-muted-foreground text-sm">
          Dense semantic search powered by NVIDIA Embeddings + NVIDIARerank or offline local embeddings
        </p>
        <div className="flex gap-2">
          <Button
            onClick={() => {
              setShowBucketImport(true);
              setShowUpload(false);
            }}
            size="sm"
            variant="outline"
          >
            <CloudUpload className="mr-1.5 h-4 w-4" />
            Import from Bucket
          </Button>
          <Button
            onClick={() => {
              setShowUpload(true);
              setShowBucketImport(false);
            }}
            size="sm"
          >
            <Upload className="mr-1.5 h-4 w-4" />
            Upload Document
          </Button>
        </div>
      </div>

      {showUpload && (
        <div className="bg-card mb-6 rounded-xl border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Upload &amp; Embed Document</h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowUpload(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label>Source Name</Label>
              <Input
                value={uploadName}
                onChange={(e) => setUploadName(e.target.value)}
                placeholder="Product Architecture Guide"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>File (PDF, TXT, MD, CSV, DOCX)</Label>
              <Input
                type="file"
                accept=".pdf,.txt,.md,.csv,.docx"
                onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Embedding Model</Label>
              <select
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
                className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
              >
                {EMBEDDING_MODELS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Chunk Size</Label>
              <Input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                min={100}
                max={10000}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Chunk Overlap</Label>
              <Input
                type="number"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
                min={0}
                max={5000}
              />
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="outline" onClick={() => setShowUpload(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleUpload}
              disabled={!uploadName || !uploadFile || uploading}
            >
              {uploading ? "Indexing..." : "Upload & Index"}
            </Button>
          </div>
        </div>
      )}

      {showBucketImport && (
        <div className="bg-card mb-6 rounded-xl border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Import from MinIO / S3-Compatible Bucket
            </h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowBucketImport(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label>Source Name</Label>
              <Input
                value={bucketSourceName}
                onChange={(e) => setBucketSourceName(e.target.value)}
                placeholder="Bucket Knowledge Corpus"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Object Bucket</Label>
              {storageConnections.length > 0 ? (
                <select
                  value={bucketName}
                  onChange={(e) => setBucketName(e.target.value)}
                  className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
                >
                  <option value="">Select a registered bucket...</option>
                  {storageConnections.map((c) => (
                    <option key={c.id} value={c.bucket_name}>
                      {c.name} (s3://{c.bucket_name})
                    </option>
                  ))}
                </select>
              ) : (
                <Input
                  value={bucketName}
                  onChange={(e) => setBucketName(e.target.value)}
                  placeholder="nimagents-docs"
                />
              )}
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Prefix / Folder (optional)</Label>
              <Input
                value={bucketPrefix}
                onChange={(e) => setBucketPrefix(e.target.value)}
                placeholder="documents/2026/"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Embedding Model</Label>
              <select
                value={bucketEmbeddingModel}
                onChange={(e) => setBucketEmbeddingModel(e.target.value)}
                className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
              >
                {EMBEDDING_MODELS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Chunk Size</Label>
              <Input
                type="number"
                value={bucketChunkSize}
                onChange={(e) => setBucketChunkSize(Number(e.target.value))}
                min={100}
                max={10000}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Chunk Overlap</Label>
              <Input
                type="number"
                value={bucketChunkOverlap}
                onChange={(e) => setBucketChunkOverlap(Number(e.target.value))}
                min={0}
                max={5000}
              />
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setShowBucketImport(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleBucketImport}
              disabled={!bucketSourceName || !bucketName || importing}
            >
              {importing ? "Importing..." : "Import & Index"}
            </Button>
          </div>
        </div>
      )}

      {loading ? (
        <LoadingSkeleton />
      ) : sources.length === 0 && !showUpload && !showBucketImport ? (
        <EmptyState
          icon={HardDrive}
          title="No vector knowledge sources yet"
          subtitle="Upload documents or import from MinIO / S3-compatible object storage"
          actionLabel="Upload Document"
          onAction={() => setShowUpload(true)}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {sources.map((source) => (
            <SourceCard
              key={source.id}
              id={source.id}
              name={source.name}
              docCount={source.doc_count}
              status={source.status}
              badge={
                source.source_type === "bucket"
                  ? `s3://${source.storage_bucket || ""}`
                  : source.embedding_model
              }
              onDelete={() => handleDelete(source.id)}
              testingId={testingId}
              setTestingId={setTestingId}
              testQuery={testQuery}
              setTestQuery={setTestQuery}
              testResults={testResults}
              setTestResults={setTestResults}
              onTestQuery={() => handleTestQuery(source.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Lexical RAG Tab ─────────────────────────────────────────────────── */

function LexicalRagTab() {
  const [sources, setSources] = useState<LexicalRagSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadName, setUploadName] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [testQuery, setTestQuery] = useState("");
  const [testResults, setTestResults] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);

  const fetchSources = useCallback(async () => {
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/lexical-rag`);
      if (res.ok) setSources(await res.json());
    } catch (e) {
      console.error("Failed to fetch lexical RAG sources:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSources();
  }, [fetchSources]);

  const handleUpload = async () => {
    if (!uploadFile || !uploadName) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      formData.append("name", uploadName);
      formData.append("chunk_size", String(chunkSize));
      formData.append("chunk_overlap", String(chunkOverlap));
      const res = await fetch(`${getBuilderApiUrl()}/api/lexical-rag/upload`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        setShowUpload(false);
        setUploadName("");
        setUploadFile(null);
        fetchSources();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "Upload failed");
      }
    } catch (e) {
      console.error("Failed to upload:", e);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this lexical RAG source?")) return;
    try {
      await fetch(`${getBuilderApiUrl()}/api/lexical-rag/${id}`, {
        method: "DELETE",
      });
      fetchSources();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const handleTestQuery = async (sourceId: string) => {
    if (!testQuery.trim()) return;
    setTestResults(null);
    try {
      const res = await fetch(
        `${getBuilderApiUrl()}/api/lexical-rag/${sourceId}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: testQuery, k: 5 }),
        },
      );
      if (res.ok) {
        const data = await res.json();
        setTestResults(
          data.results?.map((r: any) => r.content).join("\n---\n") ||
            "No results",
        );
      }
    } catch (e) {
      console.error("Query failed:", e);
    }
  };

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-muted-foreground text-sm">
          Non-blocking BM25 keyword search powered by Meilisearch
        </p>
        <Button onClick={() => setShowUpload(true)} size="sm">
          <Upload className="mr-1.5 h-4 w-4" />
          Upload Document
        </Button>
      </div>

      {showUpload && (
        <div className="bg-card mb-6 rounded-xl border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Upload for Lexical Indexing</h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowUpload(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label>Source Name</Label>
              <Input
                value={uploadName}
                onChange={(e) => setUploadName(e.target.value)}
                placeholder="API Reference Docs"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>File (PDF, TXT, MD, CSV, DOCX)</Label>
              <Input
                type="file"
                accept=".pdf,.txt,.md,.csv,.docx"
                onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Chunk Size</Label>
              <Input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                min={100}
                max={10000}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Chunk Overlap</Label>
              <Input
                type="number"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
                min={0}
                max={5000}
              />
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button variant="outline" onClick={() => setShowUpload(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleUpload}
              disabled={!uploadName || !uploadFile || uploading}
            >
              {uploading ? "Indexing..." : "Upload & Index"}
            </Button>
          </div>
        </div>
      )}

      {loading ? (
        <LoadingSkeleton />
      ) : sources.length === 0 && !showUpload ? (
        <EmptyState
          icon={TextSearch}
          title="No lexical RAG sources"
          subtitle="Upload documents for BM25 keyword search via Meilisearch"
          actionLabel="Upload Document"
          onAction={() => setShowUpload(true)}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {sources.map((source) => (
            <SourceCard
              key={source.id}
              id={source.id}
              name={source.name}
              docCount={source.doc_count}
              status={source.status}
              badge={source.meili_index}
              onDelete={() => handleDelete(source.id)}
              testingId={testingId}
              setTestingId={setTestingId}
              testQuery={testQuery}
              setTestQuery={setTestQuery}
              testResults={testResults}
              setTestResults={setTestResults}
              onTestQuery={() => handleTestQuery(source.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Shared Components ───────────────────────────────────────────────── */

function SourceCard({
  id,
  name,
  docCount,
  status,
  badge,
  onDelete,
  testingId,
  setTestingId,
  testQuery,
  setTestQuery,
  testResults,
  setTestResults,
  onTestQuery,
}: {
  id: string;
  name: string;
  docCount: number;
  status: string;
  badge?: string;
  onDelete: () => void;
  testingId: string | null;
  setTestingId: (v: string | null) => void;
  testQuery: string;
  setTestQuery: (v: string) => void;
  testResults: string | null;
  setTestResults: (v: string | null) => void;
  onTestQuery: () => void;
}) {
  return (
    <div className="bg-card flex flex-col rounded-xl border p-5 shadow-sm">
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-2">
          <div className="bg-primary/10 text-primary flex h-9 w-9 items-center justify-center rounded-lg">
            <FileText className="h-5 w-5" />
          </div>
          <div>
            <h3 className="font-semibold">{name}</h3>
            <p className="text-muted-foreground text-xs">
              {docCount} chunks ·{" "}
              <span
                className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${statusColor(status)}`}
              >
                {status}
              </span>
              {badge && (
                <span className="bg-primary/10 text-primary ml-1.5 inline-block rounded px-1.5 py-0.5 font-mono text-[10px] font-medium">
                  {badge}
                </span>
              )}
            </p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="text-destructive hover:text-destructive"
          onClick={onDelete}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>

      {testingId === id ? (
        <div className="mt-2 flex flex-col gap-2">
          <div className="flex gap-2">
            <Input
              placeholder="Test query..."
              value={testQuery}
              onChange={(e) => setTestQuery(e.target.value)}
              className="flex-1"
              onKeyDown={(e) => e.key === "Enter" && onTestQuery()}
            />
            <Button size="sm" onClick={onTestQuery}>
              <Search className="h-3.5 w-3.5" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setTestingId(null);
                setTestResults(null);
              }}
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
          {testResults && (
            <pre className="bg-muted max-h-40 overflow-auto rounded p-2 text-xs whitespace-pre-wrap">
              {testResults}
            </pre>
          )}
        </div>
      ) : (
        <div className="mt-2 flex gap-2 border-t pt-3">
          <Button
            variant="ghost"
            size="sm"
            className="flex-1"
            onClick={() => {
              setTestingId(id);
              setTestQuery("");
              setTestResults(null);
            }}
          >
            <Search className="mr-1 h-3.5 w-3.5" />
            Test Query
          </Button>
        </div>
      )}
    </div>
  );
}

function EmptyState({
  icon: Icon,
  title,
  subtitle,
  actionLabel,
  onAction,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  subtitle: string;
  actionLabel: string;
  onAction: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <Icon className="text-muted-foreground mb-4 h-16 w-16" />
      <h2 className="mb-2 text-lg font-semibold">{title}</h2>
      <p className="text-muted-foreground mb-6 max-w-md text-sm">{subtitle}</p>
      <Button onClick={onAction}>
        <Plus className="mr-1.5 h-4 w-4" />
        {actionLabel}
      </Button>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {Array.from({ length: 2 }).map((_, i) => (
        <div key={i} className="bg-muted h-32 animate-pulse rounded-lg" />
      ))}
    </div>
  );
}
