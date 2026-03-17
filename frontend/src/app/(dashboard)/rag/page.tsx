"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

// For the hackathon demo we expose only a single, known-good embedding
// option to avoid Bedrock model-id mismatches during live indexing.
const EMBEDDING_MODELS = [
  { value: "titan-embed-text-v2", label: "Titan Embed Text v2" },
] as const;

interface RagSource {
  id: string;
  name: string;
  description: string;
  knowledge_base_id: string;
  region: string;
  s3_bucket: string;
  s3_prefix: string;
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
  s3_bucket: string;
  s3_prefix: string;
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

type Tab = "bedrock" | "local" | "lexical";

export default function RagPage() {
  const [tab, setTab] = useState<Tab>("bedrock");

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">RAG Sources</h1>
          <p className="text-muted-foreground text-sm">
            Manage knowledge bases for your agents
          </p>
        </div>
      </div>

      <div className="flex gap-1 border-b px-6 pt-2">
        {(
          [
            { key: "bedrock", icon: BookOpen, label: "Bedrock Knowledge Bases" },
            { key: "local", icon: HardDrive, label: "Local RAG" },
            { key: "lexical", icon: TextSearch, label: "Lexical RAG" },
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
        {tab === "bedrock" && <BedrockKBTab />}
        {tab === "local" && <LocalRagTab />}
        {tab === "lexical" && <LexicalRagTab />}
      </div>
    </div>
  );
}

function statusColor(status: string) {
  switch (status) {
    case "ready":
      return "text-green-600 bg-green-50 dark:bg-green-950/30 dark:text-green-400";
    case "indexing":
      return "text-yellow-600 bg-yellow-50 dark:bg-yellow-950/30 dark:text-yellow-400";
    case "error":
      return "text-red-600 bg-red-50 dark:bg-red-950/30 dark:text-red-400";
    default:
      return "text-gray-600 bg-gray-50 dark:bg-gray-800 dark:text-gray-400";
  }
}

/* ─── Bedrock Knowledge Bases Tab ────────────────────────────────────── */

function BedrockKBTab() {
  const [sources, setSources] = useState<RagSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [showRegister, setShowRegister] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [regName, setRegName] = useState("");
  const [regDescription, setRegDescription] = useState("");
  const [regKbId, setRegKbId] = useState("");
  const [regRegion, setRegRegion] = useState("");
  const [regS3Bucket, setRegS3Bucket] = useState("");
  const [regS3Prefix, setRegS3Prefix] = useState("");
  const [testQuery, setTestQuery] = useState("");
  const [testResults, setTestResults] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);
  const [addDataSourceId, setAddDataSourceId] = useState<string | null>(null);

  const fetchSources = useCallback(async () => {
    try {
      const res = await fetch(`${BUILDER_API}/api/rag`);
      if (res.ok) setSources(await res.json());
    } catch (e) {
      console.error("Failed to fetch knowledge bases:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSources();
  }, [fetchSources]);

  const handleRegister = async () => {
    if (!regName || !regKbId) return;
    setRegistering(true);
    try {
      const res = await fetch(`${BUILDER_API}/api/rag/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: regName,
          description: regDescription,
          knowledge_base_id: regKbId,
          region: regRegion,
          s3_bucket: regS3Bucket,
          s3_prefix: regS3Prefix,
        }),
      });
      if (res.ok) {
        setShowRegister(false);
        setRegName("");
        setRegDescription("");
        setRegKbId("");
        setRegRegion("");
        setRegS3Bucket("");
        setRegS3Prefix("");
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
    if (!confirm("Remove this knowledge base?")) return;
    try {
      await fetch(`${BUILDER_API}/api/rag/${id}`, { method: "DELETE" });
      fetchSources();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const handleTestQuery = async (sourceId: string) => {
    if (!testQuery.trim()) return;
    setTestResults(null);
    try {
      const res = await fetch(`${BUILDER_API}/api/rag/${sourceId}/query`, {
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
          Semantic search powered by Amazon Bedrock Knowledge Bases
        </p>
        <Button onClick={() => setShowRegister(true)} size="sm">
          <Plus className="mr-1.5 h-4 w-4" />
          Add Knowledge Base
        </Button>
      </div>

      {showRegister && (
        <div className="bg-card mb-6 rounded-lg border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Register Bedrock Knowledge Base
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
                placeholder="HR Policy Docs"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Knowledge Base ID</Label>
              <Input
                value={regKbId}
                onChange={(e) => setRegKbId(e.target.value)}
                placeholder="ABCDEF1234"
              />
            </div>
            <div className="flex flex-col gap-1.5 md:col-span-2">
              <Label>Description</Label>
              <Input
                value={regDescription}
                onChange={(e) => setRegDescription(e.target.value)}
                placeholder="Contains company HR policies, handbooks, and guidelines"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>
                S3 Bucket{" "}
                <span className="text-muted-foreground text-xs font-normal">
                  (for data uploads)
                </span>
              </Label>
              <Input
                value={regS3Bucket}
                onChange={(e) => setRegS3Bucket(e.target.value)}
                placeholder="my-kb-data-bucket"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>
                S3 Prefix{" "}
                <span className="text-muted-foreground text-xs font-normal">
                  (optional folder)
                </span>
              </Label>
              <Input
                value={regS3Prefix}
                onChange={(e) => setRegS3Prefix(e.target.value)}
                placeholder="documents/"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>
                Region{" "}
                <span className="text-muted-foreground text-xs font-normal">
                  (optional)
                </span>
              </Label>
              <Input
                value={regRegion}
                onChange={(e) => setRegRegion(e.target.value)}
                placeholder="us-east-1"
              />
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setShowRegister(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleRegister}
              disabled={!regName || !regKbId || registering}
            >
              {registering ? "Registering..." : "Register"}
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
          title="No knowledge bases registered"
          subtitle="Add your Bedrock Knowledge Base IDs to enable semantic search"
          actionLabel="Add Knowledge Base"
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
  source: RagSource;
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
      const res = await fetch(`${BUILDER_API}/api/rag/generate-metadata`, {
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
        `${BUILDER_API}/api/rag/${sourceId}/upload-data`,
        { method: "POST", body: formData },
      );
      const data = await res.json();
      if (res.ok) {
        setResult(`Uploaded to ${data.s3_uri}${data.metadata_uploaded ? " (with metadata)" : ""}`);
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
    <div className="bg-card mb-6 rounded-lg border p-6 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">
            Add Data to {source.name}
          </h2>
          <p className="text-muted-foreground text-xs">
            Upload to s3://{source.s3_bucket}/{source.s3_prefix}
          </p>
        </div>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {!source.s3_bucket ? (
        <p className="text-sm text-red-500">
          No S3 bucket configured for this KB. Edit the registration to add one.
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
            <Label>Describe content (for AI metadata generation)</Label>
            <div className="flex gap-2">
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="This file contains Q1 2026 financial reports..."
                className="flex-1"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={handleGenerate}
                disabled={!description.trim() || generating}
              >
                <Sparkles className="mr-1 h-3.5 w-3.5" />
                {generating ? "Generating..." : "AI Generate"}
              </Button>
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label>
              Metadata JSON{" "}
              <span className="text-muted-foreground text-xs font-normal">
                (Bedrock KB format)
              </span>
            </Label>
            <textarea
              value={metadataJson}
              onChange={(e) => setMetadataJson(e.target.value)}
              placeholder={'{"metadataAttributes": {"category": "...", "topic": "..."}}'}
              rows={5}
              className="border-input bg-background w-full rounded-md border px-3 py-2 font-mono text-xs"
            />
          </div>

          {result && (
            <p
              className={`text-sm ${result.startsWith("Error") ? "text-red-500" : "text-green-600"}`}
            >
              {result}
            </p>
          )}

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
            >
              <Upload className="mr-1.5 h-4 w-4" />
              {uploading ? "Uploading..." : "Upload to S3"}
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
  source: RagSource;
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
    <div className="bg-card flex flex-col rounded-lg border p-5 shadow-sm">
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
              <span className="ml-1.5 inline-block rounded bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-600 dark:bg-blue-950/30 dark:text-blue-400">
                {source.knowledge_base_id}
              </span>
              {source.region && (
                <span className="ml-1.5 inline-block rounded bg-gray-50 px-1.5 py-0.5 text-[10px] font-medium text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                  {source.region}
                </span>
              )}
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
          {source.s3_bucket && (
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

/* ─── Local RAG Tab ──────────────────────────────────────────────────── */

function LocalRagTab() {
  const [sources, setSources] = useState<LocalRagSource[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadName, setUploadName] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [embeddingModel, setEmbeddingModel] = useState("titan-embed-text-v2");
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);

  const [showS3Import, setShowS3Import] = useState(false);
  const [s3Importing, setS3Importing] = useState(false);
  const [s3Name, setS3Name] = useState("");
  const [s3Bucket, setS3Bucket] = useState("");
  const [s3Prefix, setS3Prefix] = useState("");
  const [s3Region, setS3Region] = useState("");
  const [s3ChunkSize, setS3ChunkSize] = useState(1000);
  const [s3ChunkOverlap, setS3ChunkOverlap] = useState(200);
  const [s3EmbeddingModel, setS3EmbeddingModel] = useState("titan-embed-text-v2");
  const [s3Connections, setS3Connections] = useState<
    { id: string; name: string; bucket_name: string; region: string }[]
  >([]);

  const [testQuery, setTestQuery] = useState("");
  const [testResults, setTestResults] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${BUILDER_API}/api/storage`)
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => setS3Connections(data))
      .catch(() => {});
  }, []);

  const fetchSources = useCallback(async () => {
    try {
      const res = await fetch(`${BUILDER_API}/api/local-rag`);
      if (res.ok) setSources(await res.json());
    } catch (e) {
      console.error("Failed to fetch local RAG sources:", e);
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
      const res = await fetch(`${BUILDER_API}/api/local-rag/upload`, {
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

  const handleS3Import = async () => {
    if (!s3Name || !s3Bucket) return;
    setS3Importing(true);
    try {
      const res = await fetch(`${BUILDER_API}/api/local-rag/s3-import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: s3Name,
          bucket_name: s3Bucket,
          prefix: s3Prefix,
          aws_region: s3Region,
          chunk_size: s3ChunkSize,
          chunk_overlap: s3ChunkOverlap,
          embedding_model: s3EmbeddingModel,
        }),
      });
      if (res.ok) {
        setShowS3Import(false);
        setS3Name("");
        setS3Bucket("");
        setS3Prefix("");
        setS3Region("");
        fetchSources();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "S3 import failed");
      }
    } catch (e) {
      console.error("S3 import failed:", e);
    } finally {
      setS3Importing(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this local RAG source?")) return;
    try {
      await fetch(`${BUILDER_API}/api/local-rag/${id}`, { method: "DELETE" });
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
        `${BUILDER_API}/api/local-rag/${sourceId}/query`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: testQuery, k: 3 }),
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
          Local vector search powered by FAISS + Bedrock Embeddings
        </p>
        <div className="flex gap-2">
          <Button
            onClick={() => {
              setShowS3Import(true);
              setShowUpload(false);
            }}
            size="sm"
            variant="outline"
          >
            <CloudUpload className="mr-1.5 h-4 w-4" />
            Import from S3
          </Button>
          <Button
            onClick={() => {
              setShowUpload(true);
              setShowS3Import(false);
            }}
            size="sm"
          >
            <Upload className="mr-1.5 h-4 w-4" />
            Upload Document
          </Button>
        </div>
      </div>

      {showUpload && (
        <div className="bg-card mb-6 rounded-lg border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Upload Document</h2>
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
                placeholder="Company Docs"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>File (PDF, TXT, MD, CSV, DOCX)</Label>
              <Input
                type="file"
                accept=".pdf,.txt,.md,.csv,.docx"
                onChange={(e) =>
                  setUploadFile(e.target.files?.[0] ?? null)
                }
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

      {showS3Import && (
        <div className="bg-card mb-6 rounded-lg border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Import from AWS S3</h2>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowS3Import(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="flex flex-col gap-1.5">
              <Label>Source Name</Label>
              <Input
                value={s3Name}
                onChange={(e) => setS3Name(e.target.value)}
                placeholder="S3 Company Docs"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>S3 Connection</Label>
              {s3Connections.length > 0 ? (
                <select
                  value={s3Bucket}
                  onChange={(e) => {
                    const conn = s3Connections.find(
                      (c) => c.bucket_name === e.target.value,
                    );
                    setS3Bucket(e.target.value);
                    if (conn?.region) setS3Region(conn.region);
                  }}
                  className="border-input bg-background flex h-9 rounded-md border px-3 py-1 text-sm"
                >
                  <option value="">Select a registered bucket...</option>
                  {s3Connections.map((c) => (
                    <option key={c.id} value={c.bucket_name}>
                      {c.name} (s3://{c.bucket_name})
                    </option>
                  ))}
                </select>
              ) : (
                <Input
                  value={s3Bucket}
                  onChange={(e) => setS3Bucket(e.target.value)}
                  placeholder="my-docs-bucket"
                />
              )}
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Prefix / Folder (optional)</Label>
              <Input
                value={s3Prefix}
                onChange={(e) => setS3Prefix(e.target.value)}
                placeholder="documents/2026/"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Embedding Model</Label>
              <select
                value={s3EmbeddingModel}
                onChange={(e) => setS3EmbeddingModel(e.target.value)}
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
                value={s3ChunkSize}
                onChange={(e) => setS3ChunkSize(Number(e.target.value))}
                min={100}
                max={10000}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Chunk Overlap</Label>
              <Input
                type="number"
                value={s3ChunkOverlap}
                onChange={(e) => setS3ChunkOverlap(Number(e.target.value))}
                min={0}
                max={5000}
              />
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setShowS3Import(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleS3Import}
              disabled={!s3Name || !s3Bucket || s3Importing}
            >
              {s3Importing ? "Importing..." : "Import & Index"}
            </Button>
          </div>
        </div>
      )}

      {loading ? (
        <LoadingSkeleton />
      ) : sources.length === 0 && !showUpload && !showS3Import ? (
        <EmptyState
          icon={HardDrive}
          title="No local RAG sources"
          subtitle="Upload documents or import from S3 for local vector search"
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
                source.source_type === "s3"
                  ? `s3://${source.s3_bucket || ""}`
                  : undefined
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
      const res = await fetch(`${BUILDER_API}/api/lexical-rag`);
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
      const res = await fetch(`${BUILDER_API}/api/lexical-rag/upload`, {
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
      await fetch(`${BUILDER_API}/api/lexical-rag/${id}`, {
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
        `${BUILDER_API}/api/lexical-rag/${sourceId}/query`,
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
          Keyword / BM25 search powered by Meilisearch
        </p>
        <Button onClick={() => setShowUpload(true)} size="sm">
          <Upload className="mr-1.5 h-4 w-4" />
          Upload Document
        </Button>
      </div>

      {showUpload && (
        <div className="bg-card mb-6 rounded-lg border p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Upload for Lexical Indexing
            </h2>
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
                placeholder="Company Docs"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>File (PDF, TXT, MD, CSV, DOCX)</Label>
              <Input
                type="file"
                accept=".pdf,.txt,.md,.csv,.docx"
                onChange={(e) =>
                  setUploadFile(e.target.files?.[0] ?? null)
                }
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
          subtitle="Upload documents for keyword search via Meilisearch"
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
    <div className="bg-card flex flex-col rounded-lg border p-5 shadow-sm">
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
                <span className="ml-1.5 inline-block rounded bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-600 dark:bg-blue-950/30 dark:text-blue-400">
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
    <div className="flex flex-col items-center justify-center py-20">
      <Icon className="text-muted-foreground mb-4 h-16 w-16" />
      <h2 className="mb-2 text-lg font-semibold">{title}</h2>
      <p className="text-muted-foreground mb-6 text-sm">{subtitle}</p>
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
