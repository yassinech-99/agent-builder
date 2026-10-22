"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { getBuilderApiUrl } from "@/lib/runtime-urls";
import {
  HardDrive,
  Plus,
  Trash2,
  X,
  CheckCircle2,
  XCircle,
  Loader2,
} from "lucide-react";

interface StorageConnection {
  id: string;
  name: string;
  endpoint_url: string;
  bucket_name: string;
  region: string;
  access_key_masked: string;
  secret_key_masked: string;
  has_credentials: boolean;
  secure: boolean;
  created_at: string;
}

export default function StoragePage() {
  const [connections, setConnections] = useState<StorageConnection[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testStatus, setTestStatus] = useState<
    Record<string, { ok: boolean; message: string } | "loading">
  >({});
  const [form, setForm] = useState({
    name: "",
    endpoint_url: "http://localhost:9000",
    bucket_name: "",
    region: "us-east-1",
    access_key: "",
    secret_key: "",
    skip_validation: false,
  });

  const fetchConnections = useCallback(async () => {
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/storage`);
      if (res.ok) setConnections(await res.json());
    } catch (e) {
      console.error("Failed to fetch storage connections:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchConnections();
  }, [fetchConnections]);

  const resetForm = () => {
    setForm({
      name: "",
      endpoint_url: "http://localhost:9000",
      bucket_name: "",
      region: "us-east-1",
      access_key: "",
      secret_key: "",
      skip_validation: false,
    });
    setShowForm(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/storage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (res.ok) {
        resetForm();
        fetchConnections();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "Failed to add storage bucket connection");
      }
    } catch (e) {
      console.error("Failed to save:", e);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Remove this object storage connection?")) return;
    try {
      await fetch(`${getBuilderApiUrl()}/api/storage/${id}`, { method: "DELETE" });
      fetchConnections();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const handleTest = async (id: string) => {
    setTestStatus((prev) => ({ ...prev, [id]: "loading" }));
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/storage/${id}/test`, {
        method: "POST",
      });
      const data = await res.json();
      setTestStatus((prev) => ({ ...prev, [id]: data }));
    } catch {
      setTestStatus((prev) => ({
        ...prev,
        [id]: { ok: false, message: "Network error" },
      }));
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Object Storage</h1>
          <p className="text-muted-foreground text-sm">
            Connect RustFS, Cloudflare R2, SeaweedFS, Ceph, or any S3-compatible bucket
          </p>
        </div>
        <Button
          onClick={() => {
            resetForm();
            setShowForm(true);
          }}
        >
          <Plus className="mr-1.5 h-4 w-4" />
          Add Bucket Connection
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {showForm && (
          <div className="bg-card mb-6 rounded-xl border p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Add S3-Compatible Bucket</h2>
              <Button variant="ghost" size="icon" onClick={resetForm}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <Label>Display Name</Label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  placeholder="RustFS Knowledge Corpus"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Endpoint URL</Label>
                <Input
                  value={form.endpoint_url}
                  onChange={(e) =>
                    setForm({ ...form, endpoint_url: e.target.value })
                  }
                  placeholder="http://localhost:9000"
                  className="font-mono text-xs"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Bucket Name</Label>
                <Input
                  value={form.bucket_name}
                  onChange={(e) =>
                    setForm({ ...form, bucket_name: e.target.value })
                  }
                  placeholder="nimagents-docs"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Region (optional)</Label>
                <Input
                  value={form.region}
                  onChange={(e) => setForm({ ...form, region: e.target.value })}
                  placeholder="us-east-1"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Access Key</Label>
                <Input
                  type="password"
                  value={form.access_key}
                  onChange={(e) =>
                    setForm({ ...form, access_key: e.target.value })
                  }
                  placeholder="nimagentsadmin (falls back to RUSTFS_ACCESS_KEY)"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Secret Key</Label>
                <Input
                  type="password"
                  value={form.secret_key}
                  onChange={(e) =>
                    setForm({ ...form, secret_key: e.target.value })
                  }
                  placeholder="•••••••• (falls back to RUSTFS_SECRET_KEY)"
                />
              </div>
              <div className="md:col-span-2">
                <label className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={form.skip_validation}
                    onChange={(e) =>
                      setForm({ ...form, skip_validation: e.target.checked })
                    }
                  />
                  Save without immediate live bucket reachability check
                </label>
              </div>
            </div>
            <div className="mt-4 flex justify-end gap-2">
              <Button variant="outline" onClick={resetForm}>
                Cancel
              </Button>
              <Button
                onClick={handleSave}
                disabled={!form.name || !form.bucket_name || saving}
              >
                {saving ? "Connecting..." : "Save Connection"}
              </Button>
            </div>
          </div>
        )}

        {loading ? (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="bg-muted h-32 animate-pulse rounded-lg" />
            ))}
          </div>
        ) : connections.length === 0 && !showForm ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <HardDrive className="text-muted-foreground mb-4 h-16 w-16" />
            <h2 className="mb-2 text-lg font-semibold">
              No object storage buckets registered
            </h2>
            <p className="text-muted-foreground mb-6 max-w-md text-sm">
              Connect a RustFS, Cloudflare R2, or S3-compatible bucket for document
              ingestion and large tool artifacts.
            </p>
            <Button onClick={() => setShowForm(true)}>
              <Plus className="mr-1.5 h-4 w-4" />
              Add Bucket Connection
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {connections.map((c) => {
              const ts = testStatus[c.id];
              return (
                <div
                  key={c.id}
                  className="bg-card flex flex-col rounded-xl border p-5 shadow-sm"
                >
                  <div className="mb-3 flex items-start justify-between">
                    <div className="flex items-center gap-2.5">
                      <div className="bg-primary/10 text-primary flex h-9 w-9 items-center justify-center rounded-lg">
                        <HardDrive className="h-5 w-5" />
                      </div>
                      <div className="min-w-0">
                        <h3 className="truncate font-semibold">{c.name}</h3>
                        <p className="text-muted-foreground truncate font-mono text-xs">
                          s3://{c.bucket_name}
                        </p>
                      </div>
                    </div>
                  </div>
                  <p className="text-muted-foreground mb-4 flex-1 text-xs">
                    {c.endpoint_url}
                    {c.has_credentials
                      ? ` · Key: ${c.access_key_masked}`
                      : " · Env credentials"}
                  </p>

                  {ts && ts !== "loading" && (
                    <div
                      className={`mb-3 flex items-center gap-1.5 rounded-md px-3 py-2 text-xs ${
                        ts.ok
                          ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-400"
                          : "bg-red-50 text-red-700 dark:bg-red-950/30 dark:text-red-400"
                      }`}
                    >
                      {ts.ok ? (
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 shrink-0" />
                      )}
                      <span className="truncate">{ts.message}</span>
                    </div>
                  )}

                  <div className="flex items-center gap-1 border-t pt-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-1"
                      onClick={() => handleTest(c.id)}
                      disabled={ts === "loading"}
                    >
                      {ts === "loading" ? (
                        <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
                      ) : null}
                      Test Bucket
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-destructive hover:text-destructive"
                      onClick={() => handleDelete(c.id)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
