"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { HardDrive, Plus, Trash2, X, CheckCircle2, XCircle, Loader2 } from "lucide-react";

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

interface S3Connection {
  id: string;
  name: string;
  bucket_name: string;
  region: string;
  access_key_id: string;
  secret_access_key: string;
  created_at: string;
}

export default function StoragePage() {
  const [connections, setConnections] = useState<S3Connection[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testStatus, setTestStatus] = useState<Record<string, { ok: boolean; message: string } | "loading">>({});
  const [form, setForm] = useState({
    name: "",
    bucket_name: "",
    region: "",
    access_key_id: "",
    secret_access_key: "",
  });

  const fetchConnections = useCallback(async () => {
    try {
      const res = await fetch(`${BUILDER_API}/api/storage`);
      if (res.ok) setConnections(await res.json());
    } catch (e) {
      console.error("Failed to fetch S3 connections:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchConnections();
  }, [fetchConnections]);

  const resetForm = () => {
    setForm({ name: "", bucket_name: "", region: "", access_key_id: "", secret_access_key: "" });
    setShowForm(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await fetch(`${BUILDER_API}/api/storage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (res.ok) {
        resetForm();
        fetchConnections();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || "Failed to add S3 connection");
      }
    } catch (e) {
      console.error("Failed to save:", e);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Remove this S3 connection?")) return;
    try {
      await fetch(`${BUILDER_API}/api/storage/${id}`, { method: "DELETE" });
      fetchConnections();
    } catch (e) {
      console.error("Failed to delete:", e);
    }
  };

  const handleTest = async (id: string) => {
    setTestStatus((prev) => ({ ...prev, [id]: "loading" }));
    try {
      const res = await fetch(`${BUILDER_API}/api/storage/${id}/test`, { method: "POST" });
      const data = await res.json();
      setTestStatus((prev) => ({ ...prev, [id]: data }));
    } catch {
      setTestStatus((prev) => ({ ...prev, [id]: { ok: false, message: "Network error" } }));
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Storage</h1>
          <p className="text-muted-foreground text-sm">
            Manage Amazon S3 bucket connections
          </p>
        </div>
        <Button onClick={() => { resetForm(); setShowForm(true); }}>
          <Plus className="mr-1.5 h-4 w-4" />
          Add S3 Connection
        </Button>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {showForm && (
          <div className="bg-card mb-6 rounded-lg border p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Add S3 Connection</h2>
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
                  placeholder="Production Data Bucket"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Bucket Name</Label>
                <Input
                  value={form.bucket_name}
                  onChange={(e) => setForm({ ...form, bucket_name: e.target.value })}
                  placeholder="my-company-bucket"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>AWS Region (optional)</Label>
                <Input
                  value={form.region}
                  onChange={(e) => setForm({ ...form, region: e.target.value })}
                  placeholder="us-east-1 (defaults to env)"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Access Key ID (optional)</Label>
                <Input
                  type="password"
                  value={form.access_key_id}
                  onChange={(e) => setForm({ ...form, access_key_id: e.target.value })}
                  placeholder="AKIA... (falls back to env)"
                />
              </div>
              <div className="flex flex-col gap-1.5 md:col-span-2">
                <Label>Secret Access Key (optional)</Label>
                <Input
                  type="password"
                  value={form.secret_access_key}
                  onChange={(e) => setForm({ ...form, secret_access_key: e.target.value })}
                  placeholder="wJalr... (falls back to env)"
                />
              </div>
            </div>
            <div className="mt-4 flex justify-end gap-2">
              <Button variant="outline" onClick={resetForm}>Cancel</Button>
              <Button onClick={handleSave} disabled={!form.name || !form.bucket_name || saving}>
                {saving ? "Connecting..." : "Add Connection"}
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
          <div className="flex flex-col items-center justify-center py-20">
            <HardDrive className="text-muted-foreground mb-4 h-16 w-16" />
            <h2 className="mb-2 text-lg font-semibold">No S3 connections</h2>
            <p className="text-muted-foreground mb-6 text-sm">
              Register an Amazon S3 bucket to use with RAG and agents
            </p>
            <Button onClick={() => setShowForm(true)}>
              <Plus className="mr-1.5 h-4 w-4" />
              Add S3 Connection
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {connections.map((c) => {
              const ts = testStatus[c.id];
              return (
                <div key={c.id} className="bg-card flex flex-col rounded-lg border p-5 shadow-sm">
                  <div className="mb-3 flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <div className="bg-primary/10 text-primary flex h-9 w-9 items-center justify-center rounded-lg">
                        <HardDrive className="h-5 w-5" />
                      </div>
                      <div>
                        <h3 className="font-semibold">{c.name}</h3>
                        <p className="text-muted-foreground text-xs">
                          s3://{c.bucket_name}
                        </p>
                      </div>
                    </div>
                  </div>
                  <p className="text-muted-foreground mb-4 flex-1 text-xs">
                    {c.region || "Default region"}
                    {c.access_key_id ? " · Custom credentials" : " · Env credentials"}
                  </p>

                  {ts && ts !== "loading" && (
                    <div className={`mb-3 flex items-center gap-1.5 rounded-md px-3 py-2 text-xs ${
                      ts.ok
                        ? "bg-green-50 text-green-700 dark:bg-green-950/30 dark:text-green-400"
                        : "bg-red-50 text-red-700 dark:bg-red-950/30 dark:text-red-400"
                    }`}>
                      {ts.ok ? <CheckCircle2 className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
                      {ts.message}
                    </div>
                  )}

                  <div className="flex items-center gap-1 border-t pt-3">
                    <Button variant="ghost" size="sm" className="flex-1" onClick={() => handleTest(c.id)} disabled={ts === "loading"}>
                      {ts === "loading" ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : null}
                      Test
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
