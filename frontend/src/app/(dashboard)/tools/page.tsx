"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { getBuilderApiUrl, getTenantHeaders } from "@/lib/runtime-urls";
import {
  Search,
  Wrench,
  Sparkles,
  Play,
  FileJson,
  Trash2,
  ShieldAlert,
  Terminal,
  Loader2,
  Zap,
  CheckCircle2,
} from "lucide-react";
import { toast } from "sonner";

interface ToolSpecItem {
  id: string;
  namespaced_id: string;
  name: string;
  description: string;
  category: string;
  source: string;
  is_toolkit?: boolean;
  input_examples?: Record<string, unknown>[];
  defer_loading: boolean;
  requires_approval: boolean;
  side_effect_level: string;
  token_cost: number;
  score?: number;
  telemetry?: {
    calls: number;
    errors: number;
    cache_hits: number;
    p50_ms: number;
    p95_ms: number;
    circuit_open: boolean;
  };
}

interface BudgetSummary {
  total_tools: number;
  pinned_tools: number;
  deferred_tools: number;
  eager_schema_tokens: number;
  deferred_startup_tokens: number;
  saved_tokens: number;
  saved_percent: number;
}

const SAMPLE_OPENAPI_SPEC = JSON.stringify(
  {
    openapi: "3.1.0",
    info: { title: "BillingOps", version: "1.0.0" },
    servers: [{ url: "https://api.billing-demo.internal" }],
    paths: {
      "/v1/invoices": {
        get: {
          operationId: "list_invoices",
          summary: "List customer invoices filtered by status and customer ID",
          parameters: [
            { name: "customer_id", in: "query", required: true, schema: { type: "string", example: "cus_9042" } },
            { name: "status", in: "query", required: false, schema: { type: "string", example: "paid" } },
          ],
        },
      },
      "/v1/invoices/{invoice_id}": {
        delete: {
          operationId: "void_invoice",
          summary: "Void and cancel an unpaid customer invoice",
          parameters: [
            { name: "invoice_id", in: "path", required: true, schema: { type: "string", example: "inv_1088" } },
          ],
        },
      },
    },
  },
  null,
  2,
);

const DEFAULT_ORCHESTRATE_SCRIPT = `# Parallel Zero-Trust Sandbox Hand Demo (runs via Unix socket RPC!)
results = await asyncio.gather(
    tools.calculator(expression="2**12"),
    tools.calculator(expression="sqrt(1764)"),
    tools.current_datetime(),
)
print(json.dumps({"powers_of_two": results[0], "root": results[1], "utc_now": results[2]}, indent=2))`;

export default function ToolsRegistryPage() {
  const [tools, setTools] = useState<ToolSpecItem[]>([]);
  const [budget, setBudget] = useState<BudgetSummary | null>(null);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"hybrid" | "regex" | "select">("hybrid");
  const [searchLatencyMs, setSearchLatencyMs] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  const [selectedTool, setSelectedTool] = useState<ToolSpecItem | null>(null);
  const [testArgsText, setTestArgsText] = useState('{"expression": "sqrt(144) + 2**8"}');
  const [testOutput, setTestOutput] = useState("");
  const [testingTool, setTestingTool] = useState(false);

  const [showOpenApiModal, setShowOpenApiModal] = useState(false);
  const [openApiText, setOpenApiText] = useState(SAMPLE_OPENAPI_SPEC);
  const [openApiAuth, setOpenApiAuth] = useState("");
  const [importingSpec, setImportingSpec] = useState(false);

  const [scriptCode, setScriptCode] = useState(DEFAULT_ORCHESTRATE_SCRIPT);
  const [scriptOutput, setScriptOutput] = useState("");
  const [runningScript, setRunningScript] = useState(false);

  const loadCatalog = useCallback(async () => {
    setLoading(true);
    try {
      const [catRes, budRes] = await Promise.all([
        fetch(`${getBuilderApiUrl()}/api/tools/catalog`, { headers: getTenantHeaders(), cache: "no-store" }),
        fetch(`${getBuilderApiUrl()}/api/tools/budget`, { headers: getTenantHeaders(), cache: "no-store" }),
      ]);
      if (catRes.ok) {
        const list = (await catRes.json()) as ToolSpecItem[];
        setTools(list);
        if (!selectedTool && list.length > 0) {
          setSelectedTool(list[0]);
        }
      }
      if (budRes.ok) {
        setBudget((await budRes.json()) as BudgetSummary);
      }
    } catch {
      toast.error("Could not load unified tool registry");
    } finally {
      setLoading(false);
    }
  }, [selectedTool]);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim()) {
      setSearchLatencyMs(null);
      await loadCatalog();
      return;
    }
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/tools/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getTenantHeaders() },
        body: JSON.stringify({ query: query.trim(), mode, limit: 12 }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.detail || "Search failed");
        return;
      }
      setTools(data.results || []);
      setSearchLatencyMs(data.latency_ms);
      if (data.budget) setBudget(data.budget);
    } catch {
      toast.error("Tool search request failed");
    }
  };

  const handleRunSingleTool = async () => {
    if (!selectedTool) return;
    setTestingTool(true);
    setTestOutput("");
    try {
      const parsedArgs = JSON.parse(testArgsText || "{}");
      const res = await fetch(`${getBuilderApiUrl()}/api/tools/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getTenantHeaders() },
        body: JSON.stringify({ tool_id: selectedTool.id, args: parsedArgs }),
      });
      const data = await res.json();
      if (!res.ok) {
        setTestOutput(`Error: ${data.detail || "Execution failed"}`);
      } else {
        setTestOutput(`// Completed in ${data.latency_ms}ms\n${data.output}`);
        void loadCatalog();
      }
    } catch (err: any) {
      setTestOutput(`Invalid JSON arguments: ${err?.message || String(err)}`);
    } finally {
      setTestingTool(false);
    }
  };

  const handleImportOpenApi = async () => {
    setImportingSpec(true);
    try {
      const parsedSpec = JSON.parse(openApiText);
      const res = await fetch(`${getBuilderApiUrl()}/api/tools/import-openapi`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getTenantHeaders() },
        body: JSON.stringify({ spec: parsedSpec, auth_header: openApiAuth }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.detail || "OpenAPI import failed");
        return;
      }
      toast.success(`Imported ${data.imported_count} deferred OpenAPI tools`);
      setShowOpenApiModal(false);
      setQuery("");
      await loadCatalog();
    } catch (err: any) {
      toast.error(`Invalid OpenAPI JSON: ${err?.message || "parse error"}`);
    } finally {
      setImportingSpec(false);
    }
  };

  const handleDeleteCustom = async (id: string) => {
    const res = await fetch(`${getBuilderApiUrl()}/api/tools/custom/${id}`, {
      method: "DELETE",
      headers: getTenantHeaders(),
    });
    if (res.ok) {
      toast.success("Custom tool deleted");
      if (selectedTool?.id === id) setSelectedTool(null);
      await loadCatalog();
    }
  };

  const handleRunOrchestrator = async () => {
    setRunningScript(true);
    setScriptOutput("");
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/tools/orchestrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...getTenantHeaders() },
        body: JSON.stringify({
          python_code: scriptCode,
          authorized_tool_ids: ["calculator", "current_datetime", "json_explorer"],
        }),
      });
      const data = await res.json();
      setScriptOutput(
        `// Exit: ${data.exit_code} · Host RPC Invocations: ${data.rpc_calls}\n` +
          (data.stdout || data.stderr || "(no output)"),
      );
    } catch {
      setScriptOutput("Sandbox orchestration request failed.");
    } finally {
      setRunningScript(false);
    }
  };

  const badgeColor = (level: string) => {
    if (level === "destructive") return "bg-red-500/15 text-red-600 dark:text-red-400";
    if (level === "write") return "bg-amber-500/15 text-amber-600 dark:text-amber-400";
    return "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400";
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Unified Tool Registry &amp; Sandbox Hands
          </h1>
          <p className="text-muted-foreground text-sm">
            Deferred <code>tool_search</code> (BM25 + Cosine RRF), OpenAPI 3.x importer &amp; Zero-Trust{" "}
            <code>orchestrate_tools</code>
          </p>
        </div>
        <Button onClick={() => setShowOpenApiModal((v) => !v)} variant="default" size="sm">
          <FileJson className="mr-1.5 h-4 w-4" />
          Import OpenAPI 3.x Spec
        </Button>
      </div>

      <div className="flex-1 space-y-6 overflow-auto p-6">
        {/* Token Budget Savings Strip */}
        {budget && (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
            <div className="bg-card rounded-xl border p-4 shadow-xs">
              <p className="text-muted-foreground text-xs font-medium">Registered Tools</p>
              <p className="mt-1 font-mono text-xl font-bold">{budget.total_tools}</p>
              <p className="text-muted-foreground text-xs">
                {budget.pinned_tools} pinned · {budget.deferred_tools} deferred
              </p>
            </div>
            <div className="bg-card rounded-xl border p-4 shadow-xs">
              <p className="text-muted-foreground text-xs font-medium">Eager Schema Cost</p>
              <p className="mt-1 font-mono text-xl font-bold">
                {budget.eager_schema_tokens.toLocaleString()} tok
              </p>
              <p className="text-muted-foreground text-xs">If all schemas loaded upfront</p>
            </div>
            <div className="bg-card rounded-xl border p-4 shadow-xs">
              <p className="text-muted-foreground text-xs font-medium">Deferred Startup Cost</p>
              <p className="mt-1 font-mono text-xl font-bold text-emerald-600 dark:text-emerald-400">
                {budget.deferred_startup_tokens.toLocaleString()} tok
              </p>
              <p className="text-muted-foreground text-xs">Pinned core + namespace directory</p>
            </div>
            <div className="bg-card rounded-xl border p-4 shadow-xs">
              <p className="text-muted-foreground text-xs font-medium">Context Window Saved</p>
              <p className="mt-1 flex items-center gap-1.5 font-mono text-xl font-bold text-emerald-600 dark:text-emerald-400">
                <Sparkles className="h-4 w-4" />
                {budget.saved_percent}%
              </p>
              <p className="text-muted-foreground text-xs">
                -{budget.saved_tokens.toLocaleString()} tokens/turn
              </p>
            </div>
          </div>
        )}

        {/* OpenAPI Importer Panel */}
        {showOpenApiModal && (
          <div className="bg-card space-y-4 rounded-xl border p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold">Import OpenAPI 3.0 / 3.1 Specification</h2>
              <Button variant="ghost" size="sm" onClick={() => setShowOpenApiModal(false)}>
                Close
              </Button>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <Label>Optional Authorization Header (Vault-Masked)</Label>
                <Input
                  type="password"
                  placeholder="Bearer sk_live_..."
                  value={openApiAuth}
                  onChange={(e) => setOpenApiAuth(e.target.value)}
                  className="font-mono text-xs"
                />
              </div>
              <div className="flex items-end">
                <Button onClick={handleImportOpenApi} disabled={importingSpec} className="w-full">
                  {importingSpec && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
                  Import Endpoints as Deferred Tenant Tools
                </Button>
              </div>
            </div>
            <Textarea
              rows={9}
              value={openApiText}
              onChange={(e) => setOpenApiText(e.target.value)}
              className="font-mono text-xs"
            />
          </div>
        )}

        {/* Search + Master/Detail Split */}
        <form onSubmit={handleSearch} className="flex flex-wrap gap-2">
          <div className="relative min-w-[240px] flex-1">
            <Search className="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Test sub-5ms ToolSearch (e.g. 'calculate square root', 'academic papers', 'invoice')..."
              className="pl-9 text-sm"
            />
          </div>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as "hybrid" | "regex" | "select")}
            className="border-input bg-background rounded-md border px-3 text-xs"
          >
            <option value="hybrid">Hybrid (BM25 + Cosine RRF)</option>
            <option value="regex">Regex Pattern</option>
            <option value="select">Exact ID Select</option>
          </select>
          <Button type="submit" variant="secondary">
            <Zap className="mr-1.5 h-3.5 w-3.5" />
            Search
          </Button>
          {query && (
            <Button
              type="button"
              variant="ghost"
              onClick={() => {
                setQuery("");
                setSearchLatencyMs(null);
                void loadCatalog();
              }}
            >
              Reset
            </Button>
          )}
        </form>

        {searchLatencyMs !== null && (
          <p className="text-muted-foreground text-xs">
            <code>tool_search</code> ranked top matches in <strong>{searchLatencyMs} ms</strong>
          </p>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Catalog List */}
          <div className="space-y-2 lg:col-span-2">
            {loading ? (
              <div className="text-muted-foreground rounded-xl border p-8 text-center text-sm">
                Loading unified tool catalog...
              </div>
            ) : (
              <div className="grid gap-2.5 sm:grid-cols-2">
                {tools.map((t) => {
                  const active = selectedTool?.id === t.id;
                  return (
                    <div
                      key={t.id}
                      onClick={() => {
                        setSelectedTool(t);
                        if (t.input_examples && t.input_examples.length > 0) {
                          setTestArgsText(JSON.stringify(t.input_examples[0], null, 2));
                        } else {
                          setTestArgsText("{}");
                        }
                        setTestOutput("");
                      }}
                      className={`cursor-pointer rounded-xl border p-3.5 transition-all ${
                        active ? "border-primary bg-primary/5 shadow-xs" : "hover:border-primary/40"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <Wrench className="text-muted-foreground h-3.5 w-3.5 shrink-0" />
                            <span className="truncate text-sm font-semibold">{t.name}</span>
                          </div>
                          <p className="text-muted-foreground truncate font-mono text-[11px]">
                            {t.namespaced_id}
                          </p>
                        </div>
                        <span className="bg-muted shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px]">
                          {t.token_cost} tok
                        </span>
                      </div>

                      <p className="text-muted-foreground mt-2 line-clamp-2 text-xs">
                        {t.description}
                      </p>

                      <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${badgeColor(
                            t.side_effect_level,
                          )}`}
                        >
                          {t.side_effect_level}
                        </span>
                        <span className="bg-muted rounded px-1.5 py-0.5 text-[10px]">
                          {t.defer_loading ? "deferred" : "pinned"}
                        </span>
                        {t.requires_approval && (
                          <span className="flex items-center gap-0.5 rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">
                            <ShieldAlert className="h-2.5 w-2.5" />
                            HITL
                          </span>
                        )}
                        {t.source === "openapi" && (
                          <button
                            type="button"
                            onClick={(ev) => {
                              ev.stopPropagation();
                              void handleDeleteCustom(t.id);
                            }}
                            className="text-destructive ml-auto hover:opacity-80"
                            title="Delete custom tool"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right Live Inspector Drawer */}
          <div className="bg-card h-fit space-y-4 rounded-xl border p-5 shadow-xs">
            {selectedTool ? (
              <>
                <div>
                  <span className="text-muted-foreground font-mono text-xs">
                    {selectedTool.namespaced_id}
                  </span>
                  <h3 className="text-lg font-semibold">{selectedTool.name}</h3>
                  <p className="text-muted-foreground mt-1 text-xs">{selectedTool.description}</p>
                </div>

                <div className="grid grid-cols-3 gap-2 rounded-lg border p-2.5 text-center text-xs">
                  <div>
                    <div className="text-muted-foreground">Calls</div>
                    <div className="font-mono font-bold">{selectedTool.telemetry?.calls ?? 0}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Cache Hits</div>
                    <div className="font-mono font-bold">
                      {selectedTool.telemetry?.cache_hits ?? 0}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">p95 Latency</div>
                    <div className="font-mono font-bold">
                      {selectedTool.telemetry?.p95_ms ?? 0}ms
                    </div>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <Label className="text-xs">JSON Input Arguments</Label>
                  <Textarea
                    rows={4}
                    value={testArgsText}
                    onChange={(e) => setTestArgsText(e.target.value)}
                    className="font-mono text-xs"
                  />
                </div>

                <Button
                  onClick={handleRunSingleTool}
                  disabled={testingTool}
                  className="w-full"
                  size="sm"
                >
                  {testingTool ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Play className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  Execute Live via ToolExecutor
                </Button>

                {testOutput && (
                  <pre className="bg-muted max-h-48 overflow-auto rounded-lg p-3 font-mono text-xs whitespace-pre-wrap">
                    {testOutput}
                  </pre>
                )}
              </>
            ) : (
              <p className="text-muted-foreground text-xs">Select a tool to inspect and test.</p>
            )}
          </div>
        </div>

        {/* Zero-Trust Programmatic `orchestrate_tools` Playground */}
        <div className="bg-card space-y-4 rounded-xl border p-5 shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Terminal className="h-5 w-5 text-emerald-500" />
              <div>
                <h2 className="text-base font-semibold">
                  Zero-Trust Sandbox Hands Playground (<code>orchestrate_tools</code>)
                </h2>
                <p className="text-muted-foreground text-xs">
                  Executes parallel async Python tool fan-out inside an isolated, secret-scrubbed subprocess over Unix socket RPC.
                </p>
              </div>
            </div>
            <Button onClick={handleRunOrchestrator} disabled={runningScript} size="sm">
              {runningScript ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <CheckCircle2 className="mr-1.5 h-3.5 w-3.5" />
              )}
              Run in Sandbox Hand
            </Button>
          </div>

          <Textarea
            rows={6}
            value={scriptCode}
            onChange={(e) => setScriptCode(e.target.value)}
            className="font-mono text-xs"
          />

          {scriptOutput && (
            <pre className="bg-muted overflow-auto rounded-lg p-3 font-mono text-xs whitespace-pre-wrap">
              {scriptOutput}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
