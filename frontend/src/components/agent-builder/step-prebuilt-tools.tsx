"use client";

import { useEffect, useState, useCallback } from "react";
import { useAgentConfig, ToolLoadingMode } from "@/providers/AgentConfig";
import { getBuilderApiUrl, getTenantHeaders } from "@/lib/runtime-urls";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { ChevronDown, ChevronUp, Search, ShieldAlert } from "lucide-react";

interface ToolConfigField {
  key: string;
  label: string;
  type: "text" | "password";
  hint?: string;
}

interface CatalogTool {
  id: string;
  name: string;
  description: string;
  category: string;
  required_config: ToolConfigField[];
  is_toolkit?: boolean;
  token_cost?: number;
  requires_approval?: boolean;
}

const CATEGORY_LABELS: Record<string, string> = {
  search: "Search",
  knowledge: "Knowledge & Research",
  computation: "Computation",
  code: "Code & Sandbox",
  web: "Web / HTTP",
  email: "Email & Communication",
  file: "File & Data",
  browser: "Browser Automation",
  project: "Project Management",
  creative: "Creative / Generative",
  data: "Data APIs",
  openapi: "Custom OpenAPI & REST",
  utility: "Utility",
};

const CATEGORY_ORDER = [
  "openapi",
  "search",
  "knowledge",
  "computation",
  "code",
  "web",
  "email",
  "file",
  "browser",
  "project",
  "creative",
  "data",
  "utility",
];

export function StepPrebuiltTools() {
  const { config, setConfig } = useAgentConfig();
  const [catalog, setCatalog] = useState<CatalogTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [expandedTool, setExpandedTool] = useState<string | null>(null);

  const fetchCatalog = useCallback(async () => {
    try {
      const res = await fetch(`${getBuilderApiUrl()}/api/tools/catalog`, {
        headers: getTenantHeaders(),
      });
      if (res.ok) setCatalog(await res.json());
    } catch {
      /* API not available — will show empty */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCatalog();
  }, [fetchCatalog]);

  const selected = new Set(config.enabled_prebuilt_tools);
  const approvalSet = new Set(config.approval_required_tools ?? []);

  const toggle = (id: string) => {
    setConfig((prev) => {
      const next = new Set(prev.enabled_prebuilt_tools);
      const nextApproval = new Set(prev.approval_required_tools ?? []);
      if (next.has(id)) {
        next.delete(id);
        nextApproval.delete(id);
        const { [id]: _, ...remaining } = prev.tool_configs;
        return {
          ...prev,
          enabled_prebuilt_tools: Array.from(next),
          approval_required_tools: Array.from(nextApproval),
          tool_configs: remaining,
        };
      } else {
        next.add(id);
        return { ...prev, enabled_prebuilt_tools: Array.from(next) };
      }
    });
  };

  const toggleHitl = (id: string) => {
    setConfig((prev) => {
      const nextApproval = new Set(prev.approval_required_tools ?? []);
      if (nextApproval.has(id)) nextApproval.delete(id);
      else nextApproval.add(id);
      return { ...prev, approval_required_tools: Array.from(nextApproval) };
    });
  };

  const setToolConfig = (toolId: string, key: string, value: string) => {
    setConfig((prev) => ({
      ...prev,
      tool_configs: {
        ...prev.tool_configs,
        [toolId]: { ...prev.tool_configs[toolId], [key]: value },
      },
    }));
  };

  const filteredCatalog = filter
    ? catalog.filter(
        (t) =>
          t.name.toLowerCase().includes(filter.toLowerCase()) ||
          t.description.toLowerCase().includes(filter.toLowerCase()),
      )
    : catalog;

  const grouped = CATEGORY_ORDER.reduce<Record<string, CatalogTool[]>>(
    (acc, cat) => {
      const tools = filteredCatalog.filter((t) => t.category === cat);
      if (tools.length > 0) acc[cat] = tools;
      return acc;
    },
    {},
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-muted-foreground text-sm">Loading tools...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Phase 3 Execution & Deferred Loading Controls */}
      <div className="bg-muted/40 grid grid-cols-1 gap-3 rounded-lg border p-3 sm:grid-cols-2">
        <div className="space-y-1">
          <Label className="text-xs">Schema Context Loading Mode</Label>
          <select
            value={config.tool_loading_mode ?? "auto"}
            onChange={(e) =>
              setConfig((prev) => ({
                ...prev,
                tool_loading_mode: e.target.value as ToolLoadingMode,
              }))
            }
            className="border-input bg-background flex h-8 w-full rounded-md border px-2.5 text-xs"
          >
            <option value="auto">Auto Deferred (&gt;6 tools / 1.2K tok)</option>
            <option value="deferred">Always Deferred (85%+ Token Savings)</option>
            <option value="eager">Eager (Bind All Schemas Upfront)</option>
          </select>
        </div>

        <div className="flex items-center justify-between gap-2 rounded-md border px-3 py-1.5">
          <div>
            <p className="text-xs font-medium">Sandbox <code>orchestrate_tools</code></p>
            <p className="text-muted-foreground text-[11px]">
              Parallel Python RPC fan-out
            </p>
          </div>
          <Switch
            checked={config.enable_programmatic_tools ?? true}
            onCheckedChange={(val) =>
              setConfig((prev) => ({ ...prev, enable_programmatic_tools: val }))
            }
          />
        </div>
      </div>

      <div className="relative">
        <Search className="text-muted-foreground absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2" />
        <Input
          placeholder="Filter built-in & imported OpenAPI tools..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="pl-9"
        />
      </div>

      <div className="max-h-[360px] space-y-5 overflow-y-auto pr-1">
        {Object.entries(grouped).map(([cat, tools]) => (
          <div key={cat}>
            <h3 className="text-muted-foreground mb-2 text-xs font-semibold uppercase tracking-wider">
              {CATEGORY_LABELS[cat] || cat}
            </h3>
            <div className="grid gap-2 sm:grid-cols-2">
              {tools.map((tool) => {
                const checked = selected.has(tool.id);
                const hasConfig = tool.required_config.length > 0;
                const isExpanded = expandedTool === tool.id;
                const needsHitl =
                  Boolean(tool.requires_approval) || approvalSet.has(tool.id);

                return (
                  <div
                    key={tool.id}
                    className={`rounded-lg border transition-colors ${
                      checked
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-primary/50"
                    }`}
                  >
                    <label className="flex cursor-pointer items-start gap-3 p-3">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggle(tool.id)}
                        className="mt-0.5 h-4 w-4 shrink-0"
                      />
                      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="text-sm font-medium">{tool.name}</span>
                          {tool.token_cost && (
                            <span className="bg-muted rounded px-1 py-0.5 font-mono text-[10px]">
                              {tool.token_cost}t
                            </span>
                          )}
                          {tool.is_toolkit && (
                            <span className="rounded bg-blue-500/15 px-1 py-0.5 text-[10px] font-medium text-blue-600 dark:text-blue-400">
                              Toolkit
                            </span>
                          )}
                          {needsHitl && (
                            <span className="rounded bg-amber-500/15 px-1 py-0.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">
                              HITL
                            </span>
                          )}
                        </div>
                        <span className="text-muted-foreground text-xs leading-snug">
                          {tool.description}
                        </span>
                      </div>
                      {checked && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.preventDefault();
                            setExpandedTool(isExpanded ? null : tool.id);
                          }}
                          className="text-muted-foreground mt-0.5 shrink-0"
                        >
                          {isExpanded ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </button>
                      )}
                    </label>

                    {checked && isExpanded && (
                      <div className="space-y-2.5 border-t px-3 pb-3 pt-2">
                        <label className="flex cursor-pointer items-center justify-between text-xs">
                          <span className="flex items-center gap-1 font-medium">
                            <ShieldAlert className="h-3.5 w-3.5 text-amber-500" />
                            Require Human Approval (AgentInbox)
                          </span>
                          <input
                            type="checkbox"
                            checked={needsHitl}
                            disabled={Boolean(tool.requires_approval)}
                            onChange={() => toggleHitl(tool.id)}
                          />
                        </label>
                        {hasConfig &&
                          tool.required_config.map((field) => (
                            <div key={field.key} className="flex flex-col gap-1">
                              <label className="text-xs font-medium">
                                {field.label}
                              </label>
                              <Input
                                type={field.type === "password" ? "password" : "text"}
                                placeholder={field.hint || ""}
                                value={
                                  config.tool_configs[tool.id]?.[field.key] || ""
                                }
                                onChange={(e) =>
                                  setToolConfig(tool.id, field.key, e.target.value)
                                }
                                className="h-8 text-xs"
                              />
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        {Object.keys(grouped).length === 0 && (
          <div className="text-muted-foreground py-8 text-center text-sm">
            {filter ? "No tools match your search." : "No tools available."}
          </div>
        )}
      </div>

      <p className="text-muted-foreground text-xs">
        {selected.size} tool{selected.size !== 1 ? "s" : ""} selected · Mode:{" "}
        <code>{config.tool_loading_mode ?? "auto"}</code>
      </p>
    </div>
  );
}
