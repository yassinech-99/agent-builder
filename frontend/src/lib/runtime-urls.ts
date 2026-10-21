/**
 * Centralized endpoint, multi-tenant header & storage namespace helpers for nimAgents.
 *
 * Ensures browser requests default to same-origin Next.js proxy paths (`/api/builder`
 * and `${origin}/api`) so remote preview environments, reverse proxies, and Docker
 * deployments never attempt browser calls to `localhost:8100` / `localhost:2024`,
 * while immediately honoring user overrides & active tenant/workspace scope in `/settings`.
 */

export const NIMAGENTS_STORAGE = {
  auth: "nimagents:auth",
  loaded: "nimagents:loaded",
  newsletter: "nimagents:newsletter-email",
  runtimeApiUrl: "nimagents:settings:runtimeApiUrl",
  builderApiUrl: "nimagents:settings:builderApiUrl",
  tenantId: "nimagents:settings:tenantId",
  workspaceId: "nimagents:settings:workspaceId",
  planTier: "nimagents:settings:planTier",
  knowledgePrefix: "nimagents:knowledge",
} as const;

export function getSavedRuntimeOverride(): string {
  if (typeof window === "undefined") return "";
  try {
    return (window.localStorage.getItem(NIMAGENTS_STORAGE.runtimeApiUrl) || "").trim();
  } catch {
    return "";
  }
}

export function getSavedBuilderOverride(): string {
  if (typeof window === "undefined") return "";
  try {
    return (window.localStorage.getItem(NIMAGENTS_STORAGE.builderApiUrl) || "").trim();
  } catch {
    return "";
  }
}

export function getActiveTenantScope(): {
  tenantId: string;
  workspaceId: string;
  planTier: string;
} {
  if (typeof window === "undefined") {
    return { tenantId: "default", workspaceId: "default", planTier: "pro" };
  }
  try {
    return {
      tenantId: (window.localStorage.getItem(NIMAGENTS_STORAGE.tenantId) || "default").trim() || "default",
      workspaceId: (window.localStorage.getItem(NIMAGENTS_STORAGE.workspaceId) || "default").trim() || "default",
      planTier: (window.localStorage.getItem(NIMAGENTS_STORAGE.planTier) || "pro").trim() || "pro",
    };
  } catch {
    return { tenantId: "default", workspaceId: "default", planTier: "pro" };
  }
}

export function getTenantHeaders(): Record<string, string> {
  const { tenantId, workspaceId, planTier } = getActiveTenantScope();
  return {
    "X-Tenant-ID": tenantId,
    "X-Workspace-ID": workspaceId,
    "X-Plan-Tier": planTier,
  };
}

/**
 * Base URL for nimAgents Control Plane API (`/api/models`, `/api/rag`, `/api/storage`, etc.).
 * Defaults to `/api/builder` which Next.js proxies cleanly to `http://127.0.0.1:8100`.
 */
export function getBuilderApiUrl(): string {
  const custom = getSavedBuilderOverride();
  if (custom) return custom.replace(/\/$/, "");
  const envUrl = process.env.NEXT_PUBLIC_BUILDER_API_URL;
  if (envUrl) return envUrl.replace(/\/$/, "");
  return "/api/builder";
}

/**
 * Base URL for LangGraph runtime API (`/assistants`, `/threads`, `/runs/stream`).
 * `@langchain/langgraph-sdk` requires an absolute URL with scheme + host, so in the
 * browser we default to `${window.location.origin}/api` (proxied to `127.0.0.1:2024`).
 */
export function getRuntimeApiUrl(): string {
  const custom = getSavedRuntimeOverride();
  if (custom) return custom.replace(/\/$/, "");
  const envUrl = process.env.NEXT_PUBLIC_API_URL;
  if (envUrl) return envUrl.replace(/\/$/, "");
  if (typeof window !== "undefined") {
    return `${window.location.origin}/api`;
  }
  return "http://127.0.0.1:2024/api";
}

/**
 * WebSocket URL for real-time voice streaming (`/api/voice/ws`).
 */
export function getVoiceWsUrl(modelId?: string): string {
  const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : "";
  const custom = getSavedBuilderOverride() || process.env.NEXT_PUBLIC_BUILDER_API_URL || "";
  if (custom && /^https?:\/\//i.test(custom)) {
    return `${custom.replace(/\/$/, "").replace(/^http/i, "ws")}/api/voice/ws${query}`;
  }
  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    const e2bMatch = hostname.match(/^\d+-(.+\.e2b\.(?:app|dev))$/i);
    if (e2bMatch) {
      const wsProto = protocol === "https:" ? "wss:" : "ws:";
      return `${wsProto}//8100-${e2bMatch[1]}/api/voice/ws${query}`;
    }
    const wsProto = protocol === "https:" ? "wss:" : "ws:";
    return `${wsProto}//${hostname}:8100/api/voice/ws${query}`;
  }
  return `ws://localhost:8100/api/voice/ws${query}`;
}
