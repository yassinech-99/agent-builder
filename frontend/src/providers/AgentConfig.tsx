"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";

export interface MCPServerConfig {
  url: string;
  selected_tools: string[];
}

export type ToolLoadingMode = "auto" | "deferred" | "eager";

export interface AgentConfig {
  agent_name: string;
  system_prompt: string;
  enabled_prebuilt_tools: string[];
  tool_configs: Record<string, Record<string, string>>;
  mcp_servers: MCPServerConfig[];
  model_id: string;
  rag_source_ids: string[];
  local_rag_source_ids: string[];
  lexical_rag_source_ids: string[];
  voice_enabled: boolean;
  tool_loading_mode?: ToolLoadingMode;
  enable_programmatic_tools?: boolean;
  approval_required_tools?: string[];
  enabled_skill_ids?: string[];
  session_memory?: string;
}

const DEFAULT_CONFIG: AgentConfig = {
  agent_name: "My Agent",
  system_prompt: "",
  enabled_prebuilt_tools: [],
  tool_configs: {},
  mcp_servers: [],
  model_id: "",
  rag_source_ids: [],
  local_rag_source_ids: [],
  lexical_rag_source_ids: [],
  voice_enabled: false,
  tool_loading_mode: "auto",
  enable_programmatic_tools: true,
  approval_required_tools: [],
  enabled_skill_ids: [],
  session_memory: "",
};

interface AgentConfigContextType {
  config: AgentConfig;
  setConfig: React.Dispatch<React.SetStateAction<AgentConfig>>;
  resetConfig: () => void;
}

const AgentConfigContext = createContext<AgentConfigContextType | undefined>(
  undefined,
);

export const AgentConfigProvider: React.FC<{
  children: ReactNode;
  initialConfig?: AgentConfig;
}> = ({ children, initialConfig }) => {
  const [config, setConfig] = useState<AgentConfig>(
    initialConfig ?? DEFAULT_CONFIG,
  );
  const resetConfig = () => setConfig(DEFAULT_CONFIG);

  return (
    <AgentConfigContext.Provider value={{ config, setConfig, resetConfig }}>
      {children}
    </AgentConfigContext.Provider>
  );
};

export const useAgentConfig = (): AgentConfigContextType => {
  const ctx = useContext(AgentConfigContext);
  if (!ctx)
    throw new Error("useAgentConfig must be used within AgentConfigProvider");
  return ctx;
};
