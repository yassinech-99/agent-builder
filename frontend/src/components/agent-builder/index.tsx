"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { useAgentConfig } from "@/providers/AgentConfig";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";
import { StepBasics } from "./step-basics";
import { StepPrebuiltTools } from "./step-prebuilt-tools";
import { StepMCPServers } from "./step-mcp-servers";
import { StepReview } from "./step-review";
import {
  ArrowLeft,
  ArrowRight,
  Rocket,
  Bot,
  Wrench,
  Server,
  CheckCircle2,
  Save,
} from "lucide-react";
import { toast } from "sonner";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024";

const STEPS = [
  { label: "Basics", icon: Bot },
  { label: "Prebuilt Tools", icon: Wrench },
  { label: "MCP Servers", icon: Server },
  { label: "Review & Deploy", icon: CheckCircle2 },
] as const;

export function AgentBuilder({ editId }: { editId?: string } = {}) {
  const [step, setStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const { config } = useAgentConfig();
  const router = useRouter();

  const deploy = async () => {
    setSaving(true);
    try {
      const client = createClient(API_URL, getApiKey() ?? undefined);
      const configurable = {
        agent_name: config.agent_name,
        system_prompt: config.system_prompt,
        enabled_prebuilt_tools: config.enabled_prebuilt_tools,
        tool_configs: config.tool_configs,
        mcp_servers: config.mcp_servers,
        model_id: config.model_id,
        rag_source_ids: config.rag_source_ids,
        local_rag_source_ids: config.local_rag_source_ids,
        lexical_rag_source_ids: config.lexical_rag_source_ids,
        voice_enabled: config.voice_enabled,
      };

      if (editId) {
        await client.assistants.update(editId, {
          name: config.agent_name,
          config: { configurable },
          metadata: { updated_at: new Date().toISOString() },
        });
        toast.success("Agent updated successfully");
      } else {
        await client.assistants.create({
          graphId: "agent",
          name: config.agent_name,
          config: { configurable },
          metadata: { created_at: new Date().toISOString() },
        });
        toast.success("Agent created successfully");
      }
      router.push("/agents");
    } catch (e: any) {
      console.error("Failed to save agent:", e);
      toast.error("Failed to save agent", {
        description: e?.message || "Unknown error",
      });
    } finally {
      setSaving(false);
    }
  };

  const canProceed = () => {
    if (step === 0 && !config.agent_name.trim()) return false;
    return true;
  };

  return (
    <div className="flex h-full items-center justify-center overflow-auto p-4">
      <div className="bg-background flex w-full max-w-2xl flex-col rounded-lg border shadow-lg">
        <div className="flex flex-col gap-2 border-b p-6">
          <h1 className="text-xl font-semibold tracking-tight">
            {editId ? "Edit Agent" : "New Agent"}
          </h1>
          <p className="text-muted-foreground text-sm">
            Configure your AI agent with tools and deploy it.
          </p>

          <div className="mt-4 flex gap-1">
            {STEPS.map((s, i) => {
              const Icon = s.icon;
              return (
                <button
                  key={i}
                  onClick={() => i < step && setStep(i)}
                  className={`flex flex-1 items-center justify-center gap-1.5 rounded-md py-2 text-xs font-medium transition-colors ${
                    i === step
                      ? "bg-primary text-primary-foreground"
                      : i < step
                        ? "bg-primary/10 text-primary cursor-pointer"
                        : "bg-muted text-muted-foreground"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">{s.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="min-h-[300px] p-6">
          {step === 0 && <StepBasics />}
          {step === 1 && <StepPrebuiltTools />}
          {step === 2 && <StepMCPServers />}
          {step === 3 && <StepReview />}
        </div>

        <div className="flex items-center justify-between border-t p-4">
          <Button
            variant="outline"
            onClick={() => setStep((s) => Math.max(0, s - 1))}
            disabled={step === 0}
          >
            <ArrowLeft className="mr-1 h-4 w-4" />
            Back
          </Button>

          {step < STEPS.length - 1 ? (
            <Button
              onClick={() => setStep((s) => s + 1)}
              disabled={!canProceed()}
            >
              Next
              <ArrowRight className="ml-1 h-4 w-4" />
            </Button>
          ) : (
            <Button onClick={deploy} disabled={saving}>
              {editId ? (
                <>
                  <Save className="mr-1 h-4 w-4" />
                  {saving ? "Saving..." : "Save Changes"}
                </>
              ) : (
                <>
                  <Rocket className="mr-1 h-4 w-4" />
                  {saving ? "Deploying..." : "Deploy Agent"}
                </>
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
