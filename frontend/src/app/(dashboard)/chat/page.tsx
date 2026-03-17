"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { SessionSummary, ThreadProvider, useThreads } from "@/providers/Thread";
import { ArtifactProvider } from "@/components/thread/artifact";
import { AgentSelector } from "@/components/dashboard/agent-selector";
import { SessionChooser } from "@/components/dashboard/session-chooser";
import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useQueryState } from "nuqs";
import { toast } from "sonner";
import { Mic, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { VoiceSession } from "@/components/voice/voice-session";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:2024";

function ChatContentInner() {
  const [assistantId, setAssistantId] = useQueryState("assistantId", {
    defaultValue: process.env.NEXT_PUBLIC_ASSISTANT_ID || "agent",
  });
  const [, setThreadId] = useQueryState("threadId");
  const {
    getSessionSummariesForAssistant,
    getThreadsForAssistant,
    getThreadMetadataForAssistant,
    deleteThread,
  } = useThreads();
  const [chooserOpen, setChooserOpen] = useState(false);
  const [pendingAssistantId, setPendingAssistantId] = useState<string | null>(null);
  const [pendingSessions, setPendingSessions] = useState<SessionSummary[]>([]);
  const [switching, setSwitching] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const [agentVoiceEnabled, setAgentVoiceEnabled] = useState(false);
  const initialCheckDone = useRef(false);

  useEffect(() => {
    if (initialCheckDone.current) return;
    initialCheckDone.current = true;
    const [, threadIdParam] = window.location.search
      .replace("?", "")
      .split("&")
      .map((p) => p.split("="))
      .find(([k]) => k === "threadId") ?? [];
    if (threadIdParam) return;
    const checkInitialSessions = async () => {
      if (!assistantId) return;
      try {
        const sessions = await getSessionSummariesForAssistant(assistantId);
        if (sessions.length > 0) {
          setPendingAssistantId(assistantId);
          setPendingSessions(sessions);
          setChooserOpen(true);
        }
      } catch (e) {
        console.error("Failed to check initial sessions:", e);
      }
    };
    checkInitialSessions();
  }, [assistantId, getSessionSummariesForAssistant]);

  useEffect(() => {
    const checkVoice = async () => {
      if (!assistantId || assistantId === "agent") {
        setAgentVoiceEnabled(false);
        return;
      }
      try {
        const { createClient } = await import("@/providers/client");
        const { getApiKey } = await import("@/lib/api-key");
        const client = createClient(API_URL, getApiKey() ?? undefined);
        const assistant = await client.assistants.get(assistantId);
        const cfg = (assistant as any)?.config?.configurable;
        setAgentVoiceEnabled(!!cfg?.voice_enabled);
      } catch {
        setAgentVoiceEnabled(false);
      }
    };
    checkVoice();
    setVoiceMode(false);
  }, [assistantId]);

  const targetAgentLabel = useMemo(
    () => pendingAssistantId || "selected agent",
    [pendingAssistantId],
  );

  const applySwitch = async (nextAssistantId: string, nextThreadId: string | null) => {
    await setAssistantId(nextAssistantId);
    await setThreadId(nextThreadId);
  };

  const closeChooser = () => {
    setChooserOpen(false);
    setPendingAssistantId(null);
    setPendingSessions([]);
  };

  const handleAgentChange = async (nextAssistantId: string) => {
    if (!nextAssistantId || nextAssistantId === assistantId || switching) return;
    setSwitching(true);
    try {
      const sessions = await getSessionSummariesForAssistant(nextAssistantId);
      if (sessions.length === 0) {
        await applySwitch(nextAssistantId, null);
        return;
      }
      setPendingAssistantId(nextAssistantId);
      setPendingSessions(sessions);
      setChooserOpen(true);
    } catch (error) {
      console.error("Failed to load sessions for selected agent:", error);
      toast.error("Failed to load previous sessions");
    } finally {
      setSwitching(false);
    }
  };

  const handleContinue = async (threadIdToResume: string) => {
    if (!pendingAssistantId) return;
    try {
      const threads = await getThreadsForAssistant(pendingAssistantId);
      const exists = threads.some((t) => t.thread_id === threadIdToResume);
      if (!exists) {
        toast.error("Session no longer exists. Starting a new session.");
        await applySwitch(pendingAssistantId, null);
      } else {
        const metadata = await getThreadMetadataForAssistant(
          pendingAssistantId,
          threadIdToResume,
        );
        if (!metadata) {
          toast.error("Session metadata unavailable. Starting a new session.");
          await applySwitch(pendingAssistantId, null);
        } else {
          await applySwitch(pendingAssistantId, threadIdToResume);
        }
      }
    } finally {
      closeChooser();
    }
  };

  const handleStartNew = async () => {
    if (!pendingAssistantId) return;
    try {
      await applySwitch(pendingAssistantId, null);
    } finally {
      closeChooser();
    }
  };

  return (
    <div className="relative flex h-full flex-col">
      <div className="flex items-center gap-3 border-b px-4 py-2">
        <AgentSelector
          value={assistantId}
          onChange={handleAgentChange}
        />
        {agentVoiceEnabled && (
          <div className="ml-auto flex items-center gap-1 rounded-lg border p-0.5">
            <Button
              variant={voiceMode ? "ghost" : "default"}
              size="sm"
              className="h-7 px-2"
              onClick={() => setVoiceMode(false)}
            >
              <MessageSquare className="mr-1 h-3.5 w-3.5" />
              Chat
            </Button>
            <Button
              variant={voiceMode ? "default" : "ghost"}
              size="sm"
              className="h-7 px-2"
              onClick={() => setVoiceMode(true)}
            >
              <Mic className="mr-1 h-3.5 w-3.5" />
              Voice
            </Button>
          </div>
        )}
      </div>
      <div className="flex-1 overflow-hidden">
        {voiceMode ? (
          <VoiceSession />
        ) : (
          <StreamProvider>
            <ArtifactProvider>
              <Thread />
            </ArtifactProvider>
          </StreamProvider>
        )}
      </div>
      <SessionChooser
        open={chooserOpen}
        sessions={pendingSessions}
        targetAgentLabel={targetAgentLabel}
        onContinue={handleContinue}
        onStartNew={handleStartNew}
        onCancel={closeChooser}
        onDelete={async (threadId) => {
          try {
            await deleteThread(threadId);
            setPendingSessions((prev) => prev.filter((s) => s.thread_id !== threadId));
          } catch (e) {
            console.error("Failed to delete session:", e);
          }
        }}
      />
    </div>
  );
}

function ChatContent() {
  return (
    <ThreadProvider>
      <ChatContentInner />
    </ThreadProvider>
  );
}

export default function ChatPage(): React.ReactNode {
  return (
    <Suspense
      fallback={
        <div className="flex h-full items-center justify-center">
          Loading...
        </div>
      }
    >
      <ChatContent />
    </Suspense>
  );
}
