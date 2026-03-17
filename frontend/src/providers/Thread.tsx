import { validate } from "uuid";
import { getApiKey } from "@/lib/api-key";
import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useState,
  Dispatch,
  SetStateAction,
} from "react";
import { createClient } from "./client";

export interface SessionSummary {
  thread_id: string;
  title: string;
  updated_at: string;
  message_count: number | null;
  rag_count: number;
  local_rag_count: number;
  lexical_count: number;
}

export interface ThreadMetadataSnapshot {
  thread_id: string;
  assistant_id: string;
  title: string;
  updated_at: string;
  message_count: number | null;
  rag_source_ids: string[];
  local_rag_source_ids: string[];
  lexical_rag_source_ids: string[];
}

interface ThreadContextType {
  getThreads: () => Promise<Thread[]>;
  getThreadsForAssistant: (assistantId: string) => Promise<Thread[]>;
  getThreadMetadataForAssistant: (
    assistantId: string,
    threadId: string,
  ) => Promise<ThreadMetadataSnapshot | null>;
  getSessionSummariesForAssistant: (
    assistantId: string,
  ) => Promise<SessionSummary[]>;
  deleteThread: (threadId: string) => Promise<void>;
  threads: Thread[];
  setThreads: Dispatch<SetStateAction<Thread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  } else {
    return { graph_id: assistantId };
  }
}

const DEFAULT_API_URL = "http://localhost:2024";

export function ThreadProvider({ children }: { children: ReactNode }) {
  const envApiUrl = process.env.NEXT_PUBLIC_API_URL;
  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || DEFAULT_API_URL,
  });
  const [assistantId] = useQueryState("assistantId");
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);

  const searchThreadsByAssistant = useCallback(
    async (targetAssistantId: string): Promise<Thread[]> => {
      if (!apiUrl || !targetAssistantId) return [];
      const client = createClient(apiUrl, getApiKey() ?? undefined);
      return client.threads.search({
        metadata: {
          ...getThreadSearchMetadata(targetAssistantId),
        },
        limit: 100,
      });
    },
    [apiUrl],
  );

  const getThreads = useCallback(async (): Promise<Thread[]> => {
    if (!assistantId) return [];
    return searchThreadsByAssistant(assistantId);
  }, [assistantId, searchThreadsByAssistant]);

  const getThreadsForAssistant = useCallback(
    async (targetAssistantId: string): Promise<Thread[]> => {
      return searchThreadsByAssistant(targetAssistantId);
    },
    [searchThreadsByAssistant],
  );

  const getThreadMetadataForAssistant = useCallback(
    async (
      targetAssistantId: string,
      targetThreadId: string,
    ): Promise<ThreadMetadataSnapshot | null> => {
      if (!targetAssistantId || !targetThreadId) return null;
      const results = await searchThreadsByAssistant(targetAssistantId);
      const match = results.find((t) => (t as any).thread_id === targetThreadId);
      if (!match) return null;
      const thread = match as any;
      const metadata =
        thread.metadata && typeof thread.metadata === "object"
          ? thread.metadata
          : {};
      const values =
        thread.values && typeof thread.values === "object" ? thread.values : {};
      const messages = Array.isArray(values.messages) ? values.messages : [];
      const ragIds = Array.isArray(metadata.rag_source_ids)
        ? metadata.rag_source_ids
        : [];
      const localRagIds = Array.isArray(metadata.local_rag_source_ids)
        ? metadata.local_rag_source_ids
        : [];
      const lexicalIds = Array.isArray(metadata.lexical_rag_source_ids)
        ? metadata.lexical_rag_source_ids
        : [];

      return {
        thread_id: thread.thread_id,
        assistant_id: targetAssistantId,
        title:
          typeof metadata.title === "string" && metadata.title.trim()
            ? metadata.title.trim()
            : "Untitled Session",
        updated_at:
          (typeof metadata.updated_at === "string" && metadata.updated_at) ||
          thread.updated_at ||
          thread.created_at ||
          "",
        message_count:
          typeof metadata.message_count === "number"
            ? metadata.message_count
            : messages.length > 0
              ? messages.length
              : null,
        rag_source_ids: ragIds,
        local_rag_source_ids: localRagIds,
        lexical_rag_source_ids: lexicalIds,
      };
    },
    [searchThreadsByAssistant],
  );

  const getSessionSummariesForAssistant = useCallback(
    async (targetAssistantId: string): Promise<SessionSummary[]> => {
      const results = await searchThreadsByAssistant(targetAssistantId);
      const mapped = results.map((t) => {
        const thread = t as any;
        const metadata =
          thread.metadata && typeof thread.metadata === "object"
            ? thread.metadata
            : {};
        const values =
          thread.values && typeof thread.values === "object" ? thread.values : {};
        const messages = Array.isArray(values.messages) ? values.messages : [];

        let title =
          typeof metadata.title === "string" ? metadata.title.trim() : "";
        if (!title && messages.length > 0) {
          const firstHuman =
            messages.find((m: any) => m?.type === "human") ?? messages[0];
          if (firstHuman) {
            if (typeof firstHuman.content === "string") {
              title = firstHuman.content;
            } else if (Array.isArray(firstHuman.content)) {
              title = firstHuman.content
                .filter(
                  (c: any) => c?.type === "text" && typeof c?.text === "string",
                )
                .map((c: any) => c.text)
                .join(" ");
            }
          }
        }

        const updatedAt =
          (typeof metadata.updated_at === "string" && metadata.updated_at) ||
          thread.updated_at ||
          thread.created_at ||
          "";
        const messageCount =
          typeof metadata.message_count === "number"
            ? metadata.message_count
            : messages.length > 0
              ? messages.length
              : null;
        const ragIds = Array.isArray(metadata.rag_source_ids)
          ? metadata.rag_source_ids
          : [];
        const localRagIds = Array.isArray(metadata.local_rag_source_ids)
          ? metadata.local_rag_source_ids
          : [];
        const lexicalIds = Array.isArray(metadata.lexical_rag_source_ids)
          ? metadata.lexical_rag_source_ids
          : [];

        return {
          thread_id: thread.thread_id,
          title: title || "Untitled Session",
          updated_at: updatedAt,
          message_count: messageCount,
          rag_count: ragIds.length,
          local_rag_count: localRagIds.length,
          lexical_count: lexicalIds.length,
        } satisfies SessionSummary;
      });

      return mapped.sort((a, b) => {
        const aTs = a.updated_at ? Date.parse(a.updated_at) : 0;
        const bTs = b.updated_at ? Date.parse(b.updated_at) : 0;
        if (aTs !== bTs) return bTs - aTs;
        return b.thread_id.localeCompare(a.thread_id);
      });
    },
    [searchThreadsByAssistant],
  );

  const deleteThread = useCallback(
    async (threadId: string) => {
      if (!apiUrl) return;
      const client = createClient(apiUrl, getApiKey() ?? undefined);
      await client.threads.delete(threadId);
      setThreads((prev) => prev.filter((t) => t.thread_id !== threadId));
    },
    [apiUrl],
  );

  const value = {
    getThreads,
    getThreadsForAssistant,
    getThreadMetadataForAssistant,
    getSessionSummariesForAssistant,
    deleteThread,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
  };

  return (
    <ThreadContext.Provider value={value}>{children}</ThreadContext.Provider>
  );
}

export function useThreads() {
  const context = useContext(ThreadContext);
  if (context === undefined) {
    throw new Error("useThreads must be used within a ThreadProvider");
  }
  return context;
}
