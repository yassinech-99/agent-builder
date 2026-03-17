import { v4 as uuidv4 } from "uuid";
import {
  ReactNode,
  useEffect,
  useRef,
  useState,
  FormEvent,
  useCallback,
} from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { NovaGridLogo } from "../icons/grid-agents";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  Database,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  XIcon,
  Plus,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { useThreads } from "@/providers/Thread";
import { Label } from "../ui/label";
import { Switch } from "../ui/switch";
import { useFileUpload } from "@/hooks/use-file-upload";
import { ContentBlocksPreview } from "./ContentBlocksPreview";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "./artifact";
import { detectRiskyInput } from "@/lib/guardrails";

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

type KnowledgeSource = {
  id: string;
  name: string;
  status: string;
};

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>
      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();
  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

function ChatHistoryToggle({
  isOpen,
  onToggle,
}: {
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <Button className="hover:bg-gray-100" variant="ghost" onClick={onToggle}>
      {isOpen ? (
        <PanelRightOpen className="size-5" />
      ) : (
        <PanelRightClose className="size-5" />
      )}
    </Button>
  );
}

export function Thread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const [threadId, _setThreadId] = useQueryState("threadId");
  const [assistantId] = useQueryState("assistantId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [knowledgeModalOpen, setKnowledgeModalOpen] = useState(false);
  const [knowledgeSearch, setKnowledgeSearch] = useState("");
  const [sourcesLoading, setSourcesLoading] = useState(false);
  const [vectorSources, setVectorSources] = useState<KnowledgeSource[]>([]);
  const [localRagSources, setLocalRagSources] = useState<KnowledgeSource[]>([]);
  const [lexicalSources, setLexicalSources] = useState<KnowledgeSource[]>([]);
  const [selectedVectorSourceIds, setSelectedVectorSourceIds] = useState<
    string[]
  >([]);
  const [selectedLocalRagSourceIds, setSelectedLocalRagSourceIds] = useState<
    string[]
  >([]);
  const [selectedLexicalSourceIds, setSelectedLexicalSourceIds] = useState<
    string[]
  >([]);
  const [input, setInput] = useState("");
  const {
    contentBlocks,
    setContentBlocks,
    handleFileUpload,
    dropRef,
    removeBlock,
    dragOver,
    handlePaste,
  } = useFileUpload();
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const { getThreadMetadataForAssistant } = useThreads();
  const messages = stream.messages;
  const isLoading = stream.isLoading;

  const lastError = useRef<string | undefined>(undefined);
  const metadataHydratedForThread = useRef<string | null>(null);
  const lastMetadataSelections = useRef<{
    threadId: string | null;
    rag: string[];
    local: string[];
    lexical: string[];
  }>({
    threadId: null,
    rag: [],
    local: [],
    lexical: [],
  });
  const knowledgeTouched = useRef(false);
  const [knowledgeHydrated, setKnowledgeHydrated] = useState(false);

  const setThreadId = (id: string | null) => {
    _setThreadId(id);
    closeArtifact();
    setArtifactContext({});
  };

  const getKnowledgeStorageKey = useCallback(
    (id: string | null | undefined) =>
      `novagrid:knowledge:${assistantId || "agent"}:${id || "__new__"}`,
    [assistantId],
  );

  const loadKnowledgeSelections = useCallback(
    (
      id: string | null | undefined,
    ):
      | {
          rag_source_ids: string[];
          local_rag_source_ids: string[];
          lexical_rag_source_ids: string[];
        }
      | null => {
      try {
        const raw = localStorage.getItem(getKnowledgeStorageKey(id));
        if (!raw) return null;
        const parsed = JSON.parse(raw) as {
          rag_source_ids?: string[];
          local_rag_source_ids?: string[];
          lexical_rag_source_ids?: string[];
        };
        return {
          rag_source_ids: parsed.rag_source_ids ?? [],
          local_rag_source_ids: parsed.local_rag_source_ids ?? [],
          lexical_rag_source_ids: parsed.lexical_rag_source_ids ?? [],
        };
      } catch {
        return null;
      }
    },
    [getKnowledgeStorageKey],
  );

  const persistKnowledgeSelections = useCallback((
    id: string | null | undefined,
    ragIds: string[],
    localIds: string[],
    lexicalIds: string[],
  ) => {
    try {
      localStorage.setItem(
        getKnowledgeStorageKey(id),
        JSON.stringify({
          rag_source_ids: ragIds,
          local_rag_source_ids: localIds,
          lexical_rag_source_ids: lexicalIds,
        }),
      );
    } catch {
      // no-op
    }
  }, [getKnowledgeStorageKey]);

  const applyKnowledgeSelections = (
    ragIds: string[],
    localIds: string[],
    lexicalIds: string[],
    markTouched = false,
  ) => {
    setSelectedVectorSourceIds(ragIds);
    setSelectedLocalRagSourceIds(localIds);
    setSelectedLexicalSourceIds(lexicalIds);
    knowledgeTouched.current = markTouched;
  };

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) return;
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  useEffect(() => {
    let cancelled = false;
    setKnowledgeHydrated(false);

    const hydrate = async () => {
      const localSelections = loadKnowledgeSelections(threadId);
      if (!threadId) {
        const rag = localSelections?.rag_source_ids ?? [];
        const local = localSelections?.local_rag_source_ids ?? [];
        const lexical = localSelections?.lexical_rag_source_ids ?? [];
        applyKnowledgeSelections(rag, local, lexical, false);
        metadataHydratedForThread.current = null;
        setKnowledgeHydrated(true);
        return;
      }

      try {
        const metadata = await getThreadMetadataForAssistant(
          assistantId || "agent",
          threadId,
        );
        if (cancelled) return;

        const metadataRag = metadata?.rag_source_ids ?? [];
        const metadataLocal = metadata?.local_rag_source_ids ?? [];
        const metadataLexical = metadata?.lexical_rag_source_ids ?? [];
        const hasMetadataKnowledge = metadataRag.length > 0 || metadataLocal.length > 0 || metadataLexical.length > 0;
        if (hasMetadataKnowledge) {
          if (
            localSelections &&
            (localSelections.rag_source_ids.join(",") !== metadataRag.join(",") ||
              localSelections.local_rag_source_ids.join(",") !== metadataLocal.join(",") ||
              localSelections.lexical_rag_source_ids.join(",") !==
                metadataLexical.join(","))
          ) {
            console.info(
              "Knowledge selections mismatch between metadata and local storage; metadata takes precedence",
              { threadId },
            );
          }
          applyKnowledgeSelections(metadataRag, metadataLocal, metadataLexical, false);
          persistKnowledgeSelections(threadId, metadataRag, metadataLocal, metadataLexical);
          lastMetadataSelections.current = {
            threadId,
            rag: metadataRag,
            local: metadataLocal,
            lexical: metadataLexical,
          };
          metadataHydratedForThread.current = threadId;
          setKnowledgeHydrated(true);
          return;
        }
      } catch (error) {
        console.error("Failed to hydrate thread metadata for knowledge restore", error);
      }

      try {
        const newThreadRaw = localStorage.getItem(getKnowledgeStorageKey(null));
        const hasThreadRaw = localStorage.getItem(getKnowledgeStorageKey(threadId));
        if (newThreadRaw && !hasThreadRaw) {
          localStorage.setItem(getKnowledgeStorageKey(threadId), newThreadRaw);
          localStorage.removeItem(getKnowledgeStorageKey(null));
        }
      } catch {
        // no-op
      }

      const fallback = loadKnowledgeSelections(threadId);
      applyKnowledgeSelections(
        fallback?.rag_source_ids ?? [],
        fallback?.local_rag_source_ids ?? [],
        fallback?.lexical_rag_source_ids ?? [],
        false,
      );
      metadataHydratedForThread.current = null;
      setKnowledgeHydrated(true);
    };

    hydrate();
    return () => {
      cancelled = true;
    };
  }, [
    assistantId,
    getKnowledgeStorageKey,
    getThreadMetadataForAssistant,
    loadKnowledgeSelections,
    threadId,
    persistKnowledgeSelections,
  ]);

  useEffect(() => {
    if (!knowledgeHydrated) return;
    persistKnowledgeSelections(
      threadId,
      selectedVectorSourceIds,
      selectedLocalRagSourceIds,
      selectedLexicalSourceIds,
    );
  }, [
    persistKnowledgeSelections,
    knowledgeHydrated,
    threadId,
    selectedVectorSourceIds,
    selectedLocalRagSourceIds,
    selectedLexicalSourceIds,
  ]);

  useEffect(() => {
    if (!knowledgeModalOpen) return;
    const fetchSources = async () => {
      setSourcesLoading(true);
      try {
        const [vectorRes, localRes, lexicalRes] = await Promise.all([
          fetch(`${BUILDER_API}/api/rag`),
          fetch(`${BUILDER_API}/api/local-rag`),
          fetch(`${BUILDER_API}/api/lexical-rag`),
        ]);
        if (vectorRes.ok) {
          const vectorData = await vectorRes.json();
          setVectorSources(
            (vectorData || []).map((s: any) => ({
              id: s.id,
              name: s.name,
              status: s.status,
            })),
          );
        }
        if (localRes.ok) {
          const localData = await localRes.json();
          setLocalRagSources(
            (localData || []).map((s: any) => ({
              id: s.id,
              name: s.name,
              status: s.status,
            })),
          );
        }
        if (lexicalRes.ok) {
          const lexicalData = await lexicalRes.json();
          setLexicalSources(
            (lexicalData || []).map((s: any) => ({
              id: s.id,
              name: s.name,
              status: s.status,
            })),
          );
        }
      } catch {
        toast.error("Failed to load knowledge sources");
      } finally {
        setSourcesLoading(false);
      }
    };
    fetchSources();
  }, [knowledgeModalOpen]);

  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }
    prevMessageLength.current = messages.length;
  }, [messages]);

  const getEffectiveKnowledgeSelections = () => {
    const shouldUseMetadataFallback =
      !!threadId &&
      metadataHydratedForThread.current === threadId &&
      !knowledgeTouched.current &&
      selectedVectorSourceIds.length === 0 &&
      selectedLocalRagSourceIds.length === 0 &&
      selectedLexicalSourceIds.length === 0 &&
      (lastMetadataSelections.current.rag.length > 0 ||
        lastMetadataSelections.current.local.length > 0 ||
        lastMetadataSelections.current.lexical.length > 0);

    if (shouldUseMetadataFallback) {
      return {
        rag: lastMetadataSelections.current.rag,
        local: lastMetadataSelections.current.local,
        lexical: lastMetadataSelections.current.lexical,
      };
    }
    return {
      rag: selectedVectorSourceIds,
      local: selectedLocalRagSourceIds,
      lexical: selectedLexicalSourceIds,
    };
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if ((input.trim().length === 0 && contentBlocks.length === 0) || isLoading)
      return;
    const riskHits = detectRiskyInput(input);
    if (riskHits.length > 0) {
      toast.warning("Potential sensitive content detected", {
        description:
          "Guardrails will check this request server-side. Remove secrets if this was unintentional.",
      });
    }
    setFirstTokenReceived(false);

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [
        ...(input.trim().length > 0 ? [{ type: "text", text: input }] : []),
        ...contentBlocks,
      ] as Message["content"],
    };

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);
    const context =
      Object.keys(artifactContext).length > 0 ? artifactContext : undefined;
    const effectiveKnowledge = getEffectiveKnowledgeSelections();
    const runTitle =
      !threadId && input.trim().length > 0
        ? normalizeTitle(input)
        : getExistingThreadTitle();
    const runMetadata: Record<string, unknown> = {
      assistant_id: assistantId || "agent",
      updated_at: new Date().toISOString(),
      message_count: (messages?.length ?? 0) + toolMessages.length + 1,
      rag_source_ids: effectiveKnowledge.rag,
      local_rag_source_ids: effectiveKnowledge.local,
      lexical_rag_source_ids: effectiveKnowledge.lexical,
    };
    if (runTitle) {
      runMetadata.title = runTitle;
    }

    stream.submit(
      { messages: [...toolMessages, newHumanMessage], context },
      {
        metadata: runMetadata,
        config: {
          configurable: {
            rag_source_ids: effectiveKnowledge.rag,
            local_rag_source_ids: effectiveKnowledge.local,
            lexical_rag_source_ids: effectiveKnowledge.lexical,
          },
        },
        streamMode: ["values"],
        streamSubgraphs: true,
        streamResumable: true,
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
    setContentBlocks([]);
  };

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    const effectiveKnowledge = getEffectiveKnowledgeSelections();
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      metadata: {
        assistant_id: assistantId || "agent",
        updated_at: new Date().toISOString(),
        message_count: messages.length,
        rag_source_ids: effectiveKnowledge.rag,
        local_rag_source_ids: effectiveKnowledge.local,
        lexical_rag_source_ids: effectiveKnowledge.lexical,
        title: getExistingThreadTitle(),
      },
      config: {
        configurable: {
          rag_source_ids: effectiveKnowledge.rag,
          local_rag_source_ids: effectiveKnowledge.local,
          lexical_rag_source_ids: effectiveKnowledge.lexical,
        },
      },
      streamMode: ["values"],
      streamSubgraphs: true,
      streamResumable: true,
    });
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );
  const showHistoryToggle = !chatHistoryOpen || !isLargeScreen;
  const selectedCount =
    selectedVectorSourceIds.length + selectedLocalRagSourceIds.length + selectedLexicalSourceIds.length;
  const filteredVectorSources = vectorSources.filter((s) =>
    s.name.toLowerCase().includes(knowledgeSearch.toLowerCase()),
  );
  const filteredLocalRagSources = localRagSources.filter((s) =>
    s.name.toLowerCase().includes(knowledgeSearch.toLowerCase()),
  );
  const filteredLexicalSources = lexicalSources.filter((s) =>
    s.name.toLowerCase().includes(knowledgeSearch.toLowerCase()),
  );

  const toggleVectorSource = (sourceId: string) => {
    knowledgeTouched.current = true;
    setSelectedVectorSourceIds((prev) =>
      prev.includes(sourceId)
        ? prev.filter((id) => id !== sourceId)
        : [...prev, sourceId],
    );
  };
  const toggleLocalRagSource = (sourceId: string) => {
    knowledgeTouched.current = true;
    setSelectedLocalRagSourceIds((prev) =>
      prev.includes(sourceId)
        ? prev.filter((id) => id !== sourceId)
        : [...prev, sourceId],
    );
  };
  const toggleLexicalSource = (sourceId: string) => {
    knowledgeTouched.current = true;
    setSelectedLexicalSourceIds((prev) =>
      prev.includes(sourceId)
        ? prev.filter((id) => id !== sourceId)
        : [...prev, sourceId],
    );
  };

  const normalizeTitle = (text: string) => {
    const collapsed = text.replace(/\s+/g, " ").trim();
    return collapsed.length > 80 ? `${collapsed.slice(0, 80)}...` : collapsed;
  };

  const getExistingThreadTitle = () => {
    const firstHuman = messages.find((m) => m.type === "human");
    if (!firstHuman) return "";
    if (typeof firstHuman.content === "string") {
      return normalizeTitle(firstHuman.content);
    }
    if (Array.isArray(firstHuman.content)) {
      const joined = firstHuman.content
        .filter(
          (c): c is { type: "text"; text: string } =>
            c.type === "text" && typeof c.text === "string",
        )
        .map((c) => c.text)
        .join(" ");
      return normalizeTitle(joined);
    }
    return "";
  };

  return (
    <div className="flex h-full w-full overflow-hidden">
      <div className="relative hidden lg:flex">
        <motion.div
          className="absolute z-20 h-full overflow-hidden border-r bg-white"
          style={{ width: 300 }}
          animate={{ x: chatHistoryOpen ? 0 : -300 }}
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div className="relative h-full" style={{ width: 300 }}>
            <ThreadHistory />
          </div>
        </motion.div>
      </div>

      <div
        className={cn(
          "grid w-full grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          animate={{
            marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
            width: chatHistoryOpen
              ? isLargeScreen
                ? "calc(100% - 300px)"
                : "100%"
              : "100%",
          }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          {!chatStarted && (
            <div className="absolute top-0 left-0 z-10 flex w-full items-center justify-between gap-3 p-2 pl-4">
              <div>
                {showHistoryToggle && (
                  <ChatHistoryToggle
                    isOpen={!!chatHistoryOpen}
                    onToggle={() => setChatHistoryOpen((p) => !p)}
                  />
                )}
              </div>
            </div>
          )}
          {chatStarted && (
            <div className="relative z-10 flex items-center justify-between gap-3 p-2">
              <div className="relative flex items-center justify-start gap-2">
                <div className="absolute left-0 z-10">
                  {showHistoryToggle && (
                    <ChatHistoryToggle
                      isOpen={!!chatHistoryOpen}
                      onToggle={() => setChatHistoryOpen((p) => !p)}
                    />
                  )}
                </div>
                <motion.button
                  className="flex cursor-pointer items-center gap-2"
                  onClick={() => setThreadId(null)}
                  animate={{ marginLeft: !chatHistoryOpen ? 48 : 0 }}
                  transition={{
                    type: "spring",
                    stiffness: 300,
                    damping: 30,
                  }}
                >
                  <NovaGridLogo width={28} height={28} />
                  <span className="text-xl font-semibold tracking-tight">
                    NovaGrid
                  </span>
                </motion.button>
              </div>

              <div className="flex items-center gap-4">
                <TooltipIconButton
                  size="lg"
                  className="p-4"
                  tooltip="New thread"
                  variant="ghost"
                  onClick={() => setThreadId(null)}
                >
                  <SquarePen className="size-5" />
                </TooltipIconButton>
              </div>

              <div className="from-background to-background/0 absolute inset-x-0 top-full h-5 bg-gradient-to-b" />
            </div>
          )}

          <StickToBottom className="relative flex-1 overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "mt-[25vh] flex flex-col items-stretch",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName="pt-8 pb-16 max-w-3xl mx-auto flex flex-col gap-4 w-full"
              content={
                <>
                  {messages
                    .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                    .map((message, index) =>
                      message.type === "human" ? (
                        <HumanMessage
                          key={message.id || `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                        />
                      ) : (
                        <AssistantMessage
                          key={message.id || `${message.type}-${index}`}
                          message={message}
                          isLoading={isLoading}
                          handleRegenerate={handleRegenerate}
                        />
                      ),
                    )}
                  {hasNoAIOrToolMessages && !!stream.interrupt && (
                    <AssistantMessage
                      key="interrupt-msg"
                      message={undefined}
                      isLoading={isLoading}
                      handleRegenerate={handleRegenerate}
                    />
                  )}
                  {isLoading && !firstTokenReceived && (
                    <AssistantMessageLoading />
                  )}
                </>
              }
              footer={
                <div className="bg-background sticky bottom-0 flex flex-col items-center gap-8">
                  {!chatStarted && (
                    <div className="flex items-center gap-3">
                      <NovaGridLogo className="text-primary h-8 w-8 flex-shrink-0" />
                      <h1 className="text-2xl font-semibold tracking-tight">
                        NovaGrid Agents
                      </h1>
                    </div>
                  )}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  <div
                    ref={dropRef}
                    className={cn(
                      "bg-muted relative z-10 mx-auto mb-8 w-full max-w-3xl rounded-2xl shadow-xs transition-all",
                      dragOver
                        ? "border-primary border-2 border-dotted"
                        : "border border-solid",
                    )}
                  >
                    <form
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <ContentBlocksPreview
                        blocks={contentBlocks}
                        onRemove={removeBlock}
                      />
                      <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onPaste={handlePaste}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        placeholder="Type your message..."
                        className="field-sizing-content resize-none border-none bg-transparent p-3.5 pb-0 shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />

                      <div className="flex items-center gap-6 p-2 pt-4">
                        <div>
                          <div className="flex items-center space-x-2">
                            <Switch
                              id="render-tool-calls"
                              checked={hideToolCalls ?? false}
                              onCheckedChange={setHideToolCalls}
                            />
                            <Label
                              htmlFor="render-tool-calls"
                              className="text-sm text-gray-600"
                            >
                              Hide Tool Calls
                            </Label>
                          </div>
                        </div>
                        <Label
                          htmlFor="file-input"
                          className="flex cursor-pointer items-center gap-2"
                        >
                          <Plus className="size-5 text-gray-600" />
                          <span className="text-sm text-gray-600">
                            Upload PDF or Image
                          </span>
                        </Label>
                        <Button
                          type="button"
                          variant="ghost"
                          className="h-auto p-0 hover:bg-transparent"
                          onClick={() => setKnowledgeModalOpen(true)}
                        >
                          <span className="flex items-center gap-2 text-sm text-gray-600">
                            <Database className="size-5 text-gray-600" />
                            Add Internal Knowledge
                            {selectedCount > 0 ? ` (${selectedCount})` : ""}
                          </span>
                        </Button>
                        <input
                          id="file-input"
                          type="file"
                          onChange={handleFileUpload}
                          multiple
                          accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
                          className="hidden"
                        />
                        {stream.isLoading ? (
                          <Button
                            key="stop"
                            onClick={() => stream.stop()}
                            className="ml-auto"
                          >
                            <LoaderCircle className="h-4 w-4 animate-spin" />
                            Cancel
                          </Button>
                        ) : (
                          <Button
                            type="submit"
                            className="ml-auto shadow-md transition-all"
                            disabled={
                              isLoading ||
                              (!input.trim() && contentBlocks.length === 0)
                            }
                          >
                            Send
                          </Button>
                        )}
                      </div>
                      {selectedCount > 0 && (
                        <div className="flex flex-wrap items-center gap-2 px-2 pb-2">
                          {selectedVectorSourceIds.map((id) => {
                            const src = vectorSources.find((s) => s.id === id);
                            return (
                              <button
                                key={`v-${id}`}
                                type="button"
                                onClick={() => {
                                  knowledgeTouched.current = true;
                                  setSelectedVectorSourceIds((prev) =>
                                    prev.filter((item) => item !== id),
                                  );
                                }}
                                className="bg-primary/10 text-primary inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs"
                              >
                                {src?.name ?? `Vector ${id.slice(0, 8)}...`}
                                <XIcon className="size-3" />
                              </button>
                            );
                          })}
                          {selectedLocalRagSourceIds.map((id) => {
                            const src = localRagSources.find((s) => s.id === id);
                            return (
                              <button
                                key={`lr-${id}`}
                                type="button"
                                onClick={() => {
                                  knowledgeTouched.current = true;
                                  setSelectedLocalRagSourceIds((prev) =>
                                    prev.filter((item) => item !== id),
                                  );
                                }}
                                className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-1 text-xs text-emerald-800"
                              >
                                {src?.name ?? `Local ${id.slice(0, 8)}...`}
                                <XIcon className="size-3" />
                              </button>
                            );
                          })}
                          {selectedLexicalSourceIds.map((id) => {
                            const src = lexicalSources.find((s) => s.id === id);
                            return (
                              <button
                                key={`l-${id}`}
                                type="button"
                                onClick={() => {
                                  knowledgeTouched.current = true;
                                  setSelectedLexicalSourceIds((prev) =>
                                    prev.filter((item) => item !== id),
                                  );
                                }}
                                className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-1 text-xs text-amber-800"
                              >
                                {src?.name ?? `Lexical ${id.slice(0, 8)}...`}
                                <XIcon className="size-3" />
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </form>
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>
        <div className="relative flex flex-col border-l">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col">
            <div className="grid grid-cols-[1fr_auto] border-b p-4">
              <ArtifactTitle className="truncate overflow-hidden" />
              <button onClick={closeArtifact} className="cursor-pointer">
                <XIcon className="size-5" />
              </button>
            </div>
            <ArtifactContent className="relative flex-grow" />
          </div>
        </div>
      </div>
      {knowledgeModalOpen && (
        <div className="bg-background/70 absolute inset-0 z-50 flex items-center justify-center backdrop-blur-[1px]">
          <div className="bg-background w-full max-w-4xl rounded-xl border shadow-xl">
            <div className="flex items-center justify-between border-b px-5 py-4">
              <div>
                <h2 className="text-base font-semibold">Add Internal Knowledge</h2>
                <p className="text-muted-foreground mt-0.5 text-sm">
                  Select knowledge sources for this thread.
                </p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setKnowledgeModalOpen(false)}
              >
                <XIcon className="size-4" />
              </Button>
            </div>
            <div className="px-5 pt-4">
              <input
                value={knowledgeSearch}
                onChange={(e) => setKnowledgeSearch(e.target.value)}
                placeholder="Search sources..."
                className="bg-background w-full rounded-md border px-3 py-2 text-sm outline-none"
              />
            </div>
            <div className="grid gap-4 px-5 py-4 md:grid-cols-3">
              <div className="rounded-lg border p-4">
                <h3 className="text-sm font-medium">Bedrock Knowledge Bases</h3>
                <div className="mt-2 max-h-64 space-y-2 overflow-auto pr-1">
                  {sourcesLoading && (
                    <p className="text-muted-foreground text-xs">
                      Loading sources...
                    </p>
                  )}
                  {!sourcesLoading && filteredVectorSources.length === 0 && (
                    <p className="text-muted-foreground text-xs">
                      No vector sources found.
                    </p>
                  )}
                  {filteredVectorSources.map((source) => (
                    <button
                      key={source.id}
                      type="button"
                      onClick={() => toggleVectorSource(source.id)}
                      className="flex w-full items-center justify-between rounded-md border px-2 py-2 text-left text-sm"
                    >
                      <span className="truncate">{source.name}</span>
                      <span className="ml-2 flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          {source.status}
                        </span>
                        <input
                          type="checkbox"
                          readOnly
                          checked={selectedVectorSourceIds.includes(source.id)}
                        />
                      </span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <h3 className="text-sm font-medium">Local RAG Sources</h3>
                <div className="mt-2 max-h-64 space-y-2 overflow-auto pr-1">
                  {sourcesLoading && (
                    <p className="text-muted-foreground text-xs">
                      Loading sources...
                    </p>
                  )}
                  {!sourcesLoading && filteredLocalRagSources.length === 0 && (
                    <p className="text-muted-foreground text-xs">
                      No local sources found.
                    </p>
                  )}
                  {filteredLocalRagSources.map((source) => (
                    <button
                      key={source.id}
                      type="button"
                      onClick={() => toggleLocalRagSource(source.id)}
                      className="flex w-full items-center justify-between rounded-md border px-2 py-2 text-left text-sm"
                    >
                      <span className="truncate">{source.name}</span>
                      <span className="ml-2 flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          {source.status}
                        </span>
                        <input
                          type="checkbox"
                          readOnly
                          checked={selectedLocalRagSourceIds.includes(source.id)}
                        />
                      </span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <h3 className="text-sm font-medium">Lexical RAG Sources</h3>
                <div className="mt-2 max-h-64 space-y-2 overflow-auto pr-1">
                  {sourcesLoading && (
                    <p className="text-muted-foreground text-xs">
                      Loading sources...
                    </p>
                  )}
                  {!sourcesLoading && filteredLexicalSources.length === 0 && (
                    <p className="text-muted-foreground text-xs">
                      No lexical sources found.
                    </p>
                  )}
                  {filteredLexicalSources.map((source) => (
                    <button
                      key={source.id}
                      type="button"
                      onClick={() => toggleLexicalSource(source.id)}
                      className="flex w-full items-center justify-between rounded-md border px-2 py-2 text-left text-sm"
                    >
                      <span className="truncate">{source.name}</span>
                      <span className="ml-2 flex items-center gap-2">
                        <span className="text-xs text-gray-500">
                          {source.status}
                        </span>
                        <input
                          type="checkbox"
                          readOnly
                          checked={selectedLexicalSourceIds.includes(source.id)}
                        />
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex justify-end gap-2 border-t px-5 py-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => setKnowledgeModalOpen(false)}
              >
                Done
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
