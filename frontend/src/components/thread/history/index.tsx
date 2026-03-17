import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useMemo } from "react";

import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { PanelRightOpen, PanelRightClose, Trash2 } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";

function getDateGroup(ts: string | undefined): string {
  if (!ts) return "Older";
  const parsed = Date.parse(ts);
  if (Number.isNaN(parsed)) return "Older";
  const date = new Date(parsed);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 86400000);
  const weekAgo = new Date(today.getTime() - 6 * 86400000);
  if (date >= today) return "Today";
  if (date >= yesterday) return "Yesterday";
  if (date >= weekAgo) return "This Week";
  return "Older";
}

const GROUP_ORDER = ["Today", "Yesterday", "This Week", "Older"];

function ThreadList({
  threads,
  onThreadClick,
  onDelete,
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
  onDelete?: (threadId: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");

  const grouped = useMemo(() => {
    const groups: Record<string, Thread[]> = {};
    for (const t of threads) {
      const ts = (t as any).updated_at || (t as any).created_at;
      const g = getDateGroup(ts);
      if (!groups[g]) groups[g] = [];
      groups[g].push(t);
    }
    return GROUP_ORDER.filter((g) => groups[g]?.length).map((g) => ({
      label: g,
      items: groups[g],
    }));
  }, [threads]);

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-1 overflow-y-scroll px-1 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {grouped.map((group) => (
        <div key={group.label} className="w-full">
          <p className="text-muted-foreground mb-1 mt-3 px-2 text-[10px] font-semibold uppercase tracking-wider first:mt-1">
            {group.label}
          </p>
          {group.items.map((t) => {
            let itemText = t.thread_id;
            if (
              typeof t.values === "object" &&
              t.values &&
              "messages" in t.values &&
              Array.isArray(t.values.messages) &&
              t.values.messages?.length > 0
            ) {
              const firstMessage = t.values.messages[0];
              itemText = getContentString(firstMessage.content);
            }
            const isActive = t.thread_id === threadId;
            return (
              <div
                key={t.thread_id}
                className={`group flex w-full items-center rounded-md transition-colors ${isActive ? "bg-muted" : "hover:bg-muted/50"}`}
              >
                <Button
                  variant="ghost"
                  className="min-w-0 flex-1 items-start justify-start text-left font-normal"
                  onClick={(e) => {
                    e.preventDefault();
                    onThreadClick?.(t.thread_id);
                    if (t.thread_id === threadId) return;
                    setThreadId(t.thread_id);
                  }}
                >
                  <p className="truncate text-ellipsis">{itemText}</p>
                </Button>
                {onDelete && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-destructive hover:text-destructive h-7 w-7 shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                    onClick={() => onDelete(t.thread_id)}
                  >
                    <Trash2 className="size-3" />
                  </Button>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton
          key={`skeleton-${i}`}
          className="h-10 w-[280px]"
        />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading, deleteThread } =
    useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, []);

  const handleDelete = async (threadId: string) => {
    if (!confirm("Delete this session?")) return;
    try {
      await deleteThread(threadId);
    } catch (e) {
      console.error("Failed to delete thread:", e);
    }
  };

  return (
    <>
      <div className="shadow-inner-right hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-6 border-r-[1px] border-slate-300 lg:flex">
        <div className="flex w-full items-center justify-between px-4 pt-1.5">
          <Button
            className="hover:bg-gray-100"
            variant="ghost"
            onClick={() => setChatHistoryOpen((p) => !p)}
          >
            {chatHistoryOpen ? (
              <PanelRightOpen className="size-5" />
            ) : (
              <PanelRightClose className="size-5" />
            )}
          </Button>
          <h1 className="text-xl font-semibold tracking-tight">
            Thread History
          </h1>
        </div>
        {threadsLoading ? (
          <ThreadHistoryLoading />
        ) : (
          <ThreadList threads={threads} onDelete={handleDelete} />
        )}
      </div>
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent
            side="left"
            className="flex lg:hidden"
          >
            <SheetHeader>
              <SheetTitle>Thread History</SheetTitle>
            </SheetHeader>
            <ThreadList
              threads={threads}
              onThreadClick={() => setChatHistoryOpen((o) => !o)}
              onDelete={handleDelete}
            />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
