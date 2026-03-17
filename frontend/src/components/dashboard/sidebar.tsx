"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  MessageSquare,
  Bot,
  Cpu,
  Database,
  HardDrive,
  Settings,
  Shield,
  ChevronLeft,
  ChevronRight,
  LogOut,
  Mic,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { NovaGridLogo } from "@/components/icons/grid-agents";

const NAV_ITEMS = [
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/agents", label: "Agents", icon: Bot },
  { href: "/models", label: "Models", icon: Cpu },
  { href: "/rag", label: "RAG", icon: Database },
  { href: "/storage", label: "Storage", icon: HardDrive },
  { href: "/voice", label: "Voice", icon: Mic },
  { href: "/security", label: "Security", icon: Shield },
  { href: "/settings", label: "Settings", icon: Settings },
] as const;

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [collapsed, setCollapsed] = useState(false);

  const handleMockLogout = () => {
    sessionStorage.removeItem("novagrid:mock-auth");
    sessionStorage.removeItem("novagrid:loaded");
    router.push("/");
  };

  return (
    <aside
      className={cn(
        "bg-muted/40 flex h-screen flex-col border-r transition-all duration-200",
        collapsed ? "w-16" : "w-56",
      )}
    >
      <div
        className={cn(
          "flex items-center border-b px-4 py-4",
          collapsed ? "justify-center" : "justify-between",
        )}
      >
        {!collapsed && (
          <Link href="/chat" className="flex items-center gap-2">
            <NovaGridLogo width={24} height={24} />
            <span className="text-lg font-bold tracking-tight">NovaGrid</span>
          </Link>
        )}
        {collapsed && (
          <Link href="/chat">
            <NovaGridLogo width={24} height={24} />
          </Link>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={() => setCollapsed((c) => !c)}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      <nav className="flex flex-1 flex-col gap-1 p-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active =
            pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
                collapsed && "justify-center px-2",
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && <span>{label}</span>}
            </Link>
          );
        })}
      </nav>

      <div className="border-t p-3">
        {!collapsed ? (
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <p className="text-muted-foreground text-xs">NovaGrid Agents v1.0</p>
              <p className="text-muted-foreground/80 text-[10px]">
                Logged in as demo
              </p>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={handleMockLogout}
            >
              <LogOut className="mr-1 h-3.5 w-3.5" />
              Logout
            </Button>
          </div>
        ) : (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={handleMockLogout}
          >
            <LogOut className="h-4 w-4" />
          </Button>
        )}
      </div>
    </aside>
  );
}
