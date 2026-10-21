"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  MessageSquare,
  Bot,
  Wrench,
  Cpu,
  Database,
  HardDrive,
  Settings,
  Shield,
  ChevronLeft,
  ChevronRight,
  LogOut,
  Mic,
  Moon,
  Sun,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { useTheme } from "next-themes";
import { NimAgentsLogo } from "@/components/icons/nim-agents";
import { NIMAGENTS_STORAGE } from "@/lib/runtime-urls";

const NAV_ITEMS = [
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/agents", label: "Agents", icon: Bot },
  { href: "/tools", label: "Tools", icon: Wrench },
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
  const { resolvedTheme, setTheme } = useTheme();
  const [collapsed, setCollapsed] = useState(false);

  const handleLogout = () => {
    sessionStorage.removeItem(NIMAGENTS_STORAGE.auth);
    sessionStorage.removeItem(NIMAGENTS_STORAGE.loaded);
    router.push("/");
  };

  const toggleTheme = () => {
    setTheme(resolvedTheme === "dark" ? "light" : "dark");
  };

  return (
    <aside
      className={cn(
        "bg-sidebar text-sidebar-foreground flex h-screen flex-col border-r transition-all duration-200",
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
            <NimAgentsLogo width={26} height={26} className="text-primary" />
            <span className="text-lg font-bold tracking-tight">nimAgents</span>
          </Link>
        )}
        {collapsed && (
          <Link href="/chat" title="nimAgents">
            <NimAgentsLogo width={26} height={26} className="text-primary" />
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
                  ? "bg-primary text-primary-foreground shadow-xs"
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
          <div className="flex items-center justify-between gap-1">
            <div className="min-w-0">
              <p className="text-muted-foreground truncate text-xs font-medium">
                nimAgents v1.0
              </p>
              <p className="text-muted-foreground/80 truncate text-[10px]">
                NVIDIA NIM Ready
              </p>
            </div>
            <div className="flex items-center gap-1">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={toggleTheme}
                title="Toggle dark/light theme"
              >
                <Sun className="h-3.5 w-3.5 scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
                <Moon className="absolute h-3.5 w-3.5 scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={handleLogout}
              >
                <LogOut className="mr-1 h-3.5 w-3.5" />
                Exit
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={toggleTheme}
              title="Toggle theme"
            >
              <Sun className="h-4 w-4 dark:hidden" />
              <Moon className="hidden h-4 w-4 dark:block" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={handleLogout}
              title="Exit"
            >
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>
    </aside>
  );
}
