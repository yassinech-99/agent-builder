"use client";

import { ReactNode } from "react";
import { Sidebar } from "./sidebar";
import { Toaster } from "@/components/ui/sonner";

export function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-screen w-full overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">
        {children}
      </main>
      <Toaster />
    </div>
  );
}
