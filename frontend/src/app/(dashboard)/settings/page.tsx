"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState("");
  const [builderApiUrl, setBuilderApiUrl] = useState("");

  useEffect(() => {
    setApiUrl(
      localStorage.getItem("settings:apiUrl") ||
        process.env.NEXT_PUBLIC_API_URL ||
        "http://localhost:2024",
    );
    setBuilderApiUrl(
      localStorage.getItem("settings:builderApiUrl") ||
        process.env.NEXT_PUBLIC_BUILDER_API_URL ||
        "http://localhost:8100",
    );
  }, []);

  const handleSave = () => {
    localStorage.setItem("settings:apiUrl", apiUrl);
    localStorage.setItem("settings:builderApiUrl", builderApiUrl);
    toast.success("Settings saved");
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b px-6 py-4">
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground text-sm">
          Configure your platform settings
        </p>
      </div>

      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-xl space-y-6">
          <div className="bg-card rounded-lg border p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Server Configuration</h2>
            <div className="space-y-4">
              <div className="flex flex-col gap-1.5">
                <Label>Agent Runtime URL</Label>
                <Input
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  placeholder="http://localhost:2024"
                />
                <p className="text-muted-foreground text-xs">
                  URL of the agent runtime server for execution
                </p>
              </div>
              <div className="flex flex-col gap-1.5">
                <Label>Builder API URL</Label>
                <Input
                  value={builderApiUrl}
                  onChange={(e) => setBuilderApiUrl(e.target.value)}
                  placeholder="http://localhost:8100"
                />
                <p className="text-muted-foreground text-xs">
                  URL of the Builder API for models, RAG, and MCP discovery
                </p>
              </div>
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={handleSave}>Save Settings</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
