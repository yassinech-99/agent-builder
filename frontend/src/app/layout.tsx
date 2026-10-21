import type { Metadata } from "next";
import "./globals.css";
import React from "react";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { ThemeProvider } from "next-themes";

export const metadata: Metadata = {
  title: "nimAgents — Multi-Provider NVIDIA NIM Agent Studio",
  description:
    "Build, orchestrate, and scale AI agents on NVIDIA NIM, NVIDIA Dynamo, OpenAI, Anthropic, OpenRouter, vLLM, and Ollama.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <NuqsAdapter>{children}</NuqsAdapter>
        </ThemeProvider>
      </body>
    </html>
  );
}
