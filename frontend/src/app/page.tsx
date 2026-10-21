"use client";

import { FormEvent, useEffect, useState } from "react";
import { NimAgentsLogo } from "@/components/icons/nim-agents";
import { RuntimeSplash } from "@/components/runtime-splash";
import { NIMAGENTS_STORAGE } from "@/lib/runtime-urls";
import { Cpu, Database, Layers, ShieldCheck, ArrowRight } from "lucide-react";

const DEMO_USER = "demo";
const DEMO_PASS = "demo123";

export default function HomePage() {
  const [isAuthed, setIsAuthed] = useState(false);
  const [email, setEmail] = useState("");
  const [newsletterDone, setNewsletterDone] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    const logged = sessionStorage.getItem(NIMAGENTS_STORAGE.auth);
    if (logged === "true") {
      setIsAuthed(true);
    }
  }, []);

  const handleNewsletter = (e: FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    localStorage.setItem(NIMAGENTS_STORAGE.newsletter, email.trim());
    setNewsletterDone(true);
    setEmail("");
  };

  const handleLogin = (e: FormEvent) => {
    e.preventDefault();
    if (username.trim() === DEMO_USER && password === DEMO_PASS) {
      sessionStorage.setItem(NIMAGENTS_STORAGE.auth, "true");
      setAuthError(null);
      setIsAuthed(true);
      return;
    }
    setAuthError("Invalid credentials. Use demo / demo123.");
  };

  const handleQuickDemoAccess = () => {
    sessionStorage.setItem(NIMAGENTS_STORAGE.auth, "true");
    setAuthError(null);
    setIsAuthed(true);
  };

  if (isAuthed) {
    return <RuntimeSplash />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-50 via-white to-emerald-50/40 px-6 py-10 dark:from-zinc-950 dark:via-zinc-950 dark:to-emerald-950/20">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <NimAgentsLogo width={38} height={38} className="text-primary" />
            <div>
              <h1 className="text-2xl font-bold tracking-tight">nimAgents</h1>
              <p className="text-muted-foreground text-sm">
                NVIDIA NIM &amp; Multi-Provider AI Agent Studio
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleQuickDemoAccess}
            className="border-primary/30 bg-primary/10 text-primary hover:bg-primary/20 inline-flex items-center gap-1.5 rounded-full border px-4 py-1.5 text-xs font-medium transition-colors"
          >
            Launch Studio
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </header>

        <section className="grid gap-8 md:grid-cols-[1.25fr_0.75fr]">
          <div className="space-y-6">
            <div className="bg-card/85 rounded-2xl border p-8 shadow-sm backdrop-blur">
              <p className="text-primary mb-2 text-xs font-semibold uppercase tracking-widest">
                Open GPU-Native Agent Platform
              </p>
              <h2 className="text-4xl font-semibold leading-tight tracking-tight">
                Reasoning agents powered by NVIDIA NIM, Dynamo, and any open model endpoint
              </h2>
              <p className="text-muted-foreground mt-4 text-base leading-relaxed">
                Build, test, and scale tool-using agents across Nemotron, DeepSeek-R1,
                Qwen3, OpenAI, Anthropic, vLLM, SGLang, and Ollama — with live
                reasoning traces, hybrid RAG + reranking, and MCP discovery.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="bg-card/80 rounded-xl border p-5 shadow-sm">
                <div className="text-primary mb-2 flex items-center gap-2 font-medium">
                  <Cpu className="h-4 w-4" />
                  <h3>NVIDIA NIM &amp; Dynamo Core</h3>
                </div>
                <p className="text-muted-foreground text-sm">
                  Live <code>/v1/models</code> discovery, role routing, fallback chains,
                  and collapsible <code>&lt;think&gt;</code> token streaming.
                </p>
              </div>
              <div className="bg-card/80 rounded-xl border p-5 shadow-sm">
                <div className="text-primary mb-2 flex items-center gap-2 font-medium">
                  <Database className="h-4 w-4" />
                  <h3>Hybrid Knowledge + Reranking</h3>
                </div>
                <p className="text-muted-foreground text-sm">
                  Combine NVIDIA Nemotron Embeddings, NVIDIARerank, Meilisearch BM25,
                  and S3/MinIO object buckets.
                </p>
              </div>
              <div className="bg-card/80 rounded-xl border p-5 shadow-sm">
                <div className="text-primary mb-2 flex items-center gap-2 font-medium">
                  <Layers className="h-4 w-4" />
                  <h3>Multi-Provider Freedom</h3>
                </div>
                <p className="text-muted-foreground text-sm">
                  Switch seamlessly between Hosted NIM, self-hosted vLLM/SGLang/Ollama,
                  OpenRouter, OpenAI, and Anthropic.
                </p>
              </div>
              <div className="bg-card/80 rounded-xl border p-5 shadow-sm">
                <div className="text-primary mb-2 flex items-center gap-2 font-medium">
                  <ShieldCheck className="h-4 w-4" />
                  <h3>Guardrails &amp; MCP Ecosystem</h3>
                </div>
                <p className="text-muted-foreground text-sm">
                  Real-time PII/secret protection, Playwright browser automation, and
                  streamable HTTP MCP servers.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-card/85 rounded-xl border p-6 shadow-sm">
              <h3 className="text-lg font-semibold">Workspace Sign In</h3>
              <p className="text-muted-foreground mt-1 text-sm">
                Verify runtime services and open your agent workspace.
              </p>
              <form onSubmit={handleLogin} className="mt-4 flex flex-col gap-3">
                <input
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  type="text"
                  placeholder="Username"
                  className="bg-background w-full rounded-md border px-3 py-2 text-sm outline-none focus:ring-2"
                />
                <input
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  type="password"
                  placeholder="Password"
                  className="bg-background w-full rounded-md border px-3 py-2 text-sm outline-none focus:ring-2"
                />
                <button
                  type="submit"
                  className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-md px-4 py-2 text-sm font-medium transition-colors"
                >
                  Sign In to nimAgents
                </button>
                <button
                  type="button"
                  onClick={handleQuickDemoAccess}
                  className="hover:bg-muted rounded-md border px-4 py-2 text-xs font-medium transition-colors"
                >
                  Continue with Demo Workspace (demo / demo123)
                </button>
              </form>
              {authError && (
                <p className="mt-2 text-xs text-red-600 dark:text-red-400">
                  {authError}
                </p>
              )}
            </div>

            <div className="bg-card/85 rounded-xl border p-6 shadow-sm">
              <h3 className="text-base font-semibold">Release Updates</h3>
              <p className="text-muted-foreground mt-1 text-xs">
                Follow new NIM model presets, memory modules, and runtime updates.
              </p>
              <form onSubmit={handleNewsletter} className="mt-3 flex flex-col gap-2.5">
                <input
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  type="email"
                  placeholder="you@organization.org"
                  className="bg-background w-full rounded-md border px-3 py-2 text-sm outline-none"
                />
                <button
                  type="submit"
                  className="bg-secondary text-secondary-foreground hover:bg-secondary/80 rounded-md px-4 py-2 text-xs font-medium"
                >
                  Subscribe
                </button>
              </form>
              {newsletterDone && (
                <p className="mt-2.5 text-xs text-emerald-600 dark:text-emerald-400">
                  Subscribed to nimAgents release notes.
                </p>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
