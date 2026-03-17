"use client";

import { FormEvent, useEffect, useState } from "react";
import { NovaGridLogo } from "@/components/icons/grid-agents";
import { RuntimeSplash } from "@/components/runtime-splash";

const MOCK_USER = "demo";
const MOCK_PASS = "demo123";

export default function HomePage() {
  const [isAuthed, setIsAuthed] = useState(false);
  const [email, setEmail] = useState("");
  const [newsletterDone, setNewsletterDone] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    const logged = sessionStorage.getItem("novagrid:mock-auth");
    if (logged === "true") {
      setIsAuthed(true);
    }
  }, []);

  const handleNewsletter = (e: FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    localStorage.setItem("novagrid:newsletter-email", email.trim());
    setNewsletterDone(true);
    setEmail("");
  };

  const handleMockLogin = (e: FormEvent) => {
    e.preventDefault();
    if (username.trim() === MOCK_USER && password === MOCK_PASS) {
      sessionStorage.setItem("novagrid:mock-auth", "true");
      setAuthError(null);
      setIsAuthed(true);
      return;
    }
    setAuthError("Invalid credentials. Use demo / demo123.");
  };

  if (isAuthed) {
    return <RuntimeSplash />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-6 py-10 dark:from-slate-950 dark:to-slate-900">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-10">
        <header className="flex items-center gap-3">
          <NovaGridLogo width={36} height={36} className="text-primary" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight">NovaGrid Agents</h1>
            <p className="text-muted-foreground text-sm">
              Nova-native enterprise agent workspace on AWS
            </p>
          </div>
        </header>

        <section className="grid gap-8 md:grid-cols-[1.2fr_0.8fr]">
          <div className="space-y-6">
            <div className="rounded-2xl border bg-white/80 p-8 shadow-sm backdrop-blur dark:bg-slate-900/70">
              <p className="text-primary mb-2 text-sm font-medium uppercase tracking-wider">
                Product
              </p>
              <h2 className="text-4xl font-semibold tracking-tight">
                Nova-powered agents with reasoning, retrieval, and browser automation
              </h2>
              <p className="text-muted-foreground mt-4 text-base">
                NovaGrid Agents lets teams build enterprise AI agents on AWS, powered by
                Nova 2 Lite reasoning, Nova embeddings, and Nova Act browser automation.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border bg-white/80 p-5 shadow-sm dark:bg-slate-900/60">
                <h3 className="font-medium">Nova-Powered Reasoning</h3>
                <p className="text-muted-foreground mt-2 text-sm">
                  Amazon Nova 2 Lite for fast, cost-effective agent reasoning and tool use.
                </p>
              </div>
              <div className="rounded-xl border bg-white/80 p-5 shadow-sm dark:bg-slate-900/60">
                <h3 className="font-medium">Nova Embeddings RAG</h3>
                <p className="text-muted-foreground mt-2 text-sm">
                  Nova Multimodal Embeddings with S3 and local document ingestion.
                </p>
              </div>
              <div className="rounded-xl border bg-white/80 p-5 shadow-sm dark:bg-slate-900/60">
                <h3 className="font-medium">Nova Act Automation</h3>
                <p className="text-muted-foreground mt-2 text-sm">
                  Browser workflow automation powered by Amazon Nova Act.
                </p>
              </div>
              <div className="rounded-xl border bg-white/80 p-5 shadow-sm dark:bg-slate-900/60">
                <h3 className="font-medium">AWS-Native Platform</h3>
                <p className="text-muted-foreground mt-2 text-sm">
                  Bedrock models, S3 knowledge, and enterprise-grade agent management.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="rounded-xl border bg-white/80 p-6 shadow-sm dark:bg-slate-900/70">
              <h3 className="text-lg font-semibold">Join Newsletter</h3>
              <p className="text-muted-foreground mt-1 text-sm">
                Get release notes and product updates.
              </p>
              <form onSubmit={handleNewsletter} className="mt-4 flex flex-col gap-3">
                <input
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  type="email"
                  placeholder="you@example.com"
                  className="bg-background w-full rounded-md border px-3 py-2 text-sm outline-none"
                />
                <button
                  type="submit"
                  className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-md px-4 py-2 text-sm font-medium"
                >
                  Subscribe
                </button>
              </form>
              {newsletterDone && (
                <p className="mt-3 text-xs text-green-600 dark:text-green-400">
                  Thanks. You are subscribed.
                </p>
              )}
            </div>

            <div className="rounded-xl border bg-white/80 p-6 shadow-sm dark:bg-slate-900/70">
              <h3 className="text-lg font-semibold">Mock Login</h3>
              <p className="text-muted-foreground mt-1 text-sm">
                Sign in to start runtime checks and enter the dashboard.
              </p>
              <form onSubmit={handleMockLogin} className="mt-4 flex flex-col gap-3">
                <input
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  type="text"
                  placeholder="Username"
                  className="bg-background w-full rounded-md border px-3 py-2 text-sm outline-none"
                />
                <input
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  type="password"
                  placeholder="Password"
                  className="bg-background w-full rounded-md border px-3 py-2 text-sm outline-none"
                />
                <button
                  type="submit"
                  className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-md px-4 py-2 text-sm font-medium"
                >
                  Continue
                </button>
              </form>
              <p className="text-muted-foreground mt-3 text-xs">
                Demo credentials: <strong>demo</strong> / <strong>demo123</strong>
              </p>
              {authError && (
                <p className="mt-2 text-xs text-red-600 dark:text-red-400">
                  {authError}
                </p>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
