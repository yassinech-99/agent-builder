"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getBuilderApiUrl, getVoiceWsUrl } from "@/lib/runtime-urls";
import { Hand, Mic, MicOff, Radio, Send, Volume2 } from "lucide-react";

export interface VoiceMessage {
  role: "user" | "assistant" | "system";
  text: string;
  timestamp: number;
}

export type SessionState =
  | "idle"
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking"
  | "error";

interface VoiceGatewayConfig {
  sample_rate_hz: number;
  pcm_format: string;
  max_ring_chunks: number;
  barge_in_supported: boolean;
  asr: { model: string; endpoint_configured: boolean; mode: string };
  tts: { model: string; endpoint_configured: boolean; mode: string };
  voices: Array<{ id: string; label: string; lang: string }>;
}

// Inline WebAudio `AudioWorkletProcessor` downsampling hardware audio to 16kHz signed 16-bit PCM (`pcm_s16le`)
const PCM16_WORKLET_SOURCE = [
  "class NimPcm16CaptureProcessor extends AudioWorkletProcessor {",
  "  constructor() {",
  "    super();",
  "    this._acc = [];",
  "    this._targetRate = 16000;",
  "  }",
  "  process(inputs) {",
  "    const ch = inputs && inputs[0] && inputs[0][0];",
  "    if (!ch || ch.length === 0) return true;",
  "    let sumSq = 0;",
  "    const ratio = sampleRate / this._targetRate;",
  "    const outLen = Math.max(1, Math.floor(ch.length / ratio));",
  "    for (let i = 0; i < outLen; i++) {",
  "      const srcIdx = Math.min(ch.length - 1, Math.floor(i * ratio));",
  "      const s = Math.max(-1, Math.min(1, ch[srcIdx]));",
  "      sumSq += s * s;",
  "      this._acc.push(s < 0 ? s * 0x8000 : s * 0x7fff);",
  "    }",
  "    const rms = Math.sqrt(sumSq / outLen);",
  "    if (this._acc.length >= 4096) {",
  "      const pcm = new Int16Array(this._acc.splice(0, 4096));",
  "      this.port.postMessage({ pcmBuffer: pcm.buffer, rms }, [pcm.buffer]);",
  "    } else {",
  "      this.port.postMessage({ rms });",
  "    }",
  "    return true;",
  "  }",
  "}",
  "registerProcessor('nim-pcm16-capture', NimPcm16CaptureProcessor);",
].join("\n");

export function StatusBadge({ state }: { state: SessionState }) {
  const config: Record<SessionState, { label: string; className: string }> = {
    idle: {
      label: "Offline",
      className: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
    },
    connecting: {
      label: "Connecting",
      className:
        "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
    },
    listening: {
      label: "Listening (16kHz PCM)",
      className:
        "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
    },
    thinking: {
      label: "Thinking",
      className:
        "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
    },
    speaking: {
      label: "Speaking",
      className:
        "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
    },
    error: {
      label: "Error",
      className: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
    },
  };
  const c = config[state];
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${c.className}`}
    >
      {state === "listening" && (
        <span className="mr-1.5 h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
      )}
      {c.label}
    </span>
  );
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

interface VoiceSessionProps {
  inline?: boolean;
  threadId?: string | null;
}

export function VoiceSession({ inline = false, threadId }: VoiceSessionProps) {
  const [state, setState] = useState<SessionState>("idle");
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [textInput, setTextInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [micActive, setMicActive] = useState(false);
  const [rmsLevel, setRmsLevel] = useState(0);
  const [selectedVoice, setSelectedVoice] = useState("English-US.Female-1");
  const [gatewayCfg, setGatewayCfg] = useState<VoiceGatewayConfig | null>(null);
  const [bargeInsCount, setBargeInsCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const captureCtxRef = useRef<AudioContext | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const activeSourceRef = useRef<AudioBufferSourceNode | null>(null);

  useEffect(() => {
    fetch(`${getBuilderApiUrl()}/api/voice/config`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setGatewayCfg(data);
      })
      .catch(() => {});
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const stopActivePlayback = useCallback(() => {
    try {
      activeSourceRef.current?.stop();
    } catch {
      // already stopped
    }
    activeSourceRef.current = null;
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const stopMic = useCallback((flush = true) => {
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (captureCtxRef.current) {
      captureCtxRef.current.close().catch(() => {});
      captureCtxRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (flush && wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "flush_audio" }));
    }
    setMicActive(false);
    setRmsLevel(0);
  }, []);

  useEffect(() => {
    return () => {
      stopActivePlayback();
      stopMic(false);
      audioContextRef.current?.close().catch(() => {});
    };
  }, [stopActivePlayback, stopMic]);

  const addMessage = useCallback((role: VoiceMessage["role"], text: string) => {
    setMessages((prev) => [...prev, { role, text, timestamp: Date.now() }]);
  }, []);

  const playAudioChunk = useCallback(
    (base64Audio: string) => {
      try {
        stopActivePlayback();
        if (!audioContextRef.current) {
          audioContextRef.current = new AudioContext();
        }
        const ctx = audioContextRef.current;
        const buffer = base64ToArrayBuffer(base64Audio);
        ctx
          .decodeAudioData(buffer.slice(0))
          .then((audioBuffer) => {
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);
            activeSourceRef.current = source;
            source.onended = () => {
              if (activeSourceRef.current === source) {
                activeSourceRef.current = null;
              }
            };
            source.start(0);
          })
          .catch(() => {});
      } catch {
        // Audio playback unavailable
      }
    },
    [stopActivePlayback],
  );

  const speakBrowserFallback = useCallback(
    (text: string) => {
      if (gatewayCfg?.tts.endpoint_configured) return;
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
      try {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text.slice(0, 500));
        utter.rate = 1.05;
        window.speechSynthesis.speak(utter);
      } catch {
        // ignore if unsupported
      }
    },
    [gatewayCfg],
  );

  const triggerBargeIn = useCallback(() => {
    stopActivePlayback();
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "barge_in" }));
      setBargeInsCount((c) => c + 1);
    }
  }, [stopActivePlayback]);

  const startMic = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // Starting speech while assistant is thinking/speaking triggers automatic barge-in!
    if (state === "speaking" || state === "thinking") {
      triggerBargeIn();
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      mediaStreamRef.current = stream;

      const audioCtx = new AudioContext();
      captureCtxRef.current = audioCtx;
      const blob = new Blob([PCM16_WORKLET_SOURCE], {
        type: "application/javascript",
      });
      const moduleUrl = URL.createObjectURL(blob);
      await audioCtx.audioWorklet.addModule(moduleUrl);
      URL.revokeObjectURL(moduleUrl);

      const source = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, "nim-pcm16-capture");
      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (ev) => {
        const { pcmBuffer, rms } = ev.data || {};
        if (typeof rms === "number") {
          setRmsLevel(Math.min(1, rms * 5));
        }
        if (pcmBuffer && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: "audio",
              format: "pcm16",
              sample_rate: 16000,
              data: arrayBufferToBase64(pcmBuffer),
            }),
          );
        }
      };

      source.connect(workletNode);
      setMicActive(true);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Microphone access denied";
      setError(`Microphone unavailable (${msg}) — text stream remains active.`);
    }
  }, [state, triggerBargeIn]);

  const toggleMic = useCallback(() => {
    if (micActive) {
      stopMic(true);
    } else {
      startMic();
    }
  }, [micActive, stopMic, startMic]);

  const connect = useCallback(() => {
    setState("connecting");
    setError(null);

    const baseWs = getVoiceWsUrl();
    const sep = baseWs.includes("?") ? "&" : "?";
    const params = new URLSearchParams({
      voice: selectedVoice,
      ...(threadId ? { thread_id: threadId } : {}),
    });
    const ws = new WebSocket(`${baseWs}${sep}${params.toString()}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setState("listening");
      addMessage(
        "system",
        threadId
          ? `Connected to nimAgents Riva Voice Gateway (shared session ${threadId.slice(0, 8)}).`
          : "Connected to nimAgents Riva Voice Gateway (16kHz AudioWorklet PCM).",
      );
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "transcript":
            addMessage(msg.role, msg.text);
            if (msg.role === "assistant") {
              speakBrowserFallback(msg.text);
            }
            break;
          case "audio":
            if (msg.data) playAudioChunk(msg.data);
            break;
          case "status":
            setState(msg.state as SessionState);
            break;
          case "barge_in_ack":
            stopActivePlayback();
            addMessage("system", "Interrupted active response (Barge-In).");
            break;
          case "backpressure":
            setError(
              `Ring buffer backpressure guard evicted oldest frames (${msg.dropped_frames} dropped).`,
            );
            break;
          case "error":
            setError(msg.detail);
            break;
        }
      } catch {
        // ignore invalid frames
      }
    };

    ws.onclose = () => {
      setState("idle");
      wsRef.current = null;
      setMicActive(false);
    };

    ws.onerror = () => {
      setError("Voice WebSocket connection failed.");
      setState("error");
    };
  }, [addMessage, playAudioChunk, selectedVoice, speakBrowserFallback, stopActivePlayback, threadId]);

  const disconnect = useCallback(() => {
    stopActivePlayback();
    stopMic(false);
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setState("idle");
    addMessage("system", "Voice session ended.");
  }, [addMessage, stopActivePlayback, stopMic]);

  const sendText = useCallback(() => {
    if (!textInput.trim() || !wsRef.current) return;
    stopActivePlayback();
    wsRef.current.send(
      JSON.stringify({ type: "text", data: textInput.trim() }),
    );
    addMessage("user", textInput.trim());
    setTextInput("");
  }, [textInput, addMessage, stopActivePlayback]);

  const isConnected =
    state !== "idle" && state !== "error" && state !== "connecting";

  const micButton = isConnected && (
    <Button
      variant={micActive ? "destructive" : "outline"}
      size="sm"
      onClick={toggleMic}
      title={micActive ? "Flush 16kHz PCM utterance" : "Capture 16kHz PCM via AudioWorklet"}
    >
      {micActive ? (
        <MicOff className="h-3.5 w-3.5" />
      ) : (
        <Mic className="h-3.5 w-3.5" />
      )}
    </Button>
  );

  const bargeInButton = isConnected && (state === "thinking" || state === "speaking") && (
    <Button
      variant="secondary"
      size="sm"
      onClick={triggerBargeIn}
      title="Instant Barge-In: cancel in-flight generation & audio immediately"
    >
      <Hand className="mr-1 h-3.5 w-3.5 text-amber-500" />
      Interrupt
    </Button>
  );

  if (inline) {
    return (
      <div className="bg-card/95 flex flex-col gap-3 rounded-xl border p-3.5 shadow-sm backdrop-blur">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Radio className="text-primary h-4 w-4" />
            <span className="text-xs font-semibold">Live Riva Voice Mode</span>
            <StatusBadge state={state} />
          </div>
          <div className="flex items-center gap-1.5">
            {isConnected ? (
              <>
                {micButton}
                {bargeInButton}
                <Button variant="outline" size="sm" onClick={disconnect}>
                  Disconnect
                </Button>
              </>
            ) : (
              <Button size="sm" onClick={connect} disabled={state === "connecting"}>
                <Mic className="mr-1 h-3.5 w-3.5" />
                {state === "connecting" ? "Connecting..." : "Connect Voice"}
              </Button>
            )}
          </div>
        </div>

        {micActive && (
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground font-mono text-[10px]">
              16kHz PCM Level
            </span>
            <div className="bg-muted h-1.5 flex-1 overflow-hidden rounded-full">
              <div
                className="bg-primary h-full transition-all duration-75"
                style={{ width: `${Math.round(rmsLevel * 100)}%` }}
              />
            </div>
          </div>
        )}

        {messages.length > 0 && (
          <div className="max-h-44 space-y-1.5 overflow-auto rounded-lg border p-2.5">
            {messages.slice(-8).map((msg, i) => (
              <div
                key={i}
                className={`flex ${
                  msg.role === "user"
                    ? "justify-end"
                    : msg.role === "system"
                      ? "justify-center"
                      : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[85%] rounded-lg px-2.5 py-1 text-xs ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : msg.role === "system"
                        ? "text-muted-foreground italic"
                        : "bg-muted"
                  }`}
                >
                  {msg.text}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        {error && <p className="text-xs text-amber-600 dark:text-amber-400">{error}</p>}

        {isConnected && (
          <div className="flex gap-2">
            <Input
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendText()}
              placeholder="Speak or send a live prompt over WebSocket..."
              className="h-8 flex-1 text-xs"
            />
            <Button size="sm" onClick={sendText} disabled={!textInput.trim()}>
              <Send className="h-3.5 w-3.5" />
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-6 py-2.5">
        <div className="flex flex-wrap items-center gap-2.5">
          <StatusBadge state={state} />
          {gatewayCfg && (
            <span className="text-muted-foreground font-mono text-xs">
              ASR: {gatewayCfg.asr.model.split("/").pop()} · TTS:{" "}
              {gatewayCfg.tts.model.split("/").pop()} · Barge-Ins: {bargeInsCount}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={selectedVoice}
            onChange={(e) => setSelectedVoice(e.target.value)}
            disabled={isConnected}
            className="border-input bg-background h-8 rounded-md border px-2.5 text-xs"
          >
            {(gatewayCfg?.voices || [
              { id: "English-US.Female-1", label: "Riva Magpie — Aria (US Female)" },
            ]).map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
          {bargeInButton}
          {isConnected ? (
            <Button variant="outline" size="sm" onClick={disconnect}>
              End Session
            </Button>
          ) : (
            <Button size="sm" onClick={connect} disabled={state === "connecting"}>
              <Mic className="mr-1.5 h-3.5 w-3.5" />
              {state === "connecting" ? "Connecting..." : "Connect Session"}
            </Button>
          )}
        </div>
      </div>

      {micActive && (
        <div className="bg-muted/40 flex items-center gap-3 border-b px-6 py-1.5">
          <span className="text-primary font-mono text-xs font-medium">
            AudioWorklet 16,000 Hz PCM_S16LE Stream
          </span>
          <div className="bg-muted h-2 max-w-md flex-1 overflow-hidden rounded-full">
            <div
              className="bg-primary h-full transition-all duration-75"
              style={{ width: `${Math.round(rmsLevel * 100)}%` }}
            />
          </div>
        </div>
      )}

      <div className="flex-1 overflow-auto p-6">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <Volume2 className="text-primary mb-4 h-16 w-16 opacity-80" />
            <h2 className="mb-2 text-lg font-semibold">
              nimAgents Full-Duplex Riva Voice Gateway
            </h2>
            <p className="text-muted-foreground mb-6 max-w-lg text-sm">
              Streams 16kHz PCM audio from a WebAudio <code>AudioWorklet</code> ring buffer to NVIDIA Riva Speech NIMs (<code>Parakeet ASR</code> + <code>Magpie TTS</code>) with sub-5ms barge-in interruption and NemoGuard pre/post shields.
            </p>
            <Button onClick={connect} disabled={state === "connecting"}>
              <Mic className="mr-1.5 h-4 w-4" />
              Start Voice Session
            </Button>
          </div>
        ) : (
          <div className="mx-auto max-w-2xl space-y-4">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${
                  msg.role === "user"
                    ? "justify-end"
                    : msg.role === "system"
                      ? "justify-center"
                      : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2.5 text-sm ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : msg.role === "system"
                        ? "text-muted-foreground bg-transparent text-xs italic"
                        : "bg-muted"
                  }`}
                >
                  {msg.text}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {error && (
        <div className="border-t border-amber-200 bg-amber-50 px-6 py-2.5 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-300">
          {error}
        </div>
      )}

      {isConnected && (
        <div className="border-t px-6 py-4">
          <div className="mx-auto flex max-w-2xl gap-2">
            {micButton}
            {bargeInButton}
            <Input
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendText()}
              placeholder="Speak with 16kHz AudioWorklet mic or type a prompt..."
              className="flex-1"
            />
            <Button onClick={sendText} disabled={!textInput.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
