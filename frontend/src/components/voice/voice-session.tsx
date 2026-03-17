"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Mic, MicOff, Send, Volume2 } from "lucide-react";

const BUILDER_API =
  process.env.NEXT_PUBLIC_BUILDER_API_URL || "http://localhost:8100";

export interface VoiceMessage {
  role: "user" | "assistant" | "system";
  text: string;
  timestamp: number;
}

export type SessionState = "idle" | "connecting" | "listening" | "thinking" | "speaking" | "error";

export function StatusBadge({ state }: { state: SessionState }) {
  const config: Record<SessionState, { label: string; className: string }> = {
    idle: { label: "Offline", className: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400" },
    connecting: { label: "Connecting", className: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400" },
    listening: { label: "Listening", className: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400" },
    thinking: { label: "Thinking", className: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400" },
    speaking: { label: "Speaking", className: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400" },
    error: { label: "Error", className: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400" },
  };
  const c = config[state];
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${c.className}`}>
      {state === "listening" && (
        <span className="mr-1.5 h-2 w-2 animate-pulse rounded-full bg-green-500" />
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
}

export function VoiceSession({ inline = false }: VoiceSessionProps) {
  const [state, setState] = useState<SessionState>("idle");
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [textInput, setTextInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [micActive, setMicActive] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    return () => {
      stopMic();
      audioContextRef.current?.close().catch(() => {});
    };
  }, []);

  const addMessage = useCallback((role: VoiceMessage["role"], text: string) => {
    setMessages((prev) => [...prev, { role, text, timestamp: Date.now() }]);
  }, []);

  const playAudioChunk = useCallback((base64Audio: string) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }
      const ctx = audioContextRef.current;
      const buffer = base64ToArrayBuffer(base64Audio);
      ctx.decodeAudioData(buffer.slice(0)).then((audioBuffer) => {
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        source.start(0);
      }).catch(() => {});
    } catch {
      // Audio playback not available
    }
  }, []);

  const stopMic = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    setMicActive(false);
  }, []);

  const startMic = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

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

      const recorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });
      recorderRef.current = recorder;

      recorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          const buffer = await event.data.arrayBuffer();
          const base64 = arrayBufferToBase64(buffer);
          wsRef.current.send(JSON.stringify({ type: "audio", data: base64 }));
        }
      };

      recorder.start(500);
      setMicActive(true);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Microphone access denied";
      setError(`Mic error: ${msg}`);
    }
  }, []);

  const toggleMic = useCallback(() => {
    if (micActive) {
      stopMic();
    } else {
      startMic();
    }
  }, [micActive, stopMic, startMic]);

  const connect = useCallback(async () => {
    setState("connecting");
    setError(null);

    try {
      await navigator.mediaDevices.getUserMedia({ audio: true }).then((s) => {
        s.getTracks().forEach((t) => t.stop());
      });
    } catch {
      setError("Microphone permission is required for voice mode.");
      setState("error");
      return;
    }

    const wsUrl = BUILDER_API.replace(/^http/, "ws") + "/api/voice/ws";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setState("listening");
      addMessage("system", "Voice session started.");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case "transcript":
            addMessage(msg.role, msg.text);
            break;
          case "audio":
            if (msg.data) playAudioChunk(msg.data);
            break;
          case "status":
            setState(msg.state as SessionState);
            break;
          case "tool_use":
            addMessage("system", `Tool call: ${msg.name}`);
            break;
          case "error":
            setError(msg.detail);
            break;
        }
      } catch {
        // ignore
      }
    };

    ws.onclose = () => {
      setState("idle");
      wsRef.current = null;
      stopMic();
    };

    ws.onerror = () => {
      setError("WebSocket connection failed");
      setState("error");
    };
  }, [addMessage, playAudioChunk, stopMic]);

  const disconnect = useCallback(() => {
    stopMic();
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setState("idle");
    addMessage("system", "Voice session ended.");
  }, [addMessage, stopMic]);

  const sendText = useCallback(() => {
    if (!textInput.trim() || !wsRef.current) return;
    wsRef.current.send(
      JSON.stringify({ type: "text", data: textInput.trim() }),
    );
    addMessage("user", textInput.trim());
    setTextInput("");
  }, [textInput, addMessage]);

  const isConnected = state !== "idle" && state !== "error" && state !== "connecting";

  const micButton = isConnected && (
    <Button
      variant={micActive ? "destructive" : "outline"}
      size="sm"
      onClick={toggleMic}
      title={micActive ? "Mute microphone" : "Unmute microphone"}
    >
      {micActive ? <MicOff className="h-3.5 w-3.5" /> : <Mic className="h-3.5 w-3.5" />}
    </Button>
  );

  if (inline) {
    return (
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <StatusBadge state={state} />
          {isConnected ? (
            <>
              {micButton}
              <Button variant="destructive" size="sm" onClick={disconnect}>
                <MicOff className="mr-1 h-3.5 w-3.5" />
                End
              </Button>
            </>
          ) : (
            <Button size="sm" onClick={connect} disabled={state === "connecting"}>
              <Mic className="mr-1 h-3.5 w-3.5" />
              {state === "connecting" ? "Connecting..." : "Start Voice"}
            </Button>
          )}
        </div>

        {messages.length > 0 && (
          <div className="max-h-60 space-y-2 overflow-auto rounded-lg border p-3">
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
                  className={`max-w-[80%] rounded-lg px-3 py-1.5 text-sm ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : msg.role === "system"
                        ? "text-muted-foreground text-xs italic"
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

        {error && (
          <p className="text-xs text-red-600">{error}</p>
        )}

        {isConnected && (
          <div className="flex gap-2">
            <Input
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendText()}
              placeholder="Type to the voice agent..."
              className="flex-1"
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
      <div className="flex-1 overflow-auto p-6">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <Volume2 className="text-muted-foreground mb-4 h-16 w-16" />
            <h2 className="mb-2 text-lg font-semibold">Nova 2 Sonic Voice Agent</h2>
            <p className="text-muted-foreground mb-6 max-w-md text-sm">
              Start a voice session to interact with your AI agent using natural
              language. Microphone permission will be requested.
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
        <div className="border-t border-red-200 bg-red-50 px-6 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-400">
          {error}
        </div>
      )}

      {isConnected && (
        <div className="border-t px-6 py-4">
          <div className="mx-auto flex max-w-2xl gap-2">
            {micButton}
            <Input
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendText()}
              placeholder="Type a message to the voice agent..."
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
