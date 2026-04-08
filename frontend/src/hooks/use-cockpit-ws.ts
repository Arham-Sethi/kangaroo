/**
 * WebSocket hook for the multi-model Cockpit.
 *
 * Connects to `ws://host/api/v1/ws/{sessionId}?token=<jwt>`
 * and handles the full cockpit streaming protocol:
 *   - connected, model_response, model_queued, prompt_received
 *   - ping/pong heartbeat (every 25s)
 *   - auto-reconnect with exponential backoff
 *   - error and disconnect handling
 *
 * Usage:
 *   const { connected, messages, sendPrompt, addModel, removeModel }
 *     = useCockpitWS(sessionId);
 */

"use client";

import * as React from "react";
import { getAccessToken } from "@/lib/api-client";

// ── Types ──────────────────────────────────────────────────────────────────

/** A streamed model response message accumulating content. */
export interface ModelStreamMessage {
  readonly id: string;
  readonly model: string;
  readonly role: "user" | "assistant";
  readonly content: string;
  readonly done: boolean;
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly costUsd: number;
  readonly error?: string;
  readonly timestamp: number;
}

/** Connection status reported by the hook. */
export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

/** Server message envelope from the WS stream. */
interface ServerMessage {
  readonly type: string;
  readonly data: Record<string, unknown>;
  readonly message_id?: string;
  readonly timestamp?: number;
}

/** Per-model streaming state (content accumulator). */
interface StreamAccumulator {
  content: string;
  done: boolean;
  promptTokens: number;
  completionTokens: number;
  costUsd: number;
  error?: string;
}

// ── Constants ──────────────────────────────────────────────────────────────

const WS_BASE_URL =
  (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1")
    .replace(/^http/, "ws");

const HEARTBEAT_INTERVAL_MS = 25_000;
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;
const MAX_RECONNECT_ATTEMPTS = 10;

// ── Hook ───────────────────────────────────────────────────────────────────

export interface UseCockpitWSReturn {
  /** Current connection status. */
  readonly status: ConnectionStatus;
  /** Whether the WS is connected and authenticated. */
  readonly connected: boolean;
  /** All chat messages (user + assistant) in chronological order. */
  readonly messages: readonly ModelStreamMessage[];
  /** Models currently being queued/streamed (loading indicators). */
  readonly streamingModels: ReadonlySet<string>;
  /** Session data from the server (session_id, models, cost, etc.). */
  readonly sessionData: Record<string, unknown> | null;
  /** Send a prompt to the specified models. */
  readonly sendPrompt: (content: string, models: readonly string[]) => void;
  /** Request adding a model to the cockpit session. */
  readonly addModel: (modelId: string, role?: string) => void;
  /** Request removing a model from the cockpit session. */
  readonly removeModel: (modelId: string) => void;
  /** Clear all messages (local-only, does not affect server). */
  readonly clearMessages: () => void;
  /** Total accumulated cost across all model responses. */
  readonly totalCost: number;
}

export function useCockpitWS(sessionId: string | null): UseCockpitWSReturn {
  const [status, setStatus] = React.useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = React.useState<readonly ModelStreamMessage[]>([]);
  const [streamingModels, setStreamingModels] = React.useState<ReadonlySet<string>>(new Set());
  const [sessionData, setSessionData] = React.useState<Record<string, unknown> | null>(null);
  const [totalCost, setTotalCost] = React.useState(0);

  const wsRef = React.useRef<WebSocket | null>(null);
  const heartbeatRef = React.useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectAttemptRef = React.useRef(0);
  const reconnectTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const accumulatorsRef = React.useRef<Map<string, StreamAccumulator>>(new Map());
  const messageIdCounterRef = React.useRef(0);

  // Stable ref for the current prompt being tracked per-model
  const promptIdRef = React.useRef<string>("");

  const nextMessageId = React.useCallback((): string => {
    messageIdCounterRef.current += 1;
    return `msg-${Date.now()}-${messageIdCounterRef.current}`;
  }, []);

  // ── Cleanup helpers ────────────────────────────────────────────────────

  const clearHeartbeat = React.useCallback(() => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  }, []);

  const clearReconnectTimer = React.useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  // ── Message handler ────────────────────────────────────────────────────

  const handleServerMessage = React.useCallback(
    (event: MessageEvent) => {
      let msg: ServerMessage;
      try {
        msg = JSON.parse(event.data as string) as ServerMessage;
      } catch {
        return; // Ignore malformed messages
      }

      switch (msg.type) {
        case "connected": {
          setStatus("connected");
          reconnectAttemptRef.current = 0;
          if (msg.data.session) {
            setSessionData(msg.data.session as Record<string, unknown>);
          }
          break;
        }

        case "prompt_received": {
          // Server acknowledged the prompt
          break;
        }

        case "model_queued": {
          const modelId = msg.data.model as string;
          setStreamingModels((prev) => new Set([...prev, modelId]));
          // Initialize accumulator for this model
          const key = `${promptIdRef.current}-${modelId}`;
          accumulatorsRef.current.set(key, {
            content: "",
            done: false,
            promptTokens: 0,
            completionTokens: 0,
            costUsd: 0,
          });
          break;
        }

        case "model_response": {
          const modelId = msg.data.model as string;
          const delta = msg.data.content as string;
          const done = msg.data.done as boolean;
          const error = msg.data.error as string | undefined;

          const key = `${promptIdRef.current}-${modelId}`;
          const acc = accumulatorsRef.current.get(key);

          if (acc) {
            acc.content += delta;
            acc.done = done;

            if (done) {
              acc.promptTokens = (msg.data.prompt_tokens as number) || 0;
              acc.completionTokens = (msg.data.completion_tokens as number) || 0;
              acc.costUsd = (msg.data.cost_usd as number) || 0;
              if (error) acc.error = error;
            }

            // Update the messages array with current accumulated state
            const finalContent = acc.content;
            const finalDone = acc.done;
            const finalPromptTokens = acc.promptTokens;
            const finalCompletionTokens = acc.completionTokens;
            const finalCostUsd = acc.costUsd;
            const finalError = acc.error;

            setMessages((prev) => {
              const existingIdx = prev.findIndex(
                (m) => m.id === key && m.role === "assistant",
              );
              const updated: ModelStreamMessage = {
                id: key,
                model: modelId,
                role: "assistant",
                content: finalContent,
                done: finalDone,
                promptTokens: finalPromptTokens,
                completionTokens: finalCompletionTokens,
                costUsd: finalCostUsd,
                error: finalError,
                timestamp: Date.now(),
              };

              if (existingIdx >= 0) {
                return prev.map((m, i) => (i === existingIdx ? updated : m));
              }
              return [...prev, updated];
            });

            if (done) {
              setStreamingModels((prev) => {
                const next = new Set(prev);
                next.delete(modelId);
                return next;
              });
              setTotalCost((prev) => prev + (acc.costUsd || 0));
            }
          }
          break;
        }

        case "session_update": {
          setSessionData(msg.data as Record<string, unknown>);
          break;
        }

        case "pong": {
          // Heartbeat acknowledged
          break;
        }

        case "error": {
          const errorMsg = msg.data.message as string;
          // Add error as a system message
          setMessages((prev) => [
            ...prev,
            {
              id: `error-${Date.now()}`,
              model: "system",
              role: "assistant",
              content: `Error: ${errorMsg}`,
              done: true,
              promptTokens: 0,
              completionTokens: 0,
              costUsd: 0,
              timestamp: Date.now(),
            },
          ]);
          break;
        }
      }
    },
    [],
  );

  // ── Connect / Reconnect ────────────────────────────────────────────────

  const connect = React.useCallback(() => {
    if (!sessionId) return;

    const token = getAccessToken();
    if (!token) {
      setStatus("error");
      return;
    }

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.onclose = null; // Prevent reconnect loop
      wsRef.current.close();
    }

    setStatus("connecting");
    const url = `${WS_BASE_URL}/ws/${sessionId}?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      // Start heartbeat
      clearHeartbeat();
      heartbeatRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, HEARTBEAT_INTERVAL_MS);
    };

    ws.onmessage = handleServerMessage;

    ws.onerror = () => {
      setStatus("error");
    };

    ws.onclose = (event) => {
      clearHeartbeat();
      setStatus("disconnected");

      // Don't reconnect on auth errors
      if (event.code === 4001 || event.code === 4003) {
        return;
      }

      // Exponential backoff reconnect
      if (reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(
          RECONNECT_BASE_MS * Math.pow(2, reconnectAttemptRef.current),
          RECONNECT_MAX_MS,
        );
        reconnectAttemptRef.current += 1;
        clearReconnectTimer();
        reconnectTimerRef.current = setTimeout(connect, delay);
      }
    };

    wsRef.current = ws;
  }, [sessionId, handleServerMessage, clearHeartbeat, clearReconnectTimer]);

  // ── Lifecycle ──────────────────────────────────────────────────────────

  React.useEffect(() => {
    connect();

    return () => {
      clearHeartbeat();
      clearReconnectTimer();
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect, clearHeartbeat, clearReconnectTimer]);

  // ── Actions ────────────────────────────────────────────────────────────

  const sendPrompt = React.useCallback(
    (content: string, models: readonly string[]) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      // Track this prompt round
      promptIdRef.current = `prompt-${Date.now()}`;

      // Add user message locally
      const userMsg: ModelStreamMessage = {
        id: nextMessageId(),
        model: "user",
        role: "user",
        content,
        done: true,
        promptTokens: 0,
        completionTokens: 0,
        costUsd: 0,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);

      // Send to server
      wsRef.current.send(
        JSON.stringify({
          type: "prompt",
          content,
          models: [...models],
        }),
      );
    },
    [nextMessageId],
  );

  const addModel = React.useCallback((modelId: string, role = "general") => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(
      JSON.stringify({ type: "add_model", model_id: modelId, role }),
    );
  }, []);

  const removeModel = React.useCallback((modelId: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(
      JSON.stringify({ type: "remove_model", model_id: modelId }),
    );
  }, []);

  const clearMessages = React.useCallback(() => {
    setMessages([]);
    setTotalCost(0);
    accumulatorsRef.current.clear();
  }, []);

  return {
    status,
    connected: status === "connected",
    messages,
    streamingModels,
    sessionData,
    sendPrompt,
    addModel,
    removeModel,
    clearMessages,
    totalCost,
  };
}
