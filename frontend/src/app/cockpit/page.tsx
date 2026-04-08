"use client";

import * as React from "react";
import {
  Monitor,
  Send,
  Plus,
  X,
  DollarSign,
  Loader2,
  Bot,
  Wifi,
  WifiOff,
  RotateCcw,
  Trash2,
} from "lucide-react";

import { AppShell } from "@/components/layout/app-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useAuthStore } from "@/stores/auth-store";
import { useCockpitSession } from "@/hooks/use-cockpit-session";
import { useCockpitWS, type ModelStreamMessage } from "@/hooks/use-cockpit-ws";

// ── Available Models ───────────────────────────────────────────────────────

const AVAILABLE_MODELS = [
  { id: "gpt-4o", name: "GPT-4o", color: "bg-green-500" },
  { id: "claude-sonnet-4", name: "Claude Sonnet 4", color: "bg-orange-500" },
  { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", color: "bg-blue-500" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", color: "bg-emerald-500" },
  { id: "claude-haiku-4", name: "Claude Haiku 4", color: "bg-amber-500" },
  { id: "gemini-2.0-flash", name: "Gemini Flash", color: "bg-cyan-500" },
] as const;

function getModelInfo(modelId: string) {
  return AVAILABLE_MODELS.find((m) => m.id === modelId);
}

// ── Connection Status Indicator ────────────────────────────────────────────

function ConnectionIndicator({ connected, status }: { connected: boolean; status: string }) {
  if (connected) {
    return (
      <Badge variant="outline" className="flex items-center gap-1.5 text-green-600 border-green-200">
        <Wifi className="h-3 w-3" />
        Connected
      </Badge>
    );
  }

  if (status === "connecting") {
    return (
      <Badge variant="outline" className="flex items-center gap-1.5 text-amber-600 border-amber-200">
        <Loader2 className="h-3 w-3 animate-spin" />
        Connecting
      </Badge>
    );
  }

  return (
    <Badge variant="outline" className="flex items-center gap-1.5 text-red-600 border-red-200">
      <WifiOff className="h-3 w-3" />
      Disconnected
    </Badge>
  );
}

// ── Streaming Message Bubble ───────────────────────────────────────────────

function MessageBubble({ msg }: { msg: ModelStreamMessage }) {
  const isUser = msg.role === "user";
  const model = getModelInfo(msg.model);
  const isStreaming = !msg.done && msg.role === "assistant";

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : ""}`}>
      {!isUser && (
        <div
          className={`mt-1 h-6 w-6 flex-shrink-0 rounded-full ${model?.color ?? "bg-zinc-400"}`}
        />
      )}
      <div
        className={`max-w-[80%] rounded-lg px-4 py-3 ${
          isUser
            ? "bg-orange-500 text-white"
            : "border border-zinc-200 bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900"
        }`}
      >
        {!isUser && (
          <div className="mb-1 flex items-center gap-2">
            <p className="text-xs font-semibold text-zinc-500">
              {model?.name ?? msg.model}
            </p>
            {msg.done && msg.costUsd > 0 && (
              <span className="text-[10px] text-zinc-400">
                ${msg.costUsd.toFixed(4)} | {msg.promptTokens + msg.completionTokens} tok
              </span>
            )}
          </div>
        )}
        <p className="text-sm whitespace-pre-wrap">
          {msg.content}
          {isStreaming && (
            <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-orange-500" />
          )}
        </p>
        {msg.error && (
          <p className="mt-1 text-xs text-red-500">
            {msg.error}
          </p>
        )}
      </div>
    </div>
  );
}

// ── Main Cockpit Content ───────────────────────────────────────────────────

function CockpitContent() {
  const { subscription } = useAuthStore();
  const effectiveTier = subscription?.effective_tier ?? "free";
  const isPro = effectiveTier !== "free";

  const { sessionId, resetSession } = useCockpitSession();
  const {
    connected,
    status,
    messages,
    streamingModels,
    sendPrompt,
    addModel: wsAddModel,
    removeModel: wsRemoveModel,
    clearMessages,
    totalCost,
  } = useCockpitWS(isPro ? sessionId : null);

  const [selectedModels, setSelectedModels] = React.useState<readonly string[]>([
    "gpt-4o",
    "claude-sonnet-4",
  ]);
  const [input, setInput] = React.useState("");
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  const maxModels =
    effectiveTier === "pro"
      ? 2
      : effectiveTier === "pro_team"
        ? 4
        : effectiveTier === "enterprise"
          ? 6
          : 0;

  const isStreaming = streamingModels.size > 0;

  // Auto-scroll to bottom on new messages
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleAddModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) return;
    if (selectedModels.length >= maxModels) return;
    setSelectedModels([...selectedModels, modelId]);
    wsAddModel(modelId);
  };

  const handleRemoveModel = (modelId: string) => {
    if (selectedModels.length <= 1) return; // Keep at least 1
    setSelectedModels(selectedModels.filter((id) => id !== modelId));
    wsRemoveModel(modelId);
  };

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming || !connected) return;
    sendPrompt(trimmed, selectedModels);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Paywall ────────────────────────────────────────────────────────────

  if (!isPro) {
    return (
      <div className="mx-auto max-w-2xl animate-fade-in">
        <Card>
          <CardContent className="flex flex-col items-center gap-4 py-16 text-center">
            <Monitor className="h-16 w-16 text-zinc-300" />
            <div>
              <h2 className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
                Multi-Model Cockpit
              </h2>
              <p className="mt-2 text-zinc-500">
                Send one prompt to multiple AI models simultaneously and compare
                responses side-by-side.
              </p>
            </div>
            <Badge variant="pro">Pro Feature</Badge>
            <Button asChild>
              <a href="/pricing">Upgrade to Pro</a>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ── Main UI ────────────────────────────────────────────────────────────

  return (
    <div className="mx-auto max-w-6xl space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            Cockpit
          </h1>
          <p className="mt-1 text-zinc-500">
            Multi-model parallel chat — compare AI responses in real time.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <ConnectionIndicator connected={connected} status={status} />
          <Badge variant="secondary" className="flex items-center gap-1">
            <DollarSign className="h-3 w-3" />$
            {totalCost.toFixed(4)}
          </Badge>
          <Badge variant="outline">
            {selectedModels.length}/{maxModels} models
          </Badge>
        </div>
      </div>

      {/* Model Selector */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Active Models</CardTitle>
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={clearMessages}
                title="Clear chat"
              >
                <Trash2 className="mr-1 h-3 w-3" />
                Clear
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={resetSession}
                title="New session"
              >
                <RotateCcw className="mr-1 h-3 w-3" />
                New Session
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {selectedModels.map((modelId) => {
              const model = getModelInfo(modelId);
              const isModelStreaming = streamingModels.has(modelId);
              return (
                <Badge
                  key={modelId}
                  variant="secondary"
                  className="flex items-center gap-1.5 py-1.5"
                >
                  {isModelStreaming ? (
                    <Loader2 className="h-2 w-2 animate-spin text-orange-500" />
                  ) : (
                    <div
                      className={`h-2 w-2 rounded-full ${model?.color ?? "bg-zinc-400"}`}
                    />
                  )}
                  {model?.name ?? modelId}
                  <button
                    onClick={() => handleRemoveModel(modelId)}
                    className="ml-1 text-zinc-400 hover:text-zinc-600"
                    disabled={selectedModels.length <= 1}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              );
            })}
            {selectedModels.length < maxModels && (
              <div className="relative group">
                <Button variant="outline" size="sm" className="h-7 text-xs">
                  <Plus className="mr-1 h-3 w-3" /> Add Model
                </Button>
                <div className="absolute left-0 top-full z-10 mt-1 hidden w-48 rounded-lg border bg-white p-1 shadow-lg group-hover:block dark:border-zinc-700 dark:bg-zinc-900">
                  {AVAILABLE_MODELS.filter(
                    (m) => !selectedModels.includes(m.id),
                  ).map((model) => (
                    <button
                      key={model.id}
                      onClick={() => handleAddModel(model.id)}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-800"
                    >
                      <div
                        className={`h-2 w-2 rounded-full ${model.color}`}
                      />
                      {model.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Chat Area */}
      <Card className="min-h-[400px]">
        <CardContent className="flex flex-col p-0">
          {/* Messages */}
          <div
            className="flex-1 space-y-4 overflow-y-auto p-6 scrollbar-thin"
            style={{ maxHeight: "500px" }}
          >
            {messages.length === 0 && (
              <div className="flex flex-col items-center gap-3 py-16 text-center">
                <Bot className="h-12 w-12 text-zinc-300" />
                <p className="text-sm text-zinc-500">
                  {connected
                    ? `Send a message to query ${selectedModels.length} models simultaneously.`
                    : "Connecting to Kangaroo Shift..."}
                </p>
              </div>
            )}
            {messages.map((msg) => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}
            {isStreaming && (
              <div className="flex items-center gap-2 text-sm text-zinc-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                Streaming from {streamingModels.size} model
                {streamingModels.size > 1 ? "s" : ""}...
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-zinc-200 p-4 dark:border-zinc-700">
            <div className="flex gap-2">
              <textarea
                className="flex-1 resize-none rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm placeholder:text-zinc-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
                placeholder={
                  connected
                    ? "Ask all models at once... (Shift+Enter for new line)"
                    : "Waiting for connection..."
                }
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={2}
                disabled={!connected}
              />
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isStreaming || !connected}
                className="self-end"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function CockpitPage() {
  return (
    <AppShell>
      <React.Suspense
        fallback={
          <div className="flex items-center justify-center py-24">
            <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
          </div>
        }
      >
        <CockpitContent />
      </React.Suspense>
    </AppShell>
  );
}
