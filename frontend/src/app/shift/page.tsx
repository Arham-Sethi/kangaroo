"use client";

import * as React from "react";
import {
  ArrowRightLeft,
  Copy,
  Upload,
  Sparkles,
  CheckCircle2,
  Loader2,
  FileText,
  X,
  AlertCircle,
} from "lucide-react";

import { AppShell } from "@/components/layout/app-shell";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useCreateShift } from "@/hooks/use-api-query";
import { apiUpload } from "@/lib/api-client";

const MODELS = [
  { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "OpenAI" },
  { id: "claude-sonnet-4", name: "Claude Sonnet 4", provider: "Anthropic" },
  { id: "claude-haiku-4", name: "Claude Haiku 4", provider: "Anthropic" },
  { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", provider: "Google" },
  { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "Google" },
  { id: "llama-3.3-70b", name: "Llama 3.3 70B", provider: "Meta" },
  { id: "mistral-large", name: "Mistral Large", provider: "Mistral" },
  { id: "deepseek-v3", name: "DeepSeek V3", provider: "DeepSeek" },
  { id: "command-r-plus", name: "Command R+", provider: "Cohere" },
] as const;

interface ImportResult {
  readonly source_type: string;
  readonly detected_format: string;
  readonly original_size_bytes: number;
  readonly entity_count: number;
  readonly message_count: number;
}

function ModelSelector({
  label,
  selected,
  onSelect,
  excludeId,
}: {
  readonly label: string;
  readonly selected: string;
  readonly onSelect: (id: string) => void;
  readonly excludeId?: string;
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{label}</label>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        {MODELS.filter((m) => m.id !== excludeId).map((model) => (
          <button
            key={model.id}
            onClick={() => onSelect(model.id)}
            className={`rounded-lg border p-3 text-left transition-all ${
              selected === model.id
                ? "border-orange-500 bg-orange-50 ring-2 ring-orange-200 dark:bg-orange-950 dark:ring-orange-800"
                : "border-zinc-200 hover:border-zinc-300 dark:border-zinc-700 dark:hover:border-zinc-600"
            }`}
          >
            <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">{model.name}</p>
            <p className="text-xs text-zinc-500">{model.provider}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

function ShiftContent() {
  const [sourceModel, setSourceModel] = React.useState("gpt-4o");
  const [targetModel, setTargetModel] = React.useState("claude-sonnet-4");
  const [context, setContext] = React.useState("");
  const [title, setTitle] = React.useState("");
  const [uploadedFile, setUploadedFile] = React.useState<File | null>(null);
  const [uploading, setUploading] = React.useState(false);
  const [uploadResult, setUploadResult] = React.useState<ImportResult | null>(null);
  const [uploadError, setUploadError] = React.useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const createShift = useCreateShift();

  const handleShift = () => {
    if (!context.trim()) return;
    createShift.mutate({
      source_model: sourceModel,
      target_model: targetModel,
      context: context.trim(),
      title: title.trim() || undefined,
    });
  };

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      setContext(text);
    } catch {
      // Clipboard access denied
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = [".json", ".txt", ".md"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!validTypes.includes(ext)) {
      setUploadError("Unsupported file type. Use .json, .txt, or .md files.");
      return;
    }

    // Validate file size (5MB max)
    if (file.size > 5 * 1024 * 1024) {
      setUploadError("File too large. Maximum size is 5MB.");
      return;
    }

    setUploadedFile(file);
    setUploadError(null);
    setUploading(true);

    try {
      const result = await apiUpload<ImportResult>("/import/file", file);
      setUploadResult(result);

      // Also read the file content for the context textarea
      const text = await file.text();
      setContext(text);
      if (!title.trim()) {
        setTitle(file.name.replace(/\.[^.]+$/, ""));
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed.");
      setUploadedFile(null);
    } finally {
      setUploading(false);
    }
  };

  const clearUpload = () => {
    setUploadedFile(null);
    setUploadResult(null);
    setUploadError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">Context Shift</h1>
        <p className="mt-1 text-zinc-500">Transfer your conversation context seamlessly between AI models.</p>
      </div>

      {createShift.isSuccess && (
        <Card className="border-emerald-200 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950">
          <CardContent className="flex items-center gap-3 p-5">
            <CheckCircle2 className="h-6 w-6 text-emerald-600" />
            <div>
              <p className="font-semibold text-emerald-800 dark:text-emerald-200">Shift completed!</p>
              <p className="text-sm text-emerald-600 dark:text-emerald-400">
                Your context has been transferred to {MODELS.find((m) => m.id === targetModel)?.name ?? targetModel}.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {createShift.isError && (
        <Card className="border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950">
          <CardContent className="flex items-center gap-3 p-5">
            <AlertCircle className="h-6 w-6 text-red-600" />
            <div>
              <p className="font-semibold text-red-800 dark:text-red-200">Shift failed</p>
              <p className="text-sm text-red-600 dark:text-red-400">
                {createShift.error instanceof Error ? createShift.error.message : "An error occurred."}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ArrowRightLeft className="h-5 w-5 text-orange-500" />
            Select Models
          </CardTitle>
          <CardDescription>Choose the source and target AI models for your context transfer.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <ModelSelector label="Source Model (where context is from)" selected={sourceModel} onSelect={setSourceModel} excludeId={targetModel} />
          <div className="flex items-center gap-4">
            <div className="flex-1 border-t border-zinc-200 dark:border-zinc-700" />
            <ArrowRightLeft className="h-5 w-5 text-zinc-400" />
            <div className="flex-1 border-t border-zinc-200 dark:border-zinc-700" />
          </div>
          <ModelSelector label="Target Model (where context is going)" selected={targetModel} onSelect={setTargetModel} excludeId={sourceModel} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-orange-500" />
            Context
          </CardTitle>
          <CardDescription>Paste or type the conversation context you want to transfer.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Title (optional)</label>
            <Input placeholder="e.g., React architecture discussion" value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Context Content</label>
              <div className="flex gap-2">
                <Button variant="ghost" size="sm" onClick={handlePaste} className="text-xs">
                  <Copy className="mr-1 h-3 w-3" /> Paste
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-xs"
                  disabled={uploading}
                  onClick={() => fileInputRef.current?.click()}
                >
                  {uploading ? (
                    <><Loader2 className="mr-1 h-3 w-3 animate-spin" /> Uploading...</>
                  ) : (
                    <><Upload className="mr-1 h-3 w-3" /> Upload</>
                  )}
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json,.txt,.md"
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </div>
            </div>

            {/* Upload result badge */}
            {uploadedFile && uploadResult && (
              <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 dark:border-emerald-800 dark:bg-emerald-950">
                <FileText className="h-4 w-4 text-emerald-600 shrink-0" />
                <span className="text-xs text-emerald-800 dark:text-emerald-200 flex-1 truncate">
                  {uploadedFile.name} &middot; {uploadResult.detected_format} &middot;{" "}
                  {uploadResult.message_count} messages &middot; {uploadResult.entity_count} entities
                </span>
                <button onClick={clearUpload} className="text-emerald-600 hover:text-emerald-800">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            {/* Upload error */}
            {uploadError && (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 dark:border-red-800 dark:bg-red-950">
                <AlertCircle className="h-4 w-4 text-red-600 shrink-0" />
                <span className="text-xs text-red-800 dark:text-red-200 flex-1">{uploadError}</span>
                <button onClick={() => setUploadError(null)} className="text-red-600 hover:text-red-800">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            <textarea
              className="min-h-[200px] w-full rounded-lg border border-zinc-300 bg-white px-4 py-3 text-sm placeholder:text-zinc-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100 scrollbar-thin"
              placeholder="Paste your conversation here, or upload a file..."
              value={context}
              onChange={(e) => setContext(e.target.value)}
            />
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span>{context.length.toLocaleString()} characters</span>
              <span>~{Math.ceil(context.length / 4).toLocaleString()} tokens</span>
            </div>
          </div>

          <Button className="w-full" size="lg" onClick={handleShift} disabled={!context.trim() || createShift.isPending}>
            {createShift.isPending ? (
              <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Shifting Context...</>
            ) : (
              <><ArrowRightLeft className="mr-2 h-4 w-4" /> Shift Context</>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

export default function ShiftPage() {
  return (
    <AppShell>
      <ShiftContent />
    </AppShell>
  );
}
