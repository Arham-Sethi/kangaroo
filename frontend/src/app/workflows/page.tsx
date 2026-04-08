"use client";

import * as React from "react";
import {
  Zap,
  ArrowRight,
  Plus,
  GitBranch,
  Play,
  Loader2,
  Clock,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  Trash2,
  DollarSign,
} from "lucide-react";

import { AppShell } from "@/components/layout/app-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useAuthStore } from "@/stores/auth-store";
import {
  useWorkflows,
  useCreateWorkflow,
  useRunWorkflow,
  useDeleteWorkflow,
} from "@/hooks/use-api-query";
import type { WorkflowDefinition, WorkflowRunResult } from "@/types/api";

// ── Available Models ───────────────────────────────────────────────────────

const MODEL_OPTIONS = [
  { id: "gpt-4o", name: "GPT-4o" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini" },
  { id: "claude-sonnet-4", name: "Claude Sonnet 4" },
  { id: "claude-haiku-4", name: "Claude Haiku 4" },
  { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro" },
  { id: "gemini-2.0-flash", name: "Gemini Flash" },
] as const;

// ── Create Workflow Dialog ─────────────────────────────────────────────────

interface CreateWorkflowFormProps {
  readonly templateType: "chain" | "consensus";
  readonly onClose: () => void;
}

function CreateWorkflowForm({ templateType, onClose }: CreateWorkflowFormProps) {
  const createWorkflow = useCreateWorkflow();

  const [name, setName] = React.useState(
    templateType === "chain" ? "My Chain Pipeline" : "My Consensus Query",
  );
  const [description, setDescription] = React.useState("");
  const [systemContext, setSystemContext] = React.useState("");

  // Chain-specific state
  const [steps, setSteps] = React.useState([
    { model_id: "gpt-4o", instruction: "Write the initial draft" },
    { model_id: "claude-sonnet-4", instruction: "Review and improve" },
  ]);

  // Consensus-specific state
  const [models, setModels] = React.useState(["gpt-4o", "claude-sonnet-4", "gemini-2.5-pro"]);

  const handleAddStep = () => {
    setSteps([...steps, { model_id: "gpt-4o", instruction: "" }]);
  };

  const handleRemoveStep = (idx: number) => {
    if (steps.length <= 1) return;
    setSteps(steps.filter((_, i) => i !== idx));
  };

  const handleStepChange = (idx: number, field: string, value: string) => {
    setSteps(steps.map((s, i) => (i === idx ? { ...s, [field]: value } : s)));
  };

  const handleToggleModel = (modelId: string) => {
    if (models.includes(modelId)) {
      if (models.length <= 2) return; // Need at least 2
      setModels(models.filter((m) => m !== modelId));
    } else {
      setModels([...models, modelId]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await createWorkflow.mutateAsync({
      name,
      workflow_type: templateType,
      description,
      system_context: systemContext,
      steps: templateType === "chain" ? steps : undefined,
      models: templateType === "consensus" ? models : undefined,
    });
    onClose();
  };

  return (
    <Card className="border-orange-200 dark:border-orange-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">
          {templateType === "chain" ? "New Chain Pipeline" : "New Consensus Query"}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
              required
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Description (optional)
            </label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
            />
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
              System Context (optional, shared across all steps/models)
            </label>
            <textarea
              value={systemContext}
              onChange={(e) => setSystemContext(e.target.value)}
              rows={2}
              className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
            />
          </div>

          {templateType === "chain" && (
            <div className="space-y-3">
              <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Steps (output of each step feeds the next)
              </label>
              {steps.map((step, idx) => (
                <div key={idx} className="flex items-start gap-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-700">
                  <span className="mt-2 text-xs font-bold text-zinc-400">{idx + 1}.</span>
                  <div className="flex-1 space-y-2">
                    <select
                      value={step.model_id}
                      onChange={(e) => handleStepChange(idx, "model_id", e.target.value)}
                      className="w-full rounded border border-zinc-300 bg-white px-2 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
                    >
                      {MODEL_OPTIONS.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </select>
                    <input
                      type="text"
                      value={step.instruction}
                      onChange={(e) => handleStepChange(idx, "instruction", e.target.value)}
                      placeholder="Instruction for this step..."
                      className="w-full rounded border border-zinc-300 bg-white px-2 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => handleRemoveStep(idx)}
                    className="mt-2 text-zinc-400 hover:text-red-500"
                    disabled={steps.length <= 1}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))}
              <Button type="button" variant="outline" size="sm" onClick={handleAddStep}>
                <Plus className="mr-1 h-3 w-3" /> Add Step
              </Button>
            </div>
          )}

          {templateType === "consensus" && (
            <div>
              <label className="mb-2 block text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Models (select at least 2 to compare)
              </label>
              <div className="flex flex-wrap gap-2">
                {MODEL_OPTIONS.map((m) => {
                  const selected = models.includes(m.id);
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => handleToggleModel(m.id)}
                      className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                        selected
                          ? "border-orange-300 bg-orange-100 text-orange-700 dark:border-orange-600 dark:bg-orange-900 dark:text-orange-200"
                          : "border-zinc-200 text-zinc-500 hover:border-zinc-300 dark:border-zinc-700 dark:text-zinc-400"
                      }`}
                    >
                      {m.name}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={createWorkflow.isPending}>
              {createWorkflow.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : null}
              Create Workflow
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

// ── Workflow Card (with Run + Results) ──────────────────────────────────────

interface WorkflowCardProps {
  readonly workflow: WorkflowDefinition;
}

function WorkflowCard({ workflow }: WorkflowCardProps) {
  const runWorkflow = useRunWorkflow();
  const deleteWorkflow = useDeleteWorkflow();

  const [prompt, setPrompt] = React.useState("");
  const [expanded, setExpanded] = React.useState(false);
  const [latestRun, setLatestRun] = React.useState<WorkflowRunResult | null>(null);

  const isChain = workflow.workflow_type === "chain";

  const handleRun = async () => {
    if (!prompt.trim()) return;
    const result = await runWorkflow.mutateAsync({
      workflowId: workflow.id,
      prompt: prompt.trim(),
    });
    setLatestRun(result);
    setExpanded(true);
  };

  const handleDelete = () => {
    deleteWorkflow.mutate(workflow.id);
  };

  return (
    <Card>
      <CardContent className="p-5">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-300">
              {isChain ? (
                <ArrowRight className="h-5 w-5" />
              ) : (
                <GitBranch className="h-5 w-5" />
              )}
            </div>
            <div>
              <p className="font-semibold text-zinc-900 dark:text-zinc-100">
                {workflow.name}
              </p>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-[10px]">
                  {isChain ? `${workflow.steps.length} steps` : `${workflow.models.length} models`}
                </Badge>
                {workflow.description && (
                  <span className="text-xs text-zinc-400">{workflow.description}</span>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={handleDelete}
            className="text-zinc-400 hover:text-red-500"
            title="Delete workflow"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>

        {/* Steps/Models Preview */}
        <div className="mt-3 flex flex-wrap gap-1">
          {isChain
            ? workflow.steps.map((step, idx) => (
                <React.Fragment key={idx}>
                  <Badge variant="secondary" className="text-[10px]">
                    {MODEL_OPTIONS.find((m) => m.id === step.model_id)?.name ?? step.model_id}
                  </Badge>
                  {idx < workflow.steps.length - 1 && (
                    <ArrowRight className="h-3 w-3 self-center text-zinc-400" />
                  )}
                </React.Fragment>
              ))
            : workflow.models.map((modelId) => (
                <Badge key={modelId} variant="secondary" className="text-[10px]">
                  {MODEL_OPTIONS.find((m) => m.id === modelId)?.name ?? modelId}
                </Badge>
              ))}
        </div>

        {/* Run Input */}
        <div className="mt-4 flex gap-2">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter prompt to run..."
            className="flex-1 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
            onKeyDown={(e) => e.key === "Enter" && handleRun()}
          />
          <Button onClick={handleRun} disabled={!prompt.trim() || runWorkflow.isPending} size="sm">
            {runWorkflow.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Latest Run Result */}
        {latestRun && (
          <div className="mt-4 rounded-lg border border-zinc-200 dark:border-zinc-700">
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex w-full items-center justify-between p-3 text-sm"
            >
              <div className="flex items-center gap-2">
                {latestRun.status === "completed" ? (
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                ) : (
                  <XCircle className="h-4 w-4 text-red-500" />
                )}
                <span className="font-medium text-zinc-700 dark:text-zinc-300">
                  {latestRun.status === "completed" ? "Completed" : "Failed"}
                </span>
                <span className="flex items-center gap-1 text-xs text-zinc-400">
                  <Clock className="h-3 w-3" />
                  {latestRun.total_latency_ms.toFixed(0)}ms
                </span>
                <span className="text-xs text-zinc-400">
                  {latestRun.total_prompt_tokens + latestRun.total_completion_tokens} tokens
                </span>
              </div>
              {expanded ? (
                <ChevronUp className="h-4 w-4 text-zinc-400" />
              ) : (
                <ChevronDown className="h-4 w-4 text-zinc-400" />
              )}
            </button>

            {expanded && (
              <div className="border-t border-zinc-200 p-4 dark:border-zinc-700">
                <RunResultDisplay run={latestRun} isChain={isChain} />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Run Result Display ─────────────────────────────────────────────────────

function RunResultDisplay({
  run,
  isChain,
}: {
  readonly run: WorkflowRunResult;
  readonly isChain: boolean;
}) {
  const result = run.result;

  if (isChain) {
    const steps = (result.steps as readonly Record<string, unknown>[]) ?? [];
    const finalOutput = result.final_output as string ?? "";

    return (
      <div className="space-y-4">
        {steps.map((step, idx) => {
          const modelName =
            MODEL_OPTIONS.find((m) => m.id === step.model_id)?.name ??
            (step.model_id as string);
          return (
            <div key={idx} className="space-y-1">
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-[10px]">
                  Step {(step.step_index as number) + 1}
                </Badge>
                <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                  {modelName}
                </span>
                {step.instruction ? (
                  <span className="text-xs italic text-zinc-400">
                    {String(step.instruction)}
                  </span>
                ) : null}
              </div>
              <div className="rounded-md bg-zinc-50 p-3 text-sm whitespace-pre-wrap dark:bg-zinc-900">
                {step.error
                  ? (
                    <span className="text-red-500">
                      Error: {step.error as string}
                    </span>
                  )
                  : (step.content as string)}
              </div>
            </div>
          );
        })}
        {finalOutput && (
          <div className="border-t border-zinc-200 pt-3 dark:border-zinc-700">
            <p className="mb-1 text-xs font-semibold text-zinc-500">Final Output</p>
            <div className="rounded-md bg-orange-50 p-3 text-sm whitespace-pre-wrap dark:bg-orange-950">
              {finalOutput}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Consensus result
  const agreementLevel = result.agreement_level as string ?? "unknown";
  const agreementScore = result.agreement_score as number ?? 0;
  const summary = result.summary as string ?? "";
  const commonThemes = (result.common_themes as readonly string[]) ?? [];
  const differences = (result.differences as readonly string[]) ?? [];
  const responses = (result.responses as Record<string, Record<string, unknown>>) ?? {};

  return (
    <div className="space-y-4">
      {/* Agreement Summary */}
      <div className="flex items-center gap-3">
        <Badge
          variant={
            agreementLevel === "strong"
              ? "default"
              : agreementLevel === "partial"
                ? "secondary"
                : "outline"
          }
        >
          {agreementLevel} agreement
        </Badge>
        <span className="text-xs text-zinc-500">
          {(agreementScore * 100).toFixed(0)}% overlap
        </span>
      </div>
      {summary && (
        <p className="text-sm text-zinc-600 dark:text-zinc-400">{summary}</p>
      )}

      {/* Common Themes */}
      {commonThemes.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-semibold text-zinc-500">Common Themes</p>
          <div className="flex flex-wrap gap-1">
            {commonThemes.map((theme) => (
              <Badge key={theme} variant="secondary" className="text-[10px]">
                {theme}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Differences */}
      {differences.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-semibold text-zinc-500">Unique Points</p>
          <div className="flex flex-wrap gap-1">
            {differences.map((diff) => (
              <Badge key={diff} variant="outline" className="text-[10px]">
                {diff}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Per-Model Responses */}
      <div className="space-y-3 border-t border-zinc-200 pt-3 dark:border-zinc-700">
        <p className="text-xs font-semibold text-zinc-500">Model Responses</p>
        {Object.entries(responses).map(([modelId, resp]) => {
          const modelName =
            MODEL_OPTIONS.find((m) => m.id === modelId)?.name ?? modelId;
          return (
            <div key={modelId} className="space-y-1">
              <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                {modelName}
              </span>
              <div className="rounded-md bg-zinc-50 p-3 text-sm whitespace-pre-wrap dark:bg-zinc-900">
                {resp.error
                  ? (
                    <span className="text-red-500">
                      Error: {resp.error as string}
                    </span>
                  )
                  : (resp.content as string)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main Workflow Content ──────────────────────────────────────────────────

function WorkflowContent() {
  const { subscription } = useAuthStore();
  const effectiveTier = subscription?.effective_tier ?? "free";
  const isPro = effectiveTier !== "free";

  const { data: workflows, isLoading } = useWorkflows();
  const [creatingType, setCreatingType] = React.useState<"chain" | "consensus" | null>(null);

  if (!isPro) {
    return (
      <div className="mx-auto max-w-2xl animate-fade-in">
        <Card>
          <CardContent className="flex flex-col items-center gap-4 py-16 text-center">
            <Zap className="h-16 w-16 text-zinc-300" />
            <div>
              <h2 className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
                Workflow Builder
              </h2>
              <p className="mt-2 text-zinc-500">
                Create automated multi-step pipelines: chain models, run consensus
                queries, and orchestrate complex AI workflows.
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

  return (
    <div className="mx-auto max-w-4xl space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
            Workflows
          </h1>
          <p className="mt-1 text-zinc-500">
            Build and run automated multi-model pipelines.
          </p>
        </div>
      </div>

      {/* Template Cards */}
      {!creatingType && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card
            className="group cursor-pointer transition-shadow hover:shadow-md"
            onClick={() => setCreatingType("chain")}
          >
            <CardContent className="flex items-center gap-4 p-5">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-orange-100 text-orange-600 group-hover:bg-orange-500 group-hover:text-white dark:bg-orange-900 dark:text-orange-300">
                <ArrowRight className="h-6 w-6" />
              </div>
              <div>
                <p className="font-semibold text-zinc-900 dark:text-zinc-100">
                  Chain Pipeline
                </p>
                <p className="text-sm text-zinc-500">
                  Sequential: Model A output feeds Model B
                </p>
              </div>
            </CardContent>
          </Card>
          <Card
            className="group cursor-pointer transition-shadow hover:shadow-md"
            onClick={() => setCreatingType("consensus")}
          >
            <CardContent className="flex items-center gap-4 p-5">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-orange-100 text-orange-600 group-hover:bg-orange-500 group-hover:text-white dark:bg-orange-900 dark:text-orange-300">
                <GitBranch className="h-6 w-6" />
              </div>
              <div>
                <p className="font-semibold text-zinc-900 dark:text-zinc-100">
                  Consensus Query
                </p>
                <p className="text-sm text-zinc-500">
                  Ask multiple models, measure agreement
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Create Form */}
      {creatingType && (
        <CreateWorkflowForm
          templateType={creatingType}
          onClose={() => setCreatingType(null)}
        />
      )}

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
        </div>
      )}

      {/* Existing Workflows */}
      {workflows && workflows.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
            Your Workflows
          </h2>
          {workflows.map((wf) => (
            <WorkflowCard key={wf.id} workflow={wf} />
          ))}
        </div>
      )}

      {/* Empty State */}
      {workflows && workflows.length === 0 && !creatingType && (
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
            <Zap className="h-10 w-10 text-zinc-300" />
            <div>
              <p className="font-medium text-zinc-600 dark:text-zinc-400">
                No workflows yet
              </p>
              <p className="text-sm text-zinc-400">
                Create your first workflow from a template above.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function WorkflowsPage() {
  return (
    <AppShell>
      <WorkflowContent />
    </AppShell>
  );
}
