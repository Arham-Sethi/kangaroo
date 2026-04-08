/**
 * React Query hooks for API data fetching.
 *
 * Each hook wraps a specific API endpoint with proper typing,
 * caching, and error handling via @tanstack/react-query.
 */

"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "@/lib/api-client";
import type {
  AllTiersResponse,
  UsageResponse,
  LimitsResponse,
  SessionSummary,
  SessionDetail,
  SearchResult,
  DigestResponse,
  GapResponse,
  WorkflowDefinition,
  WorkflowRunResult,
  DailyUsageEntry,
  ModelDistributionEntry,
  AnalyticsSummary,
  ApiKeyInfo,
  ApiKeyCreated,
  TeamInfo,
  TeamMemberInfo,
} from "@/types/api";

// ── Query Keys ──────────────────────────────────────────────────────────────

export const queryKeys = {
  tiers: ["tiers"] as const,
  usage: ["usage"] as const,
  limits: ["limits"] as const,
  sessions: ["sessions"] as const,
  session: (id: string) => ["session", id] as const,
  search: (q: string) => ["search", q] as const,
  digest: ["digest"] as const,
  gaps: ["gaps"] as const,
  workflows: ["workflows"] as const,
  workflow: (id: string) => ["workflow", id] as const,
  workflowRuns: (id: string) => ["workflow-runs", id] as const,
  dailyUsage: (days: number) => ["daily-usage", days] as const,
  modelDistribution: ["model-distribution"] as const,
  analyticsSummary: ["analytics-summary"] as const,
  apiKeys: ["api-keys"] as const,
  teams: ["teams"] as const,
  teamMembers: (id: string) => ["team-members", id] as const,
} as const;

// ── Billing Hooks ───────────────────────────────────────────────────────────

export function useTiers() {
  return useQuery({
    queryKey: queryKeys.tiers,
    queryFn: () => api.get<AllTiersResponse>("/billing/tiers", { public: true }),
    staleTime: 1000 * 60 * 60, // 1 hour — tiers rarely change
  });
}

export function useUsage() {
  return useQuery({
    queryKey: queryKeys.usage,
    queryFn: () => api.get<UsageResponse>("/billing/usage"),
    staleTime: 1000 * 30, // 30 seconds
    refetchInterval: 1000 * 60, // Refresh every minute
  });
}

export function useLimits() {
  return useQuery({
    queryKey: queryKeys.limits,
    queryFn: () => api.get<LimitsResponse>("/billing/limits"),
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

// ── Session Hooks ───────────────────────────────────────────────────────────

export function useSessions() {
  return useQuery({
    queryKey: queryKeys.sessions,
    queryFn: () => api.get<readonly SessionSummary[]>("/sessions"),
    staleTime: 1000 * 15,
  });
}

export function useSession(id: string) {
  return useQuery({
    queryKey: queryKeys.session(id),
    queryFn: () => api.get<SessionDetail>(`/sessions/${id}`),
    enabled: !!id,
  });
}

// ── Brain Hooks ─────────────────────────────────────────────────────────────

export function useSearch(query: string) {
  return useQuery({
    queryKey: queryKeys.search(query),
    queryFn: () =>
      api.get<readonly SearchResult[]>(`/search?q=${encodeURIComponent(query)}`),
    enabled: query.length >= 2,
    staleTime: 1000 * 60,
  });
}

export function useDigest() {
  return useQuery({
    queryKey: queryKeys.digest,
    queryFn: () => api.post<DigestResponse>("/brain/digest"),
    staleTime: 1000 * 60 * 30, // 30 min — digest is expensive
  });
}

export function useGaps() {
  return useQuery({
    queryKey: queryKeys.gaps,
    queryFn: () => api.post<GapResponse>("/brain/gaps"),
    staleTime: 1000 * 60 * 30,
  });
}

// ── Mutation Hooks ──────────────────────────────────────────────────────────

export function useCreateShift() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: {
      source_model: string;
      target_model: string;
      context: string;
      title?: string;
    }) => api.post<SessionSummary>("/sessions", payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.sessions });
      queryClient.invalidateQueries({ queryKey: queryKeys.usage });
    },
  });
}

// ── Workflow Hooks ──────────────────────────────────────────────────────

export function useWorkflows() {
  return useQuery({
    queryKey: queryKeys.workflows,
    queryFn: () => api.get<readonly WorkflowDefinition[]>("/workflows"),
    staleTime: 1000 * 30,
  });
}

export function useWorkflow(id: string) {
  return useQuery({
    queryKey: queryKeys.workflow(id),
    queryFn: () => api.get<WorkflowDefinition>(`/workflows/${id}`),
    enabled: !!id,
  });
}

export function useWorkflowRuns(id: string) {
  return useQuery({
    queryKey: queryKeys.workflowRuns(id),
    queryFn: () => api.get<readonly WorkflowRunResult[]>(`/workflows/${id}/runs`),
    enabled: !!id,
    staleTime: 1000 * 15,
  });
}

export function useCreateWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: {
      name: string;
      workflow_type: "chain" | "consensus";
      description?: string;
      steps?: readonly { model_id: string; instruction: string; system_context?: string }[];
      models?: readonly string[];
      system_context?: string;
    }) => api.post<WorkflowDefinition>("/workflows", payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.workflows });
    },
  });
}

export function useRunWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      workflowId,
      prompt,
    }: {
      workflowId: string;
      prompt: string;
    }) => api.post<WorkflowRunResult>(`/workflows/${workflowId}/run`, { prompt }),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.workflowRuns(variables.workflowId),
      });
      queryClient.invalidateQueries({ queryKey: queryKeys.usage });
    },
  });
}

export function useDeleteWorkflow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (workflowId: string) =>
      api.delete<void>(`/workflows/${workflowId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.workflows });
    },
  });
}

// ── Analytics Hooks ────────────────────────────────────────────────────────

export function useDailyUsage(days = 30) {
  return useQuery({
    queryKey: queryKeys.dailyUsage(days),
    queryFn: () =>
      api.get<readonly DailyUsageEntry[]>(`/analytics/daily-usage?days=${days}`),
    staleTime: 1000 * 60 * 5,
  });
}

export function useModelDistribution() {
  return useQuery({
    queryKey: queryKeys.modelDistribution,
    queryFn: () =>
      api.get<readonly ModelDistributionEntry[]>("/analytics/model-distribution"),
    staleTime: 1000 * 60 * 5,
  });
}

export function useAnalyticsSummary() {
  return useQuery({
    queryKey: queryKeys.analyticsSummary,
    queryFn: () => api.get<AnalyticsSummary>("/analytics/summary"),
    staleTime: 1000 * 60 * 2,
  });
}

// ── API Key Hooks ──────────────────────────────────────────────────────────

export function useApiKeys() {
  return useQuery({
    queryKey: queryKeys.apiKeys,
    queryFn: async () => {
      const res = await api.get<{ keys: readonly ApiKeyInfo[]; total: number }>("/api-keys");
      return res.keys;
    },
    staleTime: 1000 * 30,
  });
}

export function useCreateApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: { name: string; scopes?: readonly string[] }) =>
      api.post<ApiKeyCreated>("/api-keys", payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.apiKeys });
    },
  });
}

export function useDeleteApiKey() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (keyId: string) => api.delete<void>(`/api-keys/${keyId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.apiKeys });
    },
  });
}

// ── Billing Mutation Hooks ─────────────────────────────────────────────────

export function useCreateCheckout() {
  return useMutation({
    mutationFn: (payload: {
      tier: string;
      billing_cycle: "monthly" | "annual";
      success_url?: string;
      cancel_url?: string;
    }) => api.post<{ checkout_url: string; session_id: string }>("/billing/checkout", payload),
  });
}

export function useCreatePortalSession() {
  return useMutation({
    mutationFn: () =>
      api.post<{ portal_url: string }>("/billing/portal"),
  });
}

// ── Profile Hooks ──────────────────────────────────────────────────────────

export function useUpdateProfile() {
  return useMutation({
    mutationFn: (payload: { display_name?: string; settings?: Record<string, unknown> }) =>
      api.patch<unknown>("/auth/profile", payload),
  });
}

// ── Team Hooks ─────────────────────────────────────────────────────────────

export function useTeams() {
  return useQuery({
    queryKey: queryKeys.teams,
    queryFn: () => api.get<readonly TeamInfo[]>("/teams"),
    staleTime: 1000 * 30,
  });
}

export function useTeamMembers(teamId: string) {
  return useQuery({
    queryKey: queryKeys.teamMembers(teamId),
    queryFn: () => api.get<readonly TeamMemberInfo[]>(`/teams/${teamId}/members`),
    enabled: !!teamId,
    staleTime: 1000 * 15,
  });
}

export function useCreateTeam() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (payload: { name: string; description?: string }) =>
      api.post<TeamInfo>("/teams", payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.teams });
    },
  });
}

export function useInviteTeamMember() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      teamId,
      email,
      role,
    }: {
      teamId: string;
      email: string;
      role?: string;
    }) => api.post<TeamMemberInfo>(`/teams/${teamId}/invite`, { email, role }),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.teamMembers(variables.teamId),
      });
    },
  });
}
