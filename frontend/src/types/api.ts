/**
 * Shared API response types mirroring the backend Pydantic schemas.
 * Every type is readonly — no mutation on the frontend.
 */

// ── Auth ────────────────────────────────────────────────────────────────────

export interface AuthTokens {
  readonly access_token: string;
  readonly token_type: string;
}

export interface UserProfile {
  readonly id: string;
  readonly email: string;
  readonly name: string;
  readonly subscription_tier: SubscriptionTier;
  readonly created_at: string;
}

// ── Billing ─────────────────────────────────────────────────────────────────

export type SubscriptionTier = "free" | "pro" | "pro_team" | "enterprise";

export interface TierPricing {
  readonly monthly_price_usd: number;
  readonly annual_price_usd: number;
  readonly annual_monthly_equivalent: number;
  readonly per_seat: boolean;
  readonly min_seats: number;
  readonly max_seats: number;
}

export interface TierInfo {
  readonly tier: SubscriptionTier;
  readonly display_name: string;
  readonly pricing: TierPricing;
  readonly shifts_per_month: number;
  readonly active_sessions: number;
  readonly brain_queries_per_month: number;
  readonly tokens_per_month: number;
  readonly retention_days: number;
  readonly cockpit_models_max: number;
  readonly chain_steps_max: number;
  readonly consensus_models_max: number;
  readonly team_seats_included: number;
  readonly api_keys_max: number;
  readonly webhooks_max: number;
  readonly file_upload_mb: number;
  readonly smart_dispatch: boolean;
  readonly digest_daily: boolean;
  readonly knowledge_gap_alerts: boolean;
  readonly sso_enabled: boolean;
  readonly support_level: string;
}

export interface AllTiersResponse {
  readonly tiers: readonly TierInfo[];
}

export interface UsageMetricInfo {
  readonly metric: string;
  readonly current: number;
  readonly limit: number;
  readonly remaining: number;
  readonly is_unlimited: boolean;
}

export interface UsageResponse {
  readonly tier: SubscriptionTier;
  readonly period: string;
  readonly metrics: readonly UsageMetricInfo[];
}

export interface TrialStatusResponse {
  readonly is_active: boolean;
  readonly started_at: number;
  readonly expires_at: number;
  readonly days_remaining: number;
  readonly trial_tier: string;
  readonly has_been_offered: boolean;
}

export interface SubscriptionResponse {
  readonly tier: SubscriptionTier;
  readonly effective_tier: SubscriptionTier;
  readonly display_name: string;
  readonly is_trial_active: boolean;
  readonly trial_days_remaining: number;
  readonly trial_expired: boolean;
}

export interface LimitsResponse {
  readonly tier: SubscriptionTier;
  readonly display_name: string;
  readonly limits: Record<string, string | number | boolean>;
}

// ── Sessions ────────────────────────────────────────────────────────────────

export interface SessionSummary {
  readonly id: string;
  readonly title: string;
  readonly source_model: string;
  readonly target_model: string;
  readonly status: string;
  readonly created_at: string;
  readonly updated_at: string;
  readonly context_size: number;
}

export interface SessionDetail extends SessionSummary {
  readonly messages: readonly Message[];
  readonly metadata: Record<string, unknown>;
}

export interface Message {
  readonly id: string;
  readonly role: "user" | "assistant" | "system";
  readonly content: string;
  readonly model: string;
  readonly timestamp: string;
  readonly token_count: number;
}

// ── Brain ───────────────────────────────────────────────────────────────────

export interface SearchResult {
  readonly id: string;
  readonly content: string;
  readonly score: number;
  readonly metadata: Record<string, unknown>;
}

export interface DigestResponse {
  readonly entities: readonly string[];
  readonly decisions: readonly string[];
  readonly tasks: readonly string[];
  readonly generated_at: string;
}

export interface GapResponse {
  readonly undecided_topics: readonly string[];
  readonly stalled_tasks: readonly string[];
  readonly unclear_entities: readonly string[];
}

// ── Cockpit ─────────────────────────────────────────────────────────────────

export interface CockpitMessage {
  readonly model: string;
  readonly role: string;
  readonly content: string;
  readonly cost_usd: number;
  readonly tokens_used: number;
  readonly latency_ms: number;
}

export interface CockpitSession {
  readonly id: string;
  readonly models: readonly string[];
  readonly messages: readonly CockpitMessage[];
  readonly total_cost_usd: number;
}

// ── Workflows ──────────────────────────────────────────────────────────────

export type WorkflowType = "chain" | "consensus";

export interface WorkflowStepConfig {
  readonly model_id: string;
  readonly instruction: string;
  readonly system_context: string;
}

export interface WorkflowDefinition {
  readonly id: string;
  readonly name: string;
  readonly workflow_type: WorkflowType;
  readonly description: string;
  readonly steps: readonly WorkflowStepConfig[];
  readonly models: readonly string[];
  readonly system_context: string;
  readonly user_id: string;
  readonly created_at: number;
  readonly updated_at: number;
}

export interface ChainStepResult {
  readonly step_index: number;
  readonly model_id: string;
  readonly instruction: string;
  readonly content: string;
  readonly input_text: string;
  readonly prompt_tokens: number;
  readonly completion_tokens: number;
  readonly latency_ms: number;
  readonly error: string;
}

export interface WorkflowRunResult {
  readonly id: string;
  readonly workflow_id: string;
  readonly prompt: string;
  readonly workflow_type: WorkflowType;
  readonly status: "completed" | "failed";
  readonly result: Record<string, unknown>;
  readonly total_latency_ms: number;
  readonly total_prompt_tokens: number;
  readonly total_completion_tokens: number;
  readonly created_at: number;
}

// ── Analytics ──────────────────────────────────────────────────────────────

export interface DailyUsageEntry {
  readonly date: string;
  readonly shifts: number;
  readonly sessions: number;
  readonly tokens: number;
}

export interface ModelDistributionEntry {
  readonly model: string;
  readonly count: number;
  readonly direction: "source" | "target" | "capture";
}

export interface AnalyticsSummary {
  readonly total_sessions: number;
  readonly active_sessions: number;
  readonly total_shifts: number;
  readonly total_tokens: number;
  readonly shifts_this_month: number;
  readonly subscription_tier: string;
}

// ── API Keys ──────────────────────────────────────────────────────────────

export interface ApiKeyInfo {
  readonly id: string;
  readonly name: string;
  readonly key_prefix: string;
  readonly scopes: readonly string[];
  readonly is_active: boolean;
  readonly last_used: string | null;
  readonly created_at: string;
}

export interface ApiKeyCreated extends ApiKeyInfo {
  readonly key: string;
}

// ── Teams ─────────────────────────────────────────────────────────────────

export interface TeamInfo {
  readonly id: string;
  readonly name: string;
  readonly slug: string;
  readonly description: string;
  readonly max_members: number;
  readonly member_count: number;
  readonly created_at: string;
}

export interface TeamMemberInfo {
  readonly user_id: string;
  readonly email: string;
  readonly display_name: string;
  readonly role: string;
  readonly invited_by: string | null;
  readonly joined_at: string | null;
}

// ── API Error ───────────────────────────────────────────────────────────────

export interface ApiError {
  readonly error: string;
  readonly message: string;
  readonly detail?: Record<string, unknown>;
}
