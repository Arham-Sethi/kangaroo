/**
 * Shared types for the Kangaroo Shift browser extension.
 * All types are readonly — no mutation.
 */

// ── Detected Platform ───────────────────────────────────────────────────────

export type Platform = "chatgpt" | "claude" | "gemini";

export type MessageRole = "user" | "assistant" | "system";

// ── Scraped Message ─────────────────────────────────────────────────────────

export interface ScrapedMessage {
  readonly role: MessageRole;
  readonly content: string;
  readonly model: string;
  readonly timestamp: number;
  /** Unique hash to deduplicate (content + role + index). */
  readonly hash: string;
}

// ── Scraped Conversation ────────────────────────────────────────────────────

export interface ScrapedConversation {
  readonly id: string;
  readonly platform: Platform;
  readonly model: string;
  readonly title: string;
  readonly url: string;
  readonly messages: readonly ScrapedMessage[];
  readonly scrapedAt: number;
  readonly messageCount: number;
}

// ── Extension Messages (content ↔ background) ──────────────────────────────

export type ExtensionMessage =
  | { type: "CONVERSATION_UPDATE"; payload: ScrapedConversation }
  | { type: "SCRAPE_NOW"; payload: { platform: Platform } }
  | { type: "GET_STATUS" }
  | { type: "STATUS_RESPONSE"; payload: ExtensionStatus }
  | { type: "AUTH_TOKEN"; payload: { token: string } }
  | { type: "LOGOUT" }
  | { type: "SYNC_NOW" }
  | { type: "SYNC_COMPLETE"; payload: { count: number; errors: number } }
  | { type: "CAPTURE_TOGGLED"; payload: { enabled: boolean } };

// ── Extension Status ────────────────────────────────────────────────────────

export interface ExtensionStatus {
  readonly isAuthenticated: boolean;
  readonly captureEnabled: boolean;
  readonly pendingSync: number;
  readonly lastSyncAt: number | null;
  readonly totalCaptured: number;
  readonly activePlatform: Platform | null;
}

// ── Storage Schema ──────────────────────────────────────────────────────────

export interface StorageData {
  readonly authToken: string | null;
  readonly apiBaseUrl: string;
  readonly captureEnabled: boolean;
  readonly lastSyncAt: number | null;
  readonly totalCaptured: number;
  readonly pendingConversations: readonly ScrapedConversation[];
}

export const DEFAULT_STORAGE: StorageData = {
  authToken: null,
  apiBaseUrl: "http://localhost:8000/api/v1",
  captureEnabled: true,
  lastSyncAt: null,
  totalCaptured: 0,
  pendingConversations: [],
};
