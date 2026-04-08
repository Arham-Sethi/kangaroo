/**
 * Chrome storage wrapper — typed, async, immutable.
 *
 * Uses chrome.storage.local for persistence across sessions.
 * All writes create new objects (immutable pattern).
 */

import { DEFAULT_STORAGE, type StorageData, type ScrapedConversation } from "./types";

// ── Read ────────────────────────────────────────────────────────────────────

export async function getStorage(): Promise<StorageData> {
  const result = await chrome.storage.local.get(null);
  return {
    authToken: result.authToken ?? DEFAULT_STORAGE.authToken,
    apiBaseUrl: result.apiBaseUrl ?? DEFAULT_STORAGE.apiBaseUrl,
    captureEnabled: result.captureEnabled ?? DEFAULT_STORAGE.captureEnabled,
    lastSyncAt: result.lastSyncAt ?? DEFAULT_STORAGE.lastSyncAt,
    totalCaptured: result.totalCaptured ?? DEFAULT_STORAGE.totalCaptured,
    pendingConversations: result.pendingConversations ?? DEFAULT_STORAGE.pendingConversations,
  };
}

export async function getAuthToken(): Promise<string | null> {
  const data = await getStorage();
  return data.authToken;
}

export async function isAuthenticated(): Promise<boolean> {
  const token = await getAuthToken();
  return token !== null && token.length > 0;
}

// ── Write ───────────────────────────────────────────────────────────────────

export async function setAuthToken(token: string | null): Promise<void> {
  await chrome.storage.local.set({ authToken: token });
}

export async function setCaptureEnabled(enabled: boolean): Promise<void> {
  await chrome.storage.local.set({ captureEnabled: enabled });
}

export async function setApiBaseUrl(url: string): Promise<void> {
  await chrome.storage.local.set({ apiBaseUrl: url });
}

// ── Queue Management ────────────────────────────────────────────────────────

export async function enqueueConversation(conv: ScrapedConversation): Promise<number> {
  const data = await getStorage();

  // Deduplicate by conversation ID — replace if newer
  const existing = data.pendingConversations.filter((c) => c.id !== conv.id);
  const updated = [...existing, conv];

  await chrome.storage.local.set({
    pendingConversations: updated,
    totalCaptured: data.totalCaptured + 1,
  });

  return updated.length;
}

export async function dequeueConversations(count: number): Promise<readonly ScrapedConversation[]> {
  const data = await getStorage();
  const batch = data.pendingConversations.slice(0, count);
  const remaining = data.pendingConversations.slice(count);

  await chrome.storage.local.set({
    pendingConversations: remaining,
    lastSyncAt: Date.now(),
  });

  return batch;
}

export async function getPendingCount(): Promise<number> {
  const data = await getStorage();
  return data.pendingConversations.length;
}

// ── Clear ───────────────────────────────────────────────────────────────────

export async function clearAll(): Promise<void> {
  await chrome.storage.local.clear();
}

export async function clearPending(): Promise<void> {
  await chrome.storage.local.set({ pendingConversations: [] });
}
