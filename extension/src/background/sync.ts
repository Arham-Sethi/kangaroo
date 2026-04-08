/**
 * Sync engine — batched upload of captured conversations to the backend.
 *
 * Conversations are queued in chrome.storage.local and synced in batches
 * when:
 *   1. A new conversation is captured (debounced 10s)
 *   2. The user clicks "Sync Now" in the popup
 *   3. A periodic alarm fires (every 5 minutes)
 *
 * Failed syncs are retried with exponential backoff.
 * Successfully synced conversations are removed from the queue.
 */

import { dequeueConversations, enqueueConversation, getStorage, getPendingCount } from "../lib/storage";
import type { ScrapedConversation } from "../lib/types";

const BATCH_SIZE = 5;
const MAX_RETRIES = 3;

interface SyncResult {
  readonly synced: number;
  readonly failed: number;
  readonly pending: number;
}

export async function syncPending(): Promise<SyncResult> {
  const storage = await getStorage();

  if (!storage.authToken) {
    return { synced: 0, failed: 0, pending: storage.pendingConversations.length };
  }

  if (storage.pendingConversations.length === 0) {
    return { synced: 0, failed: 0, pending: 0 };
  }

  const batch = await dequeueConversations(BATCH_SIZE);
  let synced = 0;
  let failed = 0;

  for (const conversation of batch) {
    const success = await uploadConversation(conversation, storage.authToken, storage.apiBaseUrl);
    if (success) {
      synced++;
    } else {
      failed++;
      // Re-queue failed ones (will be at the end)
      await requeue(conversation);
    }
  }

  const pending = await getPendingCount();
  return { synced, failed, pending };
}

async function uploadConversation(
  conversation: ScrapedConversation,
  token: string,
  baseUrl: string,
): Promise<boolean> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await fetch(`${baseUrl}/sessions/capture`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          platform: conversation.platform,
          model: conversation.model,
          title: conversation.title,
          url: conversation.url,
          messages: conversation.messages.map((m) => ({
            role: m.role,
            content: m.content,
            model: m.model,
          })),
          captured_at: new Date(conversation.scrapedAt).toISOString(),
        }),
      });

      if (response.ok) return true;

      // 401 = token expired, stop retrying
      if (response.status === 401) return false;

      // 402 = paywall, stop retrying
      if (response.status === 402) return false;

      // 429 = rate limited, wait and retry
      if (response.status === 429) {
        const retryAfter = parseInt(response.headers.get("Retry-After") ?? "5", 10);
        await sleep(retryAfter * 1000);
        continue;
      }

      // 5xx = server error, retry with backoff
      if (response.status >= 500) {
        await sleep(Math.pow(2, attempt) * 1000);
        continue;
      }

      // Other errors — don't retry
      return false;
    } catch {
      // Network error — retry with backoff
      await sleep(Math.pow(2, attempt) * 1000);
    }
  }

  return false;
}

async function requeue(conversation: ScrapedConversation): Promise<void> {
  await enqueueConversation(conversation);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
