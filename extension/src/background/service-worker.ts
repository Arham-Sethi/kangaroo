/**
 * Background service worker — message hub and sync coordinator.
 *
 * Responsibilities:
 *   1. Receive CONVERSATION_UPDATE from content scripts
 *   2. Queue conversations in chrome.storage.local
 *   3. Trigger periodic sync via chrome.alarms
 *   4. Handle popup messages (GET_STATUS, SYNC_NOW, AUTH_TOKEN, etc.)
 *   5. Update badge with pending count
 */

import {
  enqueueConversation,
  getStorage,
  setAuthToken,
  setCaptureEnabled,
  getPendingCount,
  clearAll,
} from "../lib/storage";
import { syncPending } from "./sync";
import type { ExtensionMessage, ExtensionStatus, Platform } from "../lib/types";

// ── Alarm Names ─────────────────────────────────────────────────────────────

const SYNC_ALARM = "kangaroo-sync";
const SYNC_INTERVAL_MINUTES = 5;

// ── Install / Startup ───────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  // Set up periodic sync alarm
  chrome.alarms.create(SYNC_ALARM, {
    periodInMinutes: SYNC_INTERVAL_MINUTES,
  });
  updateBadge();
});

chrome.runtime.onStartup.addListener(() => {
  chrome.alarms.create(SYNC_ALARM, {
    periodInMinutes: SYNC_INTERVAL_MINUTES,
  });
  updateBadge();
});

// ── Alarm Handler ───────────────────────────────────────────────────────────

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === SYNC_ALARM) {
    const storage = await getStorage();
    if (storage.authToken && storage.captureEnabled && storage.pendingConversations.length > 0) {
      await syncPending();
      await updateBadge();
    }
  }
});

// ── Message Handler ─────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener(
  (message: ExtensionMessage, sender, sendResponse) => {
    handleMessage(message, sender).then(sendResponse);
    return true; // Keep channel open for async response
  },
);

async function handleMessage(
  message: ExtensionMessage,
  _sender: chrome.runtime.MessageSender,
): Promise<unknown> {
  switch (message.type) {
    case "CONVERSATION_UPDATE": {
      const storage = await getStorage();
      if (!storage.captureEnabled) return { queued: false };

      const queueSize = await enqueueConversation(message.payload);
      await updateBadge();

      // Debounced sync: if we have enough in the queue, sync immediately
      if (queueSize >= 3) {
        syncPending().then(() => updateBadge());
      }

      return { queued: true, queueSize };
    }

    case "GET_STATUS": {
      return await getStatus();
    }

    case "AUTH_TOKEN": {
      await setAuthToken(message.payload.token);
      // Trigger sync now that we have auth
      syncPending().then(() => updateBadge());
      return { success: true };
    }

    case "LOGOUT": {
      await setAuthToken(null);
      await updateBadge();
      return { success: true };
    }

    case "SYNC_NOW": {
      const result = await syncPending();
      await updateBadge();
      return result;
    }

    case "CAPTURE_TOGGLED": {
      await setCaptureEnabled(message.payload.enabled);
      return { success: true };
    }

    default:
      return { error: "Unknown message type" };
  }
}

// ── Status ──────────────────────────────────────────────────────────────────

async function getStatus(): Promise<ExtensionStatus> {
  const storage = await getStorage();

  // Detect active platform from current tab
  let activePlatform: Platform | null = null;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.url) {
      activePlatform = detectPlatformFromUrl(tab.url);
    }
  } catch {
    // No active tab access
  }

  return {
    isAuthenticated: storage.authToken !== null && storage.authToken.length > 0,
    captureEnabled: storage.captureEnabled,
    pendingSync: storage.pendingConversations.length,
    lastSyncAt: storage.lastSyncAt,
    totalCaptured: storage.totalCaptured,
    activePlatform,
  };
}

function detectPlatformFromUrl(url: string): Platform | null {
  if (url.includes("chatgpt.com") || url.includes("chat.openai.com")) return "chatgpt";
  if (url.includes("claude.ai")) return "claude";
  if (url.includes("gemini.google.com") || url.includes("aistudio.google.com")) return "gemini";
  return null;
}

// ── Badge ───────────────────────────────────────────────────────────────────

async function updateBadge(): Promise<void> {
  const pending = await getPendingCount();

  if (pending > 0) {
    await chrome.action.setBadgeText({ text: String(pending) });
    await chrome.action.setBadgeBackgroundColor({ color: "#f97316" }); // Orange
  } else {
    await chrome.action.setBadgeText({ text: "" });
  }
}
