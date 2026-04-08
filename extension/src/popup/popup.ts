/**
 * Popup script — handles UI interactions for the extension popup.
 *
 * Communicates with the background service worker via chrome.runtime.sendMessage.
 * Updates the UI based on extension status.
 */

import type { ExtensionStatus, ExtensionMessage } from "../lib/types";

// ── DOM Elements ────────────────────────────────────────────────────────────

const authView = document.getElementById("auth-view")!;
const mainView = document.getElementById("main-view")!;
const loginBtn = document.getElementById("login-btn") as HTMLButtonElement;
const logoutBtn = document.getElementById("logout-btn") as HTMLButtonElement;
const syncBtn = document.getElementById("sync-btn") as HTMLButtonElement;
const scrapeBtn = document.getElementById("scrape-btn") as HTMLButtonElement;
const captureToggle = document.getElementById("capture-toggle")!;
const platformBadge = document.getElementById("platform-badge")!;
const pendingCount = document.getElementById("pending-count")!;
const totalCaptured = document.getElementById("total-captured")!;
const lastSync = document.getElementById("last-sync")!;
const emailInput = document.getElementById("login-email") as HTMLInputElement;
const passwordInput = document.getElementById("login-password") as HTMLInputElement;
const errorMsg = document.getElementById("error-msg")!;

// ── State ───────────────────────────────────────────────────────────────────

let currentStatus: ExtensionStatus | null = null;

// ── Initialize ──────────────────────────────────────────────────────────────

async function init(): Promise<void> {
  const status = await sendMessage({ type: "GET_STATUS" }) as ExtensionStatus;
  currentStatus = status;
  render(status);
}

function render(status: ExtensionStatus): void {
  if (status.isAuthenticated) {
    authView.classList.add("hidden");
    mainView.classList.remove("hidden");
  } else {
    authView.classList.remove("hidden");
    mainView.classList.add("hidden");
    return;
  }

  // Platform
  if (status.activePlatform) {
    const names: Record<string, string> = {
      chatgpt: "ChatGPT",
      claude: "Claude",
      gemini: "Gemini",
    };
    platformBadge.textContent = names[status.activePlatform] ?? status.activePlatform;
    platformBadge.classList.add("platform");
  } else {
    platformBadge.textContent = "No AI page";
    platformBadge.classList.remove("platform");
  }

  // Capture toggle
  if (status.captureEnabled) {
    captureToggle.classList.add("on");
  } else {
    captureToggle.classList.remove("on");
  }

  // Stats
  pendingCount.textContent = String(status.pendingSync);
  pendingCount.className = `status-value ${status.pendingSync > 0 ? "orange" : "green"}`;
  totalCaptured.textContent = String(status.totalCaptured);

  if (status.lastSyncAt) {
    const ago = Math.round((Date.now() - status.lastSyncAt) / 60000);
    lastSync.textContent = ago < 1 ? "Just now" : `${ago}m ago`;
  } else {
    lastSync.textContent = "Never";
  }

  // Button states
  scrapeBtn.disabled = !status.activePlatform;
}

// ── Event Handlers ──────────────────────────────────────────────────────────

loginBtn.addEventListener("click", async () => {
  const email = emailInput.value.trim();
  const password = passwordInput.value;

  if (!email || !password) {
    showError("Please enter email and password.");
    return;
  }

  loginBtn.disabled = true;
  loginBtn.textContent = "Signing in...";
  hideError();

  try {
    // Call the backend auth API directly from popup
    const storage = await chrome.storage.local.get("apiBaseUrl");
    const baseUrl = storage.apiBaseUrl || "http://localhost:8000/api/v1";

    const res = await fetch(`${baseUrl}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail?.message ?? data.message ?? "Invalid credentials");
    }

    const data = await res.json();
    await sendMessage({ type: "AUTH_TOKEN", payload: { token: data.access_token } });

    // Refresh status
    await init();
  } catch (err) {
    showError(err instanceof Error ? err.message : "Login failed");
  } finally {
    loginBtn.disabled = false;
    loginBtn.textContent = "Sign In";
  }
});

logoutBtn.addEventListener("click", async () => {
  await sendMessage({ type: "LOGOUT" });
  await init();
});

syncBtn.addEventListener("click", async () => {
  syncBtn.disabled = true;
  syncBtn.textContent = "Syncing...";

  const result = await sendMessage({ type: "SYNC_NOW" }) as { synced: number; failed: number; pending: number };

  syncBtn.textContent = result.synced > 0 ? `Synced ${result.synced}!` : "Sync Now";
  syncBtn.disabled = false;

  // Refresh
  setTimeout(() => init(), 1000);
});

scrapeBtn.addEventListener("click", async () => {
  if (!currentStatus?.activePlatform) return;

  scrapeBtn.disabled = true;
  scrapeBtn.textContent = "Capturing...";

  // Send scrape request to the active tab's content script
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) {
    try {
      await chrome.tabs.sendMessage(tab.id, {
        type: "SCRAPE_NOW",
        payload: { platform: currentStatus.activePlatform },
      });
    } catch {
      // Content script may not be loaded
    }
  }

  scrapeBtn.textContent = "Captured!";
  setTimeout(() => {
    scrapeBtn.textContent = "Capture Page";
    scrapeBtn.disabled = false;
    init();
  }, 1500);
});

captureToggle.addEventListener("click", async () => {
  const isOn = captureToggle.classList.contains("on");
  const newState = !isOn;

  await sendMessage({ type: "CAPTURE_TOGGLED", payload: { enabled: newState } });

  if (newState) {
    captureToggle.classList.add("on");
  } else {
    captureToggle.classList.remove("on");
  }
});

// ── Helpers ─────────────────────────────────────────────────────────────────

function sendMessage(message: ExtensionMessage): Promise<unknown> {
  return chrome.runtime.sendMessage(message);
}

function showError(msg: string): void {
  errorMsg.textContent = msg;
  errorMsg.style.display = "block";
}

function hideError(): void {
  errorMsg.style.display = "none";
}

// ── Start ───────────────────────────────────────────────────────────────────

init();
