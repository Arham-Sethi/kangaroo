/**
 * Content script for ChatGPT (chatgpt.com / chat.openai.com).
 *
 * Injected automatically by the manifest. Sets up the DOM observer
 * and sends captured conversations to the background service worker.
 */

import { ConversationObserver } from "./observer";
import { parseChatGPTMessages, getChatGPTTitle, detectChatGPTModel } from "../parsers/chatgpt";
import type { ScrapedConversation } from "../lib/types";

const CONTAINER_SELECTORS = [
  'main',
  '[class*="conversation-content"]',
  '[role="presentation"]',
  '#__next main',
] as const;

function onCapture(conversation: ScrapedConversation): void {
  chrome.runtime.sendMessage({
    type: "CONVERSATION_UPDATE",
    payload: conversation,
  });
}

// ── Initialize ──────────────────────────────────────────────────────────────

const observer = new ConversationObserver({
  platform: "chatgpt",
  parseMessages: parseChatGPTMessages,
  getTitle: getChatGPTTitle,
  detectModel: detectChatGPTModel,
  containerSelectors: CONTAINER_SELECTORS,
  debounceMs: 2000, // Wait 2s after last DOM change (streaming)
  onCapture,
});

// Start observing when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => observer.start());
} else {
  observer.start();
}

// Listen for manual scrape requests from popup/background
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "SCRAPE_NOW" && message.payload.platform === "chatgpt") {
    const result = observer.scrapeNow();
    sendResponse({ success: !!result, messageCount: result?.messageCount ?? 0 });
  }
  return true; // Keep channel open for async
});

// Re-observe on SPA navigation (ChatGPT uses client-side routing)
let lastUrl = window.location.href;
const urlObserver = new MutationObserver(() => {
  if (window.location.href !== lastUrl) {
    lastUrl = window.location.href;
    observer.stop();
    // Small delay for new page content to render
    setTimeout(() => observer.start(), 1000);
  }
});

urlObserver.observe(document.body, { childList: true, subtree: true });
