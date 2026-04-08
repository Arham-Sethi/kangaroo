/**
 * Content script for Claude (claude.ai).
 *
 * Injected automatically by the manifest. Sets up the DOM observer
 * and sends captured conversations to the background service worker.
 */

import { ConversationObserver } from "./observer";
import { parseClaudeMessages, getClaudeTitle, detectClaudeModel } from "../parsers/claude";
import type { ScrapedConversation } from "../lib/types";

const CONTAINER_SELECTORS = [
  'main',
  '[class*="conversation"]',
  '[class*="chat-container"]',
  '#__next main',
] as const;

function onCapture(conversation: ScrapedConversation): void {
  chrome.runtime.sendMessage({
    type: "CONVERSATION_UPDATE",
    payload: conversation,
  });
}

const observer = new ConversationObserver({
  platform: "claude",
  parseMessages: parseClaudeMessages,
  getTitle: getClaudeTitle,
  detectModel: detectClaudeModel,
  containerSelectors: CONTAINER_SELECTORS,
  debounceMs: 2000,
  onCapture,
});

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => observer.start());
} else {
  observer.start();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "SCRAPE_NOW" && message.payload.platform === "claude") {
    const result = observer.scrapeNow();
    sendResponse({ success: !!result, messageCount: result?.messageCount ?? 0 });
  }
  return true;
});

// SPA navigation watcher
let lastUrl = window.location.href;
const urlObserver = new MutationObserver(() => {
  if (window.location.href !== lastUrl) {
    lastUrl = window.location.href;
    observer.stop();
    setTimeout(() => observer.start(), 1000);
  }
});

urlObserver.observe(document.body, { childList: true, subtree: true });
