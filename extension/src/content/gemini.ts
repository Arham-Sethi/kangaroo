/**
 * Content script for Gemini (gemini.google.com / aistudio.google.com).
 *
 * Injected automatically by the manifest. Sets up the DOM observer
 * and sends captured conversations to the background service worker.
 */

import { ConversationObserver } from "./observer";
import { parseGeminiMessages, getGeminiTitle, detectGeminiModel } from "../parsers/gemini";
import type { ScrapedConversation } from "../lib/types";

const CONTAINER_SELECTORS = [
  'main',
  'conversation-turn',
  '[class*="conversation"]',
  '[class*="chat-window"]',
] as const;

function onCapture(conversation: ScrapedConversation): void {
  chrome.runtime.sendMessage({
    type: "CONVERSATION_UPDATE",
    payload: conversation,
  });
}

const observer = new ConversationObserver({
  platform: "gemini",
  parseMessages: parseGeminiMessages,
  getTitle: getGeminiTitle,
  detectModel: detectGeminiModel,
  containerSelectors: CONTAINER_SELECTORS,
  debounceMs: 2500, // Gemini streams slower
  onCapture,
});

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => observer.start());
} else {
  observer.start();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "SCRAPE_NOW" && message.payload.platform === "gemini") {
    const result = observer.scrapeNow();
    sendResponse({ success: !!result, messageCount: result?.messageCount ?? 0 });
  }
  return true;
});

let lastUrl = window.location.href;
const urlObserver = new MutationObserver(() => {
  if (window.location.href !== lastUrl) {
    lastUrl = window.location.href;
    observer.stop();
    setTimeout(() => observer.start(), 1000);
  }
});

urlObserver.observe(document.body, { childList: true, subtree: true });
