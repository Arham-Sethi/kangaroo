/**
 * Gemini DOM parser — extracts messages from gemini.google.com.
 *
 * Selector strategy:
 *   - Gemini uses Web Components (custom elements)
 *   - User messages: message-content with "user" data attribute
 *   - Model messages: message-content with "model" data attribute
 *   - Gemini's DOM uses model-response and user-query custom tags
 *
 * Google's Gemini UI has more custom element usage than ChatGPT/Claude,
 * requiring deeper shadow DOM traversal in some cases.
 */

import type { ScrapedMessage, MessageRole } from "../lib/types";
import { hashMessage } from "../lib/hasher";

// ── Selectors ───────────────────────────────────────────────────────────────

const USER_TURN_SELECTORS = [
  'user-query',
  '[data-turn-role="user"]',
  'div[class*="query-text"]',
  'div[class*="user-message"]',
  '.query-content',
] as const;

const MODEL_TURN_SELECTORS = [
  'model-response',
  '[data-turn-role="model"]',
  'div[class*="response-text"]',
  'div[class*="model-response"]',
  '.response-content',
] as const;

const TURN_PAIR_SELECTORS = [
  'conversation-turn',
  '[class*="turn-container"]',
  'div[class*="conversation-turn"]',
] as const;

const MODEL_NAME_SELECTORS = [
  '[class*="model-badge"]',
  'span[class*="model-name"]',
  '[data-model-name]',
] as const;

// ── Core Parser ─────────────────────────────────────────────────────────────

export function parseGeminiMessages(): readonly ScrapedMessage[] {
  const messages: ScrapedMessage[] = [];
  const model = detectGeminiModel();

  // Strategy 1: Explicit user-query / model-response custom elements
  const userTurns = queryAll(USER_TURN_SELECTORS);
  const modelTurns = queryAll(MODEL_TURN_SELECTORS);

  if (userTurns.length > 0 || modelTurns.length > 0) {
    const allTurns: Array<{ el: Element; role: MessageRole }> = [
      ...userTurns.map((el) => ({ el, role: "user" as const })),
      ...modelTurns.map((el) => ({ el, role: "assistant" as const })),
    ];

    // Sort by DOM position
    allTurns.sort((a, b) => {
      const pos = a.el.compareDocumentPosition(b.el);
      return pos & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
    });

    allTurns.forEach(({ el, role }, index) => {
      const content = extractContent(el);
      if (content.length > 0) {
        messages.push({
          role,
          content,
          model: role === "user" ? "user-input" : model,
          timestamp: Date.now(),
          hash: hashMessage(role, content, index),
        });
      }
    });

    return messages;
  }

  // Strategy 2: Turn pair containers
  const pairs = queryAll(TURN_PAIR_SELECTORS);
  pairs.forEach((pair, index) => {
    // Each turn pair typically has a user query followed by a model response
    const role: MessageRole = index % 2 === 0 ? "user" : "assistant";
    const content = extractContent(pair);

    if (content.length > 0) {
      messages.push({
        role,
        content,
        model: role === "user" ? "user-input" : model,
        timestamp: Date.now(),
        hash: hashMessage(role, content, index),
      });
    }
  });

  return messages;
}

export function getGeminiTitle(): string {
  const title = document.title
    .replace(/ - Gemini$/, "")
    .replace(/ - Google AI Studio$/, "")
    .replace(/^Gemini$/, "")
    .trim();

  if (title.length > 0 && title !== "Gemini") return title;

  // Try to find conversation title in sidebar or header
  const titleEl = document.querySelector(
    '[class*="conversation-title"], [class*="chat-title"]'
  );
  if (titleEl?.textContent?.trim()) {
    return titleEl.textContent.trim();
  }

  return "Gemini Conversation";
}

export function detectGeminiModel(): string {
  // Check model name elements
  for (const selector of MODEL_NAME_SELECTORS) {
    const el = document.querySelector(selector);
    if (el?.textContent) {
      const text = el.textContent.trim().toLowerCase();
      if (text.includes("2.5") && text.includes("pro")) return "gemini-2.5-pro";
      if (text.includes("2.0") && text.includes("flash")) return "gemini-2.0-flash";
      if (text.includes("1.5") && text.includes("pro")) return "gemini-1.5-pro";
      if (text.includes("ultra")) return "gemini-ultra";
    }
  }

  // Check data attributes
  const modelAttr = document.querySelector("[data-model-name]")?.getAttribute("data-model-name");
  if (modelAttr) return modelAttr;

  // URL-based detection
  const url = window.location.href.toLowerCase();
  if (url.includes("aistudio")) return "gemini-2.5-pro";
  if (url.includes("flash")) return "gemini-2.0-flash";

  return "gemini-2.0-flash"; // Default
}

// ── Internal Helpers ────────────────────────────────────────────────────────

function extractContent(el: Element): string {
  // Gemini may use shadow DOM — try to pierce it
  let text = extractFromShadowOrLight(el);

  // Clean
  return text.replace(/\n{3,}/g, "\n\n").replace(/[ \t]+/g, " ").trim();
}

function extractFromShadowOrLight(el: Element): string {
  // Check shadow root first
  if (el.shadowRoot) {
    const clone = el.shadowRoot.cloneNode(true) as DocumentFragment;
    removeNoise(clone);
    return clone.textContent?.trim() ?? "";
  }

  // Light DOM
  const clone = el.cloneNode(true) as Element;
  removeNoise(clone);
  return clone.textContent?.trim() ?? "";
}

function removeNoise(root: Element | DocumentFragment): void {
  root.querySelectorAll("button, nav, [class*='action'], [class*='copy'], [class*='toolbar']").forEach((n) => n.remove());
}

function queryAll(selectors: readonly string[]): Element[] {
  for (const selector of selectors) {
    try {
      const els = document.querySelectorAll(selector);
      if (els.length > 0) return Array.from(els);
    } catch {
      // Invalid selector
    }
  }
  return [];
}
