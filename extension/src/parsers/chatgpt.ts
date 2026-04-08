/**
 * ChatGPT DOM parser — extracts messages from chatgpt.com.
 *
 * Selector strategy:
 *   - Conversation turns: [data-testid^="conversation-turn-"]
 *   - User messages: [data-message-author-role="user"]
 *   - Assistant messages: [data-message-author-role="assistant"]
 *   - Model indicator: button text in model selector, or meta in page
 *   - Title: <title> or first user message (truncated)
 *
 * ChatGPT's DOM is React-rendered and uses data-testid attributes
 * which are relatively stable across updates.
 *
 * Fallback selectors are included for when primary selectors change.
 */

import type { ScrapedMessage, MessageRole } from "../lib/types";
import { hashMessage } from "../lib/hasher";

// ── Selectors (ordered by reliability) ──────────────────────────────────────

const TURN_SELECTORS = [
  '[data-testid^="conversation-turn-"]',
  'article[data-scroll-anchor]',
  'div.group\\/conversation-turn',
  '[class*="ConversationTurn"]',
] as const;

const USER_ROLE_SELECTORS = [
  '[data-message-author-role="user"]',
  '[class*="user-turn"]',
] as const;

const ASSISTANT_ROLE_SELECTORS = [
  '[data-message-author-role="assistant"]',
  '[class*="assistant-turn"]',
] as const;

const MODEL_SELECTORS = [
  'button[class*="model-switcher"] span',
  '[data-testid="model-switcher"] span',
  'span[class*="text-token-text"]',
] as const;

// ── Core Parser ─────────────────────────────────────────────────────────────

export function parseChatGPTMessages(): readonly ScrapedMessage[] {
  const messages: ScrapedMessage[] = [];
  const model = detectChatGPTModel();

  // Strategy 1: data-testid conversation turns
  const turns = queryWithFallback(TURN_SELECTORS);

  if (turns.length > 0) {
    turns.forEach((turn, index) => {
      const parsed = parseTurn(turn, index, model);
      if (parsed) {
        messages.push(parsed);
      }
    });
    return messages;
  }

  // Strategy 2: Role-based selectors
  const userEls = queryWithFallback(USER_ROLE_SELECTORS);
  const assistantEls = queryWithFallback(ASSISTANT_ROLE_SELECTORS);

  const allMessages: Array<{ el: Element; role: MessageRole }> = [
    ...Array.from(userEls).map((el) => ({ el, role: "user" as const })),
    ...Array.from(assistantEls).map((el) => ({ el, role: "assistant" as const })),
  ];

  // Sort by DOM position
  allMessages.sort((a, b) => {
    const pos = a.el.compareDocumentPosition(b.el);
    return pos & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
  });

  allMessages.forEach(({ el, role }, index) => {
    const content = extractTextContent(el);
    if (content.length > 0) {
      messages.push({
        role,
        content,
        model,
        timestamp: Date.now(),
        hash: hashMessage(role, content, index),
      });
    }
  });

  return messages;
}

export function getChatGPTTitle(): string {
  // Try page title first
  const title = document.title.replace(/ \| ChatGPT$/, "").replace(/^ChatGPT$/, "").trim();
  if (title.length > 0 && title !== "ChatGPT") return title;

  // Fallback: first user message truncated
  const firstUser = document.querySelector('[data-message-author-role="user"]');
  if (firstUser) {
    const text = extractTextContent(firstUser);
    return text.slice(0, 80) + (text.length > 80 ? "..." : "");
  }

  return "ChatGPT Conversation";
}

export function detectChatGPTModel(): string {
  for (const selector of MODEL_SELECTORS) {
    const el = document.querySelector(selector);
    if (el?.textContent) {
      const text = el.textContent.trim().toLowerCase();
      if (text.includes("4o-mini")) return "gpt-4o-mini";
      if (text.includes("4o")) return "gpt-4o";
      if (text.includes("4.5")) return "gpt-4.5";
      if (text.includes("o3")) return "o3";
      if (text.includes("o1")) return "o1";
      if (text.includes("gpt-4")) return "gpt-4";
    }
  }
  return "gpt-4o"; // Default
}

// ── Internal Helpers ────────────────────────────────────────────────────────

function parseTurn(turn: Element, index: number, model: string): ScrapedMessage | null {
  const role = detectRole(turn);
  if (!role) return null;

  const content = extractTextContent(turn);
  if (content.length === 0) return null;

  return {
    role,
    content,
    model: role === "user" ? "user-input" : model,
    timestamp: Date.now(),
    hash: hashMessage(role, content, index),
  };
}

function detectRole(el: Element): MessageRole | null {
  // Check data attributes
  const authorRole = el.querySelector("[data-message-author-role]")?.getAttribute("data-message-author-role");
  if (authorRole === "user") return "user";
  if (authorRole === "assistant") return "assistant";
  if (authorRole === "system") return "system";

  // Check class-based hints
  const html = el.innerHTML.toLowerCase();
  if (html.includes("user-turn") || html.includes("human")) return "user";
  if (html.includes("assistant-turn") || html.includes("bot")) return "assistant";

  // Alternate turns: even = user, odd = assistant (heuristic)
  return null;
}

function extractTextContent(el: Element): string {
  // Skip code blocks' copy buttons, timestamps, etc.
  const clone = el.cloneNode(true) as Element;

  // Remove known noise
  clone.querySelectorAll("button, [data-testid='copy-turn-action-button'], time, nav").forEach((noise) => {
    noise.remove();
  });

  const text = clone.textContent?.trim() ?? "";

  // Clean up excessive whitespace
  return text.replace(/\n{3,}/g, "\n\n").replace(/[ \t]+/g, " ").trim();
}

function queryWithFallback(selectors: readonly string[]): NodeListOf<Element> {
  for (const selector of selectors) {
    try {
      const els = document.querySelectorAll(selector);
      if (els.length > 0) return els;
    } catch {
      // Invalid selector — try next
    }
  }
  return document.querySelectorAll("__none__");
}
