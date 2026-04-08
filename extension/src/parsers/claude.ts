/**
 * Claude.ai DOM parser — extracts messages from claude.ai.
 *
 * Selector strategy:
 *   - Conversation container: [class*="conversation"], main
 *   - User messages: [data-is-streaming="false"] with human icon
 *   - Assistant messages: [data-is-streaming] with Claude icon
 *   - Claude uses a relatively clean DOM with distinct message blocks
 *
 * Claude renders messages in div blocks with clear "human" vs "assistant"
 * distinctions via class names and aria labels.
 */

import type { ScrapedMessage, MessageRole } from "../lib/types";
import { hashMessage } from "../lib/hasher";

// ── Selectors ───────────────────────────────────────────────────────────────

const MESSAGE_CONTAINER_SELECTORS = [
  '[class*="message-"]',
  '[data-testid*="message"]',
  'div[class*="ConversationMessage"]',
  'div.group',
] as const;

const HUMAN_INDICATORS = [
  '[data-testid="human-turn"]',
  '[class*="human"]',
  '[class*="user-message"]',
] as const;

const ASSISTANT_INDICATORS = [
  '[data-testid="assistant-turn"]',
  '[class*="assistant"]',
  '[class*="ai-message"]',
] as const;

const MODEL_SELECTORS = [
  '[class*="model-name"]',
  'button[class*="model"] span',
  '[data-testid="model-selector"]',
] as const;

// ── Core Parser ─────────────────────────────────────────────────────────────

export function parseClaudeMessages(): readonly ScrapedMessage[] {
  const messages: ScrapedMessage[] = [];
  const model = detectClaudeModel();

  // Strategy 1: Look for clearly marked human/assistant turns
  const humanTurns = queryAll(HUMAN_INDICATORS);
  const assistantTurns = queryAll(ASSISTANT_INDICATORS);

  if (humanTurns.length > 0 || assistantTurns.length > 0) {
    const allTurns: Array<{ el: Element; role: MessageRole }> = [
      ...humanTurns.map((el) => ({ el, role: "user" as const })),
      ...assistantTurns.map((el) => ({ el, role: "assistant" as const })),
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

  // Strategy 2: Generic message containers with heuristic role detection
  const containers = queryAll(MESSAGE_CONTAINER_SELECTORS);
  containers.forEach((el, index) => {
    const role = inferRole(el, index);
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

export function getClaudeTitle(): string {
  // Claude puts the conversation title in the page title
  const title = document.title
    .replace(/ - Claude$/, "")
    .replace(/^Claude$/, "")
    .trim();

  if (title.length > 0 && title !== "Claude") return title;

  // Fallback: breadcrumb or sidebar active item
  const breadcrumb = document.querySelector('[class*="breadcrumb"] span, [class*="ConversationTitle"]');
  if (breadcrumb?.textContent?.trim()) {
    return breadcrumb.textContent.trim();
  }

  return "Claude Conversation";
}

export function detectClaudeModel(): string {
  for (const selector of MODEL_SELECTORS) {
    const el = document.querySelector(selector);
    if (el?.textContent) {
      const text = el.textContent.trim().toLowerCase();
      if (text.includes("opus")) return "claude-opus-4-20250514";
      if (text.includes("sonnet")) return "claude-sonnet-4-20250514";
      if (text.includes("haiku")) return "claude-haiku-4-20250414";
    }
  }

  // Check the URL for model hints
  const url = window.location.href.toLowerCase();
  if (url.includes("opus")) return "claude-opus-4-20250514";
  if (url.includes("haiku")) return "claude-haiku-4-20250414";

  return "claude-sonnet-4-20250514"; // Default
}

// ── Internal Helpers ────────────────────────────────────────────────────────

function inferRole(el: Element, index: number): MessageRole {
  const text = el.className.toLowerCase() + " " + (el.getAttribute("data-testid") ?? "");

  if (text.includes("human") || text.includes("user")) return "user";
  if (text.includes("assistant") || text.includes("ai") || text.includes("claude")) return "assistant";

  // Check for avatar/icon hints
  const hasAvatar = el.querySelector('img[alt*="Claude"], svg[class*="claude"]');
  if (hasAvatar) return "assistant";

  // Alternating heuristic: index 0 = user, 1 = assistant, etc.
  return index % 2 === 0 ? "user" : "assistant";
}

function extractContent(el: Element): string {
  const clone = el.cloneNode(true) as Element;

  // Remove noise
  clone.querySelectorAll("button, nav, [class*='action'], [class*='copy']").forEach((n) => n.remove());

  const text = clone.textContent?.trim() ?? "";
  return text.replace(/\n{3,}/g, "\n\n").replace(/[ \t]+/g, " ").trim();
}

function queryAll(selectors: readonly string[]): Element[] {
  const results: Element[] = [];
  for (const selector of selectors) {
    try {
      const els = document.querySelectorAll(selector);
      if (els.length > 0) {
        results.push(...Array.from(els));
        break; // Use first matching selector set
      }
    } catch {
      // Invalid selector
    }
  }
  return results;
}
