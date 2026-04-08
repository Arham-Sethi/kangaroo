/**
 * DOM Observer engine — watches for new messages and triggers scraping.
 *
 * Uses MutationObserver to detect when new conversation turns are added
 * to the DOM. Debounces scraping to avoid hammering on rapid updates
 * (e.g., streaming responses).
 *
 * The observer is platform-agnostic — it watches the entire conversation
 * container and delegates to platform-specific parsers.
 */

import type { Platform, ScrapedConversation, ScrapedMessage } from "../lib/types";
import { generateConversationId } from "../lib/hasher";

export interface ObserverConfig {
  readonly platform: Platform;
  readonly parseMessages: () => readonly ScrapedMessage[];
  readonly getTitle: () => string;
  readonly detectModel: () => string;
  /** Container selectors to observe (tries each until one matches). */
  readonly containerSelectors: readonly string[];
  /** Debounce delay in ms before scraping after DOM change. */
  readonly debounceMs: number;
  /** Callback when a conversation snapshot is captured. */
  readonly onCapture: (conversation: ScrapedConversation) => void;
}

export class ConversationObserver {
  private observer: MutationObserver | null = null;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private lastMessageCount = 0;
  private readonly config: ObserverConfig;

  constructor(config: ObserverConfig) {
    this.config = config;
  }

  start(): boolean {
    const container = this.findContainer();
    if (!container) {
      // Retry in 2 seconds — page may still be loading
      setTimeout(() => this.start(), 2000);
      return false;
    }

    this.observer = new MutationObserver(() => {
      this.debouncedScrape();
    });

    this.observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    // Initial scrape
    this.scrapeNow();
    return true;
  }

  stop(): void {
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }

  /** Force an immediate scrape and capture. */
  scrapeNow(): ScrapedConversation | null {
    const messages = this.config.parseMessages();
    if (messages.length === 0) return null;

    // Only capture if message count changed (avoids duplicate syncs)
    if (messages.length === this.lastMessageCount) return null;
    this.lastMessageCount = messages.length;

    const conversation: ScrapedConversation = {
      id: generateConversationId(this.config.platform, window.location.href),
      platform: this.config.platform,
      model: this.config.detectModel(),
      title: this.config.getTitle(),
      url: window.location.href,
      messages,
      scrapedAt: Date.now(),
      messageCount: messages.length,
    };

    this.config.onCapture(conversation);
    return conversation;
  }

  private debouncedScrape(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = setTimeout(() => {
      this.scrapeNow();
    }, this.config.debounceMs);
  }

  private findContainer(): Element | null {
    for (const selector of this.config.containerSelectors) {
      try {
        const el = document.querySelector(selector);
        if (el) return el;
      } catch {
        // Invalid selector
      }
    }
    return null;
  }
}
