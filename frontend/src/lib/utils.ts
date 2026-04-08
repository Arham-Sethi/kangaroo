import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes with conflict resolution. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format a number with commas (1000 → "1,000"). */
export function formatNumber(n: number): string {
  if (n === -1) return "Unlimited";
  return n.toLocaleString("en-US");
}

/** Format USD price ("$12.00"). */
export function formatPrice(usd: number): string {
  if (usd === 0) return "Free";
  return `$${usd.toFixed(usd % 1 === 0 ? 0 : 2)}`;
}

/** Truncate text to maxLen with ellipsis. */
export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + "\u2026";
}

/** Sleep for ms (use in loading states / animations). */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
