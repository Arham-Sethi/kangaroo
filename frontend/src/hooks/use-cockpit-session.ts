/**
 * Cockpit session management hook.
 *
 * Generates a stable session ID from the URL search params or creates
 * a new one, and persists it across navigations. The session ID is
 * passed to `useCockpitWS` which handles the actual WebSocket
 * connection to the backend.
 *
 * Usage:
 *   const { sessionId, isNewSession, resetSession } = useCockpitSession();
 */

"use client";

import * as React from "react";
import { useSearchParams, useRouter } from "next/navigation";

function generateSessionId(): string {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  let id = "";
  for (let i = 0; i < 16; i++) {
    id += chars[Math.floor(Math.random() * chars.length)];
  }
  return id;
}

export interface UseCockpitSessionReturn {
  /** The current cockpit session ID. */
  readonly sessionId: string;
  /** Whether this session was freshly created (no prior URL param). */
  readonly isNewSession: boolean;
  /** Reset to a fresh session (generates new ID, updates URL). */
  readonly resetSession: () => void;
}

export function useCockpitSession(): UseCockpitSessionReturn {
  const searchParams = useSearchParams();
  const router = useRouter();

  const existingId = searchParams.get("session");
  const isNewSession = !existingId;

  const [sessionId] = React.useState<string>(() => {
    return existingId || generateSessionId();
  });

  // Sync session ID to URL on first render if missing
  React.useEffect(() => {
    if (!existingId) {
      const url = new URL(window.location.href);
      url.searchParams.set("session", sessionId);
      router.replace(url.pathname + url.search, { scroll: false });
    }
  }, [existingId, sessionId, router]);

  const resetSession = React.useCallback(() => {
    const newId = generateSessionId();
    const url = new URL(window.location.href);
    url.searchParams.set("session", newId);
    // Full page navigation to reset all state
    window.location.href = url.toString();
  }, []);

  return {
    sessionId,
    isNewSession,
    resetSession,
  };
}
