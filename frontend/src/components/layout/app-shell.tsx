"use client";

import * as React from "react";
import { Sidebar } from "./sidebar";
import { TopBar } from "./top-bar";
import { useAuthStore } from "@/stores/auth-store";

/**
 * AppShell — authenticated layout with sidebar + top bar.
 * Wraps all dashboard pages.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const { initialized, initialize, user } = useAuthStore();

  React.useEffect(() => {
    initialize();
  }, [initialize]);

  // Loading state
  if (!initialized) {
    return (
      <div className="flex h-screen items-center justify-center bg-white dark:bg-zinc-950">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 animate-spin rounded-full border-4 border-orange-200 border-t-orange-500" />
          <p className="text-sm text-zinc-500">Loading...</p>
        </div>
      </div>
    );
  }

  // Not logged in — redirect handled by middleware, but show loading
  if (!user) {
    if (typeof window !== "undefined") {
      window.location.href = "/auth/login";
    }
    return null;
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <Sidebar />
      <div className="lg:pl-64">
        <TopBar />
        <main className="p-6">{children}</main>
      </div>
    </div>
  );
}
