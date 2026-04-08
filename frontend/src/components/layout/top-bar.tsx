"use client";

import * as React from "react";
import { LogOut, User, ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/stores/auth-store";

export function TopBar() {
  const { user, subscription, logout } = useAuthStore();
  const [menuOpen, setMenuOpen] = React.useState(false);

  const displayTier = subscription?.display_name ?? "Free";
  const effectiveTier = subscription?.effective_tier ?? "free";

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-zinc-200 bg-white/80 px-6 backdrop-blur-sm dark:border-zinc-800 dark:bg-zinc-950/80">
      {/* Left spacer for mobile menu button */}
      <div className="w-10 lg:w-0" />

      {/* Right side */}
      <div className="ml-auto flex items-center gap-4">
        {/* Tier badge */}
        <Badge
          variant={effectiveTier === "free" ? "secondary" : "pro"}
          className="hidden sm:inline-flex"
        >
          {displayTier}
          {subscription?.is_trial_active && " (Trial)"}
        </Badge>

        {/* User menu */}
        <div className="relative">
          <Button
            variant="ghost"
            size="sm"
            className="flex items-center gap-2"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-300">
              <User className="h-4 w-4" />
            </div>
            <span className="hidden text-sm font-medium sm:inline">
              {user?.name ?? user?.email ?? "User"}
            </span>
            <ChevronDown className="h-4 w-4 text-zinc-400" />
          </Button>

          {menuOpen && (
            <>
              <div
                className="fixed inset-0 z-40"
                onClick={() => setMenuOpen(false)}
              />
              <div className="absolute right-0 top-full z-50 mt-2 w-48 rounded-lg border border-zinc-200 bg-white p-1 shadow-lg dark:border-zinc-700 dark:bg-zinc-900">
                <div className="border-b border-zinc-100 px-3 py-2 dark:border-zinc-800">
                  <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                    {user?.name ?? "User"}
                  </p>
                  <p className="text-xs text-zinc-500">{user?.email}</p>
                </div>
                <button
                  onClick={() => {
                    setMenuOpen(false);
                    logout();
                  }}
                  className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950"
                >
                  <LogOut className="h-4 w-4" />
                  Log Out
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
