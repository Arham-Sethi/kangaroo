"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ArrowRightLeft,
  Brain,
  Monitor,
  BarChart3,
  Settings,
  Zap,
  CreditCard,
  Menu,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/stores/auth-store";

interface NavItem {
  readonly label: string;
  readonly href: string;
  readonly icon: React.ElementType;
  readonly badge?: string;
  readonly proOnly?: boolean;
}

const NAV_ITEMS: readonly NavItem[] = [
  { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { label: "Shift", href: "/shift", icon: ArrowRightLeft },
  { label: "Brain", href: "/brain", icon: Brain },
  { label: "Cockpit", href: "/cockpit", icon: Monitor, proOnly: true },
  { label: "Workflows", href: "/workflows", icon: Zap, proOnly: true },
  { label: "Analytics", href: "/analytics", icon: BarChart3 },
];

const BOTTOM_NAV: readonly NavItem[] = [
  { label: "Settings", href: "/settings", icon: Settings },
  { label: "Billing", href: "/settings/billing", icon: CreditCard },
];

export function Sidebar() {
  const pathname = usePathname();
  const subscription = useAuthStore((s) => s.subscription);
  const [mobileOpen, setMobileOpen] = React.useState(false);

  const effectiveTier = subscription?.effective_tier ?? "free";
  const isPro = effectiveTier !== "free";

  return (
    <>
      {/* Mobile toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed left-4 top-4 z-50 lg:hidden"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
      >
        {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {/* Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed left-0 top-0 z-40 flex h-full w-64 flex-col border-r border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-950",
          "transition-transform duration-200 lg:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center gap-2 border-b border-zinc-200 px-6 dark:border-zinc-800">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-500 text-white font-bold text-sm">
            K
          </div>
          <span className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
            Kangaroo Shift
          </span>
        </div>

        {/* Main nav */}
        <nav className="flex-1 space-y-1 px-3 py-4">
          {NAV_ITEMS.map((item) => {
            const isActive =
              pathname === item.href || pathname.startsWith(item.href + "/");
            const isLocked = item.proOnly && !isPro;

            return (
              <Link
                key={item.href}
                href={isLocked ? "/pricing" : item.href}
                onClick={() => setMobileOpen(false)}
                className={cn(
                  "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-orange-50 text-orange-600 dark:bg-orange-950 dark:text-orange-400"
                    : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100",
                  isLocked && "opacity-60",
                )}
              >
                <item.icon className="h-5 w-5 flex-shrink-0" />
                <span className="flex-1">{item.label}</span>
                {isLocked && (
                  <Badge variant="pro" className="text-[10px] px-1.5 py-0">
                    PRO
                  </Badge>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Bottom nav */}
        <div className="border-t border-zinc-200 px-3 py-4 dark:border-zinc-800">
          {BOTTOM_NAV.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-orange-50 text-orange-600 dark:bg-orange-950 dark:text-orange-400"
                    : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800",
                )}
              >
                <item.icon className="h-5 w-5" />
                {item.label}
              </Link>
            );
          })}
        </div>

        {/* Trial banner */}
        {subscription?.is_trial_active && (
          <div className="mx-3 mb-4 rounded-lg bg-gradient-to-r from-orange-500 to-amber-500 p-3 text-white">
            <p className="text-xs font-semibold">Pro Trial Active</p>
            <p className="text-xs opacity-90">
              {Math.ceil(subscription.trial_days_remaining)} days remaining
            </p>
            <Link
              href="/pricing"
              className="mt-2 block text-center text-xs font-bold underline"
            >
              Upgrade Now
            </Link>
          </div>
        )}
      </aside>
    </>
  );
}
