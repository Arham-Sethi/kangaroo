"use client";

import * as React from "react";
import Link from "next/link";
import {
  ArrowRightLeft,
  Brain,
  Monitor,
  Zap,
  ArrowRight,
  AlertCircle,
  Clock,
} from "lucide-react";

import { AppShell } from "@/components/layout/app-shell";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useAuthStore } from "@/stores/auth-store";
import { useUsage, useSessions } from "@/hooks/use-api-query";
import { formatNumber } from "@/lib/utils";
import { WelcomeModal, useOnboarding } from "@/components/onboarding/welcome-modal";

function UsageCard({
  label,
  current,
  limit,
  icon: Icon,
}: {
  label: string;
  current: number;
  limit: number;
  icon: React.ElementType;
}) {
  const isUnlimited = limit === -1;
  const percentage = isUnlimited ? 15 : Math.min(100, (current / limit) * 100);
  const isNearLimit = !isUnlimited && percentage >= 80;

  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-300">
              <Icon className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-medium text-zinc-600 dark:text-zinc-400">
                {label}
              </p>
              <p className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
                {formatNumber(current)}
                <span className="text-sm font-normal text-zinc-400">
                  {" / "}
                  {isUnlimited ? "Unlimited" : formatNumber(limit)}
                </span>
              </p>
            </div>
          </div>
          {isNearLimit && <AlertCircle className="h-5 w-5 text-amber-500" />}
        </div>
        <Progress
          value={percentage}
          className="mt-3"
          indicatorClassName={
            isNearLimit ? "bg-amber-500" : isUnlimited ? "bg-emerald-500" : undefined
          }
        />
      </CardContent>
    </Card>
  );
}

function QuickAction({
  href,
  icon: Icon,
  label,
  description,
}: {
  href: string;
  icon: React.ElementType;
  label: string;
  description: string;
}) {
  return (
    <Link href={href}>
      <Card className="group cursor-pointer transition-shadow hover:shadow-md">
        <CardContent className="flex items-center gap-4 p-5">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-orange-100 text-orange-600 transition-colors group-hover:bg-orange-500 group-hover:text-white dark:bg-orange-900 dark:text-orange-300">
            <Icon className="h-6 w-6" />
          </div>
          <div className="flex-1">
            <p className="font-semibold text-zinc-900 dark:text-zinc-100">{label}</p>
            <p className="text-sm text-zinc-500">{description}</p>
          </div>
          <ArrowRight className="h-5 w-5 text-zinc-300 transition-transform group-hover:translate-x-1 group-hover:text-orange-500" />
        </CardContent>
      </Card>
    </Link>
  );
}

function DashboardContent() {
  const { user, subscription } = useAuthStore();
  const { data: usage } = useUsage();
  const { data: sessions } = useSessions();

  const shiftsMetric = usage?.metrics.find((m) => m.metric === "shifts");
  const brainMetric = usage?.metrics.find((m) => m.metric === "brain_queries");
  const tokensMetric = usage?.metrics.find((m) => m.metric === "tokens_processed");

  return (
    <div className="mx-auto max-w-6xl space-y-8 animate-fade-in">
      {/* Welcome */}
      <div>
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
          Welcome back, {user?.name?.split(" ")[0] ?? "there"}
        </h1>
        <p className="mt-1 text-zinc-500">
          Here&apos;s your Kangaroo Shift overview for this month.
        </p>
      </div>

      {/* Trial banner */}
      {subscription?.is_trial_active && (
        <div className="rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 p-5 text-white shadow-lg">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-lg font-bold">
                Pro Trial — {Math.ceil(subscription.trial_days_remaining)} days left
              </p>
              <p className="text-sm opacity-90">
                You have full Pro access. Upgrade to keep all your features.
              </p>
            </div>
            <Link href="/pricing">
              <Button variant="secondary" size="sm">
                Upgrade Now
              </Button>
            </Link>
          </div>
        </div>
      )}

      {/* Expired trial nudge */}
      {subscription?.trial_expired && subscription.tier === "free" && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 dark:border-amber-800 dark:bg-amber-950">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="font-semibold text-amber-800 dark:text-amber-200">
                Your Pro trial has ended
              </p>
              <p className="text-sm text-amber-600 dark:text-amber-400">
                Upgrade to unlock unlimited shifts, cockpit, and smart dispatch.
              </p>
            </div>
            <Link href="/pricing">
              <Button size="sm">See Plans</Button>
            </Link>
          </div>
        </div>
      )}

      {/* Usage stats */}
      <div>
        <h2 className="mb-4 text-lg font-semibold text-zinc-800 dark:text-zinc-200">
          Monthly Usage
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <UsageCard
            label="Context Shifts"
            current={shiftsMetric?.current ?? 0}
            limit={shiftsMetric?.limit ?? 10}
            icon={ArrowRightLeft}
          />
          <UsageCard
            label="Brain Queries"
            current={brainMetric?.current ?? 0}
            limit={brainMetric?.limit ?? 15}
            icon={Brain}
          />
          <UsageCard
            label="Tokens Processed"
            current={tokensMetric?.current ?? 0}
            limit={tokensMetric?.limit ?? 1000000}
            icon={Zap}
          />
        </div>
      </div>

      {/* Quick actions */}
      <div>
        <h2 className="mb-4 text-lg font-semibold text-zinc-800 dark:text-zinc-200">
          Quick Actions
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <QuickAction href="/shift" icon={ArrowRightLeft} label="New Shift" description="Transfer context between models" />
          <QuickAction href="/brain" icon={Brain} label="Brain Explorer" description="Search and recall your knowledge" />
          <QuickAction href="/cockpit" icon={Monitor} label="Open Cockpit" description="Multi-model parallel chat" />
        </div>
      </div>

      {/* Recent sessions */}
      <div>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">Recent Sessions</h2>
          <Link href="/shift" className="text-sm text-orange-500 hover:text-orange-600">View all</Link>
        </div>
        {sessions && sessions.length > 0 ? (
          <div className="space-y-3">
            {sessions.slice(0, 5).map((session) => (
              <Card key={session.id}>
                <CardContent className="flex items-center justify-between p-4">
                  <div className="flex items-center gap-3">
                    <ArrowRightLeft className="h-4 w-4 text-zinc-400" />
                    <div>
                      <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        {session.title || "Untitled Shift"}
                      </p>
                      <p className="text-xs text-zinc-500">
                        {session.source_model} &rarr; {session.target_model}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={session.status === "completed" ? "success" : "secondary"}>
                      {session.status}
                    </Badge>
                    <span className="text-xs text-zinc-400">
                      <Clock className="mr-1 inline h-3 w-3" />
                      {new Date(session.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
              <ArrowRightLeft className="h-10 w-10 text-zinc-300" />
              <div>
                <p className="font-medium text-zinc-600 dark:text-zinc-400">No shifts yet</p>
                <p className="text-sm text-zinc-400">Start your first context shift to see it here.</p>
              </div>
              <Link href="/shift">
                <Button size="sm">
                  Create First Shift <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { showOnboarding, completeOnboarding } = useOnboarding();

  return (
    <AppShell>
      {showOnboarding && <WelcomeModal onComplete={completeOnboarding} />}
      <DashboardContent />
    </AppShell>
  );
}
