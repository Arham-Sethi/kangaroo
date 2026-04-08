"use client";

import * as React from "react";
import {
  BarChart3,
  TrendingUp,
  ArrowRightLeft,
  Brain,
  Zap,
  Calendar,
  Loader2,
  Database,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";

import { AppShell } from "@/components/layout/app-shell";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  useUsage,
  useDailyUsage,
  useModelDistribution,
  useAnalyticsSummary,
} from "@/hooks/use-api-query";
import { formatNumber } from "@/lib/utils";

// ── Stat Card ──────────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  subtitle,
  icon: Icon,
}: {
  readonly label: string;
  readonly value: string;
  readonly subtitle?: string;
  readonly icon: React.ElementType;
}) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-zinc-500">{label}</p>
            <p className="mt-1 text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              {value}
            </p>
            {subtitle && (
              <p className="mt-1 text-xs text-zinc-400">{subtitle}</p>
            )}
          </div>
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-300">
            <Icon className="h-6 w-6" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ── Main Content ───────────────────────────────────────────────────────────

function AnalyticsContent() {
  const { data: usage } = useUsage();
  const { data: summary, isLoading: summaryLoading } = useAnalyticsSummary();
  const { data: dailyUsage, isLoading: dailyLoading } = useDailyUsage(30);
  const { data: modelDist } = useModelDistribution();

  const shifts = summary?.total_shifts ?? 0;
  const sessions = summary?.total_sessions ?? 0;
  const tokens = summary?.total_tokens ?? 0;
  const shiftsThisMonth = summary?.shifts_this_month ?? 0;

  // Aggregate model distribution for the bar chart
  const modelChartData = React.useMemo(() => {
    if (!modelDist) return [];
    const grouped: Record<string, number> = {};
    for (const entry of modelDist) {
      grouped[entry.model] = (grouped[entry.model] ?? 0) + entry.count;
    }
    return Object.entries(grouped)
      .map(([model, count]) => ({ model, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [modelDist]);

  return (
    <div className="mx-auto max-w-6xl space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
          Analytics
        </h1>
        <p className="mt-1 text-zinc-500">
          Track your usage patterns and productivity metrics.
        </p>
      </div>

      <div className="flex items-center gap-2">
        <Badge variant="secondary" className="flex items-center gap-1">
          <Calendar className="h-3 w-3" />
          {usage?.period ?? "Current Period"}
        </Badge>
        {summary?.subscription_tier && (
          <Badge variant="outline">
            {summary.subscription_tier} tier
          </Badge>
        )}
      </div>

      {/* Summary Stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Context Shifts"
          value={summaryLoading ? "..." : formatNumber(shifts)}
          subtitle={`${shiftsThisMonth} this month`}
          icon={ArrowRightLeft}
        />
        <StatCard
          label="Total Sessions"
          value={summaryLoading ? "..." : formatNumber(sessions)}
          subtitle={`${summary?.active_sessions ?? 0} active`}
          icon={Database}
        />
        <StatCard
          label="Tokens Processed"
          value={summaryLoading ? "..." : formatNumber(tokens)}
          icon={Zap}
        />
        <StatCard
          label="Brain Queries"
          value={formatNumber(
            usage?.metrics.find((m) => m.metric === "brain_queries")?.current ?? 0,
          )}
          icon={Brain}
        />
      </div>

      {/* Daily Usage Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Usage Over Time</CardTitle>
          <CardDescription>
            Daily shifts and sessions for the last 30 days.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {dailyLoading ? (
            <div className="flex h-64 items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
            </div>
          ) : dailyUsage && dailyUsage.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={[...dailyUsage]}>
                <defs>
                  <linearGradient id="colorShifts" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorSessions" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v: string) => v.slice(5)}
                />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    borderRadius: "8px",
                    border: "1px solid #e4e4e7",
                    fontSize: "12px",
                  }}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="shifts"
                  stroke="#f97316"
                  fillOpacity={1}
                  fill="url(#colorShifts)"
                  name="Shifts"
                />
                <Area
                  type="monotone"
                  dataKey="sessions"
                  stroke="#3b82f6"
                  fillOpacity={1}
                  fill="url(#colorSessions)"
                  name="Sessions"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-64 items-center justify-center rounded-lg border-2 border-dashed border-zinc-200 dark:border-zinc-700">
              <div className="text-center">
                <BarChart3 className="mx-auto h-10 w-10 text-zinc-300" />
                <p className="mt-2 text-sm text-zinc-500">
                  No usage data yet. Start using Kangaroo Shift to see your analytics.
                </p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Model Distribution Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Model Distribution</CardTitle>
          <CardDescription>
            Which LLM models you use most across shifts and captures.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {modelChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={modelChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
                <XAxis dataKey="model" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    borderRadius: "8px",
                    border: "1px solid #e4e4e7",
                    fontSize: "12px",
                  }}
                />
                <Bar
                  dataKey="count"
                  fill="#f97316"
                  radius={[4, 4, 0, 0]}
                  name="Usage Count"
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-48 items-center justify-center rounded-lg border-2 border-dashed border-zinc-200 dark:border-zinc-700">
              <p className="text-sm text-zinc-500">
                No model usage data yet.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function AnalyticsPage() {
  return (
    <AppShell>
      <AnalyticsContent />
    </AppShell>
  );
}
