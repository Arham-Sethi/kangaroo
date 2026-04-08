"use client";

import * as React from "react";
import Link from "next/link";
import { Check, X, ArrowRight, Zap } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn, formatPrice } from "@/lib/utils";
import { useTiers } from "@/hooks/use-api-query";

type BillingCycle = "monthly" | "annual";

function FeatureRow({ label, value }: { label: string; value: string | boolean | number }) {
  if (typeof value === "boolean") {
    return (
      <li className="flex items-center gap-2 text-sm">
        {value ? (
          <Check className="h-4 w-4 text-emerald-500" />
        ) : (
          <X className="h-4 w-4 text-zinc-300" />
        )}
        <span className={value ? "text-zinc-700 dark:text-zinc-300" : "text-zinc-400"}>
          {label}
        </span>
      </li>
    );
  }

  return (
    <li className="flex items-center gap-2 text-sm">
      <Check className="h-4 w-4 text-emerald-500" />
      <span className="text-zinc-700 dark:text-zinc-300">
        {label}:{" "}
        <strong>{value === -1 ? "Unlimited" : value}</strong>
      </span>
    </li>
  );
}

export default function PricingPage() {
  const { data } = useTiers();
  const [cycle, setCycle] = React.useState<BillingCycle>("annual");

  const tiers = data?.tiers ?? [];

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      {/* Header */}
      <div className="border-b border-zinc-200 bg-white px-6 py-4 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-500 text-white font-bold text-sm">K</div>
            <span className="text-lg font-bold">Kangaroo Shift</span>
          </Link>
          <Link href="/auth/login">
            <Button variant="ghost" size="sm">Sign In</Button>
          </Link>
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-6 py-16">
        {/* Hero */}
        <div className="mb-12 text-center">
          <Badge variant="pro" className="mb-4">Pricing</Badge>
          <h1 className="text-4xl font-bold text-zinc-900 dark:text-zinc-100">
            Choose your plan
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-zinc-500">
            Start free with a 7-day Pro trial. No credit card required.
            Upgrade when you&apos;re ready to unlock your full potential.
          </p>

          {/* Billing toggle */}
          <div className="mt-8 inline-flex items-center rounded-full border border-zinc-200 bg-white p-1 dark:border-zinc-700 dark:bg-zinc-900">
            <button
              onClick={() => setCycle("monthly")}
              className={cn(
                "rounded-full px-4 py-2 text-sm font-medium transition-colors",
                cycle === "monthly" ? "bg-orange-500 text-white" : "text-zinc-600 hover:text-zinc-900",
              )}
            >
              Monthly
            </button>
            <button
              onClick={() => setCycle("annual")}
              className={cn(
                "rounded-full px-4 py-2 text-sm font-medium transition-colors",
                cycle === "annual" ? "bg-orange-500 text-white" : "text-zinc-600 hover:text-zinc-900",
              )}
            >
              Annual
              <Badge variant="success" className="ml-2 text-[10px]">Save 25%</Badge>
            </button>
          </div>
        </div>

        {/* Tier cards */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {tiers.map((tier) => {
            const isPopular = tier.tier === "pro";
            const isTeam = tier.tier === "pro_team";
            const isEnterprise = tier.tier === "enterprise";
            const price = cycle === "annual"
              ? tier.pricing.annual_monthly_equivalent
              : tier.pricing.monthly_price_usd;

            return (
              <Card
                key={tier.tier}
                className={cn(
                  "relative flex flex-col",
                  isPopular && "border-orange-500 ring-2 ring-orange-200 dark:ring-orange-800",
                )}
              >
                {isPopular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <Badge variant="pro" className="flex items-center gap-1">
                      <Zap className="h-3 w-3" /> Most Popular
                    </Badge>
                  </div>
                )}

                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">{tier.display_name}</CardTitle>
                  <div className="mt-2">
                    {isEnterprise ? (
                      <p className="text-3xl font-bold">Custom</p>
                    ) : (
                      <p className="text-3xl font-bold">
                        {formatPrice(price)}
                        {price > 0 && (
                          <span className="text-sm font-normal text-zinc-500">
                            /{isTeam ? "seat/" : ""}mo
                          </span>
                        )}
                      </p>
                    )}
                    {tier.tier === "free" && (
                      <p className="mt-1 text-xs text-orange-500 font-medium">
                        Includes 7-day Pro trial
                      </p>
                    )}
                  </div>
                </CardHeader>

                <CardContent className="flex flex-1 flex-col">
                  <ul className="flex-1 space-y-2.5">
                    <FeatureRow label="Context shifts/mo" value={tier.shifts_per_month} />
                    <FeatureRow label="Brain queries/mo" value={tier.brain_queries_per_month} />
                    <FeatureRow label="Cockpit models" value={tier.cockpit_models_max || "View only"} />
                    <FeatureRow label="Smart dispatch" value={tier.smart_dispatch} />
                    <FeatureRow label="Chain steps" value={tier.chain_steps_max || false} />
                    <FeatureRow label="Consensus mode" value={tier.consensus_models_max > 0 || tier.consensus_models_max === -1} />
                    <FeatureRow label="Team seats" value={tier.team_seats_included || "Solo"} />
                    <FeatureRow label="Daily digest" value={tier.digest_daily} />
                    <FeatureRow label="Gap alerts" value={tier.knowledge_gap_alerts} />
                    <FeatureRow label="SSO/SAML" value={tier.sso_enabled} />
                    <FeatureRow label="API keys" value={tier.api_keys_max || false} />
                    <FeatureRow label="Support" value={tier.support_level} />
                  </ul>

                  <div className="mt-6">
                    {isEnterprise ? (
                      <Button variant="outline" className="w-full" asChild>
                        <a href="mailto:sales@kangarooshift.com">Contact Sales</a>
                      </Button>
                    ) : tier.tier === "free" ? (
                      <Button variant="secondary" className="w-full" asChild>
                        <Link href="/auth/signup">
                          Start Free Trial <ArrowRight className="ml-2 h-4 w-4" />
                        </Link>
                      </Button>
                    ) : (
                      <Button
                        className={cn("w-full", isPopular && "bg-orange-500 hover:bg-orange-600")}
                        asChild
                      >
                        <Link href="/auth/signup">
                          {isPopular ? "Get Started" : "Choose Plan"}
                          <ArrowRight className="ml-2 h-4 w-4" />
                        </Link>
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* FAQ teaser */}
        <div className="mt-16 text-center">
          <p className="text-zinc-500">
            Questions? Email us at{" "}
            <a href="mailto:support@kangarooshift.com" className="text-orange-500 hover:text-orange-600">
              support@kangarooshift.com
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
