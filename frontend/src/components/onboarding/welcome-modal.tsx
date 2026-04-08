"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  ArrowRightLeft,
  Brain,
  Monitor,
  Rocket,
  Sparkles,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/stores/auth-store";

// ── Step definitions ───────────────────────────────────────────────────────

interface OnboardingStep {
  readonly icon: React.ElementType;
  readonly title: string;
  readonly description: string;
  readonly action: string;
  readonly route: string;
}

const STEPS: readonly OnboardingStep[] = [
  {
    icon: ArrowRightLeft,
    title: "Shift your first context",
    description:
      "Paste a conversation from ChatGPT, Claude, or Gemini and instantly transfer it to another model.",
    action: "Try Context Shift",
    route: "/shift",
  },
  {
    icon: Monitor,
    title: "Open the Multi-Model Cockpit",
    description:
      "Ask a question and get answers from multiple AI models simultaneously. Compare, iterate, find the best.",
    action: "Open Cockpit",
    route: "/cockpit",
  },
  {
    icon: Brain,
    title: "Explore your AI Brain",
    description:
      "Search across all your conversations, get daily digests, and discover knowledge gaps automatically.",
    action: "View Dashboard",
    route: "/dashboard",
  },
] as const;

// ── Onboarding hook ────────────────────────────────────────────────────────

const ONBOARDING_KEY = "ks_onboarding_completed";

export function useOnboarding() {
  const [showOnboarding, setShowOnboarding] = React.useState(false);
  const { user } = useAuthStore();

  React.useEffect(() => {
    if (!user) return;
    const completed = localStorage.getItem(ONBOARDING_KEY);
    if (!completed) {
      setShowOnboarding(true);
    }
  }, [user]);

  const completeOnboarding = React.useCallback(() => {
    localStorage.setItem(ONBOARDING_KEY, "true");
    setShowOnboarding(false);
  }, []);

  return { showOnboarding, completeOnboarding } as const;
}

// ── Modal Component ────────────────────────────────────────────────────────

export function WelcomeModal({
  onComplete,
}: {
  readonly onComplete: () => void;
}) {
  const router = useRouter();
  const { user } = useAuthStore();
  const [step, setStep] = React.useState(0);

  const isWelcome = step === 0;
  const isSteps = step >= 1 && step <= STEPS.length;
  const isDone = step > STEPS.length;
  const currentStep = isSteps ? STEPS[step - 1] : null;

  const handleNext = () => {
    if (isDone) {
      onComplete();
      return;
    }
    setStep((prev) => prev + 1);
  };

  const handleGoToFeature = (route: string) => {
    onComplete();
    router.push(route);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in">
      <div className="relative mx-4 w-full max-w-lg rounded-2xl bg-white p-8 shadow-2xl dark:bg-zinc-900">
        {/* Close */}
        <button
          onClick={onComplete}
          className="absolute right-4 top-4 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-300"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Welcome Screen */}
        {isWelcome && (
          <div className="text-center">
            <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-orange-500 to-amber-500 text-white">
              <Rocket className="h-8 w-8" />
            </div>
            <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              Welcome to Kangaroo Shift{user?.name ? `, ${user.name.split(" ")[0]}` : ""}!
            </h2>
            <p className="mt-3 text-zinc-500">
              Your 7-day Pro trial is active. Let&apos;s explore the key features
              that will supercharge your AI workflow.
            </p>
            <Button className="mt-6 w-full" size="lg" onClick={handleNext}>
              Let&apos;s Go
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        )}

        {/* Feature Steps */}
        {isSteps && currentStep && (
          <div className="text-center">
            {/* Progress dots */}
            <div className="mb-6 flex justify-center gap-2">
              {STEPS.map((_, i) => (
                <div
                  key={i}
                  className={`h-2 w-2 rounded-full transition-colors ${
                    i < step ? "bg-orange-500" : i === step - 1 ? "bg-orange-500" : "bg-zinc-200 dark:bg-zinc-700"
                  }`}
                />
              ))}
            </div>

            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-300">
              <currentStep.icon className="h-7 w-7" />
            </div>

            <h3 className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
              {currentStep.title}
            </h3>
            <p className="mt-2 text-sm text-zinc-500">{currentStep.description}</p>

            <div className="mt-6 flex gap-3">
              <Button
                variant="outline"
                className="flex-1"
                onClick={handleNext}
              >
                {step < STEPS.length ? "Next" : "Finish"}
              </Button>
              <Button
                className="flex-1"
                onClick={() => handleGoToFeature(currentStep.route)}
              >
                {currentStep.action}
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        {/* Done */}
        {isDone && (
          <div className="text-center">
            <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-600 dark:bg-emerald-900 dark:text-emerald-300">
              <Sparkles className="h-8 w-8" />
            </div>
            <h2 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
              You&apos;re all set!
            </h2>
            <p className="mt-3 text-zinc-500">
              Your AI Second Brain is ready. Start shifting contexts, exploring
              the cockpit, or let the brain learn from your conversations.
            </p>
            <Button className="mt-6 w-full" size="lg" onClick={onComplete}>
              Go to Dashboard
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
