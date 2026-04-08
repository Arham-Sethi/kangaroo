import Link from "next/link";
import {
  ArrowRightLeft,
  Brain,
  Monitor,
  Zap,
  Shield,
  ArrowRight,
  Check,
} from "lucide-react";

const FEATURES = [
  {
    icon: ArrowRightLeft,
    title: "Context Shift",
    description:
      "Transfer conversations between GPT-4, Claude, Gemini and more without losing context or nuance.",
  },
  {
    icon: Brain,
    title: "AI Second Brain",
    description:
      "Automatic knowledge extraction, semantic search, daily digests, and gap detection across all your AI interactions.",
  },
  {
    icon: Monitor,
    title: "Multi-Model Cockpit",
    description:
      "Query multiple AI models simultaneously, compare responses side-by-side, and find consensus.",
  },
  {
    icon: Zap,
    title: "Smart Dispatch",
    description:
      "Automatically route tasks to the best model based on capabilities, cost, and your preferences.",
  },
  {
    icon: Shield,
    title: "Enterprise Security",
    description:
      "AES-256 encryption, audit logs, SSO/SAML, role-based access control, and SOC 2 compliance.",
  },
] as const;

const TRIAL_PERKS = [
  "Unlimited context shifts",
  "Multi-model cockpit access",
  "Smart dispatch & chains",
  "Daily brain digest",
  "No credit card required",
] as const;

export default function HomePage() {
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950">
      {/* Nav */}
      <header className="border-b border-zinc-100 dark:border-zinc-800">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-orange-500 text-white font-bold">
              K
            </div>
            <span className="text-xl font-bold text-zinc-900 dark:text-zinc-100">
              Kangaroo Shift
            </span>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/pricing"
              className="text-sm font-medium text-zinc-600 hover:text-zinc-900 dark:text-zinc-400"
            >
              Pricing
            </Link>
            <Link
              href="/auth/login"
              className="text-sm font-medium text-zinc-600 hover:text-zinc-900 dark:text-zinc-400"
            >
              Sign In
            </Link>
            <Link
              href="/auth/signup"
              className="inline-flex items-center rounded-lg bg-orange-500 px-4 py-2 text-sm font-medium text-white hover:bg-orange-600 transition-colors"
            >
              Start Free Trial
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="px-6 py-24 text-center">
        <div className="mx-auto max-w-3xl">
          <div className="mb-6 inline-flex items-center rounded-full border border-orange-200 bg-orange-50 px-4 py-1.5 text-sm font-medium text-orange-600 dark:border-orange-800 dark:bg-orange-950">
            <Zap className="mr-1.5 h-4 w-4" />
            7-day Pro trial — free, no card required
          </div>
          <h1 className="text-5xl font-bold leading-tight text-zinc-900 dark:text-zinc-100 sm:text-6xl">
            Your AI
            <span className="bg-gradient-to-r from-orange-500 to-amber-500 bg-clip-text text-transparent">
              {" "}Second Brain
            </span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-zinc-500">
            Seamlessly switch between ChatGPT, Claude, and Gemini without losing
            context. Kangaroo Shift remembers everything so you can focus on
            what matters.
          </p>
          <div className="mt-10 flex flex-col items-center gap-4 sm:flex-row sm:justify-center">
            <Link
              href="/auth/signup"
              className="inline-flex items-center rounded-lg bg-orange-500 px-6 py-3 text-base font-semibold text-white shadow-lg shadow-orange-500/25 hover:bg-orange-600 transition-all hover:shadow-xl"
            >
              Start Free Trial
              <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
            <Link
              href="/pricing"
              className="inline-flex items-center rounded-lg border border-zinc-300 px-6 py-3 text-base font-medium text-zinc-700 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-300"
            >
              View Pricing
            </Link>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="border-t border-zinc-100 bg-zinc-50 px-6 py-20 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mx-auto max-w-6xl">
          <h2 className="text-center text-3xl font-bold text-zinc-900 dark:text-zinc-100">
            Everything you need to master AI
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-center text-zinc-500">
            One platform to manage all your AI interactions, knowledge, and team workflows.
          </p>
          <div className="mt-12 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((feature) => (
              <div
                key={feature.title}
                className="rounded-xl border border-zinc-200 bg-white p-6 transition-shadow hover:shadow-md dark:border-zinc-700 dark:bg-zinc-950"
              >
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-300">
                  <feature.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
                  {feature.title}
                </h3>
                <p className="mt-2 text-sm text-zinc-500">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="px-6 py-20">
        <div className="mx-auto max-w-3xl rounded-2xl bg-gradient-to-r from-orange-500 to-amber-500 p-10 text-center text-white shadow-xl">
          <h2 className="text-3xl font-bold">Start your free Pro trial today</h2>
          <p className="mt-3 text-lg opacity-90">
            Experience the full power of Kangaroo Shift for 7 days.
          </p>
          <ul className="mx-auto mt-6 flex max-w-md flex-wrap justify-center gap-x-6 gap-y-2">
            {TRIAL_PERKS.map((perk) => (
              <li key={perk} className="flex items-center gap-1.5 text-sm">
                <Check className="h-4 w-4" /> {perk}
              </li>
            ))}
          </ul>
          <Link
            href="/auth/signup"
            className="mt-8 inline-flex items-center rounded-lg bg-white px-6 py-3 text-base font-semibold text-orange-600 shadow-lg hover:bg-orange-50 transition-colors"
          >
            Get Started Free
            <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-zinc-100 px-6 py-8 dark:border-zinc-800">
        <div className="mx-auto flex max-w-6xl items-center justify-between text-sm text-zinc-500">
          <p>&copy; 2026 Kangaroo Shift. All rights reserved.</p>
          <div className="flex gap-6">
            <Link href="/pricing" className="hover:text-zinc-900 dark:hover:text-zinc-300">Pricing</Link>
            <a href="mailto:support@kangarooshift.com" className="hover:text-zinc-900 dark:hover:text-zinc-300">Support</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
