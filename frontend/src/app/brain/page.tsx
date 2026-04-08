"use client";

import * as React from "react";
import {
  Brain,
  Search,
  FileText,
  AlertTriangle,
  Sparkles,
  BookOpen,
  ListChecks,
  HelpCircle,
  Loader2,
} from "lucide-react";

import { AppShell } from "@/components/layout/app-shell";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useSearch, useDigest, useGaps } from "@/hooks/use-api-query";

function SearchSection() {
  const [query, setQuery] = React.useState("");
  const { data: results, isFetching } = useSearch(query);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5 text-orange-500" />
          Knowledge Search
        </CardTitle>
        <CardDescription>
          Search across all your stored contexts and conversations.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-400" />
          <Input
            className="pl-10"
            placeholder="Search your brain..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {isFetching && (
            <Loader2 className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-orange-500" />
          )}
        </div>

        {results && results.length > 0 && (
          <div className="space-y-2">
            {results.map((result) => (
              <div
                key={result.id}
                className="rounded-lg border border-zinc-200 p-4 transition-colors hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
              >
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                    {result.content.slice(0, 100)}
                    {result.content.length > 100 && "..."}
                  </p>
                  <Badge variant="secondary" className="text-xs">
                    {(result.score * 100).toFixed(0)}%
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        )}

        {query.length >= 2 && results && results.length === 0 && !isFetching && (
          <div className="py-8 text-center">
            <Search className="mx-auto h-8 w-8 text-zinc-300" />
            <p className="mt-2 text-sm text-zinc-500">No results found for &quot;{query}&quot;</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DigestSection() {
  const { data: digest, isFetching, refetch } = useDigest();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-orange-500" />
              Daily Digest
            </CardTitle>
            <CardDescription>
              Key entities, decisions, and tasks from your recent contexts.
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            {isFetching ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {digest ? (
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-2">
              <h4 className="flex items-center gap-2 text-sm font-medium text-zinc-600 dark:text-zinc-400">
                <BookOpen className="h-4 w-4" />
                Entities ({digest.entities.length})
              </h4>
              <div className="flex flex-wrap gap-1.5">
                {digest.entities.slice(0, 10).map((entity) => (
                  <Badge key={entity} variant="secondary" className="text-xs">
                    {entity}
                  </Badge>
                ))}
                {digest.entities.length === 0 && (
                  <p className="text-xs text-zinc-400">None found</p>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="flex items-center gap-2 text-sm font-medium text-zinc-600 dark:text-zinc-400">
                <Sparkles className="h-4 w-4" />
                Decisions ({digest.decisions.length})
              </h4>
              <ul className="space-y-1">
                {digest.decisions.slice(0, 5).map((d, i) => (
                  <li key={i} className="text-xs text-zinc-700 dark:text-zinc-300">
                    {d}
                  </li>
                ))}
                {digest.decisions.length === 0 && (
                  <p className="text-xs text-zinc-400">None found</p>
                )}
              </ul>
            </div>

            <div className="space-y-2">
              <h4 className="flex items-center gap-2 text-sm font-medium text-zinc-600 dark:text-zinc-400">
                <ListChecks className="h-4 w-4" />
                Tasks ({digest.tasks.length})
              </h4>
              <ul className="space-y-1">
                {digest.tasks.slice(0, 5).map((t, i) => (
                  <li key={i} className="text-xs text-zinc-700 dark:text-zinc-300">
                    {t}
                  </li>
                ))}
                {digest.tasks.length === 0 && (
                  <p className="text-xs text-zinc-400">None found</p>
                )}
              </ul>
            </div>
          </div>
        ) : (
          <div className="py-8 text-center">
            <FileText className="mx-auto h-8 w-8 text-zinc-300" />
            <p className="mt-2 text-sm text-zinc-500">
              Click refresh to generate your daily digest.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function GapsSection() {
  const { data: gaps, isFetching, refetch } = useGaps();

  const totalGaps =
    (gaps?.undecided_topics.length ?? 0) +
    (gaps?.stalled_tasks.length ?? 0) +
    (gaps?.unclear_entities.length ?? 0);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-500" />
              Knowledge Gaps
              {totalGaps > 0 && (
                <Badge variant="warning" className="text-xs">
                  {totalGaps}
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Topics that need attention, stalled tasks, and unclear entities.
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            {isFetching ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <HelpCircle className="h-4 w-4" />
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {gaps && totalGaps > 0 ? (
          <div className="space-y-4">
            {gaps.undecided_topics.length > 0 && (
              <div>
                <h4 className="mb-2 text-sm font-medium text-amber-700 dark:text-amber-400">
                  Undecided Topics
                </h4>
                <div className="flex flex-wrap gap-2">
                  {gaps.undecided_topics.map((t, i) => (
                    <Badge key={i} variant="warning" className="text-xs">
                      {t}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {gaps.stalled_tasks.length > 0 && (
              <div>
                <h4 className="mb-2 text-sm font-medium text-red-700 dark:text-red-400">
                  Stalled Tasks
                </h4>
                <ul className="space-y-1">
                  {gaps.stalled_tasks.map((t, i) => (
                    <li key={i} className="text-xs text-zinc-700 dark:text-zinc-300">
                      {t}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {gaps.unclear_entities.length > 0 && (
              <div>
                <h4 className="mb-2 text-sm font-medium text-zinc-600 dark:text-zinc-400">
                  Unclear Entities
                </h4>
                <div className="flex flex-wrap gap-2">
                  {gaps.unclear_entities.map((e, i) => (
                    <Badge key={i} variant="outline" className="text-xs">
                      {e}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="py-8 text-center">
            <AlertTriangle className="mx-auto h-8 w-8 text-zinc-300" />
            <p className="mt-2 text-sm text-zinc-500">
              {gaps ? "No knowledge gaps detected!" : "Click refresh to scan for gaps."}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function BrainContent() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">
          Brain Explorer
        </h1>
        <p className="mt-1 text-zinc-500">
          Your AI second brain — search, digest, and identify knowledge gaps.
        </p>
      </div>

      <SearchSection />
      <DigestSection />
      <GapsSection />
    </div>
  );
}

export default function BrainPage() {
  return (
    <AppShell>
      <BrainContent />
    </AppShell>
  );
}
