"use client";

import * as React from "react";
import Link from "next/link";
import {
  User,
  Key,
  CreditCard,
  Users,
  Copy,
  Check,
  Eye,
  EyeOff,
  Trash2,
  Loader2,
  Plus,
  Shield,
  AlertCircle,
  UserPlus,
  Crown,
} from "lucide-react";

import { AppShell } from "@/components/layout/app-shell";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useAuthStore } from "@/stores/auth-store";
import {
  useUpdateProfile,
  useApiKeys,
  useCreateApiKey,
  useDeleteApiKey,
  useTeams,
  useTeamMembers,
  useCreateTeam,
  useInviteTeamMember,
  useCreatePortalSession,
} from "@/hooks/use-api-query";

// ── Toast-like feedback ────────────────────────────────────────────────────

interface FeedbackState {
  readonly message: string;
  readonly type: "success" | "error";
}

function useFeedback() {
  const [feedback, setFeedback] = React.useState<FeedbackState | null>(null);
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = React.useCallback((message: string, type: "success" | "error" = "success") => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setFeedback({ message, type });
    timerRef.current = setTimeout(() => setFeedback(null), 4000);
  }, []);

  const clear = React.useCallback(() => setFeedback(null), []);

  return { feedback, show, clear } as const;
}

function FeedbackBanner({ feedback }: { readonly feedback: FeedbackState | null }) {
  if (!feedback) return null;
  const isError = feedback.type === "error";
  return (
    <div
      className={`rounded-lg border px-4 py-3 text-sm flex items-center gap-2 animate-fade-in ${
        isError
          ? "border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
          : "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
      }`}
    >
      {isError ? <AlertCircle className="h-4 w-4 shrink-0" /> : <Check className="h-4 w-4 shrink-0" />}
      {feedback.message}
    </div>
  );
}

// ── Profile Section ────────────────────────────────────────────────────────

function ProfileSection() {
  const { user } = useAuthStore();
  const [name, setName] = React.useState(user?.name ?? "");
  const [email] = React.useState(user?.email ?? "");
  const updateProfile = useUpdateProfile();
  const { feedback, show } = useFeedback();

  const handleSave = () => {
    if (!name.trim()) return;
    updateProfile.mutate(
      { display_name: name.trim() },
      {
        onSuccess: () => show("Profile updated successfully."),
        onError: (err) =>
          show(err instanceof Error ? err.message : "Failed to update profile.", "error"),
      },
    );
  };

  const isDirty = name.trim() !== (user?.name ?? "");

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="h-5 w-5 text-orange-500" />
          Profile
        </CardTitle>
        <CardDescription>Manage your personal information.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FeedbackBanner feedback={feedback} />
        <div className="space-y-2">
          <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Name</label>
          <Input value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Email</label>
          <Input value={email} disabled className="opacity-60" />
        </div>
        <Button size="sm" onClick={handleSave} disabled={!isDirty || updateProfile.isPending}>
          {updateProfile.isPending ? (
            <><Loader2 className="mr-2 h-3 w-3 animate-spin" /> Saving...</>
          ) : (
            "Save Changes"
          )}
        </Button>
      </CardContent>
    </Card>
  );
}

// ── API Keys Section ───────────────────────────────────────────────────────

function ApiKeysSection() {
  const { data: keys, isLoading } = useApiKeys();
  const createKey = useCreateApiKey();
  const deleteKey = useDeleteApiKey();
  const [newKeyName, setNewKeyName] = React.useState("");
  const [revealedKey, setRevealedKey] = React.useState<string | null>(null);
  const [copiedId, setCopiedId] = React.useState<string | null>(null);
  const { feedback, show } = useFeedback();

  const handleCreate = () => {
    if (!newKeyName.trim()) return;
    createKey.mutate(
      { name: newKeyName.trim() },
      {
        onSuccess: (data) => {
          setNewKeyName("");
          if (data && typeof data === "object" && "key" in data) {
            setRevealedKey((data as { key: string }).key);
            show("API key created. Copy it now \u2014 it won't be shown again.");
          } else {
            show("API key created.");
          }
        },
        onError: (err) =>
          show(err instanceof Error ? err.message : "Failed to create key.", "error"),
      },
    );
  };

  const handleDelete = (keyId: string) => {
    deleteKey.mutate(keyId, {
      onSuccess: () => show("API key revoked."),
      onError: (err) =>
        show(err instanceof Error ? err.message : "Failed to revoke key.", "error"),
    });
  };

  const handleCopy = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // clipboard denied
    }
  };

  const formatDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return iso;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Key className="h-5 w-5 text-orange-500" />
          API Keys
        </CardTitle>
        <CardDescription>Manage API keys for programmatic access to the Kangaroo Shift API.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FeedbackBanner feedback={feedback} />

        {/* New key just created — show full key */}
        {revealedKey && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950">
            <p className="text-sm font-medium text-amber-800 dark:text-amber-200 mb-2">
              Copy your API key now. It will not be shown again.
            </p>
            <div className="flex items-center gap-2">
              <code className="flex-1 rounded bg-amber-100 px-3 py-2 text-xs font-mono text-amber-900 dark:bg-amber-900 dark:text-amber-100 break-all">
                {revealedKey}
              </code>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => handleCopy(revealedKey, "__new__")}
                className="shrink-0"
              >
                {copiedId === "__new__" ? (
                  <Check className="h-4 w-4 text-emerald-600" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </div>
            <Button variant="ghost" size="sm" className="mt-2 text-xs" onClick={() => setRevealedKey(null)}>
              Dismiss
            </Button>
          </div>
        )}

        {/* Create key form */}
        <div className="flex gap-2">
          <Input
            placeholder="Key name (e.g., Production)"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
          />
          <Button size="sm" disabled={!newKeyName.trim() || createKey.isPending} onClick={handleCreate}>
            {createKey.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4 mr-1" />}
            Create
          </Button>
        </div>

        {/* Key list */}
        {isLoading ? (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-5 w-5 animate-spin text-zinc-400" />
          </div>
        ) : keys && keys.length > 0 ? (
          <div className="space-y-2">
            {keys.map((key) => (
              <div
                key={key.id}
                className="flex items-center justify-between rounded-lg border p-3 dark:border-zinc-700"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium truncate">{key.name}</p>
                    {key.is_active ? (
                      <Badge variant="outline" className="text-emerald-600 border-emerald-200 text-[10px]">
                        Active
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="text-zinc-400 border-zinc-300 text-[10px]">
                        Revoked
                      </Badge>
                    )}
                  </div>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    ks_{key.key_prefix}... &middot; Created {formatDate(key.created_at)}
                    {key.last_used ? ` \u00b7 Last used ${formatDate(key.last_used)}` : ""}
                  </p>
                </div>
                {key.is_active && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-red-500 hover:text-red-600 shrink-0"
                    onClick={() => handleDelete(key.id)}
                    disabled={deleteKey.isPending}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-zinc-500 py-2">No API keys yet. Create one to get started.</p>
        )}
      </CardContent>
    </Card>
  );
}

// ── Billing Section ────────────────────────────────────────────────────────

function BillingSection() {
  const { subscription } = useAuthStore();
  const portalSession = useCreatePortalSession();

  const handleManageBilling = () => {
    portalSession.mutate(undefined, {
      onSuccess: (data) => {
        if (data?.portal_url) {
          window.location.href = data.portal_url;
        }
      },
    });
  };

  const isPaid = subscription?.tier !== "free";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CreditCard className="h-5 w-5 text-orange-500" />
          Subscription
        </CardTitle>
        <CardDescription>Manage your plan and billing.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between rounded-lg border p-4 dark:border-zinc-700">
          <div>
            <p className="font-medium text-zinc-900 dark:text-zinc-100">
              {subscription?.display_name ?? "Free"} Plan
            </p>
            <p className="text-sm text-zinc-500">
              {subscription?.is_trial_active
                ? `Pro trial: ${Math.ceil(subscription.trial_days_remaining)} days remaining`
                : isPaid
                  ? "Full access to all features"
                  : "Limited features"}
            </p>
          </div>
          <div className="flex gap-2">
            {isPaid && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleManageBilling}
                disabled={portalSession.isPending}
              >
                {portalSession.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  "Manage Billing"
                )}
              </Button>
            )}
            <Link href="/pricing">
              <Button variant={isPaid ? "ghost" : "default"} size="sm">
                {isPaid ? "Change Plan" : "Upgrade"}
              </Button>
            </Link>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ── Team Section ───────────────────────────────────────────────────────────

const ROLE_LABELS: Record<string, { label: string; color: string }> = {
  owner: { label: "Owner", color: "text-amber-600 border-amber-200" },
  admin: { label: "Admin", color: "text-blue-600 border-blue-200" },
  member: { label: "Member", color: "text-zinc-600 border-zinc-200" },
  viewer: { label: "Viewer", color: "text-zinc-400 border-zinc-200" },
};

function TeamSection() {
  const { subscription } = useAuthStore();
  const effectiveTier = subscription?.effective_tier ?? "free";
  const hasTeamFeature =
    effectiveTier === "pro" || effectiveTier === "pro_team" || effectiveTier === "enterprise";

  const { data: teams, isLoading: teamsLoading } = useTeams();
  const createTeam = useCreateTeam();
  const inviteMember = useInviteTeamMember();
  const { feedback, show } = useFeedback();

  const [newTeamName, setNewTeamName] = React.useState("");
  const [inviteEmail, setInviteEmail] = React.useState("");
  const [showCreateTeam, setShowCreateTeam] = React.useState(false);

  // Use first team as active team (MVP: single team)
  const activeTeam = teams && teams.length > 0 ? teams[0] : null;
  const { data: members, isLoading: membersLoading } = useTeamMembers(activeTeam?.id ?? "");

  const handleCreateTeam = () => {
    if (!newTeamName.trim()) return;
    createTeam.mutate(
      { name: newTeamName.trim() },
      {
        onSuccess: () => {
          setNewTeamName("");
          setShowCreateTeam(false);
          show("Team created successfully.");
        },
        onError: (err) =>
          show(err instanceof Error ? err.message : "Failed to create team.", "error"),
      },
    );
  };

  const handleInvite = () => {
    if (!inviteEmail.trim() || !activeTeam) return;
    inviteMember.mutate(
      { teamId: activeTeam.id, email: inviteEmail.trim(), role: "member" },
      {
        onSuccess: () => {
          setInviteEmail("");
          show(`Invitation sent to ${inviteEmail.trim()}.`);
        },
        onError: (err) =>
          show(err instanceof Error ? err.message : "Failed to send invitation.", "error"),
      },
    );
  };

  if (!hasTeamFeature) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5 text-orange-500" />
            Team
          </CardTitle>
          <CardDescription>Manage team members and roles.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6">
            <Users className="mx-auto h-8 w-8 text-zinc-300" />
            <p className="mt-2 text-sm text-zinc-500">
              Team features require a Pro plan or higher.
            </p>
            <Link href="/pricing">
              <Button size="sm" className="mt-3">Upgrade</Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5 text-orange-500" />
          Team
        </CardTitle>
        <CardDescription>
          {activeTeam
            ? `${activeTeam.name} \u00b7 ${activeTeam.member_count}/${activeTeam.max_members} members`
            : "Create a team to collaborate."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FeedbackBanner feedback={feedback} />

        {teamsLoading ? (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-5 w-5 animate-spin text-zinc-400" />
          </div>
        ) : !activeTeam ? (
          /* No team yet — prompt to create */
          showCreateTeam ? (
            <div className="space-y-3">
              <Input
                placeholder="Team name (e.g., Engineering)"
                value={newTeamName}
                onChange={(e) => setNewTeamName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreateTeam()}
              />
              <div className="flex gap-2">
                <Button size="sm" disabled={!newTeamName.trim() || createTeam.isPending} onClick={handleCreateTeam}>
                  {createTeam.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-1" /> : null}
                  Create Team
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setShowCreateTeam(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-center py-4">
              <Users className="mx-auto h-8 w-8 text-zinc-300" />
              <p className="mt-2 text-sm text-zinc-500">No team yet.</p>
              <Button size="sm" className="mt-3" onClick={() => setShowCreateTeam(true)}>
                <Plus className="h-4 w-4 mr-1" /> Create Team
              </Button>
            </div>
          )
        ) : (
          /* Has team — show members + invite */
          <>
            <div className="flex gap-2">
              <Input
                placeholder="Invite by email"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleInvite()}
              />
              <Button size="sm" disabled={!inviteEmail.trim() || inviteMember.isPending} onClick={handleInvite}>
                {inviteMember.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <><UserPlus className="h-4 w-4 mr-1" /> Invite</>
                )}
              </Button>
            </div>

            {membersLoading ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="h-5 w-5 animate-spin text-zinc-400" />
              </div>
            ) : members && members.length > 0 ? (
              <div className="space-y-2">
                {members.map((member) => {
                  const roleInfo = ROLE_LABELS[member.role] ?? ROLE_LABELS.member;
                  return (
                    <div
                      key={member.user_id}
                      className="flex items-center justify-between rounded-lg border p-3 dark:border-zinc-700"
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="text-sm font-medium truncate">
                            {member.display_name || member.email}
                          </p>
                          {member.role === "owner" && <Crown className="h-3.5 w-3.5 text-amber-500" />}
                        </div>
                        <p className="text-xs text-zinc-500 truncate">{member.email}</p>
                      </div>
                      <Badge variant="outline" className={`text-[10px] ${roleInfo.color}`}>
                        {roleInfo.label}
                      </Badge>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-sm text-zinc-500">No team members yet. Invite people to collaborate.</p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// ── Page Layout ────────────────────────────────────────────────────────────

function SettingsContent() {
  return (
    <div className="mx-auto max-w-3xl space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-100">Settings</h1>
        <p className="mt-1 text-zinc-500">Manage your account, keys, team, and billing.</p>
      </div>

      <ProfileSection />
      <ApiKeysSection />
      <BillingSection />
      <TeamSection />
    </div>
  );
}

export default function SettingsPage() {
  return (
    <AppShell>
      <SettingsContent />
    </AppShell>
  );
}
