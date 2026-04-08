/**
 * Auth store — Zustand store for authentication state.
 *
 * Manages:
 *   - Current user profile
 *   - Login / logout flow
 *   - Token lifecycle
 *   - Trial status
 */

import { create } from "zustand";

import { api, setAccessToken, getAccessToken } from "@/lib/api-client";
import type {
  AuthTokens,
  UserProfile,
  SubscriptionResponse,
  TrialStatusResponse,
} from "@/types/api";

interface AuthState {
  /** Current user or null if logged out. */
  user: UserProfile | null;
  /** Subscription info including trial status. */
  subscription: SubscriptionResponse | null;
  /** Trial status details. */
  trial: TrialStatusResponse | null;
  /** Whether initial auth check has completed. */
  initialized: boolean;
  /** Whether a login/register request is in flight. */
  loading: boolean;
  /** Last auth error message. */
  error: string | null;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
  initialize: () => Promise<void>;
  refreshSubscription: () => Promise<void>;
  startTrial: () => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  subscription: null,
  trial: null,
  initialized: false,
  loading: false,
  error: null,

  login: async (email: string, password: string) => {
    set({ loading: true, error: null });
    try {
      const tokens = await api.post<AuthTokens>("/auth/login", {
        email,
        password,
      });
      setAccessToken(tokens.access_token);

      const user = await api.get<UserProfile>("/auth/me");
      set({ user, loading: false });

      // Fetch subscription in background
      get().refreshSubscription();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Login failed";
      set({ loading: false, error: message });
      throw err;
    }
  },

  register: async (email: string, password: string, name: string) => {
    set({ loading: true, error: null });
    try {
      const tokens = await api.post<AuthTokens>("/auth/register", {
        email,
        password,
        name,
      });
      setAccessToken(tokens.access_token);

      const user = await api.get<UserProfile>("/auth/me");
      set({ user, loading: false });

      // Auto-start trial for new signups
      get().startTrial();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Registration failed";
      set({ loading: false, error: message });
      throw err;
    }
  },

  logout: () => {
    setAccessToken(null);
    set({
      user: null,
      subscription: null,
      trial: null,
      error: null,
    });
    if (typeof window !== "undefined") {
      window.location.href = "/auth/login";
    }
  },

  initialize: async () => {
    const token = getAccessToken();
    if (!token) {
      set({ initialized: true });
      return;
    }

    try {
      const user = await api.get<UserProfile>("/auth/me");
      set({ user, initialized: true });
      get().refreshSubscription();
    } catch {
      // Token expired or invalid
      setAccessToken(null);
      set({ initialized: true });
    }
  },

  refreshSubscription: async () => {
    try {
      const [subscription, trial] = await Promise.all([
        api.get<SubscriptionResponse>("/billing/current"),
        api.get<TrialStatusResponse>("/billing/trial/status"),
      ]);
      set({ subscription, trial });
    } catch {
      // Non-critical — don't break the app
    }
  },

  startTrial: async () => {
    try {
      const trial = await api.post<TrialStatusResponse>(
        "/billing/trial/start",
      );
      set({ trial });
      get().refreshSubscription();
    } catch {
      // Non-critical
    }
  },

  clearError: () => set({ error: null }),
}));
