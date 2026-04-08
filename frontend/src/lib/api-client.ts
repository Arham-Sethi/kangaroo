/**
 * API client — typed fetch wrapper for the Kangaroo Shift backend.
 *
 * All requests go through `apiFetch()` which handles:
 *   - Base URL resolution from env
 *   - JWT token injection
 *   - JSON serialization/deserialization
 *   - Error normalization (ApiError)
 *   - 401 → automatic redirect to login
 */

import type { ApiError } from "@/types/api";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

// ── Error Class ─────────────────────────────────────────────────────────────

export class ApiRequestError extends Error {
  readonly status: number;
  readonly body: ApiError;

  constructor(status: number, body: ApiError) {
    super(body.message || `API error ${status}`);
    this.name = "ApiRequestError";
    this.status = status;
    this.body = body;
  }

  get isPaywall(): boolean {
    return this.status === 402;
  }

  get isRateLimit(): boolean {
    return this.status === 429;
  }

  get isUnauthorized(): boolean {
    return this.status === 401;
  }
}

// ── Token Store ─────────────────────────────────────────────────────────────

let accessToken: string | null = null;

export function setAccessToken(token: string | null): void {
  accessToken = token;
  if (token) {
    if (typeof window !== "undefined") {
      localStorage.setItem("ks_token", token);
    }
  } else {
    if (typeof window !== "undefined") {
      localStorage.removeItem("ks_token");
    }
  }
}

export function getAccessToken(): string | null {
  if (accessToken) return accessToken;
  if (typeof window !== "undefined") {
    accessToken = localStorage.getItem("ks_token");
  }
  return accessToken;
}

// ── Core Fetch ──────────────────────────────────────────────────────────────

interface FetchOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  headers?: Record<string, string>;
  /** Skip auth token injection (for public endpoints). */
  public?: boolean;
}

export async function apiFetch<T>(
  path: string,
  options: FetchOptions = {},
): Promise<T> {
  const { method = "GET", body, headers: extraHeaders, public: isPublic } = options;
  const url = `${API_BASE_URL}${path}`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };

  if (!isPublic) {
    const token = getAccessToken();
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
  }

  const res = await fetch(url, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  // Handle empty responses (204 No Content)
  if (res.status === 204) {
    return undefined as T;
  }

  const data = await res.json();

  if (!res.ok) {
    const apiError: ApiError = {
      error: data.error ?? data.detail?.error ?? "unknown_error",
      message:
        data.message ?? data.detail?.message ?? `Request failed (${res.status})`,
      detail: data.detail ?? data,
    };

    // Auto-redirect on 401
    if (res.status === 401 && typeof window !== "undefined") {
      setAccessToken(null);
      window.location.href = "/auth/login";
    }

    throw new ApiRequestError(res.status, apiError);
  }

  return data as T;
}

// ── Convenience Methods ─────────────────────────────────────────────────────

// ── File Upload ────────────────────────────────────────────────────────────

export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const formData = new FormData();
  formData.append("file", file);

  const headers: Record<string, string> = {};
  const token = getAccessToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(url, {
    method: "POST",
    headers,
    body: formData,
  });

  if (res.status === 204) {
    return undefined as T;
  }

  const data = await res.json();

  if (!res.ok) {
    const apiError: ApiError = {
      error: data.error ?? "upload_error",
      message: data.message ?? data.detail ?? `Upload failed (${res.status})`,
      detail: data.detail ?? data,
    };
    throw new ApiRequestError(res.status, apiError);
  }

  return data as T;
}

// ── Convenience Methods ─────────────────────────────────────────────────────

export const api = {
  get: <T>(path: string, opts?: Omit<FetchOptions, "method">) =>
    apiFetch<T>(path, { ...opts, method: "GET" }),

  post: <T>(path: string, body?: unknown, opts?: Omit<FetchOptions, "method" | "body">) =>
    apiFetch<T>(path, { ...opts, method: "POST", body }),

  put: <T>(path: string, body?: unknown, opts?: Omit<FetchOptions, "method" | "body">) =>
    apiFetch<T>(path, { ...opts, method: "PUT", body }),

  patch: <T>(path: string, body?: unknown, opts?: Omit<FetchOptions, "method" | "body">) =>
    apiFetch<T>(path, { ...opts, method: "PATCH", body }),

  delete: <T>(path: string, opts?: Omit<FetchOptions, "method">) =>
    apiFetch<T>(path, { ...opts, method: "DELETE" }),
} as const;
