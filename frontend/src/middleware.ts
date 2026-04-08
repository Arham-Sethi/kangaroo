import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/** Public routes that don't require authentication. */
const PUBLIC_ROUTES = new Set([
  "/",
  "/auth/login",
  "/auth/signup",
  "/pricing",
]);

/** Route prefix check for public paths. */
function isPublicRoute(pathname: string): boolean {
  if (PUBLIC_ROUTES.has(pathname)) return true;
  // Static assets and API routes
  if (pathname.startsWith("/_next")) return true;
  if (pathname.startsWith("/api")) return true;
  if (pathname.includes(".")) return true; // Static files
  return false;
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (isPublicRoute(pathname)) {
    return NextResponse.next();
  }

  // Check for auth token in cookie or localStorage isn't available in middleware,
  // so we check for the ks_token cookie (set by client-side JS on login)
  const token = request.cookies.get("ks_token")?.value;

  if (!token) {
    const loginUrl = new URL("/auth/login", request.url);
    loginUrl.searchParams.set("redirect", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
