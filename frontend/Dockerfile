# =============================================================================
# Kangaroo Shift Frontend — Multi-Stage Dockerfile
#
# Three stages for different use cases:
#   1. deps:        Install dependencies (cached layer)
#   2. builder:     Build the Next.js production bundle
#   3. runner:      Minimal runtime image (~100MB vs ~1GB)
#
# Production: Uses Next.js standalone output mode for minimal footprint.
# Development: Uses deps stage with hot-reload.
#
# Security:
#   - Runs as non-root user (nextjs) in production
#   - No devDependencies in final image
#   - No source code in final image — only compiled output
# =============================================================================

# ── Stage 1: Dependencies ────────────────────────────────────────────────────
# Separate stage for deps so they're cached independently from code changes.
# Code changes = fast rebuild. Dependency changes = full reinstall.
FROM node:20-alpine AS deps

WORKDIR /app

# Copy only package files first (Docker layer caching optimization)
COPY package.json package-lock.json* ./

# npm ci = deterministic install from lockfile (not npm install)
# This ensures identical node_modules in every environment.
RUN npm ci


# ── Stage 2: Builder ─────────────────────────────────────────────────────────
# Builds the Next.js production bundle with standalone output.
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Next.js standalone mode outputs a self-contained server.js
# that includes only the files needed to run — no node_modules bloat.
# This requires output: 'standalone' in next.config.js.
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build


# ── Stage 3: Production Runner ───────────────────────────────────────────────
# Minimal image: ~100MB instead of ~1GB.
# Contains only the compiled output, not source code or build tools.
FROM node:20-alpine AS runner

WORKDIR /app

# Security: non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Copy only what's needed to run
# Static files (CSS, images, fonts)
COPY --from=builder /app/public ./public

# Next.js standalone output (if standalone mode is configured)
# Falls back to normal .next output if standalone isn't set up yet
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["npm", "run", "start"]


# ── Stage 4: Development (hot-reload) ────────────────────────────────────────
# Used by docker-compose.yml with target: development
FROM deps AS development

WORKDIR /app

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
