# Kangaroo Shift — Production Deployment Guide

## Pre-Launch Checklist

### 1. Infrastructure
- [ ] Provision PostgreSQL 16 with pgvector extension (Supabase / RDS / Cloud SQL)
- [ ] Provision Redis 7 (Upstash / ElastiCache / Memorystore)
- [ ] Provision Meilisearch (Meilisearch Cloud or self-hosted)
- [ ] Set up domain name (e.g., kangarooshift.com)
- [ ] Configure DNS A/CNAME records
- [ ] Obtain TLS certificate (Let's Encrypt or AWS ACM)

### 2. Stripe Setup
- [ ] Create Stripe account at https://dashboard.stripe.com
- [ ] Create products and prices:
  - **Pro Monthly** — $16/month (set price ID in `STRIPE_PRICE_PRO_MONTHLY`)
  - **Pro Annual** — $144/year ($12/mo) (set price ID in `STRIPE_PRICE_PRO_ANNUAL`)
  - **Team Monthly** — $14/seat/month (set price ID in `STRIPE_PRICE_TEAM_MONTHLY`)
  - **Team Annual** — $108/seat/year ($9/mo) (set price ID in `STRIPE_PRICE_TEAM_ANNUAL`)
- [ ] Set up Stripe webhook endpoint: `https://your-domain.com/api/v1/billing/webhook`
  - Events: `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.payment_failed`
- [ ] Copy webhook signing secret to `STRIPE_WEBHOOK_SECRET`
- [ ] Configure Customer Portal at https://dashboard.stripe.com/settings/billing/portal

### 3. LLM API Keys
- [ ] OpenAI API key (GPT-4o, GPT-4o-mini)
- [ ] Anthropic API key (Claude Sonnet 4, Claude Haiku 4)
- [ ] Google AI API key (Gemini 2.0 Flash, Gemini 2.5 Pro)
- [ ] (Optional) Together AI / Fireworks for Llama, Mistral, DeepSeek

### 4. Environment Variables
Copy `.env.example` to `.env` and fill in ALL production values:
```bash
cp .env.example .env
# Edit .env with production values
```

**Critical production settings:**
- `SECRET_KEY` — 64+ character random string
- `JWT_SECRET_KEY` — 32+ character random string
- `DATABASE_URL` — production database URL
- `REDIS_URL` — production Redis URL
- `ALLOWED_ORIGINS` — your production domain only
- `DEBUG=false`
- `LOG_LEVEL=info`

### 5. Database Migration
```bash
cd backend
alembic upgrade head
```

### 6. Deploy
```bash
# Build and start production containers
docker compose -f infra/docker-compose.production.yml up -d --build

# Verify health
curl https://your-domain.com/api/v1/health
curl https://your-domain.com
```

### 7. Post-Deploy Verification
- [ ] Landing page loads at `/`
- [ ] Signup flow works at `/auth/signup`
- [ ] Login flow works at `/auth/login`
- [ ] Context Shift page loads, file upload works
- [ ] Cockpit WebSocket connects, multi-model streaming works
- [ ] Workflows create + execute (chain and consensus)
- [ ] Analytics dashboard shows real data
- [ ] Settings page: profile save, API key CRUD, team invite
- [ ] Stripe checkout creates subscription
- [ ] Stripe portal opens for billing management
- [ ] Stripe webhook processes events

---

## Architecture (Production)

```
[CDN / CloudFlare]
        |
   [Nginx / ALB]
    /         \
[Frontend]  [Backend x2]
 Next.js     FastAPI + Gunicorn
    |              |
    |     [PostgreSQL + pgvector]
    |     [Redis 7]
    |     [Meilisearch]
    |
[Stripe]  [OpenAI]  [Anthropic]  [Google AI]
```

## Scaling Notes

| Threshold | Action |
|-----------|--------|
| 1K users | Single server (2 backend replicas) is fine |
| 10K users | Add read replicas for PostgreSQL, Redis cluster |
| 50K users | Kubernetes with HPA, separate WebSocket service |
| 100K+ users | Multi-region, dedicated model proxy layer |

## Monitoring
- **Health endpoint**: `GET /health` on both backend and frontend
- **Structured logs**: JSON format via structlog, ship to Datadog/Grafana
- **Error tracking**: Sentry for both frontend and backend
- **Uptime**: UptimeRobot or Better Uptime for /health endpoint
- **APM**: Datadog APM or New Relic for performance monitoring
