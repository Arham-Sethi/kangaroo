"""API v1 routes -- versioned API for forward compatibility.

Router hierarchy:
    /api/v1/auth      -- Registration, login, token refresh
    /api/v1/contexts  -- Context generation, entity extraction, summarization
    /api/v1/shifts    -- Context transfer between LLMs
    /api/v1/import    -- Conversation import (text, JSON, file upload)
"""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.contexts import router as contexts_router
from app.api.v1.shifts import router as shifts_router
from app.api.v1.import_export import router as import_export_router
from app.api.v1.sessions import router as sessions_router
from app.api.v1.api_keys import router as api_keys_router
from app.api.v1.webhooks import router as webhooks_router

router = APIRouter()

# Authentication
router.include_router(auth_router, prefix="/auth", tags=["auth"])

# Context engine endpoints
router.include_router(contexts_router, prefix="/contexts", tags=["contexts"])

# Context shifting (the core product feature)
router.include_router(shifts_router, prefix="/shifts", tags=["shifts"])

# Conversation import
router.include_router(import_export_router, prefix="/import", tags=["import"])

# Session management
router.include_router(sessions_router, prefix="/sessions", tags=["sessions"])

# API key management
router.include_router(api_keys_router, prefix="/api-keys", tags=["api-keys"])

# Webhook management
router.include_router(webhooks_router, prefix="/webhooks", tags=["webhooks"])

# Future routers:
# router.include_router(proxy_router, prefix="/proxy", tags=["proxy"])
# router.include_router(search_router, prefix="/search", tags=["search"])
# router.include_router(teams_router, prefix="/teams", tags=["teams"])
# router.include_router(webhooks_router, prefix="/webhooks", tags=["webhooks"])
# router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
# router.include_router(workflows_router, prefix="/workflows", tags=["workflows"])
