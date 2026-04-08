"""Usage analytics and reporting endpoints.

Endpoints:
    GET /analytics/daily-usage         -- Daily usage metrics (last 30 days)
    GET /analytics/model-distribution  -- Model usage distribution
    GET /analytics/summary             -- High-level summary stats
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text, cast, Date
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.core.models.db import AuditLog, Session, ShiftRecord, User

logger = structlog.get_logger()
router = APIRouter()


# ── Response Schemas ─────────────────────────────────────────────────────────


# Using dicts for simplicity — production would use Pydantic models.


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/daily-usage")
async def daily_usage(
    days: int = Query(default=30, ge=1, le=90),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Get daily usage metrics for the last N days.

    Returns per-day:
        - date: ISO date string
        - shifts: Number of shifts performed
        - sessions: Number of sessions created
        - tokens: Total tokens processed
    """
    user_id = current_user.id
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Shifts per day
    shift_result = await db.execute(
        select(
            cast(ShiftRecord.created_at, Date).label("day"),
            func.count(ShiftRecord.id).label("count"),
            func.coalesce(func.sum(ShiftRecord.source_tokens + ShiftRecord.target_tokens), 0).label("tokens"),
        )
        .where(ShiftRecord.user_id == user_id, ShiftRecord.created_at >= since)
        .group_by(cast(ShiftRecord.created_at, Date))
    )
    shift_rows = {str(row.day): {"shifts": row.count, "tokens": int(row.tokens)} for row in shift_result}

    # Sessions per day
    session_result = await db.execute(
        select(
            cast(Session.created_at, Date).label("day"),
            func.count(Session.id).label("count"),
        )
        .where(Session.user_id == user_id, Session.created_at >= since, Session.deleted_at.is_(None))
        .group_by(cast(Session.created_at, Date))
    )
    session_rows = {str(row.day): row.count for row in session_result}

    # Build daily data (fill in zeros for days with no activity)
    daily = []
    for i in range(days):
        date = (datetime.now(timezone.utc) - timedelta(days=days - 1 - i)).date()
        date_str = str(date)
        shift_data = shift_rows.get(date_str, {"shifts": 0, "tokens": 0})
        daily.append({
            "date": date_str,
            "shifts": shift_data["shifts"],
            "sessions": session_rows.get(date_str, 0),
            "tokens": shift_data["tokens"],
        })

    return daily


@router.get("/model-distribution")
async def model_distribution(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Get distribution of which LLM models are used.

    Returns:
        List of {model, count, direction} sorted by count descending.
    """
    user_id = current_user.id

    # Source model distribution
    source_result = await db.execute(
        select(
            ShiftRecord.source_llm.label("model"),
            func.count(ShiftRecord.id).label("count"),
        )
        .where(ShiftRecord.user_id == user_id)
        .group_by(ShiftRecord.source_llm)
    )
    source_rows = [
        {"model": row.model, "count": row.count, "direction": "source"}
        for row in source_result
    ]

    # Target model distribution
    target_result = await db.execute(
        select(
            ShiftRecord.target_llm.label("model"),
            func.count(ShiftRecord.id).label("count"),
        )
        .where(ShiftRecord.user_id == user_id)
        .group_by(ShiftRecord.target_llm)
    )
    target_rows = [
        {"model": row.model, "count": row.count, "direction": "target"}
        for row in target_result
    ]

    # Also include session source_llm for captures
    session_result = await db.execute(
        select(
            Session.source_llm.label("model"),
            func.count(Session.id).label("count"),
        )
        .where(Session.user_id == user_id, Session.deleted_at.is_(None))
        .group_by(Session.source_llm)
    )
    session_rows = [
        {"model": row.model, "count": row.count, "direction": "capture"}
        for row in session_result
    ]

    # Merge all and sort
    all_rows = source_rows + target_rows + session_rows
    all_rows.sort(key=lambda r: r["count"], reverse=True)
    return all_rows


@router.get("/summary")
async def summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get high-level summary statistics.

    Returns:
        total_sessions, total_shifts, total_tokens,
        shifts_this_month, active_sessions
    """
    user_id = current_user.id

    # Total sessions
    total_sessions_result = await db.execute(
        select(func.count(Session.id)).where(
            Session.user_id == user_id, Session.deleted_at.is_(None),
        )
    )
    total_sessions = total_sessions_result.scalar_one()

    # Active sessions (not archived)
    active_sessions_result = await db.execute(
        select(func.count(Session.id)).where(
            Session.user_id == user_id,
            Session.deleted_at.is_(None),
            Session.is_archived.is_(False),
        )
    )
    active_sessions = active_sessions_result.scalar_one()

    # Total shifts
    total_shifts_result = await db.execute(
        select(func.count(ShiftRecord.id)).where(ShiftRecord.user_id == user_id)
    )
    total_shifts = total_shifts_result.scalar_one()

    # Total tokens across all shifts
    total_tokens_result = await db.execute(
        select(
            func.coalesce(
                func.sum(ShiftRecord.source_tokens + ShiftRecord.target_tokens), 0
            )
        ).where(ShiftRecord.user_id == user_id)
    )
    total_tokens = int(total_tokens_result.scalar_one())

    # Shifts this month (from user record)
    shifts_this_month = current_user.shifts_this_month

    return {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "total_shifts": total_shifts,
        "total_tokens": total_tokens,
        "shifts_this_month": shifts_this_month,
        "subscription_tier": current_user.subscription_tier,
    }
