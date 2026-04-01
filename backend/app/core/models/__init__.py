"""Data models — SQLAlchemy ORM and Pydantic schemas.

This package contains three layers of data models:
    - db.py: SQLAlchemy ORM models (database tables)
    - ucs.py: Universal Context Schema (the product's core data format)
    - schemas.py: API request/response schemas (public contract)

These are deliberately separate to enforce boundaries:
    - Database models can change without affecting the API contract
    - UCS format is versioned independently
    - API schemas control what data enters and leaves the system
"""

from app.core.models.db import (
    APIKey,
    AuditLog,
    Base,
    Context,
    Session,
    ShiftRecord,
    Team,
    TeamMember,
    User,
    Webhook,
)
from app.core.models.ucs import (
    UCSValidator,
    UniversalContextSchema,
)

__all__ = [
    "Base",
    "User",
    "Session",
    "Context",
    "APIKey",
    "Team",
    "TeamMember",
    "Webhook",
    "AuditLog",
    "ShiftRecord",
    "UniversalContextSchema",
    "UCSValidator",
]
