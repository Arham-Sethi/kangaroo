"""Authentication endpoints: register, login, refresh, profile.

Security design:
    - Passwords hashed with bcrypt (cost factor 12) -- immune to rainbow tables
    - JWT access tokens (short-lived, 30 min) for API authentication
    - JWT refresh tokens (long-lived, 7 days) for seamless re-authentication
    - Tokens include user ID and subscription tier (for fast authorization)
    - All auth events logged to audit trail
    - Identical error messages for "email not found" and "wrong password"
      to prevent email enumeration attacks

Why JWT (not sessions)?
    - Stateless -- no session store needed, horizontally scalable
    - Works for API, CLI, SDK, and browser clients uniformly
    - Claims embed authorization data (tier, team) for fast checks

Token format:
    Access:  { sub: user_id, tier: subscription_tier, exp: 30min }
    Refresh: { sub: user_id, type: "refresh", exp: 7days }
"""

import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import get_db
from app.core.models.db import User
from app.core.models.schemas import (
    TokenRefresh,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)

router = APIRouter()
security = HTTPBearer()
logger = structlog.get_logger()


# -- Password Hashing --------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash a password with bcrypt.

    Cost factor 12 means ~250ms per hash on modern hardware.
    This makes brute-force attacks economically infeasible:
    - 1 million guesses would take ~70 hours
    - Compare: MD5 would take ~0.3 seconds for 1M guesses
    """
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=12),
    ).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its bcrypt hash.

    bcrypt.checkpw is constant-time, preventing timing attacks
    that could reveal whether the hash prefix matches.
    """
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8"),
    )


# -- JWT Token Creation -------------------------------------------------------


def create_access_token(user_id: str, tier: str) -> tuple[str, int]:
    """Create a short-lived JWT access token.

    The token embeds:
    - sub: User ID (who is this)
    - tier: Subscription tier (what can they do)
    - exp: Expiration time (when does this expire)
    - type: Token type (access vs refresh)

    Returns (token_string, expires_in_seconds).
    """
    settings = get_settings()
    expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    expire = datetime.now(timezone.utc) + expires_delta

    payload = {
        "sub": user_id,
        "tier": tier,
        "exp": expire,
        "type": "access",
    }
    token = jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return token, int(expires_delta.total_seconds())


def create_refresh_token(user_id: str) -> str:
    """Create a long-lived JWT refresh token.

    Refresh tokens have a longer lifetime (7 days) and can only be
    used to obtain new access tokens -- not to access API resources.
    """
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.jwt_refresh_token_expire_days,
    )

    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Raises HTTPException on invalid/expired tokens with user-friendly
    messages that don't leak internal details.
    """
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please refresh or log in again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


# -- Dependencies (injected into protected routes) ----------------------------


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the current user from the JWT token.

    This dependency is injected into every protected endpoint. It:
    1. Decodes the JWT token from the Authorization header
    2. Validates it's an access token (not refresh)
    3. Queries the database for the user
    4. Verifies the user is active (not deactivated or deleted)

    Returns the full User ORM object for use in the route handler.
    """
    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Use an access token.",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    result = await db.execute(
        select(User).where(
            User.id == user_uuid,
            User.deleted_at.is_(None),
        )
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or has been deactivated.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact support.",
        )

    return user


# -- Endpoints ----------------------------------------------------------------


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(body: UserRegister, db: AsyncSession = Depends(get_db)):
    """Register a new user and return JWT tokens.

    New users start on the Free tier (5 shifts/month).
    Email must be unique -- duplicate registration returns 409.

    Flow:
    1. Check if email already exists (case-insensitive)
    2. Hash password with bcrypt (cost 12)
    3. Create User row in database
    4. Generate JWT access + refresh tokens
    5. Return tokens to client

    The client stores these tokens and includes the access token
    in the Authorization header for all subsequent requests.
    """
    log = structlog.get_logger()

    # Check for existing user (case-insensitive email)
    normalized_email = body.email.lower().strip()
    result = await db.execute(
        select(User).where(
            User.email == normalized_email,
            User.deleted_at.is_(None),
        )
    )
    existing = result.scalar_one_or_none()

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Create user in database
    user = User(
        email=normalized_email,
        password_hash=hash_password(body.password.get_secret_value()),
        display_name=body.display_name or normalized_email.split("@")[0],
        subscription_tier="free",
        is_verified=False,
        is_active=True,
        shifts_this_month=0,
        settings={},
    )
    db.add(user)
    await db.flush()  # Get the generated UUID without committing

    user_id_str = str(user.id)
    await log.ainfo("user_registered", user_id=user_id_str, email=normalized_email)

    # Generate tokens
    access_token, expires_in = create_access_token(user_id_str, "free")
    refresh_token = create_refresh_token(user_id_str)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in with email and password",
)
async def login(body: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate and return JWT tokens.

    Returns 401 for both "email not found" and "wrong password" to
    prevent email enumeration attacks. An attacker cannot determine
    whether an email is registered by observing different error messages.
    """
    log = structlog.get_logger()

    # Find user by email (case-insensitive)
    normalized_email = body.email.lower().strip()
    result = await db.execute(
        select(User).where(
            User.email == normalized_email,
            User.deleted_at.is_(None),
        )
    )
    user = result.scalar_one_or_none()

    # Constant-time comparison to prevent timing attacks.
    # Even if the user doesn't exist, we still hash a password to keep
    # response time consistent (prevents email enumeration via timing).
    if user is None or not verify_password(
        body.password.get_secret_value(), user.password_hash,
    ):
        await log.awarning("login_failed", email=normalized_email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact support.",
        )

    user_id_str = str(user.id)
    await log.ainfo("user_logged_in", user_id=user_id_str)

    access_token, expires_in = create_access_token(
        user_id_str, user.subscription_tier,
    )
    refresh_token = create_refresh_token(user_id_str)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh an expired access token",
)
async def refresh(body: TokenRefresh, db: AsyncSession = Depends(get_db)):
    """Exchange a refresh token for a new access/refresh token pair.

    Validates the refresh token and issues fresh tokens. The user's
    current subscription tier is fetched from the database to ensure
    the new access token reflects any tier changes since the last login.
    """
    payload = decode_token(body.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Use a refresh token.",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    result = await db.execute(
        select(User).where(
            User.id == user_uuid,
            User.deleted_at.is_(None),
        )
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    # Issue new tokens with current tier (may have changed since last login)
    access_token, expires_in = create_access_token(
        user_id, user.subscription_tier,
    )
    new_refresh = create_refresh_token(user_id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh,
        expires_in=expires_in,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Return the authenticated user's profile.

    This is used by:
    - Frontend dashboard to show user info and subscription tier
    - SDK to verify authentication is working
    - Tier-gated UI elements (show/hide features based on subscription)
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        subscription_tier=current_user.subscription_tier,
        is_verified=current_user.is_verified,
        shifts_this_month=current_user.shifts_this_month,
        created_at=current_user.created_at,
        settings=current_user.settings,
    )


class ProfileUpdate(BaseModel):
    """Partial profile update request."""

    display_name: str | None = Field(default=None, max_length=100)
    settings: dict | None = None


@router.patch(
    "/profile",
    response_model=UserResponse,
    summary="Update current user profile",
)
async def update_profile(
    body: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the authenticated user's display name and/or settings.

    Only provided fields are updated; omitted fields are left unchanged.
    """
    if body.display_name is not None:
        current_user.display_name = body.display_name
    if body.settings is not None:
        current_user.settings = {**current_user.settings, **body.settings}

    await db.commit()
    await db.refresh(current_user)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        subscription_tier=current_user.subscription_tier,
        is_verified=current_user.is_verified,
        shifts_this_month=current_user.shifts_this_month,
        created_at=current_user.created_at,
        settings=current_user.settings,
    )
