"""
Authentication Routes for PDT Trading Bot Admin UI

This module provides authentication endpoints including login, logout,
token refresh, and user management.
"""

from datetime import timedelta
from typing import Dict, Any
import os
from fastapi import APIRouter, HTTPException, Depends, status, Request, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
import re

from ..auth.jwt_handler import JWTHandler
from ..auth.session_manager import SessionManager
from ..app import get_current_user, jwt_handler, session_manager

# Initialize components
auth_router = APIRouter()

# Request/Response models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str

    @validator('username')
    def username_must_be_valid(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.strip()

    @validator('password')
    def password_must_be_valid(cls, v):
        if not v or len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str

class UserInfo(BaseModel):
    """User information model."""
    user_id: str
    username: str
    role: str
    permissions: list[str]

# Default admin credentials (in production, use environment variables or database)
DEFAULT_ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change_me_in_production")
DEFAULT_ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@localhost")

# In-memory user store (in production, use a proper database)
USERS_DB = {
    DEFAULT_ADMIN_USER: {
        "user_id": "admin_001",
        "username": DEFAULT_ADMIN_USER,
        "password_hash": jwt_handler.hash_password(DEFAULT_ADMIN_PASSWORD),
        "email": DEFAULT_ADMIN_EMAIL,
        "role": "admin",
        "permissions": ["read", "write", "admin", "emergency_stop"],
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z"
    }
}

def get_client_ip(request: Request) -> str:
    """Extract client IP address from request."""
    # Check for forwarded IP (behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fall back to client host
    return request.client.host if request.client else "unknown"

def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticate a user with username and password.

    Args:
        username: Username
        password: Plain text password

    Returns:
        User data if authenticated

    Raises:
        HTTPException: If authentication fails
    """
    user_data = USERS_DB.get(username)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    if not user_data.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled"
        )

    if not jwt_handler.verify_password(password, user_data["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    return user_data

@auth_router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    login_data: LoginRequest
) -> TokenResponse:
    """
    Authenticate user and return access/refresh tokens.

    This endpoint validates user credentials and returns JWT tokens
    for authenticated sessions with rate limiting protection.
    """
    client_ip = get_client_ip(request)
    username = login_data.username

    # Check rate limiting
    is_allowed, wait_time = session_manager.check_rate_limit(f"login:{username}")
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many login attempts. Try again in {int(wait_time)} seconds"
        )

    # Check IP-based rate limiting
    is_allowed, wait_time = session_manager.check_rate_limit(f"login_ip:{client_ip}")
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many login attempts from this IP. Try again in {int(wait_time)} seconds"
        )

    try:
        # Authenticate user
        user_data = authenticate_user(username, login_data.password)

        # Record successful login
        session_manager.record_successful_login(f"login:{username}")
        session_manager.record_successful_login(f"login_ip:{client_ip}")

        # Create user session
        session_id = session_manager.create_session(
            user_id=user_data["user_id"],
            username=username,
            role=user_data["role"],
            ip_address=client_ip,
            user_agent=request.headers.get("User-Agent", "Unknown")
        )

        # Create JWT tokens
        token_data = {
            "sub": user_data["user_id"],
            "username": username,
            "role": user_data["role"],
            "session_id": session_id
        }

        tokens = jwt_handler.create_user_session(
            user_id=user_data["user_id"],
            username=username,
            role=user_data["role"]
        )

        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=1800  # 30 minutes in seconds
        )

    except HTTPException:
        # Record failed attempt
        session_manager.record_failed_attempt(f"login:{username}")
        session_manager.record_failed_attempt(f"login_ip:{client_ip}")
        raise

@auth_router.post("/refresh")
async def refresh_token(
    request: RefreshTokenRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TokenResponse:
    """
    Refresh an access token using a refresh token.

    This endpoint allows clients to obtain a new access token
    without re-authentication using a valid refresh token.
    """
    try:
        # Verify and decode refresh token
        payload = jwt_handler.verify_token(request.refresh_token)

        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )

        # Create new access token
        new_tokens = jwt_handler.refresh_access_token(request.refresh_token)

        return TokenResponse(
            access_token=new_tokens["access_token"],
            refresh_token=request.refresh_token,  # Keep same refresh token
            token_type=new_tokens["token_type"],
            expires_in=1800
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

@auth_router.post("/logout")
async def logout(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Logout user and invalidate all sessions.

    This endpoint invalidates the current session and all other
    sessions for the user for security.
    """
    try:
        # Invalidate current session
        session_manager.invalidate_user_sessions(current_user["user_id"])

        # Extract token JTI for blacklisting
        credentials = await HTTPAuthorizationCredentials.__class__(
            scheme="Bearer",
            credentials=request.headers.get("Authorization", "").replace("Bearer ", "")
        )

        if credentials.credentials:
            payload = jwt_handler.decode_token(credentials.credentials)
            if payload and "jti" in payload:
                session_manager.blacklist_token(payload["jti"])

        return {"message": "Successfully logged out"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )

@auth_router.get("/me")
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> UserInfo:
    """
    Get information about the currently authenticated user.

    This endpoint returns user profile information for the
    authenticated user.
    """
    # Get user data from our simple database
    user_data = None
    for username, data in USERS_DB.items():
        if data["user_id"] == current_user["user_id"]:
            user_data = data
            break

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserInfo(
        user_id=user_data["user_id"],
        username=user_data["username"],
        role=user_data["role"],
        permissions=user_data["permissions"]
    )

@auth_router.get("/sessions")
async def get_user_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all active sessions for the current user.

    This endpoint returns information about all active sessions
    for the authenticated user.
    """
    active_sessions = session_manager.get_active_sessions(current_user["user_id"])

    return {
        "sessions": [
            {
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "ip_address": session.ip_address,
                "user_agent": session.user_agent
            }
            for session_id, session in active_sessions.items()
        ],
        "total_count": len(active_sessions)
    }

@auth_router.delete("/sessions/{session_id}")
async def invalidate_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Invalidate a specific session.

    This endpoint allows users to invalidate specific sessions
    (useful for remote logout from other devices).
    """
    # Get the session to verify ownership
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Check if user owns this session
    if session.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to invalidate this session"
        )

    # Invalidate the session
    session_manager.invalidate_session(session_id)

    return {"message": "Session invalidated successfully"}

@auth_router.get("/security/metrics")
async def get_security_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get security metrics for monitoring.

    This endpoint returns security-related metrics including
    active sessions, failed attempts, etc.
    """
    # Check if user has admin permissions
    user_data = None
    for username, data in USERS_DB.items():
        if data["user_id"] == current_user["user_id"]:
            user_data = data
            break

    if not user_data or "admin" not in user_data.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )

    return session_manager.get_security_metrics()

@auth_router.post("/admin/create-user")
async def create_user(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    role: str = Form("user"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Create a new user (admin only).

    This endpoint allows administrators to create new user accounts.
    """
    # Check if user has admin permissions
    user_data = None
    for user, data in USERS_DB.items():
        if data["user_id"] == current_user["user_id"]:
            user_data = data
            break

    if not user_data or "admin" not in user_data.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )

    # Validate input
    if username in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        )

    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )

    # Create new user
    new_user_id = f"user_{len(USERS_DB) + 1}"
    USERS_DB[username] = {
        "user_id": new_user_id,
        "username": username,
        "password_hash": jwt_handler.hash_password(password),
        "email": email,
        "role": role,
        "permissions": ["read"] if role == "user" else ["read", "write"],
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z"
    }

    return {"message": "User created successfully", "user_id": new_user_id}
