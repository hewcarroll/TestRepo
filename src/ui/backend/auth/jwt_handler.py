"""
JWT Token Handler for PDT Trading Bot Admin UI

This module provides secure JWT token management including creation,
verification, and validation for authentication purposes.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError, DecodeError
from passlib.context import CryptContext
import hashlib

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class JWTHandler:
    """Handles JWT token creation, verification, and validation."""

    def __init__(self):
        """Initialize JWT handler with configuration."""
        self.secret_key = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token.

        Args:
            data: Payload data to encode in the token
            expires_delta: Custom expiration time (optional)

        Returns:
            Encoded JWT access token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": secrets.token_hex(16)  # Unique token ID
        })

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create a JWT refresh token.

        Args:
            data: Payload data to encode in the token

        Returns:
            Encoded JWT refresh token
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_hex(16)  # Unique token ID
        })

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Validate token type
            token_type = payload.get("type")
            if token_type not in ["access", "refresh"]:
                raise ValueError("Invalid token type")

            return payload

        except ExpiredSignatureError:
            raise ValueError("Token has expired")
        except DecodeError:
            raise ValueError("Invalid token format")
        except InvalidTokenError:
            raise ValueError("Invalid token")

    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode a JWT token without verification (for debugging).

        Args:
            token: JWT token to decode

        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except Exception:
            return None

    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """
        Get the expiry datetime of a token.

        Args:
            token: JWT token

        Returns:
            Expiry datetime or None if invalid
        """
        payload = self.decode_token(token)
        if payload and "exp" in payload:
            return datetime.fromtimestamp(payload["exp"])
        return None

    def is_token_expired(self, token: str) -> bool:
        """
        Check if a token is expired.

        Args:
            token: JWT token to check

        Returns:
            True if token is expired, False otherwise
        """
        expiry = self.get_token_expiry(token)
        if not expiry:
            return True
        return datetime.utcnow() > expiry

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password to verify against

        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token.

        Args:
            length: Length of the token in bytes

        Returns:
            Secure random token as hex string
        """
        return secrets.token_hex(length)

    def create_user_session(self, user_id: str, username: str, role: str = "admin") -> Dict[str, str]:
        """
        Create a user session with access and refresh tokens.

        Args:
            user_id: Unique user identifier
            username: Username
            role: User role (default: "admin")

        Returns:
            Dictionary containing access_token and refresh_token
        """
        token_data = {
            "sub": user_id,
            "username": username,
            "role": role
        }

        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Create a new access token using a refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            Dictionary containing new access_token

        Raises:
            ValueError: If refresh token is invalid
        """
        payload = self.verify_token(refresh_token)

        if payload.get("type") != "refresh":
            raise ValueError("Invalid token type for refresh")

        # Create new access token with same user data
        token_data = {
            "sub": payload.get("sub"),
            "username": payload.get("username"),
            "role": payload.get("role")
        }

        new_access_token = self.create_access_token(token_data)

        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }

    def blacklist_token(self, token: str, blacklist: set) -> None:
        """
        Add a token to the blacklist (for logout).

        Args:
            token: Token to blacklist
            blacklist: Set to store blacklisted tokens
        """
        token_id = self.decode_token(token)
        if token_id and "jti" in token_id:
            blacklist.add(token_id["jti"])

    def is_token_blacklisted(self, token: str, blacklist: set) -> bool:
        """
        Check if a token is blacklisted.

        Args:
            token: Token to check
            blacklist: Set of blacklisted token IDs

        Returns:
            True if token is blacklisted, False otherwise
        """
        token_id = self.decode_token(token)
        if token_id and "jti" in token_id:
            return token_id["jti"] in blacklist
        return False
