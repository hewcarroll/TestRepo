"""
Security Middleware for PDT Trading Bot Admin UI

This module provides FastAPI middleware for comprehensive security
including rate limiting, input validation, and threat protection.
"""

import time
import json
from typing import Dict, List, Any, Optional, Callable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import logging

from .rate_limiter import rate_limiter, check_api_rate_limit, check_auth_rate_limit
from .input_validator import input_validator, detect_input_threats

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Comprehensive security middleware for FastAPI."""

    def __init__(self, app):
        """Initialize security middleware."""
        self.app = app
        self.sensitive_endpoints = {
            '/api/auth/login',
            '/api/auth/logout',
            '/api/dashboard/emergency/kill-switch',
            '/api/data/bot/control'
        }

        # Security headers to add to responses
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; img-src 'self' data: https:; font-src 'self' https://cdnjs.cloudflare.com",
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }

    async def __call__(self, request: Request, call_next: Callable):
        """Process request through security middleware."""
        start_time = time.time()

        try:
            # Extract client information
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get('User-Agent', '')

            # Pre-request security checks
            await self._pre_request_checks(request, client_ip)

            # Rate limiting check
            rate_limit_result = await self._check_rate_limits(request, client_ip)
            if not rate_limit_result['allowed']:
                return self._create_rate_limit_response(rate_limit_result)

            # Input validation and threat detection
            threat_result = await self._check_request_threats(request)
            if threat_result['blocked']:
                return self._create_threat_response(threat_result)

            # Process request
            response = await call_next(request)

            # Add security headers
            self._add_security_headers(response)

            # Log security-relevant requests
            await self._log_security_event(request, response, client_ip, start_time)

            return response

        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={'detail': 'Internal security error'}
            )

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP address."""
        # Check for forwarded IP (behind proxy/load balancer)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()

        # Check for real IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip.strip()

        # Fall back to client host
        return request.client.host if request.client else 'unknown'

    async def _pre_request_checks(self, request: Request, client_ip: str) -> None:
        """Perform pre-request security checks."""
        # Check for suspicious user agents
        user_agent = request.headers.get('User-Agent', '').lower()

        suspicious_agents = [
            'sqlmap', 'nmap', 'nikto', 'dirbuster', 'gobuster',
            'masscan', 'zgrab', 'zap', 'burp', 'owasp'
        ]

        for agent in suspicious_agents:
            if agent in user_agent:
                logger.warning(f"Suspicious user agent detected: {user_agent} from {client_ip}")
                # Could add to blacklist here if needed

        # Check request size
        content_length = request.headers.get('Content-Length')
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )

    async def _check_rate_limits(self, request: Request, client_ip: str) -> Dict[str, Any]:
        """Check rate limits for request."""
        path = request.url.path

        try:
            # Determine rate limit rule based on endpoint
            if path in self.sensitive_endpoints:
                if 'auth' in path:
                    allowed, wait_time = check_auth_rate_limit(client_ip)
                elif 'emergency' in path or 'control' in path:
                    allowed, wait_time = check_emergency_rate_limit(client_ip)
                else:
                    allowed, wait_time = check_trading_rate_limit(client_ip)
            else:
                allowed, wait_time = check_api_rate_limit(client_ip)

            return {
                'allowed': allowed,
                'wait_time': wait_time,
                'client_ip': client_ip,
                'endpoint': path
            }

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return {'allowed': False, 'wait_time': 60, 'error': str(e)}

    def _create_rate_limit_response(self, rate_limit_result: Dict[str, Any]) -> JSONResponse:
        """Create rate limit exceeded response."""
        wait_time = rate_limit_result.get('wait_time', 60)

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                'detail': f'Rate limit exceeded. Try again in {int(wait_time)} seconds.',
                'retry_after': int(wait_time),
                'timestamp': datetime.utcnow().isoformat()
            },
            headers={'Retry-After': str(int(wait_time))}
        )

    async def _check_request_threats(self, request: Request) -> Dict[str, Any]:
        """Check request for security threats."""
        try:
            # Only check POST/PUT/PATCH requests with body
            if request.method not in ['POST', 'PUT', 'PATCH']:
                return {'blocked': False, 'threats': []}

            # Get request body
            body = await request.body()
            if not body:
                return {'blocked': False, 'threats': []}

            try:
                # Parse JSON body
                json_data = json.loads(body)

                # Check for threats in JSON data
                threat_result = detect_input_threats(json.dumps(json_data))

                if threat_result.get('threat_level') in ['high', 'critical']:
                    logger.warning(f"High threat level detected in request to {request.url.path}")

                    return {
                        'blocked': True,
                        'threats': threat_result,
                        'reason': 'High threat level detected in request data'
                    }

                return {'blocked': False, 'threats': threat_result}

            except json.JSONDecodeError:
                # Non-JSON body, check as raw text
                threat_result = detect_input_threats(body.decode('utf-8', errors='ignore'))

                if threat_result.get('threat_level') in ['high', 'critical']:
                    return {
                        'blocked': True,
                        'threats': threat_result,
                        'reason': 'High threat level detected in request body'
                    }

                return {'blocked': False, 'threats': threat_result}

        except Exception as e:
            logger.error(f"Threat detection error: {e}")
            return {'blocked': False, 'threats': {}, 'error': str(e)}

    def _create_threat_response(self, threat_result: Dict[str, Any]) -> JSONResponse:
        """Create threat detected response."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                'detail': 'Request blocked due to security concerns',
                'threat_level': threat_result['threats'].get('threat_level', 'unknown'),
                'timestamp': datetime.utcnow().isoformat()
            }
        )

    def _add_security_headers(self, response) -> None:
        """Add security headers to response."""
        try:
            for header_name, header_value in self.security_headers.items():
                response.headers[header_name] = header_value
        except Exception as e:
            logger.error(f"Error adding security headers: {e}")

    async def _log_security_event(self, request: Request, response, client_ip: str, start_time: float) -> None:
        """Log security-relevant events."""
        try:
            # Calculate request duration
            duration = time.time() - start_time

            # Log slow requests (>5 seconds)
            if duration > 5.0:
                logger.warning(f"Slow request detected: {request.method} {request.url.path} took {duration:.2f}s from {client_ip}")

            # Log authentication failures
            if (request.url.path.startswith('/api/auth/') and
                response.status_code in [401, 403]):
                logger.warning(f"Authentication failure: {request.method} {request.url.path} from {client_ip}")

            # Log emergency actions
            if (request.url.path in self.sensitive_endpoints and
                response.status_code == 200):
                logger.info(f"Emergency/sensitive action executed: {request.method} {request.url.path} from {client_ip}")

        except Exception as e:
            logger.error(f"Error logging security event: {e}")

# Security configuration
SECURITY_CONFIG = {
    'enable_rate_limiting': True,
    'enable_threat_detection': True,
    'enable_audit_logging': True,
    'max_request_size': 10 * 1024 * 1024,  # 10MB
    'slow_request_threshold': 5.0,  # seconds
    'suspicious_request_threshold': 10,  # requests per minute
}

# Threat intelligence data (in production, this would be updated regularly)
THREAT_INTELLIGENCE = {
    'suspicious_ips': set(),  # Would be populated from threat feeds
    'suspicious_user_agents': [
        'sqlmap', 'nmap', 'nikto', 'masscan', 'zgrab'
    ],
    'malicious_patterns': [
        r'<\s*script',
        r'union\s+select',
        r'xp_cmdshell',
        r'javascript:',
        r'vbscript:'
    ]
}

def create_security_middleware(app):
    """Create and return security middleware instance."""
    return SecurityMiddleware(app)

# Additional security utilities
async def validate_request_data(request: Request, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate request data against schema.

    Args:
        request: FastAPI request object
        schema: Validation schema

    Returns:
        Validation result
    """
    try:
        # Get request body
        body = await request.body()
        if not body:
            return {'valid': True, 'data': {}}

        # Parse JSON
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return {
                'valid': False,
                'error': 'Invalid JSON in request body'
            }

        # Basic validation
        if not isinstance(data, dict):
            return {
                'valid': False,
                'error': 'Request body must be a JSON object'
            }

        # Schema validation (simplified)
        if schema:
            for field, rules in schema.items():
                if field in data:
                    value = data[field]

                    # Required field check
                    if rules.get('required', False) and value is None:
                        return {
                            'valid': False,
                            'error': f"Field '{field}' is required"
                        }

                    # Type validation
                    expected_type = rules.get('type')
                    if expected_type == 'string' and not isinstance(value, str):
                        return {
                            'valid': False,
                            'error': f"Field '{field}' must be a string"
                        }

                    if expected_type == 'number' and not isinstance(value, (int, float)):
                        return {
                            'valid': False,
                            'error': f"Field '{field}' must be a number"
                        }

                    if expected_type == 'integer' and not isinstance(value, int):
                        return {
                            'valid': False,
                            'error': f"Field '{field}' must be an integer"
                        }

                    # Length validation for strings
                    if expected_type == 'string' and 'max_length' in rules:
                        if len(value) > rules['max_length']:
                            return {
                                'valid': False,
                                'error': f"Field '{field}' exceeds maximum length of {rules['max_length']}"
                            }

        return {'valid': True, 'data': data}

    except Exception as e:
        logger.error(f"Request validation error: {e}")
        return {
            'valid': False,
            'error': f'Validation failed: {str(e)}'
        }

async def check_request_signature(request: Request, secret_key: str) -> bool:
    """
    Validate request signature for webhook security.

    Args:
        request: FastAPI request object
        secret_key: Secret key for signature validation

    Returns:
        True if signature is valid
    """
    try:
        # Get signature from headers
        signature = request.headers.get('X-Signature')
        if not signature:
            return False

        # Get request body for signature calculation
        body = await request.body()

        # Calculate expected signature (simplified HMAC)
        import hashlib
        import hmac

        expected_signature = hmac.new(
            secret_key.encode(),
            body,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    except Exception as e:
        logger.error(f"Signature validation error: {e}")
        return False

def sanitize_response_data(data: Any) -> Any:
    """
    Sanitize response data to prevent information leakage.

    Args:
        data: Response data to sanitize

    Returns:
        Sanitized data
    """
    try:
        if isinstance(data, dict):
            # Remove sensitive fields from error responses
            sanitized = {}
            sensitive_fields = {'password', 'token', 'secret', 'key', 'credential'}

            for key, value in data.items():
                if key.lower() not in sensitive_fields:
                    sanitized[key] = sanitize_response_data(value)

            return sanitized

        elif isinstance(data, list):
            return [sanitize_response_data(item) for item in data]

        elif isinstance(data, str):
            # Truncate very long strings in responses
            if len(data) > 10000:
                return data[:10000] + '... [truncated]'
            return data

        else:
            return data

    except Exception as e:
        logger.error(f"Response sanitization error: {e}")
        return data

# Security monitoring and alerting
class SecurityMonitor:
    """Monitors security events and generates alerts."""

    def __init__(self):
        """Initialize security monitor."""
        self.security_events: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            'failed_logins': 5,
            'suspicious_requests': 10,
            'threat_detections': 3,
            'rate_limit_violations': 20
        }

        # Alert counters
        self.event_counters = {
            'failed_logins': 0,
            'suspicious_requests': 0,
            'threat_detections': 0,
            'rate_limit_violations': 0
        }

    async def record_security_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Record a security event.

        Args:
            event_type: Type of security event
            event_data: Event details
        """
        try:
            event_record = {
                'type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data': event_data
            }

            self.security_events.append(event_record)

            # Update counters
            if event_type in self.event_counters:
                self.event_counters[event_type] += 1

            # Keep only last 1000 events
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]

            # Check if alert should be generated
            await self._check_alert_thresholds(event_type)

            logger.info(f"Security event recorded: {event_type}")

        except Exception as e:
            logger.error(f"Error recording security event: {e}")

    async def _check_alert_thresholds(self, event_type: str) -> None:
        """Check if alert thresholds have been exceeded."""
        try:
            if event_type in self.alert_thresholds:
                current_count = self.event_counters[event_type]
                threshold = self.alert_thresholds[event_type]

                if current_count >= threshold:
                    await self._generate_security_alert(event_type, current_count, threshold)

                    # Reset counter after alert
                    self.event_counters[event_type] = 0

        except Exception as e:
            logger.error(f"Error checking alert thresholds: {e}")

    async def _generate_security_alert(self, event_type: str, current_count: int, threshold: int) -> None:
        """Generate security alert."""
        try:
            alert_message = f"Security alert: {event_type} threshold exceeded ({current_count} >= {threshold})"

            logger.critical(alert_message)

            # In production, this would send alerts via email, Slack, etc.
            # await self._send_security_alert(alert_message)

        except Exception as e:
            logger.error(f"Error generating security alert: {e}")

    async def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        try:
            return {
                'total_events': len(self.security_events),
                'event_counters': self.event_counters.copy(),
                'alert_thresholds': self.alert_thresholds.copy(),
                'recent_events': self.security_events[-10:],  # Last 10 events
                'generated_at': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting security summary: {e}")
            return {'error': str(e)}

# Global security monitor instance
security_monitor = SecurityMonitor()

# Export security utilities
__all__ = [
    'SecurityMiddleware',
    'create_security_middleware',
    'validate_request_data',
    'check_request_signature',
    'sanitize_response_data',
    'SecurityMonitor',
    'security_monitor',
    'SECURITY_CONFIG',
    'THREAT_INTELLIGENCE'
]
