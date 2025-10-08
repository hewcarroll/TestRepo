"""
Input Validator Module for PDT Trading Bot Admin UI

This module provides comprehensive input validation and sanitization
to protect against injection attacks and malformed data.
"""

import re
import json
import html
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import ipaddress
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Comprehensive input validation and sanitization."""

    def __init__(self):
        """Initialize input validator."""
        # Common validation patterns
        self.patterns = {
            'username': re.compile(r'^[a-zA-Z0-9_-]{3,50}$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'password': re.compile(r'^.{8,128}$'),
            'symbol': re.compile(r'^[A-Z]{1,5}$'),
            'trade_id': re.compile(r'^[a-zA-Z0-9_-]{1,50}$'),
            'ip_address': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'numeric': re.compile(r'^-?\d+(\.\d+)?$'),
            'integer': re.compile(r'^-?\d+$'),
            'positive_integer': re.compile(r'^\d+$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        }

        # Dangerous patterns to block
        self.dangerous_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<object[^>]*>.*?</object>', re.IGNORECASE | re.DOTALL),
            re.compile(r'<embed[^>]*>', re.IGNORECASE),
            re.compile(r'expression\s*\(', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'data:text/html', re.IGNORECASE),
        ]

        # SQL injection patterns
        self.sql_injection_patterns = [
            re.compile(r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)', re.IGNORECASE),
            re.compile(r'(--|#|/\*)'),
            re.compile(r'(\bor\b|\band\b)\s+\d+\s*=\s*\d+', re.IGNORECASE),
            re.compile(r'(\b(xp_|sp_|fn_)\w+)', re.IGNORECASE),
        ]

        # XSS patterns
        self.xss_patterns = [
            re.compile(r'<[^>]*>', re.IGNORECASE),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'&#', re.IGNORECASE),
            re.compile(r'%3c', re.IGNORECASE),
        ]

    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """
        Sanitize string input.

        Args:
            value: String to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return ""

        # Basic length check
        if len(value) > max_length:
            logger.warning(f"Input truncated from {len(value)} to {max_length} characters")
            value = value[:max_length]

        # HTML escape
        sanitized = html.escape(value)

        # Remove potentially dangerous content
        for pattern in self.dangerous_patterns:
            sanitized = pattern.sub('', sanitized)

        return sanitized.strip()

    def validate_username(self, username: str) -> Dict[str, Any]:
        """
        Validate username.

        Args:
            username: Username to validate

        Returns:
            Validation result
        """
        if not username:
            return {'valid': False, 'error': 'Username is required'}

        if len(username) < 3:
            return {'valid': False, 'error': 'Username must be at least 3 characters'}

        if len(username) > 50:
            return {'valid': False, 'error': 'Username must be less than 50 characters'}

        if not self.patterns['username'].match(username):
            return {'valid': False, 'error': 'Username contains invalid characters'}

        return {'valid': True, 'username': username.strip()}

    def validate_email(self, email: str) -> Dict[str, Any]:
        """
        Validate email address.

        Args:
            email: Email to validate

        Returns:
            Validation result
        """
        if not email:
            return {'valid': False, 'error': 'Email is required'}

        if len(email) > 254:  # RFC 5321 limit
            return {'valid': False, 'error': 'Email is too long'}

        if not self.patterns['email'].match(email):
            return {'valid': False, 'error': 'Invalid email format'}

        return {'valid': True, 'email': email.strip().lower()}

    def validate_password(self, password: str, min_length: int = 8) -> Dict[str, Any]:
        """
        Validate password strength.

        Args:
            password: Password to validate
            min_length: Minimum password length

        Returns:
            Validation result
        """
        if not password:
            return {'valid': False, 'error': 'Password is required'}

        if len(password) < min_length:
            return {'valid': False, 'error': f'Password must be at least {min_length} characters'}

        if len(password) > 128:
            return {'valid': False, 'error': 'Password is too long'}

        # Check for common weak passwords
        weak_passwords = {'password', '123456', 'password123', 'admin', 'qwerty'}
        if password.lower() in weak_passwords:
            return {'valid': False, 'error': 'Password is too common'}

        # Check for character variety
        has_lower = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))

        strength_score = sum([has_lower, has_upper, has_digit, has_special])

        if strength_score < 3:
            return {'valid': False, 'error': 'Password must contain at least 3 different character types'}

        return {'valid': True, 'strength': strength_score}

    def validate_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Validate trading symbol.

        Args:
            symbol: Symbol to validate

        Returns:
            Validation result
        """
        if not symbol:
            return {'valid': False, 'error': 'Symbol is required'}

        if not self.patterns['symbol'].match(symbol.upper()):
            return {'valid': False, 'error': 'Invalid symbol format'}

        return {'valid': True, 'symbol': symbol.upper()}

    def validate_quantity(self, quantity: Union[str, float, int]) -> Dict[str, Any]:
        """
        Validate quantity value.

        Args:
            quantity: Quantity to validate

        Returns:
            Validation result
        """
        try:
            # Convert to string for validation
            if isinstance(quantity, (int, float)):
                quantity_str = str(quantity)
            else:
                quantity_str = quantity

            if not quantity_str:
                return {'valid': False, 'error': 'Quantity is required'}

            # Check if numeric
            if not self.patterns['numeric'].match(quantity_str):
                return {'valid': False, 'error': 'Quantity must be a valid number'}

            quantity_float = float(quantity_str)

            # Validate range
            if quantity_float <= 0:
                return {'valid': False, 'error': 'Quantity must be positive'}

            if quantity_float > 1000000:  # Reasonable upper limit
                return {'valid': False, 'error': 'Quantity is too large'}

            return {'valid': True, 'quantity': quantity_float}

        except (ValueError, TypeError):
            return {'valid': False, 'error': 'Invalid quantity format'}

    def validate_price(self, price: Union[str, float, int]) -> Dict[str, Any]:
        """
        Validate price value.

        Args:
            price: Price to validate

        Returns:
            Validation result
        """
        try:
            if isinstance(price, (int, float)):
                price_str = str(price)
            else:
                price_str = price

            if not price_str:
                return {'valid': False, 'error': 'Price is required'}

            if not self.patterns['numeric'].match(price_str):
                return {'valid': False, 'error': 'Price must be a valid number'}

            price_float = float(price_str)

            if price_float <= 0:
                return {'valid': False, 'error': 'Price must be positive'}

            if price_float > 100000:  # Reasonable upper limit for most stocks
                return {'valid': False, 'error': 'Price seems unreasonably high'}

            return {'valid': True, 'price': price_float}

        except (ValueError, TypeError):
            return {'valid': False, 'error': 'Invalid price format'}

    def validate_ip_address(self, ip: str) -> Dict[str, Any]:
        """
        Validate IP address.

        Args:
            ip: IP address to validate

        Returns:
            Validation result
        """
        if not ip:
            return {'valid': False, 'error': 'IP address is required'}

        # Check regex pattern first
        if not self.patterns['ip_address'].match(ip):
            return {'valid': False, 'error': 'Invalid IP address format'}

        try:
            # Validate as proper IP address
            ipaddress.IPv4Address(ip)
            return {'valid': True, 'ip': ip}
        except ipaddress.AddressValueError:
            return {'valid': False, 'error': 'Invalid IP address'}

    def validate_json(self, json_str: str) -> Dict[str, Any]:
        """
        Validate and parse JSON input.

        Args:
            json_str: JSON string to validate

        Returns:
            Validation result with parsed data
        """
        if not json_str:
            return {'valid': False, 'error': 'JSON data is required'}

        if len(json_str) > 1024 * 1024:  # 1MB limit
            return {'valid': False, 'error': 'JSON data is too large'}

        try:
            # Parse JSON
            data = json.loads(json_str)

            # Basic structure validation
            if not isinstance(data, (dict, list)):
                return {'valid': False, 'error': 'JSON must be an object or array'}

            return {'valid': True, 'data': data}

        except json.JSONDecodeError as e:
            return {'valid': False, 'error': f'Invalid JSON format: {str(e)}'}

    def validate_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete trade data.

        Args:
            trade_data: Trade data to validate

        Returns:
            Validation result
        """
        errors = []
        warnings = []

        # Validate required fields
        required_fields = ['symbol', 'side', 'quantity', 'price']
        for field in required_fields:
            if field not in trade_data:
                errors.append(f"Missing required field: {field}")

        if errors:
            return {'valid': False, 'errors': errors, 'warnings': warnings}

        # Validate symbol
        symbol_result = self.validate_symbol(trade_data['symbol'])
        if not symbol_result['valid']:
            errors.append(f"Symbol: {symbol_result['error']}")

        # Validate side
        side = trade_data.get('side', '').lower()
        if side not in ['buy', 'sell']:
            errors.append("Side must be 'buy' or 'sell'")
        else:
            trade_data['side'] = side  # Normalize

        # Validate quantity
        quantity_result = self.validate_quantity(trade_data['quantity'])
        if not quantity_result['valid']:
            errors.append(f"Quantity: {quantity_result['error']}")

        # Validate price
        price_result = self.validate_price(trade_data['price'])
        if not price_result['valid']:
            errors.append(f"Price: {price_result['error']}")

        # Validate optional fields
        if 'strategy' in trade_data:
            strategy = trade_data['strategy']
            if len(strategy) > 50:
                warnings.append("Strategy name is unusually long")

        if 'timestamp' in trade_data:
            try:
                datetime.fromisoformat(trade_data['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                warnings.append("Timestamp format may be invalid")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'trade_data': trade_data if len(errors) == 0 else None
        }

    def validate_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate API request data.

        Args:
            request_data: API request data

        Returns:
            Validation result
        """
        errors = []
        warnings = []

        # Check for suspicious patterns
        json_str = json.dumps(request_data)

        # Check for SQL injection attempts
        for pattern in self.sql_injection_patterns:
            if pattern.search(json_str):
                errors.append("Potential SQL injection detected")

        # Check for XSS attempts
        for pattern in self.xss_patterns:
            if pattern.search(json_str):
                errors.append("Potential XSS attack detected")

        # Check request size
        if len(json_str) > 1024 * 1024:  # 1MB limit
            errors.append("Request data is too large")

        # Validate common fields
        if 'user_id' in request_data:
            user_id = request_data['user_id']
            if len(user_id) > 100:
                errors.append("User ID is too long")

        if 'session_id' in request_data:
            session_id = request_data['session_id']
            if not self.patterns['alphanumeric'].match(session_id):
                errors.append("Invalid session ID format")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'sanitized_data': self._sanitize_request_data(request_data) if len(errors) == 0 else None
        }

    def _sanitize_request_data(self, data: Any) -> Any:
        """
        Recursively sanitize request data.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            return self.sanitize_string(data)
        elif isinstance(data, dict):
            return {key: self._sanitize_request_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_request_data(item) for item in data]
        else:
            return data

    def validate_date_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Validate date range.

        Args:
            start_date: Start date string
            end_date: End date string

        Returns:
            Validation result
        """
        try:
            # Parse dates
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

            # Validate range
            if start >= end:
                return {'valid': False, 'error': 'Start date must be before end date'}

            # Check for reasonable range (max 1 year)
            if end - start > timedelta(days=365):
                return {'valid': False, 'error': 'Date range cannot exceed 1 year'}

            # Check if end date is not in future
            if end > datetime.utcnow():
                return {'valid': False, 'error': 'End date cannot be in the future'}

            return {
                'valid': True,
                'start_date': start,
                'end_date': end
            }

        except (ValueError, TypeError):
            return {'valid': False, 'error': 'Invalid date format'}

    def validate_pagination_params(self, limit: Union[str, int], offset: Union[str, int]) -> Dict[str, Any]:
        """
        Validate pagination parameters.

        Args:
            limit: Page size limit
            offset: Page offset

        Returns:
            Validation result
        """
        try:
            # Convert to integers
            limit_int = int(limit) if limit else 100
            offset_int = int(offset) if offset else 0

            # Validate ranges
            if limit_int < 1:
                return {'valid': False, 'error': 'Limit must be at least 1'}

            if limit_int > 1000:
                return {'valid': False, 'error': 'Limit cannot exceed 1000'}

            if offset_int < 0:
                return {'valid': False, 'error': 'Offset cannot be negative'}

            return {
                'valid': True,
                'limit': limit_int,
                'offset': offset_int
            }

        except (ValueError, TypeError):
            return {'valid': False, 'error': 'Invalid pagination parameters'}

    def detect_threats(self, input_data: str) -> Dict[str, Any]:
        """
        Detect potential security threats in input.

        Args:
            input_data: Input data to analyze

        Returns:
            Threat detection result
        """
        threats = {
            'sql_injection': False,
            'xss_attempts': False,
            'command_injection': False,
            'path_traversal': False,
            'ldap_injection': False,
            'threat_level': 'low'
        }

        if not isinstance(input_data, str):
            return threats

        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if pattern.search(input_data):
                threats['sql_injection'] = True

        # Check for XSS
        for pattern in self.xss_patterns:
            if pattern.search(input_data):
                threats['xss_attempts'] = True

        # Check for command injection
        command_patterns = [
            re.compile(r'[;&|`$()]'),
            re.compile(r'\$\(.*\)'),
            re.compile(r'`.*`'),
        ]

        for pattern in command_patterns:
            if pattern.search(input_data):
                threats['command_injection'] = True

        # Check for path traversal
        if '../' in input_data or '..\\' in input_data:
            threats['path_traversal'] = True

        # Check for LDAP injection
        ldap_patterns = [
            re.compile(r'\*\(|!\('),
            re.compile(r'\|.*\|'),
        ]

        for pattern in ldap_patterns:
            if pattern.search(input_data):
                threats['ldap_injection'] = True

        # Assess threat level
        threat_count = sum(threats.values())
        if threat_count >= 3:
            threats['threat_level'] = 'critical'
        elif threat_count >= 2:
            threats['threat_level'] = 'high'
        elif threat_count >= 1:
            threats['threat_level'] = 'medium'

        return threats

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe file operations.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        if not filename:
            return "unnamed_file"

        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]

        # Remove dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255-len(ext)-1] + '.' + ext

        return filename.strip() or "unnamed_file"

    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate URL format and safety.

        Args:
            url: URL to validate

        Returns:
            Validation result
        """
        if not url:
            return {'valid': False, 'error': 'URL is required'}

        if len(url) > 2048:
            return {'valid': False, 'error': 'URL is too long'}

        if not self.patterns['url'].match(url):
            return {'valid': False, 'error': 'Invalid URL format'}

        # Check for suspicious URLs
        suspicious_patterns = [
            'file://',
            'ftp://',
            'data:',
            'javascript:',
            'vbscript:',
        ]

        for pattern in suspicious_patterns:
            if pattern in url.lower():
                return {'valid': False, 'error': 'Suspicious URL detected'}

        return {'valid': True, 'url': url.strip()}

# Global input validator instance
input_validator = InputValidator()

# Convenience functions for common validations
def validate_login_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate login request data."""
    errors = []

    username_result = input_validator.validate_username(data.get('username', ''))
    if not username_result['valid']:
        errors.append(f"Username: {username_result['error']}")

    password_result = input_validator.validate_password(data.get('password', ''))
    if not password_result['valid']:
        errors.append(f"Password: {password_result['error']}")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'data': data if len(errors) == 0 else None
    }

def validate_trade_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate trade request data."""
    return input_validator.validate_trade_data(data)

def sanitize_user_input(text: str) -> str:
    """Sanitize general user input."""
    return input_validator.sanitize_string(text)

def detect_input_threats(text: str) -> Dict[str, Any]:
    """Detect threats in user input."""
    return input_validator.detect_threats(text)
