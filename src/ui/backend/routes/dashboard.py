"""
Dashboard Routes for PDT Trading Bot Admin UI

This module provides dashboard-specific endpoints including PDT tracking,
compliance status, and emergency controls.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel
import json

from ..app import get_current_user

# Initialize router
dashboard_router = APIRouter()

# Dashboard data models
class PDTStatus(BaseModel):
    """PDT status model."""
    is_pdt_account: bool
    day_trades_remaining: int
    day_trades_used: int
    volume_towards_threshold: float
    threshold_remaining: float
    reset_date: str
    status: str  # 'compliant', 'warning', 'violation'

class ComplianceStatus(BaseModel):
    """Compliance status model."""
    overall_status: str  # 'compliant', 'warning', 'violation'
    pdt_status: PDTStatus
    regulatory_flags: List[str]
    last_audit: str
    risk_score: float

class EmergencyControls(BaseModel):
    """Emergency controls model."""
    kill_switch_active: bool
    auto_stop_enabled: bool
    max_loss_threshold: float
    max_position_size: float
    circuit_breakers: Dict[str, bool]

class StrategyConfig(BaseModel):
    """Strategy configuration model."""
    strategy_name: str
    is_active: bool
    parameters: Dict[str, Any]
    last_updated: str

class AlertConfig(BaseModel):
    """Alert configuration model."""
    email_alerts: bool
    sms_alerts: bool
    webhook_alerts: bool
    alert_thresholds: Dict[str, float]

# Mock PDT and compliance data (in production, integrate with compliance modules)
MOCK_PDT_STATUS = {
    "is_pdt_account": True,
    "day_trades_remaining": 2,
    "day_trades_used": 1,
    "volume_towards_threshold": 18500.00,
    "threshold_remaining": 6500.00,
    "reset_date": "2024-01-02T00:00:00Z",
    "status": "warning"
}

MOCK_COMPLIANCE_STATUS = {
    "overall_status": "warning",
    "pdt_status": MOCK_PDT_STATUS,
    "regulatory_flags": ["wash_trade_warning"],
    "last_audit": "2024-01-01T08:00:00Z",
    "risk_score": 0.75
}

MOCK_EMERGENCY_CONTROLS = {
    "kill_switch_active": False,
    "auto_stop_enabled": True,
    "max_loss_threshold": -1000.00,
    "max_position_size": 10000.00,
    "circuit_breakers": {
        "max_daily_loss": False,
        "max_position_size": False,
        "unusual_volume": False,
        "price_circuit": False
    }
}

MOCK_STRATEGY_CONFIGS = [
    {
        "strategy_name": "momentum",
        "is_active": True,
        "parameters": {
            "rsi_threshold": 70,
            "position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.05
        },
        "last_updated": "2024-01-01T10:00:00Z"
    },
    {
        "strategy_name": "rsi_reversal",
        "is_active": True,
        "parameters": {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "position_size": 0.08,
            "stop_loss": 0.025
        },
        "last_updated": "2024-01-01T09:30:00Z"
    }
]

MOCK_ALERT_CONFIG = {
    "email_alerts": True,
    "sms_alerts": False,
    "webhook_alerts": True,
    "alert_thresholds": {
        "daily_loss": -500.00,
        "position_size": 15000.00,
        "unusual_activity": 0.8,
        "compliance_violation": 0.0
    }
}

@dashboard_router.get("/pdt-status", response_model=PDTStatus)
async def get_pdt_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PDTStatus:
    """
    Get PDT (Pattern Day Trader) status and compliance information.

    This endpoint returns current PDT status including day trades
    remaining, volume towards threshold, and compliance state.
    """
    try:
        # In production, integrate with compliance/pdt/pdt_tracker.py
        return PDTStatus(**MOCK_PDT_STATUS)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving PDT status: {str(e)}"
        )

@dashboard_router.get("/compliance", response_model=ComplianceStatus)
async def get_compliance_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ComplianceStatus:
    """
    Get overall compliance status and regulatory information.

    This endpoint returns comprehensive compliance information
    including PDT status, regulatory flags, and risk scores.
    """
    try:
        # In production, integrate with compliance modules
        return ComplianceStatus(**MOCK_COMPLIANCE_STATUS)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving compliance status: {str(e)}"
        )

@dashboard_router.get("/emergency-controls", response_model=EmergencyControls)
async def get_emergency_controls(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> EmergencyControls:
    """
    Get emergency controls and circuit breaker status.

    This endpoint returns the current state of emergency controls
    and circuit breakers for risk management.
    """
    try:
        return EmergencyControls(**MOCK_EMERGENCY_CONTROLS)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving emergency controls: {str(e)}"
        )

@dashboard_router.post("/emergency/kill-switch")
async def activate_kill_switch(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Activate emergency kill switch to stop all trading.

    This endpoint immediately stops all trading activities
    and closes positions if necessary.
    """
    try:
        # Check permissions
        user_data = None
        for username, data in USERS_DB.items():
            if data["user_id"] == current_user["user_id"]:
                user_data = data
                break

        if not user_data or "emergency_stop" not in user_data.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Emergency stop permissions required"
            )

        # In production, integrate with bot core to stop all trading
        global MOCK_EMERGENCY_CONTROLS, MOCK_BOT_STATUS

        MOCK_EMERGENCY_CONTROLS = {
            **MOCK_EMERGENCY_CONTROLS,
            "kill_switch_active": True
        }

        MOCK_BOT_STATUS = {
            **MOCK_BOT_STATUS,
            "is_running": False,
            "status": "emergency_stop",
            "error_message": "Emergency kill switch activated"
        }

        return {
            "message": "Emergency kill switch activated successfully",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "emergency_stop"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activating kill switch: {str(e)}"
        )

@dashboard_router.post("/emergency/circuit-breaker")
async def toggle_circuit_breaker(
    breaker_type: str = Query(..., regex="^(max_daily_loss|max_position_size|unusual_volume|price_circuit)$"),
    enabled: bool = Query(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Toggle circuit breaker settings.

    This endpoint allows enabling/disabling specific circuit
    breakers for risk management.
    """
    try:
        # Check permissions
        user_data = None
        for username, data in USERS_DB.items():
            if data["user_id"] == current_user["user_id"]:
                user_data = data
                break

        if not user_data or "write" not in user_data.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Write permissions required"
            )

        # Update circuit breaker status
        global MOCK_EMERGENCY_CONTROLS

        MOCK_EMERGENCY_CONTROLS = {
            **MOCK_EMERGENCY_CONTROLS,
            "circuit_breakers": {
                **MOCK_EMERGENCY_CONTROLS["circuit_breakers"],
                breaker_type: enabled
            }
        }

        return {
            "message": f"Circuit breaker '{breaker_type}' {'enabled' if enabled else 'disabled'}",
            "breaker_type": breaker_type,
            "enabled": enabled,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error toggling circuit breaker: {str(e)}"
        )

@dashboard_router.get("/strategies", response_model=List[StrategyConfig])
async def get_strategy_configs(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[StrategyConfig]:
    """
    Get strategy configurations and status.

    This endpoint returns all configured trading strategies
    with their parameters and activation status.
    """
    try:
        return [StrategyConfig(**config) for config in MOCK_STRATEGY_CONFIGS]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving strategy configs: {str(e)}"
        )

@dashboard_router.put("/strategies/{strategy_name}")
async def update_strategy_config(
    strategy_name: str,
    parameters: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update strategy parameters.

    This endpoint allows updating strategy parameters
    for active trading strategies.
    """
    try:
        # Check permissions
        user_data = None
        for username, data in USERS_DB.items():
            if data["user_id"] == current_user["user_id"]:
                user_data = data
                break

        if not user_data or "write" not in user_data.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Write permissions required"
            )

        # Find and update strategy
        strategy_found = False
        for strategy in MOCK_STRATEGY_CONFIGS:
            if strategy["strategy_name"] == strategy_name:
                strategy["parameters"].update(parameters)
                strategy["last_updated"] = datetime.utcnow().isoformat()
                strategy_found = True
                break

        if not strategy_found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy '{strategy_name}' not found"
            )

        # In production, integrate with bot core to update strategy
        return {
            "message": f"Strategy '{strategy_name}' updated successfully",
            "strategy_name": strategy_name,
            "updated_parameters": parameters,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating strategy: {str(e)}"
        )

@dashboard_router.get("/alerts/config", response_model=AlertConfig)
async def get_alert_config(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> AlertConfig:
    """
    Get alert configuration settings.

    This endpoint returns current alert configuration
    including notification preferences and thresholds.
    """
    try:
        return AlertConfig(**MOCK_ALERT_CONFIG)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving alert config: {str(e)}"
        )

@dashboard_router.put("/alerts/config")
async def update_alert_config(
    config: AlertConfig,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Update alert configuration settings.

    This endpoint allows updating alert preferences and
    notification thresholds.
    """
    try:
        # Check permissions
        user_data = None
        for username, data in USERS_DB.items():
            if data["user_id"] == current_user["user_id"]:
                user_data = data
                break

        if not user_data or "write" not in user_data.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Write permissions required"
            )

        # Update alert configuration
        global MOCK_ALERT_CONFIG
        MOCK_ALERT_CONFIG = config.dict()

        return {
            "message": "Alert configuration updated successfully",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating alert config: {str(e)}"
        )

@dashboard_router.get("/pdt-progress")
async def get_pdt_progress(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed PDT progress tracking data.

    This endpoint returns comprehensive PDT progress information
    for visualization and monitoring.
    """
    try:
        # Generate mock progress data for the last 30 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        progress_data = []
        current_date = start_date
        current_volume = 0

        while current_date <= end_date:
            # Simulate daily volume accumulation
            daily_volume = 500 + (hash(str(current_date.date())) % 1000)
            current_volume += daily_volume

            progress_data.append({
                "date": current_date.date().isoformat(),
                "daily_volume": round(daily_volume, 2),
                "cumulative_volume": round(current_volume, 2),
                "threshold_remaining": max(0, 25000 - current_volume),
                "progress_percentage": min(100, (current_volume / 25000) * 100)
            })

            current_date += timedelta(days=1)

        return {
            "current_status": MOCK_PDT_STATUS,
            "progress_history": progress_data,
            "target_threshold": 25000.00,
            "days_tracked": 30
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving PDT progress: {str(e)}"
        )

@dashboard_router.get("/audit-log")
async def get_audit_log(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get audit log entries for compliance and monitoring.

    This endpoint returns audit trail information for
    compliance purposes and activity monitoring.
    """
    try:
        # Check permissions
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

        # Generate mock audit log entries
        audit_entries = []
        base_time = datetime.utcnow()

        for i in range(offset, min(offset + limit, 100)):  # Max 100 mock entries
            entry_time = base_time - timedelta(minutes=i * 5)

            audit_entries.append({
                "id": f"audit_{i:03d}",
                "timestamp": entry_time.isoformat(),
                "user_id": f"user_{(i % 3) + 1}",
                "action": ["login", "logout", "trade", "config_change", "emergency_stop"][i % 5],
                "resource": ["dashboard", "strategy", "position", "settings"][i % 4],
                "details": f"Mock audit entry {i}",
                "ip_address": f"192.168.1.{(i % 254) + 1}",
                "user_agent": "Mozilla/5.0 (Mock Browser)"
            })

        return {
            "entries": audit_entries,
            "total_count": 100,
            "limit": limit,
            "offset": offset
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving audit log: {str(e)}"
        )

# Import here to avoid circular imports
from ..auth.jwt_handler import JWTHandler

# Initialize JWT handler for user database access
jwt_handler = JWTHandler()

# In-memory user store (same as in auth.py)
USERS_DB = {
    "admin": {
        "user_id": "admin_001",
        "username": "admin",
        "password_hash": jwt_handler.hash_password("change_me_in_production"),
        "email": "admin@localhost",
        "role": "admin",
        "permissions": ["read", "write", "admin", "emergency_stop"],
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z"
    }
}
