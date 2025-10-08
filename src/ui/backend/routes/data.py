"""
Trading Data Routes for PDT Trading Bot Admin UI

This module provides API endpoints for accessing trading data,
bot status, market information, and real-time updates.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, validator
import asyncio

from ..app import get_current_user

# Initialize router
data_router = APIRouter()

# Data models
class TradeData(BaseModel):
    """Trade data model."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: str
    pnl: float
    strategy: str
    status: str

class PositionData(BaseModel):
    """Position data model."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float

class PnLData(BaseModel):
    """P&L data model."""
    daily_pnl: float
    total_pnl: float
    daily_return: float
    total_return: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float

class RiskMetrics(BaseModel):
    """Risk metrics model."""
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    beta: float
    volatility: float
    exposure: float
    leverage: float

class MarketData(BaseModel):
    """Market data model."""
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    timestamp: str

class BotStatus(BaseModel):
    """Bot status model."""
    is_running: bool
    status: str  # 'active', 'paused', 'stopped', 'error'
    uptime: float
    last_update: str
    active_strategies: List[str]
    error_message: Optional[str] = None

# Mock data storage (in production, integrate with actual bot core)
MOCK_TRADES = [
    {
        "trade_id": "trade_001",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "price": 150.25,
        "timestamp": "2024-01-01T10:30:00Z",
        "pnl": 25.50,
        "strategy": "momentum",
        "status": "closed"
    },
    {
        "trade_id": "trade_002",
        "symbol": "TSLA",
        "side": "sell",
        "quantity": 50,
        "price": 245.80,
        "timestamp": "2024-01-01T11:15:00Z",
        "pnl": -12.30,
        "strategy": "rsi_reversal",
        "status": "closed"
    }
]

MOCK_POSITIONS = [
    {
        "symbol": "AAPL",
        "quantity": 100,
        "avg_price": 150.25,
        "current_price": 152.75,
        "unrealized_pnl": 250.00,
        "realized_pnl": 25.50,
        "market_value": 15275.00
    },
    {
        "symbol": "MSFT",
        "quantity": 75,
        "avg_price": 280.50,
        "current_price": 285.25,
        "unrealized_pnl": 356.25,
        "realized_pnl": 0.00,
        "market_value": 21393.75
    }
]

MOCK_PNL_DATA = {
    "daily_pnl": 614.45,
    "total_pnl": 15420.75,
    "daily_return": 0.0314,
    "total_return": 0.1542,
    "win_rate": 0.68,
    "sharpe_ratio": 1.45,
    "max_drawdown": -0.082
}

MOCK_RISK_METRICS = {
    "var_95": -1250.50,
    "var_99": -1850.75,
    "beta": 1.12,
    "volatility": 0.18,
    "exposure": 36668.75,
    "leverage": 1.25
}

MOCK_MARKET_DATA = {
    "AAPL": {
        "symbol": "AAPL",
        "price": 152.75,
        "volume": 1250000,
        "bid": 152.70,
        "ask": 152.80,
        "timestamp": "2024-01-01T12:00:00Z"
    },
    "TSLA": {
        "symbol": "TSLA",
        "price": 245.80,
        "volume": 850000,
        "bid": 245.75,
        "ask": 245.85,
        "timestamp": "2024-01-01T12:00:00Z"
    }
}

MOCK_BOT_STATUS = {
    "is_running": True,
    "status": "active",
    "uptime": 3600.5,  # seconds
    "last_update": "2024-01-01T12:00:00Z",
    "active_strategies": ["momentum", "rsi_reversal"],
    "error_message": None
}

@data_router.get("/trades", response_model=List[TradeData])
async def get_trades(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[TradeData]:
    """
    Get trade history with filtering and pagination.

    This endpoint returns historical trade data with support for
    filtering by symbol, strategy, and date range.
    """
    try:
        # Filter trades based on query parameters
        filtered_trades = MOCK_TRADES.copy()

        if symbol:
            filtered_trades = [t for t in filtered_trades if t["symbol"] == symbol]

        if strategy:
            filtered_trades = [t for t in filtered_trades if t["strategy"] == strategy]

        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            filtered_trades = [
                t for t in filtered_trades
                if datetime.fromisoformat(t["timestamp"].replace('Z', '+00:00')) >= start_dt
            ]

        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            filtered_trades = [
                t for t in filtered_trades
                if datetime.fromisoformat(t["timestamp"].replace('Z', '+00:00')) <= end_dt
            ]

        # Apply pagination
        paginated_trades = filtered_trades[offset:offset + limit]

        return [TradeData(**trade) for trade in paginated_trades]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving trades: {str(e)}"
        )

@data_router.get("/trades/{trade_id}", response_model=TradeData)
async def get_trade(
    trade_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TradeData:
    """
    Get detailed information about a specific trade.

    This endpoint returns detailed information about a single trade
    including execution details and performance metrics.
    """
    try:
        trade = next((t for t in MOCK_TRADES if t["trade_id"] == trade_id), None)

        if not trade:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trade not found"
            )

        return TradeData(**trade)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving trade: {str(e)}"
        )

@data_router.get("/positions", response_model=List[PositionData])
async def get_positions(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[PositionData]:
    """
    Get current positions.

    This endpoint returns all current open positions with
    real-time P&L and market values.
    """
    try:
        return [PositionData(**position) for position in MOCK_POSITIONS]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving positions: {str(e)}"
        )

@data_router.get("/pnl", response_model=PnLData)
async def get_pnl_data(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PnLData:
    """
    Get P&L and performance metrics.

    This endpoint returns comprehensive P&L data including
    daily and total returns, Sharpe ratio, and other metrics.
    """
    try:
        return PnLData(**MOCK_PNL_DATA)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving P&L data: {str(e)}"
        )

@data_router.get("/risk-metrics", response_model=RiskMetrics)
async def get_risk_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> RiskMetrics:
    """
    Get risk metrics and exposure data.

    This endpoint returns risk-related metrics including
    VaR, beta, volatility, and current exposure.
    """
    try:
        return RiskMetrics(**MOCK_RISK_METRICS)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving risk metrics: {str(e)}"
        )

@data_router.get("/market/{symbol}", response_model=MarketData)
async def get_market_data(
    symbol: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MarketData:
    """
    Get real-time market data for a symbol.

    This endpoint returns current market data including
    price, volume, bid/ask, and timestamp.
    """
    try:
        market_data = MOCK_MARKET_DATA.get(symbol.upper())

        if not market_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Market data not found for symbol: {symbol}"
            )

        return MarketData(**market_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market data: {str(e)}"
        )

@data_router.get("/bot-status", response_model=BotStatus)
async def get_bot_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> BotStatus:
    """
    Get current bot status and operational state.

    This endpoint returns the current operational status of
    the trading bot including uptime and active strategies.
    """
    try:
        return BotStatus(**MOCK_BOT_STATUS)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving bot status: {str(e)}"
        )

@data_router.get("/dashboard-summary")
async def get_dashboard_summary(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive dashboard summary data.

    This endpoint returns a summary of all key metrics and
    data needed for the main dashboard view.
    """
    try:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "bot_status": MOCK_BOT_STATUS,
            "pnl_data": MOCK_PNL_DATA,
            "risk_metrics": MOCK_RISK_METRICS,
            "positions": MOCK_POSITIONS,
            "recent_trades": MOCK_TRADES[:5],  # Last 5 trades
            "market_data": MOCK_MARKET_DATA
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving dashboard summary: {str(e)}"
        )

@data_router.get("/performance-history")
async def get_performance_history(
    days: int = Query(30, ge=1, le=365),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get historical performance data for charting.

    This endpoint returns historical performance data for
    generating charts and analyzing trends.
    """
    try:
        # Generate mock historical data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        historical_data = []
        current_date = start_date

        while current_date <= end_date:
            # Generate mock daily P&L data
            daily_pnl = MOCK_PNL_DATA["daily_pnl"] + (hash(str(current_date.date())) % 1000 - 500)

            historical_data.append({
                "date": current_date.date().isoformat(),
                "daily_pnl": round(daily_pnl, 2),
                "cumulative_pnl": sum(item["daily_pnl"] for item in historical_data) + daily_pnl,
                "daily_return": round(daily_pnl / 10000, 4)  # Assuming $10k portfolio
            })

            current_date += timedelta(days=1)

        return {
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "data_points": len(historical_data),
            "performance_data": historical_data
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving performance history: {str(e)}"
        )

@data_router.post("/bot/control")
async def control_bot(
    action: str = Query(..., regex="^(start|stop|pause|resume)$"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Control bot operations (start, stop, pause, resume).

    This endpoint allows authorized users to control the
    trading bot's operational state.
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
                detail="Insufficient permissions to control bot"
            )

        # In a real implementation, this would interface with the bot core
        # For now, we'll just simulate the action

        global MOCK_BOT_STATUS

        if action == "start":
            MOCK_BOT_STATUS = {
                **MOCK_BOT_STATUS,
                "is_running": True,
                "status": "active",
                "error_message": None
            }
        elif action == "stop":
            MOCK_BOT_STATUS = {
                **MOCK_BOT_STATUS,
                "is_running": False,
                "status": "stopped",
                "error_message": None
            }
        elif action == "pause":
            MOCK_BOT_STATUS = {
                **MOCK_BOT_STATUS,
                "is_running": True,
                "status": "paused",
                "error_message": None
            }
        elif action == "resume":
            MOCK_BOT_STATUS = {
                **MOCK_BOT_STATUS,
                "is_running": True,
                "status": "active",
                "error_message": None
            }

        return {
            "message": f"Bot {action} command executed successfully",
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error controlling bot: {str(e)}"
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
