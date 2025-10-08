"""
Main FastAPI Application for PDT Trading Bot Admin UI

This module serves as the entry point for the admin dashboard backend,
providing REST API endpoints, WebSocket connections, and authentication.
"""

import os
import secrets
from datetime import timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.websockets import WebSocket, WebSocketDisconnect
import uvicorn

from ..auth.jwt_handler import JWTHandler
from ..auth.session_manager import SessionManager
from .routes import auth, data, dashboard
from .websocket.connection import WebSocketManager

# Initialize core components
jwt_handler = JWTHandler()
session_manager = SessionManager()
websocket_manager = WebSocketManager()

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Create FastAPI application
app = FastAPI(
    title="PDT Trading Bot Admin Dashboard",
    description="Secure admin interface for monitoring PDT trading bot",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") == "development" else None,
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"] + os.getenv("ALLOWED_HOSTS", "").split(",")
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security scheme for JWT authentication
security = HTTPBearer(auto_error=False)

# Mount static files
app.mount("/static", StaticFiles(directory="src/ui/frontend/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="src/ui/frontend/templates")

# WebSocket endpoint for real-time updates
@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket endpoint for real-time dashboard updates."""
    # Authenticate WebSocket connection
    if not token:
        await websocket.close(code=1008, reason="Authentication required")
        return
    
    try:
        payload = jwt_handler.verify_token(token)
        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=1008, reason="Invalid token")
            return
        
        # Accept connection and add to active connections
        await websocket_manager.connect(websocket, user_id)
        
        try:
            while True:
                # Receive messages from client (if needed)
                data = await websocket.receive_text()
                
                # Handle client messages here if needed
                # For now, just maintain connection for server updates
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(user_id)
            
    except Exception as e:
        await websocket.close(code=1011, reason=f"Server error: {str(e)}")

# Authentication middleware
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt_handler.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"user_id": user_id, "token": credentials.credentials}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "admin-ui"}

# Root endpoint - serve dashboard for authenticated users
@app.get("/")
async def root(request: Request, current_user: dict = Depends(get_current_user)):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Login page endpoint
@app.get("/login")
async def login_page(request: Request):
    """Serve the login page."""
    return templates.TemplateResponse("login.html", {"request": request})

# Logout endpoint
@app.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user and invalidate session."""
    session_manager.invalidate_session(current_user["user_id"])
    return {"message": "Successfully logged out"}

# Include API routes
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(data.router, prefix="/api/data", tags=["trading-data"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])

# Global error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper error responses."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

# Background task for real-time data broadcasting
async def broadcast_trading_data():
    """Background task to broadcast real-time trading data to connected clients."""
    import asyncio
    
    while True:
        try:
            # Get latest trading data from bot core
            # This would integrate with the actual bot core
            trading_data = {
                "timestamp": "2024-01-01T00:00:00Z",
                "pnl": 0.0,
                "positions": [],
                "risk_metrics": {}
            }
            
            # Broadcast to all connected clients
            await websocket_manager.broadcast("trading_data", trading_data)
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            print(f"Error in broadcast task: {e}")
            await asyncio.sleep(5)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print("Starting PDT Trading Bot Admin UI...")
    
    # Start background tasks
    import asyncio
    asyncio.create_task(broadcast_trading_data())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    print("Shutting down PDT Trading Bot Admin UI...")
    
    # Close all WebSocket connections
    await websocket_manager.close_all()

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    
    # Run the application
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE")
    )
