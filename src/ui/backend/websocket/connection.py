"""
WebSocket Connection Manager for PDT Trading Bot Admin UI

This module provides WebSocket connection management for real-time
data streaming and live updates to the admin dashboard.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import logging

logger = logging.getLogger(__name__)

class WebSocketConnection:
    """Represents a single WebSocket connection with metadata."""

    def __init__(self, websocket: WebSocket, user_id: str, client_id: str):
        """Initialize WebSocket connection."""
        self.websocket = websocket
        self.user_id = user_id
        self.client_id = client_id
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.is_authenticated = True
        self.subscriptions: List[str] = []

    async def send_message(self, message_type: str, data: Any) -> bool:
        """
        Send a message to this connection.

        Args:
            message_type: Type of message (e.g., 'trading_data', 'alert')
            data: Message payload

        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }

            await self.websocket.send_text(json.dumps(message))
            self.last_ping = datetime.utcnow()
            return True

        except Exception as e:
            logger.error(f"Error sending message to {self.client_id}: {e}")
            return False

    async def ping(self) -> bool:
        """Send ping to check connection health."""
        try:
            await self.websocket.ping()
            self.last_ping = datetime.utcnow()
            return True
        except Exception:
            return False

    def is_alive(self, timeout_seconds: int = 30) -> bool:
        """Check if connection is alive based on last ping."""
        return (datetime.utcnow() - self.last_ping).seconds < timeout_seconds

class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, List[str]] = {}  # user_id -> [client_ids]
        self.subscriptions: Dict[str, List[str]] = {}  # subscription_type -> [client_ids]

        # Connection counters
        self.total_connections = 0
        self.message_count = 0

        # Cleanup settings
        self.cleanup_interval = 30  # seconds
        self.connection_timeout = 60  # seconds

        # Start cleanup task
        self._start_cleanup_task()

    def _generate_client_id(self) -> str:
        """Generate unique client ID."""
        self.total_connections += 1
        return f"client_{self.total_connections}_{datetime.utcnow().strftime('%H%M%S')}"

    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """
        Accept new WebSocket connection.

        Args:
            websocket: WebSocket connection
            user_id: Authenticated user ID

        Returns:
            Client ID for this connection
        """
        await websocket.accept()

        client_id = self._generate_client_id()
        connection = WebSocketConnection(websocket, user_id, client_id)

        self.active_connections[client_id] = connection

        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(client_id)

        logger.info(f"WebSocket connection established: {client_id} for user {user_id}")
        return client_id

    def disconnect(self, client_id: str) -> None:
        """
        Disconnect and cleanup WebSocket connection.

        Args:
            client_id: Client ID to disconnect
        """
        if client_id not in self.active_connections:
            return

        connection = self.active_connections[client_id]
        user_id = connection.user_id

        # Remove from active connections
        del self.active_connections[client_id]

        # Remove from user connections
        if user_id in self.user_connections:
            if client_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(client_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        # Remove from subscriptions
        for subscription_list in self.subscriptions.values():
            if client_id in subscription_list:
                subscription_list.remove(client_id)

        logger.info(f"WebSocket connection disconnected: {client_id}")

    async def subscribe(self, client_id: str, subscription_type: str) -> bool:
        """
        Subscribe a client to a specific data type.

        Args:
            client_id: Client ID
            subscription_type: Type of data to subscribe to

        Returns:
            True if subscribed successfully, False otherwise
        """
        if client_id not in self.active_connections:
            return False

        if subscription_type not in self.subscriptions:
            self.subscriptions[subscription_type] = []

        if client_id not in self.subscriptions[subscription_type]:
            self.subscriptions[subscription_type].append(client_id)

        # Add to connection subscriptions
        connection = self.active_connections[client_id]
        if subscription_type not in connection.subscriptions:
            connection.subscriptions.append(subscription_type)

        logger.debug(f"Client {client_id} subscribed to {subscription_type}")
        return True

    async def unsubscribe(self, client_id: str, subscription_type: str) -> bool:
        """
        Unsubscribe a client from a specific data type.

        Args:
            client_id: Client ID
            subscription_type: Type of data to unsubscribe from

        Returns:
            True if unsubscribed successfully, False otherwise
        """
        if client_id not in self.active_connections:
            return False

        # Remove from global subscriptions
        if subscription_type in self.subscriptions:
            if client_id in self.subscriptions[subscription_type]:
                self.subscriptions[subscription_type].remove(client_id)

        # Remove from connection subscriptions
        connection = self.active_connections[client_id]
        if subscription_type in connection.subscriptions:
            connection.subscriptions.remove(subscription_type)

        logger.debug(f"Client {client_id} unsubscribed from {subscription_type}")
        return True

    async def broadcast(self, message_type: str, data: Any,
                       exclude_client: Optional[str] = None) -> int:
        """
        Broadcast message to all subscribed clients.

        Args:
            message_type: Type of message
            data: Message payload
            exclude_client: Optional client ID to exclude

        Returns:
            Number of clients message was sent to
        """
        if message_type not in self.subscriptions:
            return 0

        subscribers = self.subscriptions[message_type].copy()
        if exclude_client and exclude_client in subscribers:
            subscribers.remove(exclude_client)

        sent_count = 0
        failed_clients = []

        for client_id in subscribers:
            if client_id in self.active_connections:
                connection = self.active_connections[client_id]
                if await connection.send_message(message_type, data):
                    sent_count += 1
                else:
                    failed_clients.append(client_id)

        # Remove failed connections
        for client_id in failed_clients:
            self.disconnect(client_id)

        self.message_count += sent_count
        logger.debug(f"Broadcast '{message_type}' sent to {sent_count} clients")
        return sent_count

    async def send_to_user(self, user_id: str, message_type: str, data: Any) -> int:
        """
        Send message to all connections for a specific user.

        Args:
            user_id: Target user ID
            message_type: Type of message
            data: Message payload

        Returns:
            Number of connections message was sent to
        """
        if user_id not in self.user_connections:
            return 0

        client_ids = self.user_connections[user_id].copy()
        sent_count = 0

        for client_id in client_ids:
            if client_id in self.active_connections:
                connection = self.active_connections[client_id]
                if await connection.send_message(message_type, data):
                    sent_count += 1

        logger.debug(f"Message '{message_type}' sent to user {user_id} on {sent_count} connections")
        return sent_count

    async def send_to_client(self, client_id: str, message_type: str, data: Any) -> bool:
        """
        Send message to a specific client.

        Args:
            client_id: Target client ID
            message_type: Type of message
            data: Message payload

        Returns:
            True if message sent successfully, False otherwise
        """
        if client_id not in self.active_connections:
            return False

        connection = self.active_connections[client_id]
        success = await connection.send_message(message_type, data)

        if not success:
            self.disconnect(client_id)

        return success

    async def broadcast_trading_update(self, trading_data: Dict[str, Any]) -> int:
        """Broadcast real-time trading data."""
        return await self.broadcast("trading_data", trading_data)

    async def broadcast_pnl_update(self, pnl_data: Dict[str, Any]) -> int:
        """Broadcast P&L updates."""
        return await self.broadcast("pnl_update", pnl_data)

    async def broadcast_position_update(self, position_data: Dict[str, Any]) -> int:
        """Broadcast position updates."""
        return await self.broadcast("position_update", position_data)

    async def broadcast_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> int:
        """Broadcast alert notifications."""
        message_data = {
            "alert_type": alert_type,
            "data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        return await self.broadcast("alert", message_data)

    async def broadcast_bot_status(self, status_data: Dict[str, Any]) -> int:
        """Broadcast bot status updates."""
        return await self.broadcast("bot_status", status_data)

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)

    def get_user_count(self) -> int:
        """Get number of unique users with active connections."""
        return len(self.user_connections)

    def get_subscription_count(self, subscription_type: Optional[str] = None) -> int:
        """Get number of subscriptions."""
        if subscription_type:
            return len(self.subscriptions.get(subscription_type, []))
        return sum(len(clients) for clients in self.subscriptions.values())

    def get_connection_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific connection."""
        connection = self.active_connections.get(client_id)
        if not connection:
            return None

        return {
            "client_id": connection.client_id,
            "user_id": connection.user_id,
            "connected_at": connection.connected_at.isoformat(),
            "last_ping": connection.last_ping.isoformat(),
            "is_alive": connection.is_alive(),
            "subscriptions": connection.subscriptions.copy()
        }

    def get_all_connections_info(self) -> Dict[str, Any]:
        """Get information about all connections."""
        connections_info = []

        for client_id, connection in self.active_connections.items():
            connections_info.append({
                "client_id": connection.client_id,
                "user_id": connection.user_id,
                "connected_at": connection.connected_at.isoformat(),
                "last_ping": connection.last_ping.isoformat(),
                "is_alive": connection.is_alive(),
                "subscriptions": connection.subscriptions.copy()
            })

        return {
            "total_connections": len(connections_info),
            "connections": connections_info,
            "total_messages_sent": self.message_count
        }

    async def cleanup_dead_connections(self) -> int:
        """Clean up dead/inactive connections."""
        dead_clients = []

        for client_id, connection in self.active_connections.items():
            if not connection.is_alive(self.connection_timeout):
                dead_clients.append(client_id)

        for client_id in dead_clients:
            self.disconnect(client_id)

        if dead_clients:
            logger.info(f"Cleaned up {len(dead_clients)} dead connections")

        return len(dead_clients)

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self.cleanup_dead_connections()
                except Exception as e:
                    logger.error(f"Error in WebSocket cleanup task: {e}")

        # Create task (assuming this runs in an async context)
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop, cleanup will be handled externally
            pass

    async def close_all(self) -> None:
        """Close all active connections."""
        logger.info("Closing all WebSocket connections")

        for connection in self.active_connections.values():
            try:
                await connection.websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection {connection.client_id}: {e}")

        self.active_connections.clear()
        self.user_connections.clear()
        self.subscriptions.clear()

    async def handle_client_message(self, client_id: str, message: str) -> None:
        """
        Handle incoming message from client.

        Args:
            client_id: Client ID that sent the message
            message: Raw message string
        """
        try:
            data = json.loads(message)
            message_type = data.get("type")
            payload = data.get("data", {})

            if message_type == "subscribe":
                subscription_type = payload.get("subscription_type")
                if subscription_type:
                    await self.subscribe(client_id, subscription_type)

            elif message_type == "unsubscribe":
                subscription_type = payload.get("subscription_type")
                if subscription_type:
                    await self.unsubscribe(client_id, subscription_type)

            elif message_type == "ping":
                # Respond with pong
                await self.send_to_client(client_id, "pong", {"timestamp": datetime.utcnow().isoformat()})

            else:
                logger.warning(f"Unknown message type '{message_type}' from client {client_id}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message from client {client_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {e}")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
