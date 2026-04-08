"""WebSocket stream manager for real-time multi-model communication.

Manages WebSocket connections, message routing, heartbeat, and
session-scoped broadcasting. Designed for horizontal scaling
via Redis pub/sub (not yet wired).

Connection lifecycle:
    1. Client connects via WebSocket with JWT auth
    2. Server assigns connection to a cockpit session
    3. Messages from LLMs are broadcast to all connections in the session
    4. Heartbeat pings keep connections alive
    5. Client disconnects or times out

Message protocol (JSON):
    Client -> Server:
        {"type": "prompt", "content": "...", "models": ["openai", "claude"]}
        {"type": "ping"}

    Server -> Client:
        {"type": "model_response", "model": "openai", "content": "...", "done": false}
        {"type": "model_response", "model": "openai", "content": "", "done": true}
        {"type": "error", "message": "..."}
        {"type": "pong"}
        {"type": "session_update", "data": {...}}

Usage:
    manager = StreamManager()
    await manager.connect(websocket, session_id, user_id)
    await manager.broadcast(session_id, message)
    await manager.disconnect(websocket, session_id)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class StreamMessage:
    """A message flowing through the stream.

    Attributes:
        type: Message type (prompt, model_response, error, ping, pong, session_update).
        data: Message payload.
        message_id: Unique message identifier.
        timestamp: Unix timestamp.
    """

    type: str
    data: dict[str, Any]
    message_id: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.message_id:
            object.__setattr__(self, "message_id", uuid4().hex[:16])
        if not self.timestamp:
            object.__setattr__(self, "timestamp", time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": self.data,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
        }


@dataclass
class ConnectionInfo:
    """Metadata about a WebSocket connection.

    Attributes:
        connection_id: Unique ID for this connection.
        user_id: Authenticated user ID.
        session_id: Cockpit session this connection belongs to.
        connected_at: Unix timestamp of connection.
        last_ping: Unix timestamp of last heartbeat.
    """

    connection_id: str
    user_id: str
    session_id: str
    connected_at: float = 0.0
    last_ping: float = 0.0

    def __post_init__(self) -> None:
        now = time.time()
        if not self.connected_at:
            self.connected_at = now
        if not self.last_ping:
            self.last_ping = now


class StreamManager:
    """Manages WebSocket connections and message routing.

    Thread-safe for asyncio. Each cockpit session has a set of
    connections that receive broadcasts.

    In production, broadcasts go through Redis pub/sub for
    horizontal scaling across multiple backend instances.
    """

    def __init__(self, heartbeat_interval: float = 30.0) -> None:
        """Initialize the stream manager.

        Args:
            heartbeat_interval: Seconds between heartbeat checks.
        """
        # session_id -> set of (websocket, connection_info)
        self._sessions: dict[str, dict[str, tuple[Any, ConnectionInfo]]] = {}
        self._heartbeat_interval = heartbeat_interval
        self._lock = asyncio.Lock()

    @property
    def active_sessions(self) -> int:
        """Number of sessions with at least one connection."""
        return len(self._sessions)

    @property
    def total_connections(self) -> int:
        """Total number of active connections across all sessions."""
        return sum(len(conns) for conns in self._sessions.values())

    async def connect(
        self,
        websocket: Any,
        session_id: str,
        user_id: str,
    ) -> ConnectionInfo:
        """Register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection object.
            session_id: Cockpit session to join.
            user_id: Authenticated user ID.

        Returns:
            ConnectionInfo for the new connection.
        """
        conn_id = uuid4().hex[:16]
        info = ConnectionInfo(
            connection_id=conn_id,
            user_id=user_id,
            session_id=session_id,
        )

        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {}
            self._sessions[session_id][conn_id] = (websocket, info)

        return info

    async def disconnect(
        self,
        session_id: str,
        connection_id: str,
    ) -> bool:
        """Remove a WebSocket connection.

        Returns True if the connection was found and removed.
        """
        async with self._lock:
            if session_id in self._sessions:
                if connection_id in self._sessions[session_id]:
                    del self._sessions[session_id][connection_id]
                    # Clean up empty sessions
                    if not self._sessions[session_id]:
                        del self._sessions[session_id]
                    return True
        return False

    async def broadcast(
        self,
        session_id: str,
        message: StreamMessage,
        exclude_connection: str | None = None,
    ) -> int:
        """Broadcast a message to all connections in a session.

        Args:
            session_id: Target session.
            message: Message to broadcast.
            exclude_connection: Optional connection ID to skip.

        Returns:
            Number of connections that received the message.
        """
        async with self._lock:
            connections = self._sessions.get(session_id, {})
            targets = [
                (ws, info)
                for conn_id, (ws, info) in connections.items()
                if conn_id != exclude_connection
            ]

        sent_count = 0
        failed_ids: list[str] = []

        for ws, info in targets:
            try:
                await ws.send_json(message.to_dict())
                sent_count += 1
            except Exception:
                failed_ids.append(info.connection_id)

        # Clean up failed connections
        for conn_id in failed_ids:
            await self.disconnect(session_id, conn_id)

        return sent_count

    async def send_to_connection(
        self,
        session_id: str,
        connection_id: str,
        message: StreamMessage,
    ) -> bool:
        """Send a message to a specific connection.

        Returns True if the message was sent successfully.
        """
        async with self._lock:
            connections = self._sessions.get(session_id, {})
            entry = connections.get(connection_id)

        if entry is None:
            return False

        ws, info = entry
        try:
            await ws.send_json(message.to_dict())
            return True
        except Exception:
            await self.disconnect(session_id, connection_id)
            return False

    def get_session_connections(self, session_id: str) -> list[ConnectionInfo]:
        """Get all connection infos for a session (non-async for read)."""
        connections = self._sessions.get(session_id, {})
        return [info for _, info in connections.values()]

    def get_connection_count(self, session_id: str) -> int:
        """Get number of connections in a session."""
        return len(self._sessions.get(session_id, {}))

    async def update_heartbeat(
        self, session_id: str, connection_id: str
    ) -> bool:
        """Update the last ping time for a connection.

        Returns True if the connection was found.
        """
        async with self._lock:
            connections = self._sessions.get(session_id, {})
            entry = connections.get(connection_id)
            if entry:
                _, info = entry
                info.last_ping = time.time()
                return True
        return False

    async def cleanup_stale(self, max_idle_seconds: float = 120.0) -> int:
        """Remove connections that haven't sent a heartbeat recently.

        Returns number of connections removed.
        """
        now = time.time()
        to_remove: list[tuple[str, str]] = []

        async with self._lock:
            for session_id, connections in self._sessions.items():
                for conn_id, (ws, info) in connections.items():
                    if now - info.last_ping > max_idle_seconds:
                        to_remove.append((session_id, conn_id))

        removed = 0
        for session_id, conn_id in to_remove:
            if await self.disconnect(session_id, conn_id):
                removed += 1

        return removed

    async def disconnect_all(self, session_id: str) -> int:
        """Disconnect all connections from a session.

        Returns number of connections removed.
        """
        async with self._lock:
            connections = self._sessions.pop(session_id, {})
        return len(connections)
