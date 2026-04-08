"""Tests for StreamManager — WebSocket connection management.

Tests cover:
    - Connect/disconnect lifecycle
    - Session tracking
    - Broadcast to session
    - Send to specific connection
    - Heartbeat updates
    - Stale connection cleanup
    - Disconnect all from session
    - Failed send handling
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.cockpit.stream import ConnectionInfo, StreamManager, StreamMessage


# -- Helpers -----------------------------------------------------------------


def _mock_ws(fail_send: bool = False) -> AsyncMock:
    """Create a mock WebSocket object."""
    ws = AsyncMock()
    if fail_send:
        ws.send_json.side_effect = Exception("Connection closed")
    return ws


# -- StreamMessage tests -----------------------------------------------------


class TestStreamMessage:
    def test_create_message(self) -> None:
        msg = StreamMessage(type="test", data={"key": "val"})
        assert msg.type == "test"
        assert msg.data == {"key": "val"}
        assert msg.message_id  # Auto-generated
        assert msg.timestamp > 0

    def test_to_dict(self) -> None:
        msg = StreamMessage(type="pong", data={})
        d = msg.to_dict()
        assert d["type"] == "pong"
        assert "message_id" in d
        assert "timestamp" in d


# -- ConnectionInfo tests ----------------------------------------------------


class TestConnectionInfo:
    def test_create(self) -> None:
        info = ConnectionInfo(
            connection_id="abc",
            user_id="user1",
            session_id="sess1",
        )
        assert info.connection_id == "abc"
        assert info.connected_at > 0
        assert info.last_ping > 0


# -- StreamManager tests -----------------------------------------------------


class TestStreamManagerConnect:
    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        mgr = StreamManager()
        ws = _mock_ws()
        info = await mgr.connect(ws, "sess1", "user1")
        assert info.session_id == "sess1"
        assert info.user_id == "user1"
        assert mgr.active_sessions == 1
        assert mgr.total_connections == 1

    @pytest.mark.asyncio
    async def test_multiple_connections_same_session(self) -> None:
        mgr = StreamManager()
        await mgr.connect(_mock_ws(), "sess1", "user1")
        await mgr.connect(_mock_ws(), "sess1", "user2")
        assert mgr.active_sessions == 1
        assert mgr.total_connections == 2

    @pytest.mark.asyncio
    async def test_multiple_sessions(self) -> None:
        mgr = StreamManager()
        await mgr.connect(_mock_ws(), "sess1", "user1")
        await mgr.connect(_mock_ws(), "sess2", "user2")
        assert mgr.active_sessions == 2
        assert mgr.total_connections == 2


class TestStreamManagerDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        mgr = StreamManager()
        ws = _mock_ws()
        info = await mgr.connect(ws, "sess1", "user1")
        result = await mgr.disconnect("sess1", info.connection_id)
        assert result is True
        assert mgr.total_connections == 0
        assert mgr.active_sessions == 0  # Empty session removed

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self) -> None:
        mgr = StreamManager()
        result = await mgr.disconnect("nope", "nope")
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_all(self) -> None:
        mgr = StreamManager()
        await mgr.connect(_mock_ws(), "sess1", "user1")
        await mgr.connect(_mock_ws(), "sess1", "user2")
        count = await mgr.disconnect_all("sess1")
        assert count == 2
        assert mgr.active_sessions == 0


class TestStreamManagerBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_to_session(self) -> None:
        mgr = StreamManager()
        ws1 = _mock_ws()
        ws2 = _mock_ws()
        await mgr.connect(ws1, "sess1", "user1")
        await mgr.connect(ws2, "sess1", "user2")

        msg = StreamMessage(type="test", data={"hello": "world"})
        count = await mgr.broadcast("sess1", msg)
        assert count == 2
        ws1.send_json.assert_called_once()
        ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_exclude_connection(self) -> None:
        mgr = StreamManager()
        ws1 = _mock_ws()
        ws2 = _mock_ws()
        info1 = await mgr.connect(ws1, "sess1", "user1")
        await mgr.connect(ws2, "sess1", "user2")

        msg = StreamMessage(type="test", data={})
        count = await mgr.broadcast("sess1", msg, exclude_connection=info1.connection_id)
        assert count == 1
        ws1.send_json.assert_not_called()
        ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed(self) -> None:
        mgr = StreamManager()
        ws_ok = _mock_ws()
        ws_fail = _mock_ws(fail_send=True)
        await mgr.connect(ws_ok, "sess1", "user1")
        await mgr.connect(ws_fail, "sess1", "user2")

        msg = StreamMessage(type="test", data={})
        count = await mgr.broadcast("sess1", msg)
        assert count == 1  # Only ws_ok succeeded
        assert mgr.total_connections == 1  # ws_fail removed

    @pytest.mark.asyncio
    async def test_broadcast_empty_session(self) -> None:
        mgr = StreamManager()
        msg = StreamMessage(type="test", data={})
        count = await mgr.broadcast("nonexistent", msg)
        assert count == 0


class TestStreamManagerSendToConnection:
    @pytest.mark.asyncio
    async def test_send_to_connection(self) -> None:
        mgr = StreamManager()
        ws = _mock_ws()
        info = await mgr.connect(ws, "sess1", "user1")
        msg = StreamMessage(type="test", data={"x": 1})
        result = await mgr.send_to_connection("sess1", info.connection_id, msg)
        assert result is True
        ws.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_nonexistent(self) -> None:
        mgr = StreamManager()
        msg = StreamMessage(type="test", data={})
        result = await mgr.send_to_connection("nope", "nope", msg)
        assert result is False


class TestStreamManagerHeartbeat:
    @pytest.mark.asyncio
    async def test_update_heartbeat(self) -> None:
        mgr = StreamManager()
        ws = _mock_ws()
        info = await mgr.connect(ws, "sess1", "user1")
        result = await mgr.update_heartbeat("sess1", info.connection_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent(self) -> None:
        mgr = StreamManager()
        result = await mgr.update_heartbeat("nope", "nope")
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_stale(self) -> None:
        mgr = StreamManager()
        ws = _mock_ws()
        info = await mgr.connect(ws, "sess1", "user1")

        # Manually set last_ping to old time
        conns = mgr._sessions["sess1"]
        _, conn_info = conns[info.connection_id]
        conn_info.last_ping = 0  # Very old

        removed = await mgr.cleanup_stale(max_idle_seconds=1.0)
        assert removed == 1
        assert mgr.total_connections == 0


class TestStreamManagerSessionInfo:
    @pytest.mark.asyncio
    async def test_get_session_connections(self) -> None:
        mgr = StreamManager()
        await mgr.connect(_mock_ws(), "sess1", "user1")
        await mgr.connect(_mock_ws(), "sess1", "user2")
        conns = mgr.get_session_connections("sess1")
        assert len(conns) == 2

    @pytest.mark.asyncio
    async def test_get_connection_count(self) -> None:
        mgr = StreamManager()
        await mgr.connect(_mock_ws(), "sess1", "user1")
        assert mgr.get_connection_count("sess1") == 1
        assert mgr.get_connection_count("nope") == 0
