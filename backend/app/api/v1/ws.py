"""WebSocket endpoint for multi-model cockpit streaming.

WS /api/v1/ws/{session_id}?token=<jwt>

Protocol:
    Client -> Server:
        {"type": "prompt", "content": "...", "models": ["openai", "claude"]}
        {"type": "ping"}

    Server -> Client:
        {"type": "connected", "data": {"connection_id": "...", "session": {...}}}
        {"type": "model_response", "data": {"model": "...", "content": "...", "done": false}}
        {"type": "pong", "data": {}}
        {"type": "error", "data": {"message": "..."}}
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import jwt as pyjwt
import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.cockpit.session import CockpitSession, CockpitStatus, ModelRole
from app.core.cockpit.stream import ConnectionInfo, StreamManager, StreamMessage
from app.core.llm.client import LLMClient
from app.core.llm.models import LLMError, LLMMessage, resolve_model

ws_logger = structlog.get_logger()

router = APIRouter()

# Module-level stream manager (singleton per process)
_stream_manager = StreamManager()

# In-memory cockpit session store (production: Redis or DB)
_cockpit_sessions: dict[str, CockpitSession] = {}


def get_stream_manager() -> StreamManager:
    """Get the global stream manager. Useful for testing."""
    return _stream_manager


async def _authenticate_ws(token: str) -> str | None:
    """Validate JWT token and return user_id, or None if invalid."""
    settings = get_settings()
    try:
        payload = pyjwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        return payload.get("sub")
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError):
        return None


@router.websocket("/{session_id}")
async def cockpit_websocket(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(...),
) -> None:
    """WebSocket endpoint for cockpit real-time streaming.

    Authentication is via JWT token passed as query parameter.
    """
    # Authenticate
    user_id = await _authenticate_ws(token)
    if user_id is None:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    await websocket.accept()

    # Get or create cockpit session
    if session_id not in _cockpit_sessions:
        _cockpit_sessions[session_id] = CockpitSession.create(
            user_id=user_id,
            models=["openai", "claude"],
        )

    cockpit = _cockpit_sessions[session_id]

    # Verify ownership
    if cockpit.user_id != user_id:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Not authorized for this session"},
        })
        await websocket.close(code=4003, reason="Unauthorized")
        return

    # Register connection
    conn_info = await _stream_manager.connect(
        websocket=websocket,
        session_id=session_id,
        user_id=user_id,
    )

    # Send connected acknowledgment
    await websocket.send_json({
        "type": "connected",
        "data": {
            "connection_id": conn_info.connection_id,
            "session": cockpit.to_dict(),
        },
    })

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON"},
                })
                continue

            msg_type = message.get("type", "")

            if msg_type == "ping":
                await _stream_manager.update_heartbeat(
                    session_id, conn_info.connection_id
                )
                await websocket.send_json({
                    "type": "pong",
                    "data": {},
                })

            elif msg_type == "prompt":
                content = message.get("content", "")
                models = message.get("models", [])

                if not content:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "Prompt content is required"},
                    })
                    continue

                # Store user message in cockpit session
                for model_id in models:
                    cockpit = cockpit.add_message(model_id, "user", content)
                _cockpit_sessions[session_id] = cockpit

                # Acknowledge prompt receipt
                await _stream_manager.broadcast(
                    session_id,
                    StreamMessage(
                        type="prompt_received",
                        data={"content": content, "models": models},
                    ),
                )

                # Notify that each model is queued
                for model_id in models:
                    await _stream_manager.broadcast(
                        session_id,
                        StreamMessage(
                            type="model_queued",
                            data={"model": model_id},
                        ),
                    )

                # Stream real LLM responses in parallel.
                # Each model runs as a separate task, streaming chunks
                # back via the StreamManager. Session state is updated
                # through _cockpit_sessions dict (no nonlocal needed).
                async def _stream_model(
                    mid: str, sid: str, user_content: str
                ) -> None:
                    """Stream a single model's response via WebSocket."""
                    llm = LLMClient()
                    msgs: list[LLMMessage] = []

                    # Include conversation history as context
                    current_session = _cockpit_sessions.get(sid)
                    if current_session:
                        model_state = current_session.models.get(mid)
                        if model_state and model_state.context:
                            msgs.append(LLMMessage(role="system", content=model_state.context))

                    msgs.append(LLMMessage(role="user", content=user_content))

                    try:
                        full_content = ""
                        prompt_tokens = 0
                        completion_tokens = 0

                        async for chunk in llm.call_streaming(mid, msgs):
                            if chunk.delta:
                                full_content += chunk.delta
                                await _stream_manager.broadcast(
                                    sid,
                                    StreamMessage(
                                        type="model_response",
                                        data={
                                            "model": mid,
                                            "content": chunk.delta,
                                            "done": False,
                                        },
                                    ),
                                )
                            if chunk.is_final:
                                prompt_tokens = chunk.prompt_tokens
                                completion_tokens = chunk.completion_tokens

                        # Get cost estimate
                        try:
                            model_info = resolve_model(mid)
                            cost = model_info.estimate_cost(prompt_tokens, completion_tokens)
                        except LLMError:
                            cost = 0.0

                        # Send completion signal
                        await _stream_manager.broadcast(
                            sid,
                            StreamMessage(
                                type="model_response",
                                data={
                                    "model": mid,
                                    "content": "",
                                    "done": True,
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "cost_usd": cost,
                                },
                            ),
                        )

                        # Update session state via the shared dict
                        sess = _cockpit_sessions.get(sid)
                        if sess:
                            sess = sess.add_message(mid, "assistant", full_content)
                            sess = sess.update_cost(
                                mid, prompt_tokens, completion_tokens, cost
                            )
                            _cockpit_sessions[sid] = sess

                    except LLMError as exc:
                        ws_logger.error("llm_stream_error", model=mid, error=str(exc))
                        await _stream_manager.broadcast(
                            sid,
                            StreamMessage(
                                type="model_response",
                                data={
                                    "model": mid,
                                    "content": "",
                                    "done": True,
                                    "error": str(exc),
                                },
                            ),
                        )
                    except Exception as exc:
                        ws_logger.exception("unexpected_stream_error", model=mid)
                        await _stream_manager.broadcast(
                            sid,
                            StreamMessage(
                                type="model_response",
                                data={
                                    "model": mid,
                                    "content": "",
                                    "done": True,
                                    "error": f"Unexpected error: {type(exc).__name__}",
                                },
                            ),
                        )

                # Launch all model streams in parallel
                tasks = [
                    asyncio.create_task(_stream_model(m, session_id, content))
                    for m in models
                ]
                # Don't await — let them stream concurrently while we keep
                # the WebSocket message loop alive. Errors are handled inside.
                asyncio.gather(*tasks, return_exceptions=True)

            elif msg_type == "add_model":
                model_id = message.get("model_id", "")
                role = message.get("role", "general")
                if model_id:
                    try:
                        model_role = ModelRole(role)
                    except ValueError:
                        model_role = ModelRole.GENERAL
                    cockpit = cockpit.add_model(model_id, model_role)
                    _cockpit_sessions[session_id] = cockpit
                    await _stream_manager.broadcast(
                        session_id,
                        StreamMessage(
                            type="session_update",
                            data=cockpit.to_dict(),
                        ),
                    )

            elif msg_type == "remove_model":
                model_id = message.get("model_id", "")
                if model_id:
                    cockpit = cockpit.remove_model(model_id)
                    _cockpit_sessions[session_id] = cockpit
                    await _stream_manager.broadcast(
                        session_id,
                        StreamMessage(
                            type="session_update",
                            data=cockpit.to_dict(),
                        ),
                    )

            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"},
                })

    except WebSocketDisconnect:
        pass
    finally:
        await _stream_manager.disconnect(session_id, conn_info.connection_id)
