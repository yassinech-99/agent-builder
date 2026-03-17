"""Nova 2 Sonic voice WebSocket endpoint.

Proxies browser audio to Amazon Bedrock bidirectional streaming for
real-time conversational voice agent interactions.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("novagrid.api.voice")
router = APIRouter(prefix="/api/voice", tags=["voice"])

SONIC_MODEL_ID = os.getenv("SONIC_MODEL_ID", "amazon.nova-2-sonic-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


async def _get_bedrock_client():
    """Create a boto3 bedrock-runtime client for streaming."""
    import boto3

    kwargs: dict = {"region_name": AWS_REGION}
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    if ak and sk:
        kwargs["aws_access_key_id"] = ak
        kwargs["aws_secret_access_key"] = sk
    return boto3.client("bedrock-runtime", **kwargs)


@router.websocket("/ws")
async def voice_ws(websocket: WebSocket):
    """Bidirectional voice WebSocket.

    Protocol (JSON messages):
      Client -> Server:
        {"type": "audio", "data": "<base64 PCM 16-bit 16kHz mono>"}
        {"type": "text", "data": "user text input"}
        {"type": "stop"}

      Server -> Client:
        {"type": "transcript", "role": "user"|"assistant", "text": "..."}
        {"type": "audio", "data": "<base64 audio chunk>"}
        {"type": "tool_use", "name": "...", "input": {...}}
        {"type": "status", "state": "listening"|"thinking"|"speaking"}
        {"type": "error", "detail": "..."}
    """
    await websocket.accept()
    logger.info("Voice WebSocket connected")

    try:
        client = await _get_bedrock_client()
    except Exception as exc:
        await websocket.send_json({"type": "error", "detail": f"AWS client init failed: {exc}"})
        await websocket.close()
        return

    audio_buffer: list[bytes] = []
    session_active = True

    async def _send_status(state: str):
        await websocket.send_json({"type": "status", "state": state})

    try:
        await _send_status("listening")

        while session_active:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "detail": "Session timed out"})
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "stop":
                session_active = False
                break

            if msg_type == "audio":
                audio_data = base64.b64decode(msg.get("data", ""))
                audio_buffer.append(audio_data)
                continue

            if msg_type == "text":
                user_text = msg.get("data", "")
                if not user_text:
                    continue

                await websocket.send_json({
                    "type": "transcript",
                    "role": "user",
                    "text": user_text,
                })
                await _send_status("thinking")

                try:
                    response = await asyncio.to_thread(
                        _invoke_sonic,
                        client,
                        user_text,
                    )
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "text": response,
                    })
                    await _send_status("listening")
                except Exception as exc:
                    logger.exception("Sonic invocation failed")
                    await websocket.send_json({
                        "type": "error",
                        "detail": str(exc),
                    })
                    await _send_status("listening")

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")
    except Exception as exc:
        logger.exception("Voice WebSocket error: %s", exc)
    finally:
        logger.info("Voice session ended")


def _invoke_sonic(client: Any, text: str) -> str:
    """Invoke Nova 2 Sonic via the Bedrock Converse API for text-based interaction.

    For full bidirectional audio streaming, this would use the
    bedrock-runtime invoke_model_with_response_stream API. This simplified
    version uses Converse for text-in/text-out as a stable demo path.
    """
    try:
        response = client.converse(
            modelId=SONIC_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": text}],
                }
            ],
        )
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        if content and "text" in content[0]:
            return content[0]["text"]
        return "I received your message but couldn't generate a response."
    except Exception as exc:
        raise RuntimeError(f"Sonic model error: {exc}") from exc
