"""Streaming Voice WebSocket gateway powered by NVIDIA Riva Speech NIM & active nimAgents chat models."""

from __future__ import annotations

import asyncio
import base64
from collections import deque
import json
import logging
import os
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent.model import get_model

logger = logging.getLogger("nimagents.api.voice")
router = APIRouter(prefix="/api/voice", tags=["voice"])

MAX_AUDIO_CHUNKS = 120
NIM_ASR_BASE_URL = os.getenv("NIM_ASR_BASE_URL", "")
NIM_TTS_BASE_URL = os.getenv("NIM_TTS_BASE_URL", "")
NIM_ASR_MODEL = os.getenv("NIM_ASR_MODEL", "nvidia/parakeet-ctc-1.1b-asr")
NIM_TTS_MODEL = os.getenv("NIM_TTS_MODEL", "nvidia/magpie-tts-multilingual")


async def _transcribe_audio_buffer(audio_bytes: bytes) -> str | None:
    """Transcribe buffered audio via an OpenAI/Riva-compatible `/audio/transcriptions` NIM endpoint."""
    if not NIM_ASR_BASE_URL or not audio_bytes:
        return None
    from api.app import get_http_client

    client = get_http_client()
    headers: dict[str, str] = {}
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{NIM_ASR_BASE_URL.rstrip('/')}/audio/transcriptions"
    files = {"file": ("utterance.webm", audio_bytes, "audio/webm")}
    data = {"model": NIM_ASR_MODEL}
    resp = await client.post(url, headers=headers, data=data, files=files, timeout=20.0)
    resp.raise_for_status()
    return (resp.json().get("text") or "").strip() or None


async def _synthesize_tts_chunk(text: str) -> bytes | None:
    """Synthesize speech audio via an OpenAI/Riva-compatible `/audio/speech` NIM endpoint."""
    if not NIM_TTS_BASE_URL or not text.strip():
        return None
    from api.app import get_http_client

    client = get_http_client()
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{NIM_TTS_BASE_URL.rstrip('/')}/audio/speech"
    payload = {
        "model": NIM_TTS_MODEL,
        "input": text,
        "voice": os.getenv("NIM_TTS_VOICE", "English-US.Female-1"),
        "response_format": "mp3",
    }
    resp = await client.post(url, headers=headers, json=payload, timeout=20.0)
    resp.raise_for_status()
    return resp.content


@router.websocket("/ws")
async def voice_ws(websocket: WebSocket):
    """Bidirectional low-latency voice & conversational WebSocket."""
    await websocket.accept()
    model_id = websocket.query_params.get("model_id") or None
    logger.info("nimAgents Voice WebSocket connected (model_id=%s)", model_id)

    try:
        llm = get_model(model_id=model_id, priority=1)
    except Exception as exc:
        await websocket.send_json(
            {"type": "error", "detail": f"Model initialization failed: {exc}"}
        )
        await websocket.close()
        return

    # Bounded ring buffer prevents per-client WebSocket RAM leaks under long sessions
    audio_buffer: deque[bytes] = deque(maxlen=MAX_AUDIO_CHUNKS)
    history: list[BaseMessage] = []

    async def _send_status(state: str):
        await websocket.send_json({"type": "status", "state": state})

    async def _run_turn(user_text: str, echo_user_transcript: bool = False):
        if echo_user_transcript:
            await websocket.send_json(
                {"type": "transcript", "role": "user", "text": user_text}
            )
        await _send_status("thinking")
        history.append(HumanMessage(content=user_text))

        try:
            chunks: list[str] = []
            async for chunk in llm.astream(history[-16:]):
                piece = getattr(chunk, "content", "")
                if isinstance(piece, str) and piece:
                    chunks.append(piece)

            raw_reply = "".join(chunks)
            reply_text = (
                re.sub(r"<think>.*?</think>", "", raw_reply, flags=re.DOTALL | re.IGNORECASE).strip()
                or "I received your message."
            )
            history.append(AIMessage(content=reply_text))

            await websocket.send_json(
                {"type": "transcript", "role": "assistant", "text": reply_text}
            )

            try:
                audio_out = await _synthesize_tts_chunk(reply_text)
                if audio_out:
                    await _send_status("speaking")
                    await websocket.send_json(
                        {
                            "type": "audio",
                            "data": base64.b64encode(audio_out).decode("ascii"),
                        }
                    )
            except Exception as tts_exc:
                logger.warning("TTS synthesis skipped: %s", tts_exc)

            await _send_status("listening")
        except Exception as exc:
            logger.exception("Voice turn generation failed")
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await _send_status("listening")

    try:
        await _send_status("listening")

        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {"type": "error", "detail": "Voice session timed out"}
                )
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")
            if msg_type == "stop":
                break

            if msg_type == "audio":
                audio_data = base64.b64decode(msg.get("data", ""))
                audio_buffer.append(audio_data)
                continue

            if msg_type == "flush_audio":
                joined = b"".join(audio_buffer)
                audio_buffer.clear()
                if joined:
                    transcript = await _transcribe_audio_buffer(joined)
                    if transcript:
                        await _run_turn(transcript, echo_user_transcript=True)
                continue

            if msg_type == "text":
                user_text = (msg.get("data") or "").strip()
                if user_text:
                    await _run_turn(user_text, echo_user_transcript=False)

    except WebSocketDisconnect:
        logger.info("nimAgents Voice WebSocket disconnected")
    except Exception as exc:
        logger.exception("Voice WebSocket error: %s", exc)
    finally:
        audio_buffer.clear()
        logger.info("nimAgents Voice session ended")
