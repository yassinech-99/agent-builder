"""Multi-provider LLM instantiation centered on NVIDIA NIM & NVIDIA Dynamo.

Supports:
  - nvidia_nim       : Hosted NVIDIA API Catalog or self-hosted NIM (`ChatNVIDIA`)
  - nvidia_dynamo    : Distributed KV-aware NVIDIA Dynamo router (`ChatNVIDIADynamo`)
  - openai           : OpenAI API (`ChatOpenAI`)
  - anthropic        : Anthropic Messages API (`ChatAnthropic`)
  - openrouter       : OpenRouter gateway (`ChatOpenAI`)
  - vllm             : Self-hosted vLLM server (`ChatOpenAI`)
  - sglang           : Self-hosted SGLang server (`ChatOpenAI`)
  - ollama           : Local / remote Ollama server (`ChatOpenAI`)
  - openai_compatible: Any OpenAI-compatible HTTP endpoint (`ChatOpenAI`)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

load_dotenv()

logger = logging.getLogger("nimagents.model")

DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

MODEL_ROLES = (
    "primary_chat",
    "fast_router",
    "compactor",
    "contextualizer",
)

SUPPORTED_PROVIDERS: dict[str, dict[str, str]] = {
    "nvidia_nim": {
        "label": "NVIDIA NIM",
        "default_base_url": DEFAULT_NIM_BASE_URL,
        "env_key": "NVIDIA_API_KEY",
        "description": "Hosted NVIDIA API Catalog or self-hosted NIM container",
    },
    "nvidia_dynamo": {
        "label": "NVIDIA Dynamo",
        "default_base_url": "http://localhost:8000/v1",
        "env_key": "NVIDIA_API_KEY",
        "description": "KV-cache-aware distributed multi-node NVIDIA Dynamo cluster",
    },
    "openai": {
        "label": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI GPT & o-series models",
    },
    "anthropic": {
        "label": "Anthropic",
        "default_base_url": "https://api.anthropic.com",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude reasoning & coding models",
    },
    "openrouter": {
        "label": "OpenRouter",
        "default_base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "description": "Unified OpenRouter multi-model catalog",
    },
    "vllm": {
        "label": "vLLM",
        "default_base_url": "http://localhost:8000/v1",
        "env_key": "VLLM_API_KEY",
        "description": "Self-hosted vLLM OpenAI-compatible server",
    },
    "sglang": {
        "label": "SGLang",
        "default_base_url": "http://localhost:30000/v1",
        "env_key": "SGLANG_API_KEY",
        "description": "Self-hosted SGLang RadixAttention server",
    },
    "ollama": {
        "label": "Ollama",
        "default_base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",
        "description": "Local or LAN Ollama model runner",
    },
    "openai_compatible": {
        "label": "OpenAI-Compatible",
        "default_base_url": "http://localhost:8080/v1",
        "env_key": "OPENAI_API_KEY",
        "description": "LiteLLM, TGI, llama.cpp, or custom OpenAI-compatible gateway",
    },
}

NIM_MODEL_PRESETS: list[dict[str, Any]] = [
    {
        "value": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "label": "Llama 3.3 Nemotron Super 49B v1.5",
        "provider": "nvidia_nim",
        "context_window": 131072,
        "supports_thinking": True,
        "recommended_role": "primary_chat",
    },
    {
        "value": "nvidia/nemotron-3-ultra-550b-a55b",
        "label": "Nemotron 3 Ultra 550B (MoE)",
        "provider": "nvidia_nim",
        "context_window": 131072,
        "supports_thinking": True,
        "recommended_role": "primary_chat",
    },
    {
        "value": "nvidia/nemotron-3-super-120b-a12b",
        "label": "Nemotron 3 Super 120B (1M Context)",
        "provider": "nvidia_nim",
        "context_window": 1048576,
        "supports_thinking": True,
        "recommended_role": "compactor",
    },
    {
        "value": "nvidia/llama-3.1-nemotron-nano-8b-v1",
        "label": "Llama 3.1 Nemotron Nano 8B",
        "provider": "nvidia_nim",
        "context_window": 131072,
        "supports_thinking": False,
        "recommended_role": "fast_router",
    },
    {
        "value": "deepseek-ai/deepseek-r1",
        "label": "DeepSeek R1 (Reasoning)",
        "provider": "nvidia_nim",
        "context_window": 131072,
        "supports_thinking": True,
        "recommended_role": "primary_chat",
    },
    {
        "value": "qwen/qwen3-235b-a22b",
        "label": "Qwen3 235B A22B (MoE)",
        "provider": "nvidia_nim",
        "context_window": 131072,
        "supports_thinking": True,
        "recommended_role": "primary_chat",
    },
    {
        "value": "meta/llama-3.3-70b-instruct",
        "label": "Llama 3.3 70B Instruct",
        "provider": "nvidia_nim",
        "context_window": 131072,
        "supports_thinking": False,
        "recommended_role": "primary_chat",
    },
]

_SAFE_MEDIA_URI_RE = re.compile(
    r"^(https://\S+|data:(?:image|application)/(?:png|jpeg|jpg|gif|webp|pdf);base64,[A-Za-z0-9+/=\s]+)$",
    re.IGNORECASE,
)


def is_safe_multimodal_uri(uri: str) -> bool:
    """Validate multimodal URI strictly as HTTPS or base64 data URI.

    Mitigates arbitrary local file path reads in multimodal content blocks
    (see GHSA-g28h-2cmm-rj9x).
    """
    if not isinstance(uri, str):
        return False
    candidate = uri.strip()
    if not candidate:
        return False
    if candidate.startswith(("file://", "/", "./", "../", "~", "\\")):
        return False
    return bool(_SAFE_MEDIA_URI_RE.match(candidate))


def sanitize_multimodal_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Strip or reject unsafe local file paths in multimodal message content blocks."""
    sanitized: list[BaseMessage] = []
    for msg in messages:
        if not isinstance(msg.content, list):
            sanitized.append(msg)
            continue

        cleaned_blocks: list[Any] = []
        changed = False
        for block in msg.content:
            if not isinstance(block, dict):
                cleaned_blocks.append(block)
                continue

            btype = block.get("type", "")
            if btype == "image_url":
                raw_val = block.get("image_url", "")
                url = raw_val.get("url", "") if isinstance(raw_val, dict) else str(raw_val)
                if not is_safe_multimodal_uri(url):
                    changed = True
                    logger.warning("Blocked unsafe image_url URI in message payload")
                    cleaned_blocks.append(
                        {"type": "text", "text": "[Blocked unsafe non-HTTPS/non-data media reference]"}
                    )
                    continue
            elif btype in ("image", "file", "media") and "path" in block:
                changed = True
                logger.warning("Blocked local path field in multimodal content block")
                cleaned_blocks.append(
                    {"type": "text", "text": "[Blocked local filesystem media path]"}
                )
                continue

            cleaned_blocks.append(block)

        if changed:
            if isinstance(msg, HumanMessage):
                sanitized.append(msg.model_copy(update={"content": cleaned_blocks}))
            else:
                sanitized.append(msg.model_copy(update={"content": cleaned_blocks}))
        else:
            sanitized.append(msg)
    return sanitized


def get_model(
    model_id: str | None = None,
    *,
    role: str | None = None,
    priority: int | None = None,
    prefix_hash: str = "",
    thread_id: str = "",
) -> Any:
    """Resolve and instantiate a LangChain chat model with optional fallback chain.

    Resolution order (strictly scoped to active `TenantContext`):
      1. Explicit `model_id` row from database
      2. Model row matching `role` (`primary_chat`, `fast_router`, `compactor`, `contextualizer`)
      3. Model row marked `is_default = 1`
      4. Environment configuration (`NIM_PROVIDER` / `NVIDIA_MODEL` / `NIM_BASE_URL` / `NVIDIA_API_KEY`)
    """
    cfg = None
    if model_id:
        cfg = _load_model_config_by_id(model_id)
    if cfg is None and role:
        cfg = _load_model_config_by_role(role)
    if cfg is None:
        cfg = _load_default_model_config()

    if cfg is not None:
        primary = create_model_from_config(
            cfg,
            priority=priority,
            prefix_hash=prefix_hash,
            thread_id=thread_id,
        )
        fallback_id = (cfg.get("fallback_model_id") or "").strip()
        if fallback_id and fallback_id != cfg.get("id"):
            fb_cfg = _load_model_config_by_id(fallback_id)
            if fb_cfg is not None:
                try:
                    fallback_model = create_model_from_config(
                        fb_cfg,
                        priority=priority,
                        prefix_hash=prefix_hash,
                        thread_id=thread_id,
                    )
                    return primary.with_fallbacks([fallback_model])
                except Exception as exc:
                    logger.warning("Failed to attach fallback model %s: %s", fallback_id, exc)
        return primary

    env_provider = os.getenv("NIM_PROVIDER", "").strip().lower()
    if not env_provider:
        if os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_BASE_URL"):
            env_provider = "nvidia_nim"
        elif os.getenv("OPENAI_API_KEY"):
            env_provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            env_provider = "anthropic"
        else:
            env_provider = "nvidia_nim"

    default_model_name = (
        os.getenv("NVIDIA_MODEL")
        if env_provider in ("nvidia_nim", "nvidia_dynamo")
        else os.getenv("DEFAULT_MODEL")
    ) or (
        DEFAULT_NIM_MODEL
        if env_provider in ("nvidia_nim", "nvidia_dynamo")
        else "gpt-4o-mini"
    )

    return create_model_from_config(
        {
            "provider": env_provider,
            "model_name": default_model_name,
            "base_url": os.getenv("NIM_BASE_URL", ""),
            "api_key": "",
            "temperature": float(os.getenv("MODEL_TEMPERATURE", "0.7")),
            "max_tokens": None,
        },
        priority=priority,
        prefix_hash=prefix_hash,
        thread_id=thread_id,
    )


def _load_model_config_by_id(model_id: str) -> dict[str, Any] | None:
    try:
        from api.core.tenancy import get_tenant_context
        from api.db import get_db

        tid = get_tenant_context().tenant_id
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE id = ? AND tenant_id = ?",
                (model_id, tid),
            ).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


def _load_model_config_by_role(role: str) -> dict[str, Any] | None:
    try:
        from api.core.tenancy import get_tenant_context
        from api.db import get_db

        tid = get_tenant_context().tenant_id
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE role = ? AND tenant_id = ? ORDER BY is_default DESC, created_at DESC LIMIT 1",
                (role, tid),
            ).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


def _load_default_model_config() -> dict[str, Any] | None:
    try:
        from api.core.tenancy import get_tenant_context
        from api.db import get_db

        tid = get_tenant_context().tenant_id
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE is_default = 1 AND tenant_id = ? ORDER BY created_at DESC LIMIT 1",
                (tid,),
            ).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


def resolve_provider_credentials(
    provider: str,
    api_key: str = "",
    base_url: str = "",
) -> tuple[str, str]:
    """Return `(resolved_api_key, resolved_base_url)` using env fallbacks."""
    meta = SUPPORTED_PROVIDERS.get(provider, SUPPORTED_PROVIDERS["nvidia_nim"])
    env_key_name = meta.get("env_key", "")
    resolved_key = (api_key or "").strip() or (
        os.getenv(env_key_name, "").strip() if env_key_name else ""
    )
    resolved_url = (base_url or "").strip()
    if not resolved_url:
        if provider in ("nvidia_nim", "nvidia_dynamo") and os.getenv("NIM_BASE_URL"):
            resolved_url = os.getenv("NIM_BASE_URL", "").strip()
        elif provider == "ollama" and os.getenv("OLLAMA_BASE_URL"):
            resolved_url = os.getenv("OLLAMA_BASE_URL", "").strip()
        else:
            resolved_url = meta.get("default_base_url", "")
    return resolved_key, resolved_url


def create_model_from_config(
    cfg: dict[str, Any],
    *,
    priority: int | None = None,
    prefix_hash: str = "",
    thread_id: str = "",
) -> Any:
    """Instantiate a LangChain BaseChatModel with KV-cache-aware replica routing."""
    from agent.inference_gateway import prefix_router

    provider = (cfg.get("provider") or "nvidia_nim").strip().lower()
    model_name = (cfg.get("model_name") or DEFAULT_NIM_MODEL).strip()
    temperature = float(cfg.get("temperature") if cfg.get("temperature") is not None else 0.7)
    max_tokens = cfg.get("max_tokens") or None
    api_key, raw_base_url = resolve_provider_credentials(
        provider,
        api_key=cfg.get("api_key") or "",
        base_url=cfg.get("base_url") or "",
    )
    base_url = prefix_router.select_endpoint(
        raw_base_url,
        model_id=model_name,
        prefix_hash=prefix_hash,
        thread_id=thread_id,
    ) or raw_base_url

    if provider == "nvidia_dynamo":
        from langchain_nvidia_ai_endpoints import ChatNVIDIADynamo

        kwargs: dict[str, Any] = {
            "model": model_name,
            "base_url": base_url or "http://localhost:8000/v1",
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        if api_key:
            kwargs["api_key"] = api_key
        if priority is not None:
            kwargs["priority"] = priority
        return ChatNVIDIADynamo(**kwargs)

    if provider == "nvidia_nim":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        kwargs = {
            "model": model_name,
            "base_url": base_url or DEFAULT_NIM_BASE_URL,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        if api_key:
            kwargs["api_key"] = api_key
        return ChatNVIDIA(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": model_name,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if api_key:
            kwargs["api_key"] = api_key
        if base_url and base_url != "https://api.anthropic.com":
            kwargs["base_url"] = base_url
        return ChatAnthropic(**kwargs)

    # OpenAI & all OpenAI-compatible providers (`openai`, `openrouter`, `vllm`, `sglang`, `ollama`, `openai_compatible`)
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "streaming": True,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if base_url and provider != "openai":
        kwargs["base_url"] = base_url
    elif base_url and base_url != "https://api.openai.com/v1":
        kwargs["base_url"] = base_url

    if api_key:
        kwargs["api_key"] = api_key
    elif provider in ("vllm", "sglang", "ollama", "openai_compatible"):
        kwargs["api_key"] = "EMPTY"

    if provider == "openrouter":
        kwargs["default_headers"] = {
            "X-Title": "nimAgents",
        }

    return ChatOpenAI(**kwargs)


# Canonical alias used by control-plane & voice streaming gateways
resolve_model = get_model

