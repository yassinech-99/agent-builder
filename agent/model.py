"""LLM model instantiation. AWS Bedrock (Nova) only."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_model(model_id: str | None = None) -> Any:
    """Return a ChatBedrockConverse for the given model_id (from SQLite), or the default from .env."""
    if model_id:
        cfg = _load_model_config(model_id)
        if cfg:
            return _create_model_from_config(cfg)

    return _create_bedrock_model(
        model_name=os.environ.get("BEDROCK_MODEL", "amazon.nova-2-lite-v1:0"),
        region=os.environ.get("AWS_REGION", "us-east-1"),
        temperature=0.7,
    )


def _load_model_config(model_id: str) -> dict | None:
    try:
        from api.db import get_db_raw

        conn = get_db_raw()
        try:
            row = conn.execute(
                "SELECT * FROM models WHERE id = ?", (model_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    except Exception:
        return None


def _create_bedrock_model(
    model_name: str = "amazon.nova-2-lite-v1:0",
    region: str = "us-east-1",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
) -> Any:
    from langchain_aws import ChatBedrockConverse

    kwargs: dict = {
        "model": model_name,
        "region_name": region,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if aws_access_key_id and aws_secret_access_key:
        kwargs["credentials"] = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
        }
    return ChatBedrockConverse(**kwargs)


def _create_model_from_config(cfg: dict) -> Any:
    model_name = cfg.get("model_name", "amazon.nova-2-lite-v1:0")
    temperature = cfg.get("temperature", 0.7)
    max_tokens = cfg.get("max_tokens")
    api_key = cfg.get("api_key") or None
    secret_key = cfg.get("aws_secret_key") or None

    return _create_bedrock_model(
        model_name=model_name,
        region=cfg.get("aws_region") or os.environ.get("AWS_REGION", "us-east-1"),
        temperature=temperature,
        max_tokens=max_tokens,
        aws_access_key_id=api_key,
        aws_secret_access_key=secret_key,
    )
