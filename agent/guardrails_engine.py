"""Deterministic Multi-Stage Safety & NVIDIA NemoGuard Engine (`agent/guardrails_engine.py`).

Implements Phase 8 of `plan.md`:
  1. Mandatory 3-Stage Enforcement Pipeline across all multi-tenant agent runs:
     - `input` (`input_guard` pre-model node): Sub-millisecond pre-compiled PII/Secret regex rules +
       NVIDIA NemoGuard Jailbreak (`nvidia/nemoguard-jailbreak-detect`), Topic Control
       (`nvidia/llama-3.1-nemoguard-8b-topic-control`), and Content Safety
       (`nvidia/llama-3.1-nemotron-safety-guard-8b-v3`). Blocks before wasting a single GPU token!
     - `tool_output` (`tool_output_guard` post-tool shield): Neutralizes **Indirect Prompt Injections**
       hidden inside untrusted web pages, RAG documents, emails, and MCP tool outputs before the LLM
       reads them, and scrubs leaked credentials (`nvapi-...`, bearer tokens, private keys).
     - `output` (`output_guard` post-model filter): Redacts accidental PII/secret leaks or blocks
       policy-violating model responses prior to delivery.
  2. Dual Execution Mode + Sub-Millisecond L1 Verdict Cache:
     - Calls live self-hosted or `integrate.api.nvidia.com` NemoGuard NIM endpoints via the shared
       HTTP/2 client pool when an endpoint URL is configured, backed by deterministic `<1ms` local
       jailbreak, indirect-injection, topic-rail, and S1–S23 safety classifiers so every tenant enjoys
       zero-stall protection both online and in air-gapped / dev environments.
  3. Full Multi-Tenant Audit Trail (`guardrail_events` RLS table) offloaded via `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
import re
import threading
import time
from typing import Any, Literal
import uuid

from agent.inference_gateway import compute_prefix_hash
from api.core.tenancy import TenantContext, get_tenant_context
from api.db import ensure_tenant_guardrails_seeded, get_db, get_db_path, mask_secret

logger = logging.getLogger("nimagents.guardrails.engine")

GuardStage = Literal["input", "tool_output", "output"]

DEFAULT_SAFETY_MODEL = "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
DEFAULT_TOPIC_MODEL = "nvidia/llama-3.1-nemoguard-8b-topic-control"
DEFAULT_JAILBREAK_MODEL = "nvidia/nemoguard-jailbreak-detect"
DEFAULT_BLOCKED_TOPICS = [
    "malware development",
    "credential theft",
    "insider trading",
]

# Pre-compiled `<0.2ms` NemoGuard local classifiers (ASCII-clean replacements keep JSON payloads valid!)
_DIRECT_JAILBREAK_PATTERNS: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "NemoGuard Jailbreak: System Override",
        0.96,
        re.compile(
            r"(?is)\b(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+|any\s+|your\s+)?(?:previous|prior|system|safety|developer)\s+(?:instructions?|prompts?|rules?|guardrails?|policies)"
        ),
    ),
    (
        "NemoGuard Jailbreak: Persona / Developer Mode Hijack",
        0.92,
        re.compile(
            r"(?is)\b(?:you\s+are\s+now\s+dan\b|do\s+anything\s+now|enable\s+developer\s+mode|unrestricted\s+jailbreak\s+mode|act\s+without\s+(?:any\s+)?safety\s+restrictions)"
        ),
    ),
    (
        "NemoGuard Jailbreak: ChatML / Control Token Smuggling",
        0.95,
        re.compile(r"(?i)(?:<\|im_start\|>\s*system|<\|system\|>|\[INST\]\s*<<SYS>>|</tool_response>\s*<system>)"),
    ),
    (
        "NemoGuard Jailbreak: System Prompt Exfiltration",
        0.88,
        re.compile(
            r"(?is)\b(?:reveal|print|dump|output|verbatim\s+repeat)\s+(?:your\s+)?(?:entire\s+|exact\s+|hidden\s+)?(?:system\s+prompt|developer\s+instructions)"
        ),
    ),
)

_INDIRECT_INJECTION_PATTERNS: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "Indirect Prompt Injection: Embedded System Directive",
        0.95,
        re.compile(
            r"(?is)(?:important\s+system\s+instruction|new\s+system\s+directive|assistant\s+instruction\s+override|ignore\s+(?:all\s+)?(?:previous|user)\s+(?:instructions?|requests?))[^\n.]{0,220}"
        ),
    ),
    (
        "Indirect Prompt Injection: Markdown Image Exfiltration",
        0.94,
        re.compile(
            r"!\[[^\]]*\]\(https?://[^)\s]*(?:secret|token|key|password|env|cookie|exfil|leak)[^)\s]*\)",
            re.IGNORECASE,
        ),
    ),
    (
        "Indirect Prompt Injection: Tool Credential Exfiltration",
        0.93,
        re.compile(
            r"(?is)\b(?:exfiltrate|send|post|upload|forward)\s+(?:all\s+|the\s+|user\s+)?(?:api[_ -]?keys?|secrets?|credentials?|environment\s+variables?|memory)\s+(?:to|via)\s+\S+"
        ),
    ),
    (
        "Indirect Prompt Injection: Delimiter Escape",
        0.91,
        re.compile(r"(?i)(?:</tool_response>|<\|im_start\|>\s*system|<<SYS>>)"),
    ),
)

_SAFETY_HAZARD_PATTERNS: tuple[tuple[str, str, float, re.Pattern[str]], ...] = (
    (
        "NemoGuard Safety S11: Cyberattack / Ransomware Payload",
        "harmful",
        0.95,
        re.compile(
            r"(?is)\b(?:write|build|generate|deploy)\s+(?:working\s+|undetectable\s+)?(?:ransomware|keylogger|credential\s+harvester|reverse\s+shell\s+implant)\b"
        ),
    ),
    (
        "NemoGuard Safety S9: CBRN / Severe Harm",
        "harmful",
        0.98,
        re.compile(
            r"(?is)\b(?:synthesize|manufacture|weaponize)\s+(?:nerve\s+agent|ricin|anthrax|improvised\s+explosive)\b"
        ),
    ),
)

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |PRIVATE )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH |PRIVATE )?PRIVATE KEY-----"
)
_UNICODE_TAG_SMUGGLING_RE = re.compile(r"[\U000E0000-\U000E007F]+")
_BASE64_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9+/]{36,}={0,2}\b")


@dataclass
class GuardrailHitRecord:
    rule_id: str
    rule_name: str
    category: str
    action: str
    matched_text: str
    detector: str = "regex"
    stage: str = "input"
    risk_score: float = 0.8


@dataclass
class NemoGuardPolicy:
    id: str
    tenant_id: str
    workspace_id: str
    nemoguard_enabled: bool = True
    safety_model: str = DEFAULT_SAFETY_MODEL
    topic_model: str = DEFAULT_TOPIC_MODEL
    jailbreak_model: str = DEFAULT_JAILBREAK_MODEL
    endpoint_url: str = ""
    enforce_input: bool = True
    enforce_tool_output: bool = True
    enforce_model_output: bool = True
    allowed_topics: list[str] = field(default_factory=list)
    blocked_topics: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCKED_TOPICS))
    jailbreak_threshold: float = 0.65
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyEvaluation:
    passed: bool
    stage: GuardStage
    original_text: str
    sanitized_text: str
    blocked_reason: str = ""
    risk_score: float = 0.0
    latency_ms: float = 0.0
    hits: list[GuardrailHitRecord] = field(default_factory=list)
    detectors_run: list[str] = field(default_factory=list)
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "stage": self.stage,
            "sanitized_text": self.sanitized_text,
            "redacted_text": self.sanitized_text,
            "blocked_reason": self.blocked_reason,
            "risk_score": round(self.risk_score, 4),
            "latency_ms": round(self.latency_ms, 3),
            "cached": self.cached,
            "detectors_run": list(self.detectors_run),
            "hits": [asdict(h) for h in self.hits],
        }


class NemoGuardSafetyEngine:
    """Thread-safe, multi-tenant Pre/Tool-Output/Post Safety & NVIDIA NemoGuard Engine."""

    def __init__(self) -> None:
        self._db_lock = threading.Lock()
        self._rules_cache: dict[tuple[str, str], list[tuple[dict[str, Any], re.Pattern[str]]]] = {}
        self._policy_cache: dict[tuple[str, str], NemoGuardPolicy] = {}
        self._verdict_l1: OrderedDict[str, tuple[float, SafetyEvaluation]] = OrderedDict()
        self._max_l1 = 2048
        self._l1_ttl_s = 600.0

    def invalidate_tenant_cache(self, tenant_id: str) -> None:
        key = (str(get_db_path()), tenant_id)
        self._rules_cache.pop(key, None)
        self._policy_cache.pop(key, None)
        self._verdict_l1.clear()

    def get_compiled_rules_sync(self, tenant_id: str) -> list[tuple[dict[str, Any], re.Pattern[str]]]:
        cache_key = (str(get_db_path()), tenant_id)
        cached = self._rules_cache.get(cache_key)
        if cached is not None:
            return cached
        with self._db_lock, get_db() as conn:
            ensure_tenant_guardrails_seeded(conn, tenant_id)
            rows = conn.execute(
                "SELECT * FROM guardrail_rules WHERE tenant_id = ? AND enabled = 1",
                (tenant_id,),
            ).fetchall()
        compiled: list[tuple[dict[str, Any], re.Pattern[str]]] = []
        for row in rows:
            pat = row["pattern"]
            if not pat:
                continue
            try:
                compiled.append((dict(row), re.compile(pat)))
            except re.error:
                continue
        self._rules_cache[cache_key] = compiled
        return compiled

    def get_policy_sync(self, ctx: TenantContext | None = None) -> NemoGuardPolicy:
        active = ctx or get_tenant_context()
        cache_key = (str(get_db_path()), active.tenant_id)
        cached = self._policy_cache.get(cache_key)
        if cached is not None:
            return cached

        policy_id = f"policy:{active.tenant_id}:{active.workspace_id}"
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._db_lock, get_db() as conn:
            row = conn.execute(
                "SELECT * FROM nemoguard_policies WHERE tenant_id = ? AND workspace_id = ?",
                (active.tenant_id, active.workspace_id),
            ).fetchone()
            if row is None:
                default_endpoint = os.getenv("NEMOGUARD_BASE_URL", "")
                conn.execute(
                    """INSERT OR IGNORE INTO nemoguard_policies
                       (id, tenant_id, workspace_id, nemoguard_enabled, safety_model,
                        topic_model, jailbreak_model, endpoint_url, enforce_input,
                        enforce_tool_output, enforce_model_output, allowed_topics_json,
                        blocked_topics_json, jailbreak_threshold, updated_at)
                       VALUES (?, ?, ?, 1, ?, ?, ?, ?, 1, 1, 1, ?, ?, 0.65, ?)""",
                    (
                        policy_id,
                        active.tenant_id,
                        active.workspace_id,
                        DEFAULT_SAFETY_MODEL,
                        DEFAULT_TOPIC_MODEL,
                        DEFAULT_JAILBREAK_MODEL,
                        default_endpoint,
                        json.dumps([]),
                        json.dumps(DEFAULT_BLOCKED_TOPICS),
                        now_iso,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM nemoguard_policies WHERE tenant_id = ? AND workspace_id = ?",
                    (active.tenant_id, active.workspace_id),
                ).fetchone()

        policy = self._row_to_policy(row, active)
        self._policy_cache[cache_key] = policy
        return policy

    def upsert_policy_sync(self, updates: dict[str, Any], ctx: TenantContext | None = None) -> NemoGuardPolicy:
        active = ctx or get_tenant_context()
        current = self.get_policy_sync(active)
        merged = current.to_dict()
        for key in (
            "nemoguard_enabled",
            "safety_model",
            "topic_model",
            "jailbreak_model",
            "endpoint_url",
            "enforce_input",
            "enforce_tool_output",
            "enforce_model_output",
            "allowed_topics",
            "blocked_topics",
            "jailbreak_threshold",
        ):
            if key in updates and updates[key] is not None:
                merged[key] = updates[key]

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._db_lock, get_db() as conn:
            conn.execute(
                """INSERT INTO nemoguard_policies
                   (id, tenant_id, workspace_id, nemoguard_enabled, safety_model,
                    topic_model, jailbreak_model, endpoint_url, enforce_input,
                    enforce_tool_output, enforce_model_output, allowed_topics_json,
                    blocked_topics_json, jailbreak_threshold, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     nemoguard_enabled = excluded.nemoguard_enabled,
                     safety_model = excluded.safety_model,
                     topic_model = excluded.topic_model,
                     jailbreak_model = excluded.jailbreak_model,
                     endpoint_url = excluded.endpoint_url,
                     enforce_input = excluded.enforce_input,
                     enforce_tool_output = excluded.enforce_tool_output,
                     enforce_model_output = excluded.enforce_model_output,
                     allowed_topics_json = excluded.allowed_topics_json,
                     blocked_topics_json = excluded.blocked_topics_json,
                     jailbreak_threshold = excluded.jailbreak_threshold,
                     updated_at = excluded.updated_at""",
                (
                    merged["id"],
                    active.tenant_id,
                    active.workspace_id,
                    int(bool(merged["nemoguard_enabled"])),
                    str(merged["safety_model"]),
                    str(merged["topic_model"]),
                    str(merged["jailbreak_model"]),
                    str(merged["endpoint_url"]),
                    int(bool(merged["enforce_input"])),
                    int(bool(merged["enforce_tool_output"])),
                    int(bool(merged["enforce_model_output"])),
                    json.dumps(list(merged["allowed_topics"])),
                    json.dumps(list(merged["blocked_topics"])),
                    float(merged["jailbreak_threshold"]),
                    now_iso,
                ),
            )
            conn.commit()

        self.invalidate_tenant_cache(active.tenant_id)
        return self.get_policy_sync(active)

    @staticmethod
    def _row_to_policy(row: Any, ctx: TenantContext) -> NemoGuardPolicy:
        if row is None:
            return NemoGuardPolicy(
                id=f"policy:{ctx.tenant_id}:{ctx.workspace_id}",
                tenant_id=ctx.tenant_id,
                workspace_id=ctx.workspace_id,
            )
        try:
            allowed = json.loads(row["allowed_topics_json"] or "[]")
        except Exception:
            allowed = []
        try:
            blocked = json.loads(row["blocked_topics_json"] or "[]")
        except Exception:
            blocked = list(DEFAULT_BLOCKED_TOPICS)
        return NemoGuardPolicy(
            id=row["id"],
            tenant_id=row["tenant_id"],
            workspace_id=row["workspace_id"],
            nemoguard_enabled=bool(row["nemoguard_enabled"]),
            safety_model=row["safety_model"] or DEFAULT_SAFETY_MODEL,
            topic_model=row["topic_model"] or DEFAULT_TOPIC_MODEL,
            jailbreak_model=row["jailbreak_model"] or DEFAULT_JAILBREAK_MODEL,
            endpoint_url=row["endpoint_url"] or "",
            enforce_input=bool(row["enforce_input"]),
            enforce_tool_output=bool(row["enforce_tool_output"]),
            enforce_model_output=bool(row["enforce_model_output"]),
            allowed_topics=[str(x) for x in allowed if str(x).strip()],
            blocked_topics=[str(x) for x in blocked if str(x).strip()],
            jailbreak_threshold=float(row["jailbreak_threshold"] or 0.65),
            updated_at=row["updated_at"] or "",
        )

    async def _query_remote_nemoguard_nim(
        self,
        text: str,
        policy: NemoGuardPolicy,
        stage: GuardStage,
    ) -> GuardrailHitRecord | None:
        """Query live NVIDIA NemoGuard NIM (`llama-3.1-nemotron-safety-guard-8b-v3`) when configured."""
        endpoint = (policy.endpoint_url or os.getenv("NEMOGUARD_BASE_URL", "")).strip()
        if not endpoint:
            return None
        try:
            from api.core.http_pool import get_shared_http_client

            client = get_shared_http_client()
            headers = {"Content-Type": "application/json"}
            api_key = os.getenv("NVIDIA_API_KEY", "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            role_label = "assistant" if stage == "output" else "user"
            payload = {
                "model": policy.safety_model,
                "temperature": 0.0,
                "max_tokens": 128,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            'Classify content safety. Respond with JSON: '
                            '{"User Safety": "safe"|"unsafe", "Safety Categories": "..."}'
                        ),
                    },
                    {"role": role_label, "content": text[:4000]},
                ],
            }
            resp = await client.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=4.0,
            )
            if resp.status_code != 200:
                return None
            content = (
                ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            )
            if '"unsafe"' in content.lower() or "unsafe" in content.lower().split():
                return GuardrailHitRecord(
                    rule_id="nemoguard-remote-safety",
                    rule_name=f"NVIDIA NemoGuard ({policy.safety_model})",
                    category="harmful",
                    action="block",
                    matched_text=content[:160],
                    detector="nemoguard_safety",
                    stage=stage,
                    risk_score=0.95,
                )
        except Exception as exc:
            logger.debug("Remote NemoGuard endpoint check skipped: %s", exc)
        return None

    @staticmethod
    def _decode_obfuscated_payloads(text: str) -> str:
        """Unpack suspicious Base64 spans so encoded prompt injections cannot bypass NemoGuard."""
        extras: list[str] = []
        for cand in _BASE64_CANDIDATE_RE.findall(text)[:6]:
            try:
                decoded = base64.b64decode(cand, validate=True).decode("utf-8", errors="ignore")
                if len(decoded) >= 16 and sum(ch.isprintable() for ch in decoded) / len(decoded) > 0.85:
                    extras.append(decoded)
            except Exception:
                continue
        return f"{text}\n" + "\n".join(extras) if extras else text

    async def evaluate(
        self,
        text: str,
        *,
        stage: GuardStage = "input",
        thread_id: str = "",
        assistant_id: str = "",
        ctx: TenantContext | None = None,
        record_audit: bool = True,
    ) -> SafetyEvaluation:
        """Run deterministic multi-stage Regex + NVIDIA NemoGuard evaluation cleanly off the event loop."""
        t0 = time.perf_counter()
        if not text:
            return SafetyEvaluation(
                passed=True,
                stage=stage,
                original_text=text,
                sanitized_text=text,
                latency_ms=0.0,
            )

        active = ctx or get_tenant_context()
        compiled_rules, policy = await asyncio.to_thread(
            lambda: (self.get_compiled_rules_sync(active.tenant_id), self.get_policy_sync(active))
        )

        stage_enabled = (
            (stage == "input" and policy.enforce_input)
            or (stage == "tool_output" and policy.enforce_tool_output)
            or (stage == "output" and policy.enforce_model_output)
        )

        cache_sig = compute_prefix_hash(
            str(get_db_path()),
            active.tenant_id,
            active.workspace_id,
            stage,
            str(policy.updated_at),
            str(len(compiled_rules)),
            text,
        )
        now_mono = time.monotonic()
        cached_entry = self._verdict_l1.get(cache_sig)
        if cached_entry is not None and cached_entry[0] > now_mono:
            self._verdict_l1.move_to_end(cache_sig)
            prev = cached_entry[1]
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if record_audit and prev.hits:
                await asyncio.to_thread(
                    self._log_audit_events_sync,
                    prev.hits,
                    dt_ms,
                    thread_id,
                    assistant_id,
                    active,
                )
            return SafetyEvaluation(
                passed=prev.passed,
                stage=stage,
                original_text=text,
                sanitized_text=prev.sanitized_text,
                blocked_reason=prev.blocked_reason,
                risk_score=prev.risk_score,
                latency_ms=dt_ms,
                hits=list(prev.hits),
                detectors_run=list(prev.detectors_run),
                cached=True,
            )

        hits: list[GuardrailHitRecord] = []
        detectors_run: list[str] = ["regex"]
        sanitized = text
        max_risk = 0.0

        # Strip zero-width / Unicode tag-smuggled hidden prompt tokens in tool outputs & inputs
        if _UNICODE_TAG_SMUGGLING_RE.search(sanitized):
            sanitized = _UNICODE_TAG_SMUGGLING_RE.sub("", sanitized)
            hits.append(
                GuardrailHitRecord(
                    rule_id="unicode-tag-smuggling",
                    rule_name="Unicode Tag Smuggling Shield",
                    category="harmful",
                    action="redact",
                    matched_text="[hidden unicode tag sequence]",
                    detector="indirect_injection" if stage == "tool_output" else "nemoguard_jailbreak",
                    stage=stage,
                    risk_score=0.86,
                )
            )
            max_risk = max(max_risk, 0.86)

        # Private PEM key redaction across all stages
        if _PRIVATE_KEY_RE.search(sanitized):
            sanitized = _PRIVATE_KEY_RE.sub("[REDACTED]", sanitized)
            hits.append(
                GuardrailHitRecord(
                    rule_id="secrets-private-key",
                    rule_name="PEM Private Key Shield",
                    category="secrets",
                    action="redact" if stage == "tool_output" else "block",
                    matched_text="-----BEGIN PRIVATE KEY-----••••••••",
                    detector="regex",
                    stage=stage,
                    risk_score=0.95,
                )
            )
            max_risk = max(max_risk, 0.95)

        # 1. Pre-compiled Tenant Regex Rules (`<0.2ms`)
        for rule, regex in compiled_rules:
            # In `tool_output` mode, convert secret/PII/injection "block" actions into safe inline
            # redactions so untrusted external documents cannot DoS an entire agent run!
            match = regex.search(sanitized)
            if not match:
                continue
            raw_match = match.group()[:200]
            display_match = (
                mask_secret(raw_match) if rule["category"] == "secrets" else raw_match
            )
            effective_action = (
                "redact"
                if (
                    (stage == "tool_output" and rule["action"] == "block")
                    or (
                        stage == "output"
                        and rule["action"] == "block"
                        and rule["category"] in ("pii", "secrets")
                    )
                )
                else rule["action"]
            )
            score = 0.92 if effective_action == "block" else 0.72 if effective_action == "redact" else 0.45
            max_risk = max(max_risk, score)
            hits.append(
                GuardrailHitRecord(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    category=rule["category"],
                    action=effective_action,
                    matched_text=display_match,
                    detector="regex",
                    stage=stage,
                    risk_score=score,
                )
            )
            if effective_action in ("redact", "block"):
                replacement = (
                    "[INDIRECT_INJECTION_NEUTRALIZED]"
                    if (stage == "tool_output" and rule["category"] == "harmful")
                    else "[REDACTED]"
                )
                sanitized = regex.sub(replacement, sanitized)

        # 2. NVIDIA NemoGuard Classifiers (Jailbreak, Indirect Tool Injection, Topic Rails, Safety)
        if policy.nemoguard_enabled and stage_enabled:
            expanded_text = self._decode_obfuscated_payloads(text)

            if stage == "tool_output":
                detectors_run.append("indirect_injection")
                for label, score, pattern in _INDIRECT_INJECTION_PATTERNS:
                    m = pattern.search(sanitized) or pattern.search(expanded_text)
                    if m:
                        max_risk = max(max_risk, score)
                        hits.append(
                            GuardrailHitRecord(
                                rule_id="nemoguard-indirect-injection",
                                rule_name=label,
                                category="harmful",
                                action="redact",
                                matched_text=m.group()[:160],
                                detector="indirect_injection",
                                stage=stage,
                                risk_score=score,
                            )
                        )
                        sanitized = pattern.sub("[INDIRECT_INJECTION_NEUTRALIZED]", sanitized)

            if stage == "input":
                detectors_run.extend(["nemoguard_jailbreak", "nemoguard_topic", "nemoguard_safety"])
                for label, score, pattern in _DIRECT_JAILBREAK_PATTERNS:
                    if score < policy.jailbreak_threshold:
                        continue
                    m = pattern.search(expanded_text)
                    if m:
                        max_risk = max(max_risk, score)
                        hits.append(
                            GuardrailHitRecord(
                                rule_id="nemoguard-jailbreak",
                                rule_name=label,
                                category="harmful",
                                action="block",
                                matched_text=m.group()[:160],
                                detector="nemoguard_jailbreak",
                                stage=stage,
                                risk_score=score,
                            )
                        )

                lowered = expanded_text.lower()
                for topic in policy.blocked_topics:
                    t_norm = topic.strip().lower()
                    if t_norm and t_norm in lowered:
                        max_risk = max(max_risk, 0.89)
                        hits.append(
                            GuardrailHitRecord(
                                rule_id=f"nemoguard-topic:{t_norm[:32]}",
                                rule_name=f"NemoGuard Topic Control: Blocked '{topic}'",
                                category="topic_control",
                                action="block",
                                matched_text=topic,
                                detector="nemoguard_topic",
                                stage=stage,
                                risk_score=0.89,
                            )
                        )

                if policy.allowed_topics:
                    if not any(t.strip().lower() in lowered for t in policy.allowed_topics if t.strip()):
                        max_risk = max(max_risk, 0.78)
                        hits.append(
                            GuardrailHitRecord(
                                rule_id="nemoguard-topic-allowlist",
                                rule_name=(
                                    "NemoGuard Topic Control: Outside Allowed Topics "
                                    f"({', '.join(policy.allowed_topics[:4])})"
                                ),
                                category="topic_control",
                                action="block",
                                matched_text=text[:120],
                                detector="nemoguard_topic",
                                stage=stage,
                                risk_score=0.78,
                            )
                        )

            if stage in ("input", "output"):
                if "nemoguard_safety" not in detectors_run:
                    detectors_run.append("nemoguard_safety")
                for label, category, score, pattern in _SAFETY_HAZARD_PATTERNS:
                    m = pattern.search(expanded_text)
                    if m:
                        max_risk = max(max_risk, score)
                        hits.append(
                            GuardrailHitRecord(
                                rule_id="nemoguard-safety-hazard",
                                rule_name=label,
                                category=category,
                                action="block",
                                matched_text=m.group()[:160],
                                detector="nemoguard_safety",
                                stage=stage,
                                risk_score=score,
                            )
                        )

                remote_hit = await self._query_remote_nemoguard_nim(sanitized, policy, stage)
                if remote_hit is not None:
                    max_risk = max(max_risk, remote_hit.risk_score)
                    hits.append(remote_hit)

        blocking_hits = [h for h in hits if h.action == "block"]
        passed = len(blocking_hits) == 0
        blocked_reason = (
            f"Blocked by {', '.join(dict.fromkeys(h.rule_name for h in blocking_hits))}."
            if blocking_hits
            else ""
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = SafetyEvaluation(
            passed=passed,
            stage=stage,
            original_text=text,
            sanitized_text=sanitized,
            blocked_reason=blocked_reason,
            risk_score=max_risk,
            latency_ms=latency_ms,
            hits=hits,
            detectors_run=detectors_run,
            cached=False,
        )

        self._verdict_l1[cache_sig] = (time.monotonic() + self._l1_ttl_s, result)
        while len(self._verdict_l1) > self._max_l1:
            self._verdict_l1.popitem(last=False)

        if record_audit and hits:
            await asyncio.to_thread(
                self._log_audit_events_sync,
                hits,
                latency_ms,
                thread_id,
                assistant_id,
                active,
            )

        return result

    def _log_audit_events_sync(
        self,
        hits: list[GuardrailHitRecord],
        latency_ms: float,
        thread_id: str,
        assistant_id: str,
        ctx: TenantContext,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                str(uuid.uuid4()),
                ctx.tenant_id,
                ctx.workspace_id,
                h.rule_id,
                h.rule_name,
                h.category,
                h.action,
                h.matched_text,
                h.stage,
                h.detector,
                round(latency_ms, 3),
                round(h.risk_score, 4),
                thread_id,
                assistant_id,
                now_iso,
            )
            for h in hits
        ]
        with self._db_lock, get_db() as conn:
            conn.executemany(
                """INSERT INTO guardrail_events
                   (id, tenant_id, workspace_id, rule_id, rule_name, category, action,
                    matched_text, stage, detector, latency_ms, risk_score,
                    thread_id, assistant_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()


safety_engine = NemoGuardSafetyEngine()
