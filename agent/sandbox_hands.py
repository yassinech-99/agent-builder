"""Programmatic Tool Calling (`orchestrate_tools`) & Zero-Trust Disposable Sandbox Hands (`agent/sandbox_hands.py`).

Implements Section 3.3 of `plan.md` (*Anthropic Advanced Tool Use + Scaling Managed Agents*):
  1. Decouples the **Brain** (stateless LangGraph orchestration & Vault credentials) from the
     **Hands** (short-lived isolated Python execution worker in tenant workspace `./workspace/{tid}/{wid}`).
  2. **Zero-Trust Host Vault Proxy (`HostToolRPCServer`):**
     - Spins up a short-lived `0600` Unix Domain Socket (`AF_UNIX`) accepting JSON-RPC calls ONLY for
       tools explicitly authorized on the agent (`allowed_callers` contains `'code_execution'`).
     - Tenant API keys, database URLs, and JWT secrets are **COMPLETELY SCRUBBED** from the child
       sandbox environment (`os.environ` inside the sandbox contains ONLY `PATH`, `HOME`, `LANG`,
       `PYTHONDONTWRITEBYTECODE`, and `NIMAGENTS_RPC_SOCK`!).
     - Even if LLM-generated code runs `import os; print(dict(os.environ))`, **zero secrets exist in
       the sandbox process** — every tool call crosses the Unix socket to the Host `ToolExecutor`!
  3. **Single-Turn Parallel Fan-Out (`asyncio.gather`):**
     - LLM writes compact async Python code calling `await call_tool("calculator", expression="...")`
       or `await tools.<tool_name>(...)` across 10–50 calls in parallel, filters/aggregates the results
       in Python, and prints **only compact final summary `stdout`** (`<= max_stdout_chars`) back to
       the model context window!
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from api.core.tenancy import (
    TenantContext,
    get_tenant_context,
    scoped_workspace_dir,
    tenant_scope,
)

logger = logging.getLogger("nimagents.sandbox.hands")

_HARNESS_PRELUDE = '''
import asyncio
import json
import os
import socket
import sys

_SOCK_PATH = os.environ["NIMAGENTS_RPC_SOCK"]


def _sync_rpc(tool_name: str, kwargs: dict):
    payload = (json.dumps({"tool": tool_name, "args": kwargs}) + "\\n").encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(25.0)
        s.connect(_SOCK_PATH)
        s.sendall(payload)
        chunks = []
        while True:
            buf = s.recv(65536)
            if not buf:
                break
            chunks.append(buf)
    resp = json.loads(b"".join(chunks).decode("utf-8"))
    if not resp.get("ok", False):
        raise RuntimeError(resp.get("error", "Unknown host RPC tool error"))
    raw = resp.get("result", "")
    try:
        return json.loads(raw)
    except Exception:
        return raw


async def call_tool(tool_name: str, _positional_dict=None, **kwargs):
    merged = {}
    if isinstance(_positional_dict, dict):
        merged.update(_positional_dict)
    merged.update(kwargs)
    return await asyncio.to_thread(_sync_rpc, tool_name, merged)


class _ToolsNamespace:
    def __getattr__(self, name: str):
        async def _method(_positional_dict=None, **kwargs):
            return await call_tool(name, _positional_dict, **kwargs)
        return _method


tools = _ToolsNamespace()
'''


class HostToolRPCServer:
    """Ephemeral Unix-socket RPC proxy executing pre-authorized tools on the host with Vault secrets."""

    def __init__(
        self,
        allowed_tools: dict[str, BaseTool],
        ctx: TenantContext,
    ) -> None:
        self.allowed_tools = allowed_tools
        self.ctx = ctx
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._server: asyncio.AbstractServer | None = None
        self.socket_path: str = ""
        self.rpc_invocations: int = 0

    async def __aenter__(self) -> "HostToolRPCServer":
        # Short `/tmp` path avoids 108-byte `AF_UNIX` `sun_path` limit on Linux
        self._tmpdir = tempfile.TemporaryDirectory(prefix="nimrpc_")
        os.chmod(self._tmpdir.name, 0o700)
        self.socket_path = str(Path(self._tmpdir.name) / "rpc.sock")
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=self.socket_path,
        )
        os.chmod(self.socket_path, 0o600)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            req = json.loads(line.decode("utf-8"))
            tool_name = str(req.get("tool") or "")
            args = req.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError("Tool args must be a JSON object")

            target = self.allowed_tools.get(tool_name)
            if target is None:
                resp = {
                    "ok": False,
                    "error": (
                        f"Tool '{tool_name}' is not authorized for programmatic sandbox execution. "
                        f"Authorized tools: {sorted(self.allowed_tools.keys())}"
                    ),
                }
            else:
                self.rpc_invocations += 1
                with tenant_scope(
                    tenant_id=self.ctx.tenant_id,
                    workspace_id=self.ctx.workspace_id,
                    user_id=self.ctx.user_id,
                    plan_tier=self.ctx.plan_tier,
                ):
                    result = await target.ainvoke(args)
                resp = {"ok": True, "result": result}
        except Exception as exc:
            resp = {"ok": False, "error": str(exc)}

        writer.write(json.dumps(resp, ensure_ascii=False, default=str).encode("utf-8"))
        await writer.drain()
        writer.close()
        await writer.wait_closed()


def build_scrubbed_sandbox_env(socket_path: str, workspace_dir: Path) -> dict[str, str]:
    """Build a strictly scrubbed environment containing ZERO host API keys or secrets."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workspace_dir),
        "TMPDIR": str(workspace_dir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "NIMAGENTS_RPC_SOCK": socket_path,
    }


async def execute_programmatic_script(
    python_code: str,
    *,
    allowed_tools: dict[str, BaseTool],
    ctx: TenantContext | None = None,
    timeout_seconds: float = 20.0,
    max_stdout_chars: int = 4000,
) -> dict[str, Any]:
    """Run LLM-authored async Python orchestration script inside an isolated, secret-free Hand process."""
    active = ctx or get_tenant_context()
    ws_dir = scoped_workspace_dir("./workspace", active)

    # Wrap top-level `await ...` cleanly inside `async def __nim_main__():`
    indented_body = "\n".join(
        f"    {line}" if line.strip() else ""
        for line in python_code.splitlines()
    ) or "    pass"
    full_program = (
        f"{_HARNESS_PRELUDE}\n"
        f"async def __nim_main__():\n"
        f"{indented_body}\n\n"
        f"asyncio.run(__nim_main__())\n"
    )

    async with HostToolRPCServer(allowed_tools=allowed_tools, ctx=active) as rpc_server:
        scrubbed_env = build_scrubbed_sandbox_env(rpc_server.socket_path, ws_dir)
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            full_program,
            cwd=str(ws_dir),
            env=scrubbed_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "ok": False,
                "exit_code": -9,
                "rpc_calls": rpc_server.rpc_invocations,
                "stdout": "",
                "stderr": f"Sandbox execution timed out after {timeout_seconds}s",
            }

        stdout_text = stdout_b.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_b.decode("utf-8", errors="replace").strip()
        if len(stdout_text) > max_stdout_chars:
            omitted = len(stdout_text) - max_stdout_chars
            stdout_text = f"{stdout_text[:max_stdout_chars]}\n...[Truncated {omitted} chars]"

        return {
            "ok": proc.returncode == 0,
            "exit_code": int(proc.returncode or 0),
            "rpc_calls": rpc_server.rpc_invocations,
            "stdout": stdout_text,
            "stderr": stderr_text[:2000],
        }


class OrchestrateToolsArgs(BaseModel):
    python_code: str = Field(
        ...,
        description=(
            "Async Python 3 script orchestrating authorized tools in parallel via "
            "`await tools.<tool_name>(arg=val)` or `await call_tool('<tool_name>', arg=val)` "
            "(supports `asyncio.gather(...)`, loops, filtering, and math). "
            "Print ONLY the final concise result via `print(...)`."
        ),
    )


def build_orchestrate_tools_tool(
    allowed_tools: dict[str, BaseTool],
    *,
    ctx: TenantContext | None = None,
    timeout_seconds: float = 20.0,
) -> StructuredTool:
    """Create the `orchestrate_tools` LangChain tool bound to the agent's authorized callable tools."""
    callable_names = sorted(allowed_tools.keys())
    desc = (
        "Execute multi-step or parallel tool calls programmatically in a single turn inside an isolated "
        "Python sandbox (`asyncio.gather(...)`, filtering, aggregation) and return ONLY compact `stdout`. "
        f"Authorized tools on `tools.<name>(...)`: {', '.join(callable_names) or 'none'}.\n"
        'Example: `res = await asyncio.gather(tools.calculator(expression="2**10"), '
        'tools.calculator(expression="sqrt(256)")); print(res)`'
    )

    async def _run_async(python_code: str) -> str:
        res = await execute_programmatic_script(
            python_code,
            allowed_tools=allowed_tools,
            ctx=ctx,
            timeout_seconds=timeout_seconds,
        )
        if not res["ok"]:
            return f"Sandbox error (exit {res['exit_code']}):\n{res['stderr']}\nPartial stdout:\n{res['stdout']}"
        return res["stdout"] or f"(Completed {res['rpc_calls']} tool RPCs with empty stdout)"

    def _run_sync(python_code: str) -> str:
        return asyncio.run(_run_async(python_code))

    return StructuredTool.from_function(
        func=_run_sync,
        coroutine=_run_async,
        name="orchestrate_tools",
        description=desc,
        args_schema=OrchestrateToolsArgs,
    )
