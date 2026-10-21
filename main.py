"""nimAgents CLI entry point."""

from __future__ import annotations

import os
import typer
from rich.console import Console
from rich.table import Table

cli = typer.Typer(
    name="nimagents",
    help="nimAgents — NVIDIA NIM-native multi-provider AI agent platform CLI.",
    add_completion=False,
)
console = Console()


@cli.command("info")
def info() -> None:
    """Display active nimAgents runtime configuration and supported providers."""
    from agent.model import NIM_MODEL_PRESETS, SUPPORTED_PROVIDERS

    console.print("[bold emerald_green]nimAgents v1.0.0[/bold emerald_green] — NVIDIA NIM Multi-Provider Runtime")
    table = Table(title="Supported Model Providers")
    table.add_column("Provider ID", style="cyan")
    table.add_column("Label")
    table.add_column("Default Base URL")
    table.add_column("Env Key")
    for pid, meta in SUPPORTED_PROVIDERS.items():
        has_key = "set" if meta.get("env_key") and os.getenv(meta["env_key"]) else "-"
        table.add_row(pid, meta["label"], meta.get("default_base_url") or "(SDK default)", f"{meta.get('env_key') or 'none'} ({has_key})")
    console.print(table)
    console.print(f"Curated NIM presets loaded: [bold]{len(NIM_MODEL_PRESETS)}[/bold]")


@cli.command("init-db")
def init_database() -> None:
    """Initialize the local nimAgents database schema."""
    from api.db import get_db_path, init_db

    init_db()
    console.print(f"[green]Initialized nimAgents database at {get_db_path()}[/green]")


@cli.command("api")
def serve_api(
    host: str = typer.Option("0.0.0.0", help="Host interface to bind."),
    port: int = typer.Option(8100, help="TCP port to listen on."),
    reload: bool = typer.Option(False, help="Enable auto-reload for development."),
) -> None:
    """Start the nimAgents Builder & Control Plane FastAPI server."""
    import uvicorn

    uvicorn.run("api.app:app", host=host, port=port, reload=reload)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
