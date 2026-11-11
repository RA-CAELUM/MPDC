"""Unified CLI for MPDC (`mpdc serve`, `mpdc top`, `mpdc gui`, `mpdc stats`, `mpdc eval`, `mpdc train`, `mpdc harvest`, `mpdc mcp`)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mpdc.config import MPDCConfig
from mpdc.evaluation.quantifier import SelfImprovementQuantifier
from mpdc.gateway.server import build_client_opencode_config, create_gateway_app
from mpdc.harvester.collector import TrajectoryHarvester
from mpdc.mcp_server.server import MPDCMCPServer
from mpdc.memory.store import MPDCStore
from mpdc.trainer.qlora import OnNodeQLoRATrainer
from mpdc.tui.app import run_btop_tui, seed_demo_datacenter_activity

app = typer.Typer(
    name="mpdc",
    help="MPDC: Mini-PC Local LLM Datacenter (Qwen3 + Multi-Node Token Switchboard + 3-Tier Self-Improving Pipeline)",
    add_completion=False,
)
console = Console()


def _load_context(config_path: Optional[str] = None) -> tuple[MPDCConfig, MPDCStore]:
    cfg = MPDCConfig.load(config_path)
    store = MPDCStore(cfg.data_path / "mpdc.sqlite3")
    return cfg, store


@app.command("serve")
def serve(
    host: Optional[str] = typer.Option(None, help="Bind host override (default 0.0.0.0)"),
    port: Optional[int] = typer.Option(None, help="Bind port override (default 8000)"),
    seed_demo: bool = typer.Option(False, "--seed-demo", help="Seed sample multi-computer nodes and skills on startup"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Start the MPDC OpenAI Gateway (`/v1/chat/completions` with SSE token streaming + Remote MCP `/mcp`)."""
    cfg, store = _load_context(config)
    if seed_demo:
        seed_demo_datacenter_activity(cfg, store)
    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port
    gateway_app = create_gateway_app(cfg)
    console.print(
        Panel(
            f"[bold green]MPDC Datacenter Gateway Online[/bold green]\n"
            f"• OpenAI SSE Stream : [cyan]http://{bind_host}:{bind_port}/v1/chat/completions[/cyan]\n"
            f"• Remote HTTP MCP   : [cyan]http://{bind_host}:{bind_port}/mcp[/cyan]\n"
            f"• HUD Overview API  : [cyan]http://{bind_host}:{bind_port}/v1/mpdc/overview[/cyan]\n"
            f"• Engine & Backbone : [yellow]{cfg.inference.engine}[/yellow] ({cfg.inference.base_model})",
            title="MPDC Local LLM Datacenter",
        )
    )
    uvicorn.run(gateway_app, host=bind_host, port=bind_port)


@app.command("top")
def top(
    seed_demo: bool = typer.Option(True, "--seed-demo/--no-seed-demo", help="Populate demo multi-computer traffic if empty"),
    once: bool = typer.Option(False, "--once", help="Render a single full-screen snapshot and exit"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Launch the native Rust ratatui btop++ 4-quadrant Terminal UI (`mpdc-tui`)."""
    cfg, store = _load_context(config)
    if seed_demo and not store.list_client_nodes():
        seed_demo_datacenter_activity(cfg, store)
    gui_dir = Path(__file__).resolve().parents[2] / "gui" / "mpdc-gui"
    args = ["cargo", "run", "--release", "--bin", "mpdc-tui", "--"]
    if once:
        args.append("--once")
    subprocess.run(args, cwd=str(gui_dir), check=True)


@app.command("gui")
def gui(
    gateway_url: str = typer.Option("http://127.0.0.1:8000", help="MPDC Gateway URL to poll"),
) -> None:
    """Build and launch the native GPU-rendered Rust egui/eframe Mission Control desktop HUD (`gui/mpdc-gui`)."""
    gui_dir = Path(__file__).resolve().parents[2] / "gui" / "mpdc-gui"
    if not gui_dir.exists():
        console.print(f"[red]GUI crate not found at {gui_dir}[/red]")
        raise typer.Exit(1)
    env = os.environ.copy()
    env["MPDC_GATEWAY_URL"] = gateway_url
    console.print(f"[bold cyan]Launching native Rust egui Mission Control HUD ({gui_dir})...[/bold cyan]")
    subprocess.run(["cargo", "run", "--release"], cwd=str(gui_dir), env=env, check=True)


@app.command("client-config")
def client_config(
    node_id: str = typer.Option("rtx5070-laptop", "--node", help="Identifier for the remote client computer"),
    datacenter_host: str = typer.Option("mpdc.local", "--host", help="LAN or Tailscale hostname/IP of the MPDC Mini-PC"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Generate a drop-in `opencode.json` for a remote laptop/computer to stream tokens & MCP tools from MPDC."""
    cfg, store = _load_context(config)
    store.upsert_client_node(node_id=node_id, hostname=node_id, ip_address=datacenter_host)
    payload = build_client_opencode_config(datacenter_host, cfg.server.port, node_id)
    typer.echo(json.dumps(payload, indent=2))


@app.command("mcp")
def mcp(
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Run the stdio JSON-RPC MCP server for OpenCode (`mpdc_plan`, `mpdc_learn`, `mpdc_stats`)."""
    cfg = MPDCConfig.load(config)
    server = MPDCMCPServer(cfg)
    asyncio.run(server.serve_stdio())


@app.command("stats")
def stats(
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON scorecard"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Display Pillar 1 Online Self-Improvement Telemetry (`Pass@1`, `ITG`, `PSR`, `Skill Lift`)."""
    cfg, store = _load_context(config)
    quantifier = SelfImprovementQuantifier(cfg, store)
    scorecard = quantifier.compute_online_scorecard()

    if json_out:
        typer.echo(json.dumps(scorecard, indent=2))
        return

    table = Table(title="MPDC Self-Improvement Longitudinal Scorecard")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold green")
    table.add_column("Interpretation", style="dim")

    table.add_row("Total Verified Tasks", str(scorecard["total_tasks"]), "Tracked OpenCode coding cycles")
    table.add_row(
        "Workflow Pass@1",
        f"{scorecard['pass_at_1']:.1%}",
        "% of tasks passing build/test on 1st attempt",
    )
    table.add_row(
        "Mean Iterations-to-Green (ITG)",
        f"{scorecard['mean_iterations_to_green']:.2f}",
        "Avg compiler/test retries per task (lower -> 1.0 is better)",
    )
    table.add_row(
        "Mean Patch Survival Rate (PSR)",
        f"{scorecard['mean_patch_survival_rate']:.1%}",
        "% of agent code surviving into your final git commit untouched",
    )
    table.add_row(
        "Skill Attribution Lift (ΔPass@1)",
        f"{scorecard['skill_attribution_lift_pass_at_1']:+.1%}",
        "Pass@1 gain when Tier 1 Episodic Skills are injected",
    )
    table.add_row(
        "Episodic Skill Bank",
        f"{scorecard['active_skills']} active / {scorecard['pruned_poison_skills']} auto-pruned",
        "Tier 1 zero-wait memory cards",
    )
    table.add_row(
        "Harvested Trajectories (80/20)",
        f"{scorecard['train_trajectories']} train / {scorecard['golden_replay_holdout']} golden replay",
        "Tier 2 SFT/DPO vs Offline Holdout Split",
    )
    table.add_row(
        "Active Model / LoRA Adapter",
        str(scorecard["active_adapter"]),
        f"Trend: {scorecard['trend']}",
    )
    console.print(table)


@app.command("harvest")
def harvest(
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Export accumulated trajectories to SFT, DPO, and Golden Replay JSONL files."""
    cfg, store = _load_context(config)
    harvester = TrajectoryHarvester(cfg, store)
    summary = harvester.export_datasets()
    console.print_json(data=summary)


@app.command("train")
def train(
    now: bool = typer.Option(True, "--now", help="Trigger on-node QLoRA training cycle immediately"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run pipeline & gated promotion without GPU weight updates"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Run Tier 3 On-Node QLoRA Training + Pillar 2 Gated Evaluation & vLLM LoRA Hot-Swap."""
    cfg, store = _load_context(config)
    trainer = OnNodeQLoRATrainer(cfg, store)
    report = asyncio.run(trainer.run_training_cycle(dry_run=dry_run))
    payload = {
        "version": report.version,
        "adapter_dir": report.adapter_dir,
        "mode": report.mode,
        "sft_samples": report.sft_samples,
        "dpo_pairs": report.dpo_pairs,
        "promoted": report.evaluation.promoted,
        "gate_decision": report.evaluation.reason,
        "hot_loaded_into_vllm": report.hot_loaded_into_vllm,
    }
    console.print_json(data=payload)


@app.command("eval")
def evaluate_cmd(
    candidate_version: str = typer.Option("qwen3-lora-candidate", help="Candidate adapter version tag"),
    candidate_path: str = typer.Option("./.mpdc_data/adapters/candidate", help="Path to PEFT adapter"),
    replay_pass_rate: float = typer.Option(0.65, help="Measured Golden Replay Pass@1"),
    perplexity: float = typer.Option(2.85, help="Measured Holdout Perplexity"),
    sanity_score: float = typer.Option(1.00, help="Measured Core Coding Sanity Score"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Run Pillar 2 Offline Gated Evaluation comparing a candidate adapter against production."""
    cfg, store = _load_context(config)
    quantifier = SelfImprovementQuantifier(cfg, store)
    res = quantifier.evaluate_candidate_adapter(
        candidate_version=candidate_version,
        candidate_path=candidate_path,
        candidate_replay_pass_rate=replay_pass_rate,
        candidate_perplexity=perplexity,
        candidate_sanity_score=sanity_score,
    )
    console.print_json(data=res.__dict__)


if __name__ == "__main__":
    app()
