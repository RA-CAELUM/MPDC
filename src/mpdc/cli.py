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

from mpdc.agents.orchestrator import MultiAgentOrchestrator, SpatialObservation
from mpdc.config import MPDCConfig, ModelWorkerConfig
from mpdc.evaluation.quantifier import SelfImprovementQuantifier
from mpdc.gateway.server import (
    build_client_opencode_config,
    build_mpdc_native_client_profile,
    create_gateway_app,
)
from mpdc.harvester.collector import TrajectoryHarvester
from mpdc.mcp_server.server import MPDCMCPServer
from mpdc.memory.store import MPDCStore
from mpdc.trainer.qlora import OnNodeQLoRATrainer
from mpdc.tui.app import seed_demo_datacenter_activity

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
    import shutil

    which_bin = shutil.which("mpdc-tui")
    gui_dir = Path(__file__).resolve().parents[2] / "gui" / "mpdc-gui"
    bin_path = gui_dir / "target" / "release" / "mpdc-tui"
    if which_bin:
        args = [which_bin]
        run_cwd = None
    elif bin_path.exists():
        args = [str(bin_path)]
        run_cwd = None
    else:
        args = ["cargo", "run", "--release", "--bin", "mpdc-tui", "--"]
        run_cwd = str(gui_dir)
    if once:
        args.append("--once")
    subprocess.run(args, cwd=run_cwd, check=True)


@app.command("gui")
def gui(
    gateway_url: str = typer.Option("http://127.0.0.1:8000", help="MPDC Gateway URL to poll"),
) -> None:
    """Build and launch the native GPU-rendered Rust egui/eframe Mission Control desktop HUD (`gui/mpdc-gui`)."""
    import shutil

    env = os.environ.copy()
    env["MPDC_GATEWAY_URL"] = gateway_url
    which_bin = shutil.which("mpdc-gui")
    gui_dir = Path(__file__).resolve().parents[2] / "gui" / "mpdc-gui"
    bin_path = gui_dir / "target" / "release" / "mpdc-gui"
    if which_bin:
        subprocess.run([which_bin], env=env, check=True)
    elif bin_path.exists():
        subprocess.run([str(bin_path)], env=env, check=True)
    else:
        subprocess.run(
            ["cargo", "run", "--release", "--bin", "mpdc-gui"],
            cwd=str(gui_dir),
            env=env,
            check=True,
        )


@app.command("code")
def code_cmd(
    prompt: Optional[str] = typer.Argument(None, help="Coding or spatial-engineering task instruction (omit to launch interactive TUI)"),
    tui: bool = typer.Option(False, "--tui", help="Launch the interactive Claude-Code-style coding & planning TUI"),
    once: bool = typer.Option(False, "--once", help="Render a single TUI snapshot and exit"),
    domain: str = typer.Option("robotics", "--domain", "-d", help="Domain tag (robotics, nixos, cpp, vla)"),
    images: Optional[list[str]] = typer.Option(None, "--image", "-i", help="Attach 2D/video frames or sim viewport renders"),
    pointclouds: Optional[list[str]] = typer.Option(None, "--pointcloud", "-p", help="Attach 3D point clouds (.ply/.pcd) or depth maps"),
    sim_state: str = typer.Option("", "--sim-state", help="Optional SE(3) pose / URDF / MJCF JSON state"),
    coding_models: Optional[str] = typer.Option(None, "--coding-models", help="Comma-separated coding model IDs (default: all active)"),
    spatial_models: Optional[str] = typer.Option(None, "--spatial-models", help="Comma-separated visual/spatial model IDs (default: all active)"),
    mode: str = typer.Option("sensor_fusion_loop", "--mode", "-m", help="Coordination mode: sensor_fusion_loop | parallel_ensemble | specialist_graph"),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON swarm trace"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Native MPDC Multi-Agent Coding & Planning TUI (`mpdc code`) + One-Shot Swarm CLI."""
    if tui or once or prompt is None:
        import shutil

        which_bin = shutil.which("mpdc-code")
        gui_dir = Path(__file__).resolve().parents[2] / "gui" / "mpdc-gui"
        local_bin = gui_dir / "target" / "release" / "mpdc-code"
        if which_bin:
            args = [which_bin]
            run_cwd = None
        elif local_bin.exists():
            args = [str(local_bin)]
            run_cwd = None
        else:
            args = ["cargo", "run", "--release", "--bin", "mpdc-code", "--"]
            run_cwd = str(gui_dir)
        if once:
            args.append("--once")
        if prompt:
            args.append(prompt)
        subprocess.run(args, cwd=run_cwd, check=True)
        return
    cfg, store = _load_context(config)
    orchestrator = MultiAgentOrchestrator(cfg, store)

    c_ids = [s.strip() for s in coding_models.split(",") if s.strip()] if coding_models else None
    s_ids = [s.strip() for s in spatial_models.split(",") if s.strip()] if spatial_models else None

    res = asyncio.run(
        orchestrator.execute_multimodal_swarm(
            prompt=prompt,
            domain=domain,
            observation=SpatialObservation(
                image_paths=images or [],
                pointcloud_paths=pointclouds or [],
                sim_state_json=sim_state,
            ),
            coding_model_ids=c_ids,
            spatial_model_ids=s_ids,
            coordination_mode=mode,
        )
    )

    if json_out:
        typer.echo(
            json.dumps(
                {
                    "coordination_mode": res.coordination_mode,
                    "spatial_scene_graph": res.spatial_scene_graph,
                    "spatial_steps": [s.__dict__ for s in res.spatial_steps],
                    "coding_steps": [s.__dict__ for s in res.coding_steps],
                    "synthesized_patch": res.synthesized_patch,
                    "skills_injected": res.skills_injected,
                },
                indent=2,
            )
        )
        return

    console.print(
        Panel(
            f"[bold #38bdf8]MPDC Native Multi-Agent Swarm[/bold #38bdf8] | Mode: [bold #a78bfa]{res.coordination_mode}[/bold #a78bfa]\n"
            f"• Spatial/Visual Agents : [cyan]{len(res.spatial_steps)}[/cyan] ({', '.join(s.agent_id for s in res.spatial_steps)})\n"
            f"• Programming Agents    : [magenta]{len(res.coding_steps)}[/magenta] ({', '.join(s.agent_id for s in res.coding_steps)})\n"
            f"• Episodic Skills Used  : [green]{len(res.skills_injected)}[/green]",
            title="MPDC // Man Portable Data Center",
        )
    )
    for st in res.spatial_steps:
        console.print(Panel(st.output, title=f"[#38bdf8][SPATIAL/VISUAL] {st.role} ({st.model_name})[/#38bdf8]"))
    for ct in res.coding_steps:
        console.print(Panel(ct.output, title=f"[#f472b6][PROGRAMMING] {ct.role} ({ct.model_name})[/#f472b6]"))


@app.command("pools")
def pools_cmd(
    add_id: Optional[str] = typer.Option(None, "--add-id", help="Register a new model worker ID"),
    add_name: Optional[str] = typer.Option(None, "--add-name", help="Model API identifier (e.g. Qwen/Qwen3-Coder-30B-A3B-Instruct)"),
    add_role: str = typer.Option("SPECIALIST_AGENT", "--add-role", help="Role label"),
    add_category: str = typer.Option("coding", "--add-category", help="Category: coding | reasoning | vision | spatial_3d | geometry"),
    add_endpoint: str = typer.Option("http://127.0.0.1:8001/v1", "--add-endpoint", help="OpenAI-compatible endpoint URL"),
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON model pool registry"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Inspect or add models to the Programming Pool and Visual/Spatial Intelligence Pool."""
    cfg, store = _load_context(config)
    orchestrator = MultiAgentOrchestrator(cfg, store)

    if add_id and add_name:
        orchestrator.register_pool_model(
            ModelWorkerConfig(
                id=add_id,
                name=add_name,
                role=add_role,
                category=add_category,
                endpoint_url=add_endpoint,
                thinking=(add_category in ("reasoning", "spatial_3d")),
            )
        )

    pools = orchestrator.list_model_pools()
    if json_out:
        typer.echo(json.dumps(pools, indent=2))
        return

    table = Table(title=f"MPDC Multi-Agent Model Pools (mode={pools['coordination_mode']})")
    table.add_column("Pool", style="#38bdf8", no_wrap=True)
    table.add_column("ID", style="bold white")
    table.add_column("Role", style="#a78bfa")
    table.add_column("Category", style="#f472b6")
    table.add_column("Model Name", style="cyan")
    table.add_column("Modalities", style="dim")
    table.add_column("Endpoint", style="green")

    for m in pools["coding_models"]:
        table.add_row(
            "PROGRAMMING",
            m["id"],
            m["role"],
            m["category"],
            m["name"],
            ",".join(m["modalities"]),
            m["endpoint_url"],
        )
    for m in pools["spatial_visual_models"]:
        table.add_row(
            "SPATIAL/VIS",
            m["id"],
            m["role"],
            m["category"],
            m["name"],
            ",".join(m["modalities"]),
            m["endpoint_url"],
        )
    console.print(table)


@app.command("client-config")
def client_config(
    node_id: str = typer.Option("rtx5070-laptop", "--node", help="Identifier for the remote client computer"),
    datacenter_host: str = typer.Option("mpdc.local", "--host", help="LAN or Tailscale hostname/IP of the MPDC Mini-PC"),
    legacy_opencode: bool = typer.Option(False, "--legacy-opencode", help="Emit legacy opencode.json instead of native MPDC client profile"),
    config: Optional[str] = typer.Option(None, help="Path to YAML config"),
) -> None:
    """Generate a native `mpdc code` multi-agent client profile for a remote laptop/workstation."""
    cfg, store = _load_context(config)
    orchestrator = MultiAgentOrchestrator(cfg, store)
    store.upsert_client_node(node_id=node_id, hostname=node_id, ip_address=datacenter_host)
    if legacy_opencode:
        payload = build_client_opencode_config(datacenter_host, cfg.server.port, node_id)
    else:
        payload = build_mpdc_native_client_profile(
            datacenter_host, cfg.server.port, node_id, orchestrator.list_model_pools()
        )
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


def run_code_entrypoint() -> None:
    """Direct entrypoint for `mpdc-code` installed via pip/uv."""
    code_cmd(prompt=None, tui=True, once=("--once" in os.sys.argv))


def run_top_entrypoint() -> None:
    """Direct entrypoint for `mpdc-tui` installed via pip/uv."""
    top(seed_demo=True, once=("--once" in os.sys.argv), config=None)


def run_gui_entrypoint() -> None:
    """Direct entrypoint for `mpdc-gui` installed via pip/uv."""
    gui(gateway_url=os.environ.get("MPDC_GATEWAY_URL", "http://127.0.0.1:8000"))


if __name__ == "__main__":
    app()
