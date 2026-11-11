"""Ultra-Dense btop/htop-style 4-Quadrant Mission Control TUI (`mpdc top`) with TrueColor RGB Gradients."""

from __future__ import annotations

import asyncio
import json
import select
import sys
import termios
import time
import tty
from typing import Optional

from rich. box import ROUNDED
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mpdc.agents.orchestrator import MultiAgentOrchestrator
from mpdc.config import MPDCConfig
from mpdc.evaluation.quantifier import SelfImprovementQuantifier
from mpdc.gateway.server import build_client_opencode_config, query_gpu_telemetry
from mpdc.harvester.collector import TrajectoryHarvester
from mpdc.memory.store import MPDCStore
from mpdc.trainer.qlora import OnNodeQLoRATrainer

SPARK_CHARS = " ▂▃▄▅▆▇█"
FRAC_BLOCKS = " ▏▎▍▌▋▊▉█"

# TrueColor RGB 3-stop palettes modeled after btop++ / htop
PALETTES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    # Low = cool cyan/green -> Mid = warm gold -> High = hot coral/red (for GPU Util, VRAM, KV, Power)
    "load": ((0, 230, 160), (255, 210, 40), (255, 65, 54)),
    # Low = red/coral -> Mid = gold -> High = neon emerald (for Pass@1, PSR, Skill Utility, Replay@1)
    "quality": ((255, 75, 75), (255, 205, 55), (0, 240, 145)),
    # Deep blue -> electric cyan -> neon mint (for tok/s throughput & share bars)
    "stream": ((45, 135, 255), (0, 215, 255), (0, 245, 160)),
}


def _lerp_rgb(
    c1: tuple[int, int, int], c2: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * t)),
        int(round(c1[1] + (c2[1] - c1[1]) * t)),
        int(round(c1[2] + (c2[2] - c1[2]) * t)),
    )


def _sample_palette(palette_name: str, pos: float) -> str:
    c_low, c_mid, c_high = PALETTES.get(palette_name, PALETTES["stream"])
    pos = max(0.0, min(1.0, pos))
    if pos < 0.5:
        r, g, b = _lerp_rgb(c_low, c_mid, pos * 2.0)
    else:
        r, g, b = _lerp_rgb(c_mid, c_high, (pos - 0.5) * 2.0)
    return f"#{r:02x}{g:02x}{b:02x}"


def _gradient_bar(
    ratio: float,
    width: int = 10,
    palette: str = "load",
    empty_color: str = "#1f242c",
) -> str:
    """Renders a btop/htop-style smooth RGB gradient bar with fractional sub-block precision."""
    clamped = max(0.0, min(1.0, ratio))
    exact_fill = clamped * width
    full_cells = int(exact_fill)
    rem = exact_fill - full_cells
    frac_idx = int(round(rem * 8))
    if frac_idx == 8:
        full_cells += 1
        frac_idx = 0

    parts: list[str] = []
    for i in range(width):
        # Color gradient position across the bar (or by ratio for quality metrics)
        grad_pos = (i / max(1, width - 1)) if palette != "quality" else clamped
        col = _sample_palette(palette, grad_pos)
        if i < full_cells:
            parts.append(f"[{col}]█[/{col}]")
        elif i == full_cells and frac_idx > 0:
            ch = FRAC_BLOCKS[frac_idx]
            parts.append(f"[{col} on {empty_color}]{ch}[/{col} on {empty_color}]")
        else:
            parts.append(f"[{empty_color}]░[/{empty_color}]")
    return "".join(parts)


def _gradient_sparkline(values: list[float], width: int = 24) -> str:
    """Renders a btop-style multi-color RGB sparkline."""
    if not values:
        return ""
    tail = values[-width:]
    lo = min(tail)
    hi = max(tail)
    span = max(1e-6, hi - lo)
    out: list[str] = []
    for v in tail:
        norm = (v - lo) / span
        idx = max(1, min(len(SPARK_CHARS) - 1, int(round(norm * (len(SPARK_CHARS) - 1)))))
        col = _sample_palette("stream", 0.25 + 0.75 * norm)
        out.append(f"[bold {col}]{SPARK_CHARS[idx]}[/bold {col}]")
    return "".join(out)


def _kv_grid(rows: list[tuple[str, str]]) -> Table:
    """Creates a strict single-line, zero-wrap 2-column grid so Panel borders never misalign."""
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column("k", style="bold #79c0ff", width=6, no_wrap=True)
    grid.add_column("v", ratio=1, no_wrap=True, overflow="ellipsis")
    for k, v in rows:
        grid.add_row(k, Text.from_markup(v, overflow="ellipsis"))
    return grid


def seed_demo_datacenter_activity(cfg: MPDCConfig, store: MPDCStore) -> str:
    """Seeds realistic multi-computer token traffic, episodic skills, and QLoRA gate history."""
    orchestrator = MultiAgentOrchestrator(cfg, store)
    harvester = TrajectoryHarvester(cfg, store)

    store.upsert_client_node(
        node_id="rtx5070-laptop",
        hostname="sgk-nixos-laptop",
        ip_address="192.168.1.42",
        prompt_tokens_delta=4200,
        completion_tokens_delta=1850,
        tok_per_sec=68.4,
        active_model=orchestrator.active_model_id(),
    )
    store.upsert_client_node(
        node_id="ros2-field-deck",
        hostname="ros2-humble-deck",
        ip_address="100.94.18.12",
        prompt_tokens_delta=2800,
        completion_tokens_delta=1120,
        tok_per_sec=64.1,
        active_model=orchestrator.active_model_id(),
    )
    store.upsert_client_node(
        node_id="workstation-sim",
        hostname="isaac-sim-rig",
        ip_address="192.168.1.88",
        prompt_tokens_delta=6100,
        completion_tokens_delta=3090,
        tok_per_sec=71.2,
        active_model="qwen3-mpdc-fast",
    )

    if not store.list_skills():
        s1 = orchestrator.distill_lesson_heuristic(
            task_prompt="ROS2 Humble sensor_msgs/LaserScan subscriber QoS mismatch",
            failed_attempt="QoSProfile(depth=10) # Default RELIABLE drops BestEffort lidar packets",
            working_solution="QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)",
            error_trace="New publisher discovered on '/scan', offering incompatible QoS",
            domain="robotics",
        )
        s2 = orchestrator.distill_lesson_heuristic(
            task_prompt="VLA action chunk transformer output shape mismatch",
            failed_attempt="actions = head(latent).reshape(batch_size, -1)",
            working_solution="actions = head(latent).view(batch_size, chunk_horizon, action_dim)",
            error_trace="RuntimeError: shape '[16, 7]' is invalid for input of size 896",
            domain="robotics",
        )
        s3 = orchestrator.distill_lesson_heuristic(
            task_prompt="NixOS CUDA systemd service libcuda.so.1 missing in environment",
            failed_attempt="serviceConfig.ExecStart = '${pkgs.vllm}/bin/vllm serve';",
            working_solution="hardware.nvidia-container-toolkit.enable = true; # bind /run/opengl-driver/lib",
            error_trace="ImportError: libcuda.so.1: cannot open shared object file",
            domain="nixos",
        )
        for i in range(8):
            harvester.ingest_coding_outcome(
                task_id=f"demo-task-{int(time.time())}-{i}",
                prompt=f"Implement ROS2 lifecycle publisher #{i}",
                agent_initial_code="qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)",
                final_committed_code="qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)",
                iterations_to_green=1 if i % 4 != 0 else 2,
                domain="robotics",
                skills_used=[s1, s2 if i % 2 == 0 else s3],
                adapter_version=orchestrator.active_model_id(),
            )
    if not store.list_adapters():
        store.register_adapter(
            version="qwen3-lora-v001",
            base_model=cfg.inference.base_model,
            adapter_path=str(cfg.data_path / "adapters" / "qwen3-lora-v001"),
            replay_pass_rate=0.68,
            holdout_perplexity=2.31,
            sanity_score=1.0,
            promoted=True,
        )
    return "Seeded 3 client computers + 3 episodic skills + 8 trajectories + promoted adapter v001."


class MPDCMissionControlRenderer:
    """Renders the ultra-dense 4-quadrant Mission Control HUD layout with strict 1-cell border alignment."""

    def __init__(self, config: Optional[MPDCConfig] = None) -> None:
        self.cfg = config or MPDCConfig.load()
        self.store = MPDCStore(self.cfg.data_path / "mpdc.sqlite3")
        self.quantifier = SelfImprovementQuantifier(self.cfg, self.store)
        self.orchestrator = MultiAgentOrchestrator(self.cfg, self.store)
        self.trainer = OnNodeQLoRATrainer(self.cfg, self.store)
        self.tps_history: list[float] = [
            142.0, 156.4, 168.0, 175.2, 189.0, 194.5, 201.2, 198.4, 205.1, 203.7
        ]
        self.selected_node_idx: int = 0
        self.events: list[str] = [
            "[bold #58a6ff]>> MPDC btop++ Mission Control online.[/bold #58a6ff]",
            "[#8b949e]Keys: [bold #ffd200]j/k[/bold #ffd200] Select Node | [bold #ffd200]p[/bold #ffd200] Pause/Resume | [bold #ffd200]o[/bold #ffd200] Export opencode.json[/#8b949e]",
            "[#8b949e]      [bold #ffd200]s[/bold #ffd200] Toggle Skill | [bold #ffd200]t[/bold #ffd200] QLoRA Sleep-Train | [bold #ffd200]d[/bold #ffd200] Seed Demo | [bold #ffd200]q[/bold #ffd200] Quit[/#8b949e]",
        ]

    def build_layout(self) -> Layout:
        gpu = query_gpu_telemetry()
        scorecard = self.quantifier.compute_online_scorecard()
        nodes = self.store.list_client_nodes()
        skills = self.store.list_skills(include_inactive=True)
        adapters = self.store.list_adapters()

        total_out = sum(n["completion_tokens"] for n in nodes)
        total_in = sum(n["prompt_tokens"] for n in nodes)
        total_reqs = sum(n["requests_count"] for n in nodes)
        online_nodes = sum(1 for n in nodes if n["status"] != "paused")
        active_tps = round(
            sum(n["last_tok_per_sec"] for n in nodes if n["status"] != "paused"), 1
        )
        self.tps_history = (self.tps_history + [active_tps])[-26:]

        root = Layout(name="root")
        root.split_column(
            Layout(name="ribbon", size=6),
            Layout(name="mid", size=9),
            Layout(name="bot", ratio=1),
            Layout(name="footer", size=1),
        )

        # ─────── ROW 0: 4-COLUMN HARDWARE & TELEMETRY RIBBON (Strict 4-row grids = 6 lines with border) ───────
        util_bar = _gradient_bar(gpu["gpu_util_pct"], 10, "load")
        vram_bar = _gradient_bar(gpu["vram_pct"], 10, "load")
        kv_bar = _gradient_bar(gpu["kv_cache_pct"], 10, "stream")
        pwr_bar = _gradient_bar(gpu["power_w"] / max(1.0, gpu["power_cap_w"]), 10, "load")

        col_gpu = Panel(
            _kv_grid(
                [
                    ("CORE", f"{util_bar} [bold]{gpu['gpu_util_pct']*100:3.0f}%[/bold] [#ffd200]{gpu['temp_c']:.0f}C[/#ffd200]"),
                    ("VRAM", f"{vram_bar} [bold]{gpu['vram_used_mb']:.0f}[/bold]/{gpu['vram_total_mb']:.0f}M"),
                    ("KV$", f"{kv_bar} [bold]{gpu['kv_cache_pct']*100:3.0f}%[/bold] Paged"),
                    ("PWR", f"{pwr_bar} [bold]{gpu['power_w']:4.1f}[/bold]/{gpu['power_cap_w']:.0f}W"),
                ]
            ),
            title=f"[bold #00f5a0]GPU · {gpu['name'][:19]}[/bold #00f5a0]",
            border_style="#238636",
            box=ROUNDED,
            padding=(0, 0),
        )

        col_eng = Panel(
            _kv_grid(
                [
                    ("ENG", f"[bold #58a6ff]{self.cfg.inference.engine.upper()}[/bold #58a6ff] (:{self.cfg.server.port}/v1)"),
                    ("BASE", f"[white]{self.cfg.inference.base_model}[/white]"),
                    ("LORA", f"[bold #d2a8ff]{self.orchestrator.active_model_id()}[/bold #d2a8ff]"),
                    ("TOKS", f"[bold #00f5a0]{total_out:,} OUT[/bold #00f5a0] | {total_in:,} IN ({total_reqs})"),
                ]
            ),
            title="[bold #58a6ff]SWITCHBOARD & LORA[/bold #58a6ff]",
            border_style="#1f6feb",
            box=ROUNDED,
            padding=(0, 0),
        )

        p1_bar = _gradient_bar(scorecard["pass_at_1"], 8, "quality")
        psr_bar = _gradient_bar(scorecard["mean_patch_survival_rate"], 8, "quality")
        lift = scorecard["skill_attribution_lift_pass_at_1"]
        col_imp = Panel(
            _kv_grid(
                [
                    ("PASS1", f"{p1_bar} [bold #00f5a0]{scorecard['pass_at_1']*100:5.1f}%[/bold #00f5a0] ({scorecard['mean_iterations_to_green']:.2f}x)"),
                    ("PSR", f"{psr_bar} [bold #00d2ff]{scorecard['mean_patch_survival_rate']*100:5.1f}%[/bold #00d2ff] intact"),
                    ("LIFT", f"[bold #ffd200]{lift*100:+5.1f}%[/bold #ffd200] | {scorecard['active_skills']} act / {scorecard['pruned_poison_skills']} prn"),
                    ("SPLIT", f"[#58a6ff]{scorecard['train_trajectories']} train[/#58a6ff] / [#d2a8ff]{scorecard['golden_replay_holdout']} eval (80/20)[/#d2a8ff]"),
                ]
            ),
            title="[bold #ffd200]SELF-IMPROVEMENT KPI[/bold #ffd200]",
            border_style="#d29922",
            box=ROUNDED,
            padding=(0, 0),
        )

        peak_tps = max(self.tps_history) if self.tps_history else 0.0
        col_spk = Panel(
            _kv_grid(
                [
                    ("RATE", f"[bold #00f5a0]{active_tps:5.1f} tok/s[/bold #00f5a0] [dim](pk {peak_tps:.0f})[/dim]"),
                    ("WAVE", _gradient_sparkline(self.tps_history, 20)),
                    ("NODES", f"[bold #00f5a0]{online_nodes}[/bold #00f5a0]/{len(nodes)} active streams"),
                    ("LAT", "[dim]TTFT 18ms | ITL 14.2ms/t[/dim]"),
                ]
            ),
            title="[bold #00f5a0]CLUSTER STREAM[/bold #00f5a0]",
            border_style="#2ea043",
            box=ROUNDED,
            padding=(0, 0),
        )

        root["ribbon"].split_row(col_gpu, col_eng, col_imp, col_spk)

        # ─────── ROW 1: Q1 OUTPUT NODES MATRIX + Q2 LIVE INSPECTOR ───────
        nodes_tbl = Table(expand=True, box=None, pad_edge=False, show_edge=False, collapse_padding=True)
        nodes_tbl.add_column("", width=1, no_wrap=True)
        nodes_tbl.add_column("ST", width=5, no_wrap=True)
        nodes_tbl.add_column("NODE_ID", style="bold white", ratio=2, no_wrap=True, overflow="ellipsis")
        nodes_tbl.add_column("HOST@IP", style="#8b949e", ratio=3, no_wrap=True, overflow="ellipsis")
        nodes_tbl.add_column("TOK/S", width=14, no_wrap=True)
        nodes_tbl.add_column("SHARE", width=12, no_wrap=True)
        nodes_tbl.add_column("IN", justify="right", width=6, no_wrap=True)
        nodes_tbl.add_column("OUT", justify="right", style="bold #00f5a0", width=6, no_wrap=True)
        nodes_tbl.add_column("MODEL", style="#d2a8ff", ratio=2, no_wrap=True, overflow="ellipsis")

        if nodes:
            self.selected_node_idx %= len(nodes)
        for idx, n in enumerate(nodes):
            sel = "[bold #ffd200]>[/bold #ffd200]" if idx == self.selected_node_idx else " "
            paused = n["status"] == "paused"
            st = "[bold #ff5555]■ PAU[/bold #ff5555]" if paused else "[bold #00f5a0]● ONL[/bold #00f5a0]"
            tps = 0.0 if paused else float(n["last_tok_per_sec"])
            tps_bar = _gradient_bar(tps / 100.0, 7, "stream")
            share = n["completion_tokens"] / max(1, total_out)
            sh_bar = _gradient_bar(share, 6, "stream")
            nodes_tbl.add_row(
                sel,
                st,
                n["node_id"],
                f"{n['hostname']}@{n['ip_address']}",
                f"{tps_bar} {tps:4.1f}",
                f"{sh_bar} {share*100:3.0f}%",
                f"{n['prompt_tokens']:,}",
                f"{n['completion_tokens']:,}",
                n["active_model"],
            )

        q1_panel = Panel(
            nodes_tbl,
            title="[bold #58a6ff]Q1 · CONNECTED OUTPUT COMPUTERS MATRIX (LAN / TAILSCALE SSE)[/bold #58a6ff]",
            subtitle="[dim]\\[j/k] Select | \\[p] Pause/Resume | \\[o] Client opencode.json[/dim]",
            border_style="#1f6feb",
            box=ROUNDED,
            padding=(0, 1),
        )

        bus_grid = Table.grid(expand=True)
        bus_grid.add_column("line", ratio=1, no_wrap=True, overflow="ellipsis")
        for line in self.events[-6:]:
            bus_grid.add_row(Text.from_markup(line, overflow="ellipsis"))

        q2_panel = Panel(
            bus_grid,
            title="[bold #00d2ff]Q2 · LIVE STREAM BUS & INSPECTOR[/bold #00d2ff]",
            border_style="#00a8cc",
            box=ROUNDED,
            padding=(0, 1),
        )
        root["mid"].split_row(
            Layout(q1_panel, ratio=3),
            Layout(q2_panel, ratio=2),
        )

        # ─────── ROW 2: Q3 EPISODIC SKILL MATRIX + Q4 QLORA ADAPTER GATE ───────
        skills_tbl = Table(expand=True, box=None, pad_edge=False, show_edge=False, collapse_padding=True)
        skills_tbl.add_column("ID", width=3, no_wrap=True)
        skills_tbl.add_column("ST", width=3, no_wrap=True)
        skills_tbl.add_column("UTILITY", width=13, no_wrap=True)
        skills_tbl.add_column("DOM", style="#79c0ff", width=7, no_wrap=True)
        skills_tbl.add_column("W/L", width=4, no_wrap=True)
        skills_tbl.add_column("ANTI-PATTERN -> VERIFIED FIX", ratio=1, no_wrap=True, overflow="ellipsis")

        for s in skills[:8]:
            st = "[bold #00f5a0]ACT[/bold #00f5a0]" if s["active"] else "[bold #ff5555]PRN[/bold #ff5555]"
            ubar = _gradient_bar(s["utility_score"], 7, "quality")
            wl = f"{s['success_count']}/{s['failure_count']}"
            skills_tbl.add_row(
                f"#{s['id']}",
                st,
                f"{ubar} {s['utility_score']:.2f}",
                s["domain"][:7],
                wl,
                f"[#ff7b72]- {s['anti_pattern'][:22]}[/#ff7b72] -> [#7ee787]+ {s['verified_pattern'][:32]}[/#7ee787]",
            )

        q3_panel = Panel(
            skills_tbl,
            title="[bold #ffd200]Q3 · TIER 1 EPISODIC SKILL MATRIX (Auto-Prunes <0.30 Utility)[/bold #ffd200]",
            subtitle="[dim]\\[s] Toggle Top Skill Active/Pruned[/dim]",
            border_style="#d29922",
            box=ROUNDED,
            padding=(0, 1),
        )

        adapters_tbl = Table(expand=True, box=None, pad_edge=False, show_edge=False, collapse_padding=True)
        adapters_tbl.add_column("GATE", width=10, no_wrap=True)
        adapters_tbl.add_column("ADAPTER_VER", style="bold white", ratio=1, no_wrap=True, overflow="ellipsis")
        adapters_tbl.add_column("REPLAY@1", width=13, no_wrap=True)
        adapters_tbl.add_column("PPL", style="#79c0ff", width=5, no_wrap=True)
        adapters_tbl.add_column("SAN", width=4, no_wrap=True)

        for a in adapters[:8]:
            gate = (
                "[bold #00f5a0]+ PROMOTED[/bold #00f5a0]"
                if a["promoted"]
                else "[bold #ff5555]x ROLLBACK[/bold #ff5555]"
            )
            rbar = _gradient_bar(a["replay_pass_rate"], 7, "quality")
            adapters_tbl.add_row(
                gate,
                a["version"],
                f"{rbar} {a['replay_pass_rate']*100:3.0f}%",
                f"{a['holdout_perplexity']:4.2f}",
                f"{a['sanity_score']*100:3.0f}%",
            )

        q4_panel = Panel(
            adapters_tbl,
            title="[bold #d2a8ff]Q4 · TIER 3 QLORA PROMOTION GATE (80/20 Replay)[/bold #d2a8ff]",
            subtitle="[dim]\\[t] Run On-Node QLoRA Sleep-Train Cycle[/dim]",
            border_style="#8957e5",
            box=ROUNDED,
            padding=(0, 1),
        )

        root["bot"].split_row(
            Layout(q3_panel, ratio=3),
            Layout(q4_panel, ratio=2),
        )

        footer_grid = Table.grid(expand=True)
        footer_grid.add_column("f", ratio=1, no_wrap=True, overflow="ellipsis")
        footer_grid.add_row(
            Text.from_markup(
                "[bold black on #00f5a0] MPDC btop++ [/bold black on #00f5a0] "
                "[bold #ffd200]\\[j/k][/bold #ffd200] Select Node  "
                "[bold #ffd200]\\[p][/bold #ffd200] Pause/Resume  "
                "[bold #ffd200]\\[o][/bold #ffd200] Client JSON  "
                "[bold #ffd200]\\[s][/bold #ffd200] Toggle Skill  "
                "[bold #ffd200]\\[t][/bold #ffd200] QLoRA Train  "
                "[bold #ffd200]\\[d][/bold #ffd200] Seed Demo  "
                "[bold #ffd200]\\[q][/bold #ffd200] Quit",
                overflow="ellipsis",
            )
        )
        root["footer"].update(footer_grid)
        return root

    def handle_keypress(self, key: str) -> bool:
        """Handles single-keystroke commands in raw tty mode. Returns False on quit."""
        if key in ("q", "Q", "\x03"):
            return False
        nodes = self.store.list_client_nodes()
        if key == "j" and nodes:
            self.selected_node_idx = (self.selected_node_idx + 1) % len(nodes)
        elif key == "k" and nodes:
            self.selected_node_idx = (self.selected_node_idx - 1) % len(nodes)
        elif key == "p" and nodes:
            target = nodes[self.selected_node_idx % len(nodes)]
            updated = self.store.toggle_client_node_pause(target["node_id"])
            if updated:
                st = updated["status"].upper()
                col = "#ff5555" if st == "PAUSED" else "#00f5a0"
                self.events.append(
                    f"[{col}]>> Node '{updated['node_id']}' ({updated['ip_address']}) -> {st}[/{col}]"
                )
        elif key == "s":
            skills = self.store.list_skills(include_inactive=True)
            if skills:
                updated = self.store.toggle_skill(int(skills[0]["id"]))
                if updated:
                    st = "ACTIVE" if updated["active"] else "PRUNED"
                    self.events.append(f"[#00d2ff]>> Skill #{updated['id']} -> {st}[/#00d2ff]")
        elif key == "o" and nodes:
            target = nodes[self.selected_node_idx % len(nodes)]
            cfg_json = build_client_opencode_config(
                "mpdc.local", self.cfg.server.port, target["node_id"]
            )
            compact = json.dumps(cfg_json)
            self.events.append(
                f"[bold #00f5a0]>> opencode.json ({target['node_id']}):[/bold #00f5a0] {compact}"
            )
        elif key == "d":
            msg = seed_demo_datacenter_activity(self.cfg, self.store)
            self.events.append(f"[bold #00f5a0]>> {msg}[/bold #00f5a0]")
        elif key == "t":
            report = asyncio.run(self.trainer.run_training_cycle(dry_run=True))
            col = "#00f5a0" if report.evaluation.promoted else "#ff5555"
            self.events.append(
                f"[bold {col}]>> QLoRA ({report.version}): {report.evaluation.reason}[/bold {col}]"
            )
        return True


def run_btop_tui(config: Optional[MPDCConfig] = None, once: bool = False) -> None:
    renderer = MPDCMissionControlRenderer(config)
    console = Console()
    if once or not sys.stdin.isatty():
        console.print(renderer.build_layout())
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        with Live(renderer.build_layout(), console=console, screen=True, refresh_per_second=4) as live:
            while True:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.35)
                if rlist:
                    ch = sys.stdin.read(1)
                    if not renderer.handle_keypress(ch):
                        break
                live.update(renderer.build_layout())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
