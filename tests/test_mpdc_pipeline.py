"""End-to-end verification tests for the MPDC 3-Tier Self-Improving Pipeline & Quantification Engine."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from mpdc.agents.orchestrator import MultiAgentOrchestrator
from mpdc.config import MPDCConfig
from mpdc.evaluation.quantifier import SelfImprovementQuantifier
from mpdc.gateway.server import create_gateway_app
from mpdc.harvester.collector import TrajectoryHarvester, compute_patch_survival_rate
from mpdc.mcp_server.server import MPDCMCPServer
from mpdc.memory.store import MPDCStore
from mpdc.trainer.qlora import OnNodeQLoRATrainer


def _make_test_config(tmp_path: Path) -> MPDCConfig:
    cfg = MPDCConfig()
    cfg.server.data_dir = str(tmp_path / "mpdc_data")
    cfg.agents.min_skill_similarity = 0.15
    return cfg


def test_tier1_episodic_skill_distillation_and_poison_pruning(tmp_path: Path) -> None:
    cfg = _make_test_config(tmp_path)
    store = MPDCStore(cfg.data_path / "mpdc.sqlite3")
    orchestrator = MultiAgentOrchestrator(cfg, store)

    skill_id = orchestrator.distill_lesson_heuristic(
        task_prompt="Fix ROS2 Humble lifecycle sensor publisher QoS mismatch",
        failed_attempt="qos = QoSProfile(depth=10)  # default RELIABLE caused subscriber drop",
        working_solution="qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)",
        error_trace="New publisher discovered on topic '/imu', offering incompatible QoS",
        domain="robotics",
    )
    assert skill_id == 1

    retrieved = store.retrieve_skills(
        "ROS2 Humble incompatible QoS sensor publisher", top_k=3, min_similarity=0.10
    )
    assert len(retrieved) == 1
    assert retrieved[0].id == skill_id
    assert "BEST_EFFORT" in retrieved[0].verified_pattern

    # Simulate 4 consecutive task failures using a poison skill -> verify auto-pruning
    bad_skill_id = store.add_skill(
        title="Bad VLA shape advice",
        domain="robotics",
        trigger_signature="VLA action tensor shape",
        anti_pattern="good code",
        verified_pattern="broken code",
    )
    harvester = TrajectoryHarvester(cfg, store)
    for i in range(4):
        harvester.ingest_coding_outcome(
            task_id=f"fail-task-{i}",
            prompt="Reshape VLA action tensor",
            agent_initial_code="pred.view(-1)",
            final_committed_code="pred.view(batch_size, horizon, action_dim)",
            iterations_to_green=3,
            skills_used=[bad_skill_id],
        )

    scorecard = SelfImprovementQuantifier(cfg, store).compute_online_scorecard()
    assert scorecard["active_skills"] == 1
    assert scorecard["pruned_poison_skills"] == 1


def test_tier2_patch_survival_rate_and_80_20_harvest_split(tmp_path: Path) -> None:
    cfg = _make_test_config(tmp_path)
    store = MPDCStore(cfg.data_path / "mpdc.sqlite3")
    harvester = TrajectoryHarvester(cfg, store)

    # Identical patch -> PSR = 1.0
    assert compute_patch_survival_rate("def foo():\n    return 42\n", "def foo():\n    return 42\n") == 1.0
    # Completely rewritten patch -> low PSR
    psr_partial = compute_patch_survival_rate(
        "x = 1\ny = 2\nz = 3\n",
        "x = 1\ny = 2\nz = 999\nw = 4\n",
    )
    assert 0.5 < psr_partial < 0.9

    # Ingest 25 solved failure trajectories -> verify both 80% Train (SFT/DPO) and 20% Golden Replay are populated
    for idx in range(25):
        harvester.ingest_coding_outcome(
            task_id=f"trajectory-{idx:03d}",
            prompt=f"Implement VLA transformer block #{idx}",
            agent_initial_code=f"def block_{idx}(): return None",
            final_committed_code=f"def block_{idx}(x): return x + 1",
            iterations_to_green=2,
            domain="robotics",
        )

    exported = harvester.export_datasets()
    assert exported["sft_samples"] > 0
    assert exported["dpo_pairs"] == exported["sft_samples"]
    assert exported["golden_replay_samples"] > 0
    assert exported["sft_samples"] + exported["golden_replay_samples"] == 25


def test_tier3_qlora_trainer_and_pillar2_promotion_gate(tmp_path: Path) -> None:
    cfg = _make_test_config(tmp_path)
    store = MPDCStore(cfg.data_path / "mpdc.sqlite3")
    harvester = TrajectoryHarvester(cfg, store)
    trainer = OnNodeQLoRATrainer(cfg, store)

    for idx in range(12):
        harvester.ingest_coding_outcome(
            task_id=f"train-item-{idx}",
            prompt=f"Configure NixOS CUDA service {idx}",
            agent_initial_code="services.mpdc.enable = false;",
            final_committed_code="services.mpdc.enable = true;",
            iterations_to_green=2,
            domain="nixos",
        )

    # 1. Candidate v001 improves Replay Pass@1 (0.25 -> 0.65) and Perplexity (4.50 -> 2.40) -> PROMOTED
    report_v1 = asyncio.run(
        trainer.run_training_cycle(
            dry_run=True,
            simulated_replay_pass_rate=0.65,
            simulated_perplexity=2.40,
            simulated_sanity_score=1.00,
        )
    )
    assert report_v1.version == "qwen3-lora-v001"
    assert report_v1.evaluation.promoted is True
    assert store.get_latest_promoted_adapter()["version"] == "qwen3-lora-v001"

    # 2. Candidate v002 regresses on Sanity Check (drops from 1.00 to 0.80) -> REJECTED (Auto-Rollback to v001)
    report_v2 = asyncio.run(
        trainer.run_training_cycle(
            dry_run=True,
            simulated_replay_pass_rate=0.70,
            simulated_perplexity=2.20,
            simulated_sanity_score=0.80,
        )
    )
    assert report_v2.version == "qwen3-lora-v002"
    assert report_v2.evaluation.promoted is False
    assert "REJECTED" in report_v2.evaluation.reason
    # Active production adapter remains v001
    assert store.get_latest_promoted_adapter()["version"] == "qwen3-lora-v001"


def test_gateway_skill_injection_and_mcp_server(tmp_path: Path) -> None:
    cfg = _make_test_config(tmp_path)
    app = create_gateway_app(cfg)
    client = TestClient(app)

    # Seed an episodic skill via MCP server `mpdc_learn`
    mcp = MPDCMCPServer(cfg)
    learn_resp = asyncio.run(
        mcp.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "mpdc_learn",
                    "arguments": {
                        "task_prompt": "Fix ROS2 QoS sensor reliability",
                        "failed_attempt": "ReliabilityPolicy.RELIABLE",
                        "working_solution": "ReliabilityPolicy.BEST_EFFORT",
                        "domain": "robotics",
                    },
                },
            }
        )
    )
    assert learn_resp is not None
    assert "learned" in learn_resp["result"]["content"][0]["text"]

    # Send an OpenCode chat completion request to `/v1/chat/completions`
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3-mpdc",
            "messages": [{"role": "user", "content": "How should I set ROS2 QoS sensor reliability?"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mpdc_metadata"]["qwen3_thinking"] is True
    assert len(body["mpdc_metadata"]["skills_injected"]) == 1


def test_multinode_sse_streaming_pause_and_hud_overview(tmp_path: Path) -> None:
    cfg = _make_test_config(tmp_path)
    app = create_gateway_app(cfg)
    client = TestClient(app)

    # 1. Register a remote laptop computer (`rtx5070-laptop`)
    reg = client.post(
        "/v1/mpdc/nodes/register",
        json={
            "node_id": "rtx5070-laptop",
            "hostname": "sgk-nixos-laptop",
            "ip_address": "192.168.1.42",
            "datacenter_host": "mpdc.local",
        },
    )
    assert reg.status_code == 200
    reg_data = reg.json()
    assert reg_data["opencode_config"]["provider"]["mpdc"]["options"]["headers"]["X-MPDC-Client-Node"] == "rtx5070-laptop"
    assert reg_data["opencode_config"]["mcp"]["mpdc-agents"]["url"] == "http://mpdc.local:8000/mcp"

    # 2. Stream tokens via SSE (`stream: true`) to `rtx5070-laptop`
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-MPDC-Client-Node": "rtx5070-laptop"},
        json={
            "model": "qwen3-mpdc",
            "stream": True,
            "messages": [{"role": "user", "content": "Write a ROS2 publisher"}],
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        raw_sse = "".join(stream_resp.iter_text())
        assert "data: [DONE]" in raw_sse
        assert "rtx5070-laptop" in raw_sse

    # 3. Pause token stream to `rtx5070-laptop` and verify 403 block
    toggle = client.post("/v1/mpdc/nodes/rtx5070-laptop/toggle")
    assert toggle.status_code == 200
    assert toggle.json()["status"] == "paused"

    blocked = client.post(
        "/v1/chat/completions",
        headers={"X-MPDC-Client-Node": "rtx5070-laptop"},
        json={
            "model": "qwen3-mpdc",
            "messages": [{"role": "user", "content": "Should be blocked while paused"}],
        },
    )
    assert blocked.status_code == 403

    # 4. Resume node and check `/v1/mpdc/overview` HUD payload
    resumed = client.post("/v1/mpdc/nodes/rtx5070-laptop/toggle")
    assert resumed.json()["status"] == "online"

    overview = client.get("/v1/mpdc/overview").json()
    assert "gpu" in overview
    assert len(overview["nodes"]) == 1
    assert overview["nodes"][0]["node_id"] == "rtx5070-laptop"
    assert overview["nodes"][0]["completion_tokens"] > 0
