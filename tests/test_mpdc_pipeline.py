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
    # No upstream model runs in tests: the gateway must fail loudly (502), not fake a reply.
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "unreachable" in detail["error"]
    assert detail["skills_injected"] == 1


def test_gateway_forwards_openai_fields_to_upstream(tmp_path: Path, monkeypatch) -> None:
    """SICA relies on `stop` (tool-call stop tokens) and prefill-continuation fields."""
    import httpx as _httpx

    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None, **kw):  # noqa: A002
        captured.update(json)
        return _httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
            request=_httpx.Request("POST", url),
        )

    monkeypatch.setattr(_httpx.AsyncClient, "post", fake_post)
    client = TestClient(create_gateway_app(_make_test_config(tmp_path)))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "messages": [
                {"role": "user", "content": "write a file\n/no_think"},
                {"role": "assistant", "content": "<TOOL_CALL>..."},
            ],
            "stop": ["</TOOL_CALL>", "</AGENT_CALL>"],
            "top_p": 0.9,
            "continue_final_message": True,
            "add_generation_prompt": False,
        },
    )
    assert resp.status_code == 200
    assert captured["stop"] == ["</TOOL_CALL>", "</AGENT_CALL>"]
    assert captured["top_p"] == 0.9
    assert captured["continue_final_message"] is True
    assert captured["add_generation_prompt"] is False
    # Caller's trailing /no_think must not get a conflicting /think prepended.
    assert captured["messages"][0]["content"] == "write a file\n/no_think"
    assert "domain" not in captured and "client_node" not in captured


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
    # No upstream model in tests -> no real completion tokens, but the request is accounted.
    assert overview["nodes"][0]["prompt_tokens"] > 0


def test_sica_local_qwen3_provider_and_think_block_stripping() -> None:
    import sys

    sica_root = str(Path(__file__).resolve().parents[1] / "modules" / "MPDC-SICA" / "base_agent")
    if sica_root not in sys.path:
        sys.path.insert(0, sica_root)

    from src.config import Settings
    from src.llm.providers.local import LocalOpenAIProvider, split_qwen_think_blocks
    from src.types.common import ArgFormat
    from src.types.llm_types import FCI, Model, Provider

    cfg = Settings()
    assert cfg.MODEL == Model.QWEN3
    assert cfg.REASONING_MODEL == Model.QWEN3_REASONER
    assert cfg.OVERSIGHT_MODEL == Model.QWEN3
    assert Model.QWEN3.provider == Provider.LOCAL
    assert Model.QWEN3.fci == FCI.UNCONSTRAINED
    assert Model.QWEN3.arg_format == ArgFormat.XML

    raw_qwen_output = (
        "<think>\nLet's inspect the directory first before editing.\n</think>\n"
        "I will list the directory contents.\n"
        "<TOOL_CALL>\n<TOOL_NAME>view_directory</TOOL_NAME>\n"
        "<TOOL_ARGS>\n<path>/home/agent/workdir</path>\n</TOOL_ARGS>"
    )
    cot, clean_text = split_qwen_think_blocks(raw_qwen_output)
    assert "inspect the directory first" in cot
    assert "<think>" not in clean_text
    assert clean_text.endswith("</TOOL_ARGS>")

    provider = LocalOpenAIProvider()
    assert str(provider.client.base_url).rstrip("/").endswith("/v1")


def test_default_config_targets_6gb_ollama_model() -> None:
    cfg = MPDCConfig.load()
    assert cfg.inference.engine == "ollama"
    assert cfg.inference.base_model == "mpdc-qwen3-4b"
    assert cfg.inference.thinking_toggle is False
    assert [m.name for m in cfg.pools.coding_models] == ["mpdc-qwen3-4b"]
    assert not any(m.active for m in cfg.pools.spatial_visual_models)


def test_gateway_strips_think_directives_when_model_lacks_toggle(tmp_path: Path, monkeypatch) -> None:
    import httpx as _httpx

    captured: dict = {}

    async def fake_post(self, url, json=None, headers=None, **kw):  # noqa: A002
        captured.update(json)
        return _httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                               "finish_reason": "stop"}]},
            request=_httpx.Request("POST", url),
        )

    monkeypatch.setattr(_httpx.AsyncClient, "post", fake_post)
    cfg = _make_test_config(tmp_path)
    cfg.inference.thinking_toggle = False
    client = TestClient(create_gateway_app(cfg))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3-mpdc", "messages": [{"role": "user", "content": "write a file\n/no_think"}]},
    )
    assert resp.status_code == 200
    assert captured["messages"][-1]["content"] == "write a file"


def test_ollama_assistant_prefill_uses_raw_chatml_continuation(tmp_path: Path, monkeypatch) -> None:
    """Ollama's chat APIs restart a trailing assistant turn; the gateway must continue it."""
    import httpx as _httpx

    calls: list = []

    async def fake_post(self, url, json=None, headers=None, **kw):  # noqa: A002
        calls.append((url, json))
        return _httpx.Response(
            200,
            json={"response": " 5 6", "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 3},
            request=_httpx.Request("POST", url),
        )

    monkeypatch.setattr(_httpx.AsyncClient, "post", fake_post)
    cfg = _make_test_config(tmp_path)
    cfg.inference.engine = "ollama"
    cfg.inference.backend_url = "http://127.0.0.1:11434/v1"
    cfg.inference.thinking_toggle = False
    client = TestClient(create_gateway_app(cfg))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "max_tokens": 50,
            "stop": ["</TOOL_CALL>"],
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "count\n/no_think"},
                {"role": "assistant", "content": "1 2 3 4"},
            ],
        },
    )
    assert resp.status_code == 200
    url, payload = calls[0]
    assert url == "http://127.0.0.1:11434/api/generate"
    assert payload["raw"] is True
    assert payload["prompt"] == (
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>user\ncount<|im_end|>\n"
        "<|im_start|>assistant\n1 2 3 4"
    )
    assert "</TOOL_CALL>" in payload["options"]["stop"]
    assert payload["options"]["num_predict"] == 50
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == " 5 6"
    assert body["usage"]["completion_tokens"] == 3


def test_sica_tools_remap_sandbox_paths_to_host_workdir(tmp_path: Path, monkeypatch) -> None:
    """mpdc-code runs SICA on the host: /home/agent/workdir must map to the project dir."""
    import sys

    sica_root = str(Path(__file__).resolve().parents[1] / "modules" / "MPDC-SICA" / "base_agent")
    if sica_root not in sys.path:
        sys.path.insert(0, sica_root)
    from src.llm.providers.local import prefill_extra_body, resolve_max_tokens
    from src.tools.execute_command import ExecuteCommand

    monkeypatch.setenv("SICA_HOST_WORKDIR", str(tmp_path))
    cmd = ExecuteCommand(
        calling_agent=None,
        intent="t",
        command="cd /home/agent/workdir/sub && ls",
        command_returns=True,
        generous_expected_duration=5.0,
    )
    assert cmd.directory_in_which_to_run_command == str(tmp_path)
    assert cmd.command == f"cd {tmp_path}/sub && ls"

    from src.tools.base_tool import unescape_json_style

    # Small models JSON-escape file contents inside XML args; repair those only.
    assert unescape_json_style('def f():\\n    return \\"x\\"\\n') == 'def f():\n    return "x"\n'
    assert unescape_json_style('print("a\\nb")') == 'print("a\\nb")'
    assert unescape_json_style("a\nb \\n c \\n") == "a\nb \\n c \\n"

    monkeypatch.delenv("LOCAL_LLM_MAX_TOKENS", raising=False)
    assert resolve_max_tokens(None, 32768) == 4096
    assert prefill_extra_body([{"role": "assistant", "content": "x"}]) == {
        "continue_final_message": True,
        "add_generation_prompt": False,
    }
    assert prefill_extra_body([{"role": "user", "content": "x"}]) == {}


def _sica_path():
    import sys

    sica_root = str(Path(__file__).resolve().parents[1] / "modules" / "MPDC-SICA" / "base_agent")
    if sica_root not in sys.path:
        sys.path.insert(0, sica_root)


def test_local_tools_edit_run_and_stay_inside_workdir(tmp_path: Path) -> None:
    _sica_path()
    from types import SimpleNamespace
    from src.tools.local_tools import Finish, ReadFile, ReplaceInFile, RunCommand, WriteFile

    agent = SimpleNamespace(_workdir=tmp_path, _logdir=tmp_path / "logs", _local_state={})
    run = lambda tool: asyncio.run(tool.run())  # noqa: E731

    assert run(WriteFile(calling_agent=agent, path="pkg/a.py", content="x = 1\ny = 2")).success
    assert (tmp_path / "pkg/a.py").read_text() == "x = 1\ny = 2\n"
    assert not run(WriteFile(calling_agent=agent, path="../escape.py", content="bad")).success
    assert not (tmp_path.parent / "escape.py").exists()

    assert run(ReplaceInFile(calling_agent=agent, path="pkg/a.py", old_text="y = 2", new_text="y = 3")).success
    missing = run(ReplaceInFile(calling_agent=agent, path="pkg/a.py", old_text="nope", new_text="z"))
    assert not missing.success and "not found" in missing.errors

    read = run(ReadFile(calling_agent=agent, path="pkg/a.py"))
    assert "    2| y = 3" in read.output

    out = run(RunCommand(calling_agent=agent, command="pwd; exit 3"))
    assert "exit code: 3" in out.output and str(tmp_path) in out.output

    run(Finish(calling_agent=agent, summary="done; tests pass"))
    assert agent._local_state["exec_complete"] is True
    assert (tmp_path / "logs" / "answer.txt").read_text() == "done; tests pass"


def test_fit_to_context_trims_old_tool_output_and_keeps_pairs(monkeypatch) -> None:
    _sica_path()
    from src.llm.providers.local import _estimate_tokens, fit_to_context

    monkeypatch.setenv("LOCAL_LLM_CONTEXT_TOKENS", "16384")
    monkeypatch.setenv("LOCAL_LLM_MAX_TOKENS", "4096")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(40):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"out{i} " + "x" * 6000})
    out = fit_to_context(msgs)
    assert _estimate_tokens(out) <= 16384 - 4096 - 1500
    assert out[:2] == msgs[:2]
    assert out[-1]["content"] == msgs[-1]["content"]  # newest result kept in full
    ids = [m.get("tool_call_id") for m in out if m["role"] == "tool"]
    calls = [m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")]
    assert ids == calls  # no orphaned tool results or calls


def test_text_tool_calls_are_recovered_for_known_tools_only() -> None:
    _sica_path()
    from src.llm.providers.local import extract_text_tool_calls

    names = {"list_files", "write_file", "finish"}
    calls, rest = extract_text_tool_calls('{"name": "list_files", "arguments": {"path": "."}}', names)
    assert [(c.tool_name, c.tool_args) for c in calls] == [("list_files", {"path": "."})] and rest == ""
    calls, rest = extract_text_tool_calls(
        'Writing it.\n```json\n{"name": "write_file", "arguments": {"path": "a.py", "content": "x = {1: 2}"}}\n```', names)
    assert calls[0].tool_args["content"] == "x = {1: 2}" and rest.strip() == "Writing it."
    assert extract_text_tool_calls('{"name": "rm_rf", "arguments": {}}', names)[0] == []
    assert extract_text_tool_calls("no calls here", names)[0] == []


def test_local_coder_loop_guard_stops_repeated_identical_calls(tmp_path: Path, monkeypatch) -> None:
    _sica_path()
    monkeypatch.setenv("SICA_HOST_WORKDIR", str(tmp_path))
    from src.agents.implementations.local_coder import LocalCoderAgent
    from src.types.llm_types import FCI, ToolCallContent

    agent = LocalCoderAgent(workdir=tmp_path, logdir=tmp_path / "logs")
    assert {t.TOOL_NAME for t in agent._available_tools} == {
        "list_files", "read_file", "write_file", "replace_in_file", "run_command", "finish"}
    call = ToolCallContent(call_id="c", tool_name="list_files", tool_args={"path": "."}, call_type=FCI.CONSTRAINED)
    for _ in range(4):
        asyncio.run(agent._handle_tool_call(call))
        assert not agent._local_state.get("needs_exit")
    asyncio.run(agent._handle_tool_call(call))
    assert agent._local_state["needs_exit"] is True


def test_local_coder_text_reply_ends_run_only_after_real_edits(tmp_path: Path, monkeypatch) -> None:
    _sica_path()
    monkeypatch.setenv("SICA_HOST_WORKDIR", str(tmp_path))
    monkeypatch.delenv("SICA_READ_ONLY", raising=False)
    from types import SimpleNamespace
    from src.agents.implementations.local_coder import LocalCoderAgent
    from src.types.llm_types import TextContent

    agent = LocalCoderAgent(workdir=tmp_path, logdir=tmp_path / "logs")
    narration = SimpleNamespace(content=[TextContent(text="Next, let's read stats.py.")])
    # Narration before any edit never ends the run, however often it repeats.
    for _ in range(3):
        assert asyncio.run(agent.accept_text_reply_as_final(narration)) is False
    agent._local_state["has_edited"] = True
    agent._local_state["text_only_replies"] = 0
    summary = SimpleNamespace(content=[TextContent(text="finish Fixed median; tests pass.")])
    assert asyncio.run(agent.accept_text_reply_as_final(summary)) is False  # first: reminder
    assert asyncio.run(agent.accept_text_reply_as_final(summary)) is True   # second: accepted
    assert (tmp_path / "logs" / "answer.txt").read_text() == "Fixed median; tests pass."


def test_multipool_coding_and_spatial_visual_swarm(tmp_path: Path) -> None:
    cfg = _make_test_config(tmp_path)
    app = create_gateway_app(cfg)
    client = TestClient(app)

    # 1. Inspect default multi-model pools (4 coding + 3 visual/spatial)
    pools = client.get("/v1/mpdc/pools").json()
    assert len(pools["coding_models"]) == 4
    assert len(pools["spatial_visual_models"]) == 3

    # 2. Register an additional 3D Spatial model dynamically
    added = client.post(
        "/v1/mpdc/pools/register",
        json={
            "id": "molmo-spatial-7b",
            "name": "allenai/Molmo-7B-D-0924",
            "role": "AFFORDANCE_KEYPOINT_AGENT",
            "category": "spatial_3d",
            "endpoint_url": "http://127.0.0.1:8006/v1",
            "thinking": True,
            "modalities": ["image", "keypoints_3d"],
            "active": True,
        },
    )
    assert added.status_code == 200
    assert len(client.get("/v1/mpdc/pools").json()["spatial_visual_models"]) == 4

    # 3. Execute multimodal swarm across Spatial/Visual Pool + Coding Pool
    swarm_resp = client.post(
        "/v1/mpdc/swarm",
        json={
            "prompt": "Fix ROS2 grasp controller Z-axis orientation using MuJoCo depth & point cloud",
            "domain": "robotics",
            "image_paths": ["sim_viewport.png"],
            "pointcloud_paths": ["wrist_cam.ply"],
            "coordination_mode": "sensor_fusion_loop",
        },
    )
    assert swarm_resp.status_code == 200
    data = swarm_resp.json()
    assert "<mpdc_spatial_scene_graph>" in data["spatial_scene_graph"]
    assert len(data["spatial_steps"]) == 4
    assert len(data["coding_steps"]) == 4
    # No model servers run in tests: failures must be surfaced, never fabricated.
    assert "target_pose_se3" not in data["spatial_scene_graph"]
    assert all("MPDC ERROR" in s["output"] for s in data["spatial_steps"])


def test_unservable_promoted_adapter_is_not_routed(tmp_path: Path) -> None:
    cfg = _make_test_config(tmp_path)
    store = MPDCStore(cfg.data_path / "mpdc.sqlite3")
    store.register_adapter(
        version="qwen3-lora-v001",
        base_model=cfg.inference.base_model,
        adapter_path=str(tmp_path / "missing-adapter"),
        replay_pass_rate=0.9,
        holdout_perplexity=1.5,
        sanity_score=1.0,
        promoted=True,
    )
    orch = MultiAgentOrchestrator(cfg, store)
    assert orch.active_model_id() == cfg.inference.base_model

    weights_dir = tmp_path / "missing-adapter"
    weights_dir.mkdir()
    (weights_dir / "adapter_model.safetensors").write_bytes(b"")
    assert orch.active_model_id() == "qwen3-lora-v001"


def test_training_without_real_metrics_never_promotes(tmp_path: Path) -> None:
    cfg = _make_test_config(tmp_path)
    store = MPDCStore(cfg.data_path / "mpdc.sqlite3")
    report = asyncio.run(OnNodeQLoRATrainer(cfg, store).run_training_cycle(dry_run=True))
    assert report.evaluation.promoted is False
    assert store.get_latest_promoted_adapter() is None


