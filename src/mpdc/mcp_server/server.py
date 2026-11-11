"""JSON-RPC 2.0 MCP Server exposing MPDC Multi-Agent & Self-Improvement tools to OpenCode."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mpdc.agents.orchestrator import MultiAgentOrchestrator
from mpdc.config import MPDCConfig
from mpdc.evaluation.quantifier import SelfImprovementQuantifier
from mpdc.memory.store import MPDCStore


MCP_TOOLS = [
    {
        "name": "mpdc_plan",
        "description": "Query the MPDC Episodic Skill Bank and run the Qwen3 Planner Agent to decompose a complex robotics or systems task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Detailed task description"},
                "domain": {"type": "string", "default": "robotics"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "mpdc_learn",
        "description": "Distill a solved bug or anti-pattern -> fix into the Tier 1 Episodic Skill Bank AND log a DPO training trajectory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_prompt": {"type": "string"},
                "failed_attempt": {"type": "string"},
                "working_solution": {"type": "string"},
                "error_trace": {"type": "string", "default": ""},
                "domain": {"type": "string", "default": "robotics"},
            },
            "required": ["task_prompt", "failed_attempt", "working_solution"],
        },
    },
    {
        "name": "mpdc_stats",
        "description": "Return quantified self-improvement metrics (Pass@1, Iterations-to-Green, Patch Survival Rate, Skill Lift, Active LoRA).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class MPDCMCPServer:
    """Standard stdio JSON-RPC MCP server for OpenCode (`opencode.json`)."""

    def __init__(self, config: MPDCConfig) -> None:
        self.config = config
        self.store = MPDCStore(config.data_path / "mpdc.sqlite3")
        self.orchestrator = MultiAgentOrchestrator(config, self.store)
        self.quantifier = SelfImprovementQuantifier(config, self.store)

    async def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params") or {}

        if req_id is None:
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mpdc-agents", "version": "0.1.0"},
                },
            }

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}

        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "mpdc_plan":
                res = await self.orchestrator.plan_task(
                    args.get("task", ""), args.get("domain", "robotics")
                )
                text = json.dumps(res, indent=2)
            elif name == "mpdc_learn":
                skill_id = self.orchestrator.distill_lesson_heuristic(
                    task_prompt=args["task_prompt"],
                    failed_attempt=args["failed_attempt"],
                    working_solution=args["working_solution"],
                    error_trace=args.get("error_trace", ""),
                    domain=args.get("domain", "robotics"),
                )
                split = self.store.add_trajectory(
                    task_id=f"mcp-learn-{skill_id}",
                    prompt=args["task_prompt"],
                    chosen_response=args["working_solution"],
                    rejected_response=args["failed_attempt"],
                    patch_survival_rate=1.0,
                    train_split_ratio=self.config.harvester.train_split_ratio,
                )
                text = json.dumps(
                    {"status": "learned", "skill_id": skill_id, "dataset_split": split},
                    indent=2,
                )
            elif name == "mpdc_stats":
                report = self.quantifier.compute_online_scorecard()
                text = json.dumps(report, indent=2)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"},
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": text}]},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        }

    async def serve_stdio(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = await self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as exc:
                err = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": str(exc)},
                }
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()
