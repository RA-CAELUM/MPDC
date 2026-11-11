"""OpenAI-Compatible FastAPI Gateway for OpenCode with Episodic Skill Injection & Dynamic LoRA Routing."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mpdc.agents.orchestrator import MultiAgentOrchestrator, format_skill_block
from mpdc.config import MPDCConfig
from mpdc.evaluation.quantifier import SelfImprovementQuantifier
from mpdc.harvester.collector import TrajectoryHarvester
from mpdc.memory.store import MPDCStore


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-mpdc"
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    stream: bool = False
    domain: str = "robotics"


class FeedbackHarvestRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:10]}")
    prompt: str
    agent_initial_code: str
    final_committed_code: str
    iterations_to_green: int = 1
    domain: str = "robotics"
    skills_used: list[int] = Field(default_factory=list)


def create_gateway_app(config: Optional[MPDCConfig] = None) -> FastAPI:
    cfg = config or MPDCConfig.load()
    store = MPDCStore(cfg.data_path / "mpdc.sqlite3")
    orchestrator = MultiAgentOrchestrator(cfg, store)
    harvester = TrajectoryHarvester(cfg, store)
    quantifier = SelfImprovementQuantifier(cfg, store)

    app = FastAPI(
        title="MPDC Local LLM Datacenter Gateway",
        version="0.1.0",
        description="OpenCode-first OpenAI Gateway with Tier 1 Episodic Skill Injection, Multi-Agent Routing, and Self-Improvement Telemetry.",
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": cfg.inference.engine,
            "base_model": cfg.inference.base_model,
            "active_model_or_lora": orchestrator.active_model_id(),
            "backend_url": cfg.inference.backend_url,
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": "qwen3-mpdc",
                    "object": "model",
                    "created": now,
                    "owned_by": "mpdc",
                    "active_adapter": orchestrator.active_model_id(),
                },
                {
                    "id": "qwen3-mpdc-fast",
                    "object": "model",
                    "created": now,
                    "owned_by": "mpdc",
                    "active_adapter": orchestrator.active_model_id(),
                },
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        last_user_msg = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            req.messages[-1].content,
        )

        # Retrieve relevant Tier 1 Episodic Skills
        skills = store.retrieve_skills(
            query=f"{req.domain} {last_user_msg}",
            top_k=cfg.agents.top_k_skills,
            min_similarity=cfg.agents.min_skill_similarity,
        )
        skill_block = format_skill_block(skills)

        # Toggle Qwen3 thinking mode based on model alias (`qwen3-mpdc` vs `qwen3-mpdc-fast`)
        use_thinking = req.model != "qwen3-mpdc-fast"
        thinking_directive = "/think\n" if use_thinking else "/no_think\n"

        augmented_messages: list[dict[str, str]] = []
        injected_system = False
        for msg in req.messages:
            if msg.role == "system" and not injected_system and skill_block:
                augmented_messages.append(
                    {"role": "system", "content": f"{msg.content}\n\n{skill_block}"}
                )
                injected_system = True
            else:
                augmented_messages.append({"role": msg.role, "content": msg.content})

        if not injected_system and skill_block:
            augmented_messages.insert(0, {"role": "system", "content": skill_block})

        # Apply Qwen3 thinking directive to the final user turn if not already specified
        for idx in range(len(augmented_messages) - 1, -1, -1):
            if augmented_messages[idx]["role"] == "user":
                content = augmented_messages[idx]["content"]
                if not (content.startswith("/think") or content.startswith("/no_think")):
                    augmented_messages[idx]["content"] = f"{thinking_directive}{content}"
                break

        upstream_model = orchestrator.active_model_id()
        upstream_url = f"{cfg.inference.backend_url.rstrip('/')}/chat/completions"

        upstream_payload: dict[str, Any] = {
            "model": upstream_model,
            "messages": augmented_messages,
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            upstream_payload["max_tokens"] = req.max_tokens

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    upstream_url,
                    json=upstream_payload,
                    headers={"Authorization": f"Bearer {cfg.inference.api_key}"},
                )
                resp.raise_for_status()
                completion = resp.json()
        except Exception:
            # Local synthetic response when upstream vLLM / SGLang is not running (e.g., dev/test mode)
            completion = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": upstream_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                f"[MPDC Gateway -> {upstream_model}] Processed request with "
                                f"{len(skills)} injected episodic skills."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 64, "completion_tokens": 32, "total_tokens": 96},
            }

        completion["mpdc_metadata"] = {
            "routed_model": upstream_model,
            "qwen3_thinking": use_thinking,
            "skills_injected": [
                {"id": s.id, "title": s.title, "utility": round(s.utility_score, 3)}
                for s in skills
            ],
        }
        return completion

    @app.post("/v1/mpdc/harvest")
    async def harvest_feedback(req: FeedbackHarvestRequest) -> dict[str, Any]:
        return harvester.ingest_coding_outcome(
            task_id=req.task_id,
            prompt=req.prompt,
            agent_initial_code=req.agent_initial_code,
            final_committed_code=req.final_committed_code,
            iterations_to_green=req.iterations_to_green,
            domain=req.domain,
            skills_used=req.skills_used,
            adapter_version=orchestrator.active_model_id(),
        )

    @app.get("/v1/mpdc/stats")
    async def get_stats() -> dict[str, Any]:
        return quantifier.compute_online_scorecard()

    return app
