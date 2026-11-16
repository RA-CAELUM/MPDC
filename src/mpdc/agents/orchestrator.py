"""Multi-Agent & Multi-Pool Orchestrator (Multiple Programming Models + Multiple Visual/Spatial Models)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from mpdc.config import MPDCConfig, ModelWorkerConfig
from mpdc.memory.store import MPDCStore, SkillCard, TaskTelemetry


def format_skill_block(skills: list[SkillCard]) -> str:
    if not skills:
        return ""
    lines = [
        "<mpdc_episodic_skills>",
        "The following verified domain skills were distilled from your past solved tasks:",
    ]
    for idx, s in enumerate(skills, 1):
        lines.append(
            f"[{idx}] {s.title} (domain={s.domain}, utility={s.utility_score:.2f}):\n"
            f"  - Trigger/Symptom: {s.trigger_signature}\n"
            f"  - Avoid Anti-Pattern: {s.anti_pattern}\n"
            f"  - Verified Pattern: {s.verified_pattern}"
        )
    lines.append("</mpdc_episodic_skills>")
    return "\n".join(lines)


@dataclass
class SpatialObservation:
    image_paths: list[str] = field(default_factory=list)
    pointcloud_paths: list[str] = field(default_factory=list)
    sim_state_json: str = ""
    camera_frame: str = "world"


@dataclass
class SwarmAgentStep:
    agent_id: str
    model_name: str
    role: str
    category: str
    output: str


@dataclass
class MultiPoolSwarmResult:
    coordination_mode: str
    spatial_scene_graph: str
    spatial_steps: list[SwarmAgentStep]
    coding_steps: list[SwarmAgentStep]
    synthesized_patch: str
    skills_injected: list[dict[str, Any]]


@dataclass
class VerificationResult:
    passed: bool
    iterations_to_green: int
    final_code: str
    first_failed_attempt: Optional[str]
    command_output: str
    skills_used: list[int]


class MultiAgentOrchestrator:
    """Coordinates Planner, Coder, Verifier, and Distiller roles on a shared Qwen3 backbone."""

    def __init__(self, config: MPDCConfig, store: MPDCStore) -> None:
        self.config = config
        self.store = store

    def _adapter_has_weights(self, adapter_path: str) -> bool:
        path = Path(adapter_path)
        candidates = [path] if path.is_absolute() else [
            self.config.data_path.parent / path,
            Path.cwd() / path,
        ]
        return any(
            (p / name).is_file()
            for p in candidates
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )

    def active_model_id(self) -> str:
        # Only route to a promoted LoRA when real trained weights exist on disk.
        # A metadata-only scaffold (dry-run "training") is not servable: vLLM
        # would 404 on the unknown model name and every call would fail.
        promoted = self.store.get_latest_promoted_adapter()
        if promoted and self._adapter_has_weights(promoted["adapter_path"]):
            return promoted["version"]
        if self.config.inference.active_adapter_name:
            return self.config.inference.active_adapter_name
        return self.config.inference.base_model

    async def _call_qwen(
        self,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: bool = True,
        temperature: float = 0.2,
    ) -> str:
        """Calls the upstream vLLM / SGLang Qwen3 endpoint with Qwen3 thinking mode toggle."""
        mode_prefix = (
            ("/think\n" if enable_thinking else "/no_think\n")
            if self.config.inference.thinking_toggle
            else ""
        )
        url = f"{self.config.inference.backend_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.active_model_id(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{mode_prefix}{user_prompt}"},
            ],
            "temperature": temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.config.inference.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            return f"[MPDC Offline Fallback: upstream {url} unreachable ({exc})]"

    async def plan_task(self, task_description: str, domain: str = "robotics") -> dict[str, Any]:
        """Architect / Planner Agent: retrieves Episodic Skills and decomposes the task."""
        skills = self.store.retrieve_skills(
            f"{domain} {task_description}",
            top_k=self.config.agents.top_k_skills,
            min_similarity=self.config.agents.min_skill_similarity,
        )
        skill_context = format_skill_block(skills)
        sys_prompt = (
            "You are the MPDC Architect Planner Agent specialized in robotics (ROS2, VLA, Isaac/MuJoCo), "
            "NixOS, and systems engineering. Produce a concise step-by-step execution plan and verification command.\n"
            f"{skill_context}"
        )
        plan_text = await self._call_qwen(
            sys_prompt,
            task_description,
            enable_thinking=self.config.inference.enable_thinking_for_planner,
        )
        return {
            "plan": plan_text,
            "skills_applied": [
                {"id": s.id, "title": s.title, "utility": round(s.utility_score, 3)}
                for s in skills
            ],
            "model_used": self.active_model_id(),
        }

    def distill_lesson_heuristic(
        self,
        task_prompt: str,
        failed_attempt: str,
        working_solution: str,
        error_trace: str = "",
        domain: str = "robotics",
    ) -> int:
        """Tier 1 Distiller: extracts a structured Skill Card from a failed-then-fixed trajectory."""
        title = f"Resolved: {task_prompt.strip()[:72]}"
        trigger = error_trace.strip()[:240] if error_trace else task_prompt.strip()[:240]
        anti_pattern = failed_attempt.strip()[:400]
        verified_pattern = working_solution.strip()[:600]
        tags = [domain, "auto-distilled"]
        return self.store.add_skill(
            title=title,
            domain=domain,
            trigger_signature=trigger,
            anti_pattern=anti_pattern,
            verified_pattern=verified_pattern,
            tags=tags,
        )

    async def verify_and_self_correct(
        self,
        task_prompt: str,
        verification_cmd: list[str],
        initial_candidate: str,
        domain: str = "robotics",
    ) -> VerificationResult:
        """Closed-loop Coder <-> Critic verification loop with automatic telemetry & harvesting."""
        skills = self.store.retrieve_skills(
            f"{domain} {task_prompt}",
            top_k=self.config.agents.top_k_skills,
            min_similarity=self.config.agents.min_skill_similarity,
        )
        skill_ids = [s.id for s in skills if s.id is not None]
        skill_block = format_skill_block(skills)

        candidate = initial_candidate
        first_failed: Optional[str] = None
        last_output = ""
        max_tries = max(1, self.config.agents.max_verification_retries)

        for attempt in range(1, max_tries + 1):
            proc = subprocess.run(
                verification_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            last_output = (proc.stdout + "\n" + proc.stderr).strip()
            if proc.returncode == 0:
                task_id = f"verify-{uuid.uuid4().hex[:10]}"
                self.store.record_task_telemetry(
                    TaskTelemetry(
                        task_id=task_id,
                        prompt=task_prompt,
                        domain=domain,
                        passed_first_try=(attempt == 1),
                        iterations_to_green=attempt,
                        patch_survival_rate=1.0,
                        skills_used=skill_ids,
                        adapter_version=self.active_model_id(),
                    )
                )
                if first_failed is not None:
                    self.distill_lesson_heuristic(
                        task_prompt=task_prompt,
                        failed_attempt=first_failed,
                        working_solution=candidate,
                        error_trace=last_output,
                        domain=domain,
                    )
                    self.store.add_trajectory(
                        task_id=task_id,
                        prompt=task_prompt,
                        chosen_response=candidate,
                        rejected_response=first_failed,
                        patch_survival_rate=1.0,
                        train_split_ratio=self.config.harvester.train_split_ratio,
                    )
                return VerificationResult(
                    passed=True,
                    iterations_to_green=attempt,
                    final_code=candidate,
                    first_failed_attempt=first_failed,
                    command_output=last_output,
                    skills_used=skill_ids,
                )

            if first_failed is None:
                first_failed = candidate

            # Ask Coder agent to self-correct using compiler/test traceback
            critic_prompt = (
                f"Task: {task_prompt}\n"
                f"Previous Attempt:\n{candidate}\n\n"
                f"Verification Command Failed ({' '.join(verification_cmd)}):\n{last_output}\n\n"
                "Provide the corrected implementation."
            )
            candidate = await self._call_qwen(
                f"You are the MPDC Specialist Coder Agent.\n{skill_block}",
                critic_prompt,
                enable_thinking=self.config.inference.enable_thinking_for_coder,
            )

        return VerificationResult(
            passed=False,
            iterations_to_green=max_tries,
            final_code=candidate,
            first_failed_attempt=first_failed,
            command_output=last_output,
            skills_used=skill_ids,
        )

    def list_model_pools(self) -> dict[str, Any]:
        return {
            "coordination_mode": self.config.pools.coordination_mode,
            "coding_models": [m.model_dump() for m in self.config.pools.coding_models],
            "spatial_visual_models": [
                m.model_dump() for m in self.config.pools.spatial_visual_models
            ],
        }

    def register_pool_model(self, worker: ModelWorkerConfig) -> dict[str, Any]:
        target_pool = (
            self.config.pools.spatial_visual_models
            if worker.category in ("vision", "spatial_3d", "geometry")
            else self.config.pools.coding_models
        )
        for idx, existing in enumerate(target_pool):
            if existing.id == worker.id:
                target_pool[idx] = worker
                return worker.model_dump()
        target_pool.append(worker)
        return worker.model_dump()

    def toggle_pool_model(self, model_id: str) -> Optional[dict[str, Any]]:
        for pool in (
            self.config.pools.coding_models,
            self.config.pools.spatial_visual_models,
        ):
            for worker in pool:
                if worker.id == model_id:
                    worker.active = not worker.active
                    return worker.model_dump()
        return None

    async def _call_pool_worker(
        self,
        worker: ModelWorkerConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> SwarmAgentStep:
        """Calls a specific programming or visual/spatial model worker endpoint."""
        mode_prefix = (
            ("/think\n" if worker.thinking else "/no_think\n")
            if self.config.inference.thinking_toggle
            else ""
        )
        url = f"{worker.endpoint_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": worker.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{mode_prefix}{user_prompt}"},
            ],
            "temperature": 0.2 if worker.category in ("coding", "geometry") else 0.4,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.config.inference.api_key}"},
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            # Never fabricate model output here: a fake pose/critique gets fed
            # downstream to the coding models as if it were real grounding.
            text = f"[MPDC ERROR: {worker.id} ({worker.name}) unreachable at {url}: {exc}]"
        return SwarmAgentStep(
            agent_id=worker.id,
            model_name=worker.name,
            role=worker.role,
            category=worker.category,
            output=text,
        )

    async def execute_multimodal_swarm(
        self,
        prompt: str,
        domain: str = "robotics",
        observation: Optional[SpatialObservation] = None,
        coding_model_ids: Optional[list[str]] = None,
        spatial_model_ids: Optional[list[str]] = None,
        coordination_mode: Optional[str] = None,
    ) -> MultiPoolSwarmResult:
        """Executes a coordinated multi-agent run across Visual/Spatial models and Programming models."""
        mode = coordination_mode or self.config.pools.coordination_mode
        obs = observation or SpatialObservation()

        skills = self.store.retrieve_skills(
            f"{domain} {prompt}",
            top_k=self.config.agents.top_k_skills,
            min_similarity=self.config.agents.min_skill_similarity,
        )
        skill_block = format_skill_block(skills)

        # 1. Select active Visual & Spatial models
        active_spatial = [
            m
            for m in self.config.pools.spatial_visual_models
            if m.active and (not spatial_model_ids or m.id in spatial_model_ids)
        ]

        # 2. Select active Programming & Reasoning models
        active_coding = [
            m
            for m in self.config.pools.coding_models
            if m.active and (not coding_model_ids or m.id in coding_model_ids)
        ]

        # 3. Run Visual & Spatial models in parallel to construct the 3D Scene Graph & Visual Critique
        spatial_prompt = (
            f"Task: {prompt}\n"
            f"Attached Images/Frames: {obs.image_paths}\n"
            f"Attached Depth/PointClouds: {obs.pointcloud_paths}\n"
            f"Sim/Sensor State ({obs.camera_frame}): {obs.sim_state_json or 'live_viewport'}"
        )
        spatial_tasks = [
            self._call_pool_worker(
                w,
                f"You are {w.role} ({w.category}) in the MPDC Spatial/Visual Pool. Output structured 3D scene grounding, SE(3) poses, bounding boxes, and visual failure diagnostics.",
                spatial_prompt,
            )
            for w in active_spatial
        ]
        spatial_steps: list[SwarmAgentStep] = (
            list(await asyncio.gather(*spatial_tasks)) if spatial_tasks else []
        )

        scene_graph_lines = ["<mpdc_spatial_scene_graph>"]
        for st in spatial_steps:
            scene_graph_lines.append(st.output)
        scene_graph_lines.append("</mpdc_spatial_scene_graph>")
        spatial_scene_graph = "\n".join(scene_graph_lines)

        # 4. Run Programming & Reasoning models conditioned on BOTH the Spatial Scene Graph and Episodic Skills
        coding_user_prompt = f"{prompt}\n\n{spatial_scene_graph}"
        coding_sys_base = (
            "You are a specialist programming agent in the MPDC Multi-Model Coding Pool. "
            "Synthesize code that satisfies both the software requirements and the 3D/visual grounding in <mpdc_spatial_scene_graph>.\n"
            f"{skill_block}"
        )

        if mode == "parallel_ensemble":
            coding_tasks = [
                self._call_pool_worker(w, f"{coding_sys_base}\nRole: {w.role}", coding_user_prompt)
                for w in active_coding
            ]
            coding_steps = list(await asyncio.gather(*coding_tasks)) if coding_tasks else []
        else:
            # Sequential / Graph pipeline: Planner -> Systems Coder -> Verifier
            coding_steps = []
            running_context = coding_user_prompt
            for w in active_coding:
                step = await self._call_pool_worker(
                    w, f"{coding_sys_base}\nRole: {w.role}", running_context
                )
                coding_steps.append(step)
                running_context = f"{running_context}\n\n[{w.role} ({w.id}) Output]:\n{step.output}"

        synthesized_patch = "\n\n".join(s.output for s in coding_steps)
        return MultiPoolSwarmResult(
            coordination_mode=mode,
            spatial_scene_graph=spatial_scene_graph,
            spatial_steps=spatial_steps,
            coding_steps=coding_steps,
            synthesized_patch=synthesized_patch,
            skills_injected=[
                {"id": s.id, "title": s.title, "utility": round(s.utility_score, 3)}
                for s in skills
            ],
        )
