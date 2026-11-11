"""Multi-Agent Orchestrator for Qwen3 (Planner, Coder, Verifier Loop, and Distiller)."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from mpdc.config import MPDCConfig
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

    def active_model_id(self) -> str:
        promoted = self.store.get_latest_promoted_adapter()
        if promoted:
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
        mode_prefix = "/think\n" if enable_thinking else "/no_think\n"
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
