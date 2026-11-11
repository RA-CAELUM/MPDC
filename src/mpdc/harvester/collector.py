"""Tier 2 Trajectory & Git-Diff Harvester (Computes Patch Survival Rate & Exports SFT/DPO Datasets)."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Optional

from mpdc.config import MPDCConfig
from mpdc.memory.store import MPDCStore, TaskTelemetry


def compute_patch_survival_rate(agent_patch: str, committed_patch: str) -> float:
    """Computes Patch Survival Rate (PSR) in [0.0, 1.0].

    PSR = SequenceMatcher ratio between normalized code lines written by the agent
    and the final code lines committed by the developer to Git.
    1.00 = Developer committed 100% of the agent's patch untouched.
    0.00 = Developer completely rewrote or discarded the agent's patch.
    """
    a_lines = [line.strip() for line in agent_patch.strip().splitlines() if line.strip()]
    c_lines = [line.strip() for line in committed_patch.strip().splitlines() if line.strip()]
    if not a_lines and not c_lines:
        return 1.0
    if not a_lines or not c_lines:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a_lines, c_lines)
    return float(round(matcher.ratio(), 4))


class TrajectoryHarvester:
    """Harvests OpenCode sessions + Git outcomes into 80% Train (SFT/DPO) and 20% Golden Replay."""

    def __init__(self, config: MPDCConfig, store: MPDCStore) -> None:
        self.config = config
        self.store = store

    def ingest_coding_outcome(
        self,
        task_id: str,
        prompt: str,
        agent_initial_code: str,
        final_committed_code: str,
        iterations_to_green: int = 1,
        domain: str = "robotics",
        skills_used: Optional[list[int]] = None,
        adapter_version: str = "base",
    ) -> dict[str, Any]:
        """Processes a completed coding task, computes PSR, updates telemetry, and builds SFT/DPO pairs."""
        psr = compute_patch_survival_rate(agent_initial_code, final_committed_code)
        passed_first_try = (
            iterations_to_green == 1
            and psr >= self.config.harvester.chosen_survival_threshold
        )

        self.store.record_task_telemetry(
            TaskTelemetry(
                task_id=task_id,
                prompt=prompt,
                domain=domain,
                passed_first_try=passed_first_try,
                iterations_to_green=iterations_to_green,
                patch_survival_rate=psr,
                skills_used=skills_used or [],
                adapter_version=adapter_version,
            )
        )

        # If the user had to significantly edit the agent's code OR it took >1 iteration,
        # we have a high-signal Preference Pair: {chosen: final_committed_code, rejected: agent_initial_code}
        rejected = (
            agent_initial_code
            if (psr < 0.95 or iterations_to_green > 1)
            else None
        )
        split = self.store.add_trajectory(
            task_id=task_id,
            prompt=prompt,
            chosen_response=final_committed_code,
            rejected_response=rejected,
            patch_survival_rate=psr,
            train_split_ratio=self.config.harvester.train_split_ratio,
        )

        return {
            "task_id": task_id,
            "patch_survival_rate": psr,
            "passed_first_try": passed_first_try,
            "iterations_to_green": iterations_to_green,
            "has_dpo_pair": rejected is not None,
            "assigned_split": split,
        }

    def export_datasets(self) -> dict[str, Any]:
        """Exports accumulated SQLite trajectories to ShareGPT/ChatML SFT, DPO, and Golden Replay JSONL files."""
        datasets_dir = self.config.data_path / "datasets"
        eval_dir = self.config.data_path / "eval"
        sft_path = datasets_dir / "sft_train.jsonl"
        dpo_path = datasets_dir / "dpo_train.jsonl"
        replay_path = eval_dir / "golden_replay.jsonl"

        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM harvested_trajectories ORDER BY id ASC"
            ).fetchall()

        sft_count = 0
        dpo_count = 0
        replay_count = 0

        with (
            open(sft_path, "w", encoding="utf-8") as f_sft,
            open(dpo_path, "w", encoding="utf-8") as f_dpo,
            open(replay_path, "w", encoding="utf-8") as f_rep,
        ):
            for r in rows:
                if r["split"] == "golden_replay":
                    f_rep.write(
                        json.dumps(
                            {
                                "task_id": r["task_id"],
                                "prompt": r["prompt"],
                                "failed_attempt": r["rejected_response"],
                                "reference_solution": r["chosen_response"],
                            }
                        )
                        + "\n"
                    )
                    replay_count += 1
                else:
                    # SFT format (Qwen3 ChatML messages)
                    f_sft.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": r["prompt"]},
                                    {"role": "assistant", "content": r["chosen_response"]},
                                ]
                            }
                        )
                        + "\n"
                    )
                    sft_count += 1
                    if r["rejected_response"]:
                        f_dpo.write(
                            json.dumps(
                                {
                                    "prompt": r["prompt"],
                                    "chosen": r["chosen_response"],
                                    "rejected": r["rejected_response"],
                                }
                            )
                            + "\n"
                        )
                        dpo_count += 1

        return {
            "sft_train_path": str(sft_path),
            "sft_samples": sft_count,
            "dpo_train_path": str(dpo_path),
            "dpo_pairs": dpo_count,
            "golden_replay_path": str(replay_path),
            "golden_replay_samples": replay_count,
        }
