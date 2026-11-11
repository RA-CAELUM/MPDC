"""Pillar 1 (Online Telemetry) & Pillar 2 (Offline Replay + Gated Promotion) Quantifier."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from mpdc.config import MPDCConfig
from mpdc.memory.store import MPDCStore


SANITY_BENCHMARK_SUITE = [
    {
        "id": "sanity-ros2-qos",
        "domain": "robotics",
        "prompt": "Write a ROS2 rclpy QoSProfile for high-rate sensor data using BEST_EFFORT reliability.",
        "required_tokens": ["QoSProfile", "ReliabilityPolicy.BEST_EFFORT"],
    },
    {
        "id": "sanity-vla-chunk",
        "domain": "robotics",
        "prompt": "Reshape a PyTorch action prediction tensor `pred` to `(batch_size, chunk_horizon, action_dim)`.",
        "required_tokens": ["view", "batch_size", "chunk_horizon", "action_dim"],
    },
    {
        "id": "sanity-nix-mkderivation",
        "domain": "nixos",
        "prompt": "Define a minimal Nix flake outputs function taking `{ self, nixpkgs }`.",
        "required_tokens": ["outputs", "nixpkgs"],
    },
]


@dataclass
class AdapterEvalComparison:
    candidate_version: str
    current_version: str
    candidate_replay_pass_rate: float
    current_replay_pass_rate: float
    candidate_perplexity: float
    current_perplexity: float
    candidate_sanity_score: float
    baseline_sanity_score: float
    promoted: bool
    reason: str


class SelfImprovementQuantifier:
    """Computes online production telemetry (`mpdc stats`) and offline gated A/B benchmarks (`mpdc eval`)."""

    def __init__(self, config: MPDCConfig, store: MPDCStore) -> None:
        self.config = config
        self.store = store

    def compute_online_scorecard(self) -> dict[str, Any]:
        """Computes Pillar 1 Online Telemetry: Pass@1, ITG, Patch Survival Rate, and Skill Attribution Lift."""
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_telemetry ORDER BY created_at ASC"
            ).fetchall()
            skills_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM episodic_skills WHERE active = 1"
            ).fetchone()["cnt"]
            pruned_count = conn.execute(
                "SELECT COUNT(*) AS cnt FROM episodic_skills WHERE active = 0"
            ).fetchone()["cnt"]
            traj_counts = conn.execute(
                "SELECT split, COUNT(*) AS cnt FROM harvested_trajectories GROUP BY split"
            ).fetchall()

        splits = {r["split"]: r["cnt"] for r in traj_counts}
        promoted = self.store.get_latest_promoted_adapter()
        active_adapter = promoted["version"] if promoted else "base"

        if not rows:
            return {
                "total_tasks": 0,
                "pass_at_1": 0.0,
                "mean_iterations_to_green": 0.0,
                "mean_patch_survival_rate": 0.0,
                "skill_attribution_lift_pass_at_1": 0.0,
                "active_skills": skills_count,
                "pruned_poison_skills": pruned_count,
                "train_trajectories": splits.get("train", 0),
                "golden_replay_holdout": splits.get("golden_replay", 0),
                "active_adapter": active_adapter,
                "trend": "insufficient_data",
            }

        total = len(rows)
        pass_1 = sum(r["passed_first_try"] for r in rows) / total
        mean_itg = sum(r["iterations_to_green"] for r in rows) / total
        mean_psr = sum(r["patch_survival_rate"] for r in rows) / total

        # Skill Attribution Lift: compare tasks that retrieved >=1 Episodic Skill vs tasks without skills
        with_skills = [r for r in rows if json.loads(r["skills_used_json"])]
        without_skills = [r for r in rows if not json.loads(r["skills_used_json"])]

        if with_skills and without_skills:
            p1_with = sum(r["passed_first_try"] for r in with_skills) / len(with_skills)
            p1_without = sum(r["passed_first_try"] for r in without_skills) / len(without_skills)
            skill_lift = round(p1_with - p1_without, 4)
        else:
            skill_lift = 0.0

        # Longitudinal trend: compare second half of tasks vs first half
        half = max(1, total // 2)
        early_p1 = sum(r["passed_first_try"] for r in rows[:half]) / len(rows[:half])
        recent_p1 = sum(r["passed_first_try"] for r in rows[half:]) / max(1, len(rows[half:]))
        delta_p1 = round(recent_p1 - early_p1, 4)

        return {
            "total_tasks": total,
            "pass_at_1": round(pass_1, 4),
            "mean_iterations_to_green": round(mean_itg, 3),
            "mean_patch_survival_rate": round(mean_psr, 4),
            "skill_attribution_lift_pass_at_1": skill_lift,
            "longitudinal_pass_at_1_delta": delta_p1,
            "active_skills": skills_count,
            "pruned_poison_skills": pruned_count,
            "train_trajectories": splits.get("train", 0),
            "golden_replay_holdout": splits.get("golden_replay", 0),
            "active_adapter": active_adapter,
            "trend": "improving" if delta_p1 >= 0 else "regressing",
        }

    def evaluate_candidate_adapter(
        self,
        candidate_version: str,
        candidate_path: str,
        candidate_replay_pass_rate: float,
        candidate_perplexity: float,
        candidate_sanity_score: float,
    ) -> AdapterEvalComparison:
        """Pillar 2 Automated Promotion Gate comparing Candidate vN+1 against Current Production vN."""
        current = self.store.get_latest_promoted_adapter()
        if current is None:
            current_version = "base"
            current_replay = 0.25
            current_ppl = 4.50
            baseline_sanity = 1.00
        else:
            current_version = current["version"]
            current_replay = float(current["replay_pass_rate"])
            current_ppl = float(current["holdout_perplexity"])
            baseline_sanity = float(current["sanity_score"])

        replay_delta = candidate_replay_pass_rate - current_replay
        sanity_drop = baseline_sanity - candidate_sanity_score
        ppl_improved = candidate_perplexity <= current_ppl

        reasons: list[str] = []
        if replay_delta < self.config.evaluation.min_replay_pass_rate_delta:
            reasons.append(
                f"Replay Pass@1 delta ({replay_delta:+.3f}) below min ({self.config.evaluation.min_replay_pass_rate_delta:+.3f})"
            )
        if sanity_drop > self.config.evaluation.max_sanity_regression:
            reasons.append(
                f"Sanity score dropped by {sanity_drop:.3f} (max allowed {self.config.evaluation.max_sanity_regression:.3f})"
            )
        if self.config.evaluation.require_perplexity_improvement and not ppl_improved:
            reasons.append(
                f"Holdout perplexity worsened ({candidate_perplexity:.3f} > {current_ppl:.3f})"
            )

        promoted = len(reasons) == 0
        reason_str = (
            f"PROMOTED: Replay Pass@1 {current_replay:.1%} -> {candidate_replay_pass_rate:.1%}, "
            f"Holdout PPL {current_ppl:.2f} -> {candidate_perplexity:.2f}, Sanity {candidate_sanity_score:.1%}."
            if promoted
            else "REJECTED (Auto-Rollback): " + "; ".join(reasons)
        )

        self.store.register_adapter(
            version=candidate_version,
            base_model=self.config.inference.base_model,
            adapter_path=candidate_path,
            replay_pass_rate=candidate_replay_pass_rate,
            holdout_perplexity=candidate_perplexity,
            sanity_score=candidate_sanity_score,
            promoted=promoted,
        )

        return AdapterEvalComparison(
            candidate_version=candidate_version,
            current_version=current_version,
            candidate_replay_pass_rate=candidate_replay_pass_rate,
            current_replay_pass_rate=current_replay,
            candidate_perplexity=candidate_perplexity,
            current_perplexity=current_ppl,
            candidate_sanity_score=candidate_sanity_score,
            baseline_sanity_score=baseline_sanity,
            promoted=promoted,
            reason=reason_str,
        )

    @staticmethod
    def estimate_token_perplexity(reference_text: str, predicted_text: str) -> float:
        """Deterministic token overlap cross-entropy proxy when running offline evaluations."""
        ref_tokens = reference_text.lower().split()
        pred_tokens = set(predicted_text.lower().split())
        if not ref_tokens:
            return 1.0
        hits = sum(1 for t in ref_tokens if t in pred_tokens)
        prob = max(0.05, min(0.99, hits / len(ref_tokens)))
        return round(math.exp(-math.log(prob)), 4)
