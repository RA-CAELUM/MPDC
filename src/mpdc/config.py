"""Configuration management for MPDC."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: str = "./.mpdc_data"


class InferenceConfig(BaseModel):
    engine: str = "vllm"  # "vllm", "sglang", or "llama_cpp"
    backend_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "mpdc-local"
    base_model: str = "Qwen/Qwen3-8B"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    enable_thinking_for_planner: bool = True
    enable_thinking_for_coder: bool = False
    enable_thinking_for_distiller: bool = True
    active_adapter_name: Optional[str] = None
    active_adapter_path: Optional[str] = None


class AgentsConfig(BaseModel):
    max_verification_retries: int = 3
    top_k_skills: int = 4
    min_skill_similarity: float = 0.25


class HarvesterConfig(BaseModel):
    train_split_ratio: float = 0.80
    min_patch_lines: int = 2
    chosen_survival_threshold: float = 0.75


class TrainerConfig(BaseModel):
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2.0e-4
    num_epochs: int = 2
    max_seq_length: int = 4096
    min_samples_to_train: int = 10
    pause_inference_during_training: bool = True


class EvaluationConfig(BaseModel):
    min_replay_pass_rate_delta: float = 0.0
    max_sanity_regression: float = 0.02
    require_perplexity_improvement: bool = True


class MPDCConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    harvester: HarvesterConfig = Field(default_factory=HarvesterConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def data_path(self) -> Path:
        env_dir = os.environ.get("MPDC_DATA_DIR")
        path = Path(env_dir) if env_dir else Path(self.server.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        for sub in ("datasets", "adapters", "eval", "logs"):
            (path / sub).mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "MPDCConfig":
        path_str = config_path or os.environ.get("MPDC_CONFIG", "config/default.yaml")
        path = Path(path_str)
        data = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        cfg = cls(**data)
        if os.environ.get("MPDC_BACKEND_URL"):
            cfg.inference.backend_url = os.environ["MPDC_BACKEND_URL"]
        if os.environ.get("MPDC_BASE_MODEL"):
            cfg.inference.base_model = os.environ["MPDC_BASE_MODEL"]
        return cfg
