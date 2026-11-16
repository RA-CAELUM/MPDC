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
    engine: str = "vllm"  # "vllm", "sglang", "llama_cpp", or "ollama"
    backend_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "mpdc-local"
    base_model: str = "Qwen/Qwen3-8B"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    # Whether the served model understands Qwen3's /think and /no_think switches.
    # Qwen3 *-Instruct-2507 / *-Thinking-2507 models do not: the directives are
    # stripped instead of being sent as literal text.
    thinking_toggle: bool = True
    enable_thinking_for_planner: bool = True
    enable_thinking_for_coder: bool = False
    enable_thinking_for_distiller: bool = True
    active_adapter_name: Optional[str] = None
    active_adapter_path: Optional[str] = None


class AgentsConfig(BaseModel):
    max_verification_retries: int = 3
    top_k_skills: int = 4
    min_skill_similarity: float = 0.25


class ModelWorkerConfig(BaseModel):
    id: str
    name: str
    role: str
    category: str  # "coding", "reasoning", "vision", "spatial_3d", "geometry"
    endpoint_url: str = "http://127.0.0.1:8001/v1"
    thinking: bool = False
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    active: bool = True


def _default_coding_pool() -> list[ModelWorkerConfig]:
    return [
        ModelWorkerConfig(
            id="qwen3-architect",
            name="Qwen/Qwen3-30B-A3B",
            role="ARCHITECT_PLANNER",
            category="reasoning",
            endpoint_url="http://127.0.0.1:8001/v1",
            thinking=True,
            modalities=["text", "scene_graph"],
            active=True,
        ),
        ModelWorkerConfig(
            id="qwen3-coder",
            name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            role="SYSTEMS_CODER",
            category="coding",
            endpoint_url="http://127.0.0.1:8001/v1",
            thinking=False,
            modalities=["text", "ast_diff", "scene_graph"],
            active=True,
        ),
        ModelWorkerConfig(
            id="deepseek-r1-verifier",
            name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            role="FORMAL_VERIFIER",
            category="reasoning",
            endpoint_url="http://127.0.0.1:8002/v1",
            thinking=True,
            modalities=["text", "compiler_trace"],
            active=True,
        ),
        ModelWorkerConfig(
            id="qwen3-fast-patch",
            name="Qwen/Qwen3-8B",
            role="FAST_PATCH_APPLIER",
            category="coding",
            endpoint_url="http://127.0.0.1:8001/v1",
            thinking=False,
            modalities=["text"],
            active=True,
        ),
    ]


def _default_spatial_visual_pool() -> list[ModelWorkerConfig]:
    return [
        ModelWorkerConfig(
            id="qwen25-vl-32b",
            name="Qwen/Qwen2.5-VL-32B-Instruct",
            role="VLM_2D_VIDEO_CRITIC",
            category="vision",
            endpoint_url="http://127.0.0.1:8003/v1",
            thinking=False,
            modalities=["image", "video", "sim_viewport"],
            active=True,
        ),
        ModelWorkerConfig(
            id="spatial-vlm-3d",
            name="SpatialVLM/Spatial-Grounded-13B",
            role="SPATIAL_3D_GROUNDER",
            category="spatial_3d",
            endpoint_url="http://127.0.0.1:8004/v1",
            thinking=True,
            modalities=["image", "rgbd", "bbox_3d", "se3_pose", "affordance"],
            active=True,
        ),
        ModelWorkerConfig(
            id="mast3r-depth-geo",
            name="Naver/MASt3R-DepthAnything-V2",
            role="GEOMETRY_POINTCLOUD_ENCODER",
            category="geometry",
            endpoint_url="http://127.0.0.1:8005/v1",
            thinking=False,
            modalities=["pointcloud", "depth_map", "urdf_mjcf", "mesh"],
            active=True,
        ),
    ]


class ModelPoolsConfig(BaseModel):
    coordination_mode: str = "sensor_fusion_loop"  # "specialist_graph", "parallel_ensemble", "sensor_fusion_loop"
    coding_models: list[ModelWorkerConfig] = Field(default_factory=_default_coding_pool)
    spatial_visual_models: list[ModelWorkerConfig] = Field(default_factory=_default_spatial_visual_pool)


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
    pools: ModelPoolsConfig = Field(default_factory=ModelPoolsConfig)
    harvester: HarvesterConfig = Field(default_factory=HarvesterConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def data_path(self) -> Path:
        env_dir = os.environ.get("MPDC_DATA_DIR")
        raw_path = Path(env_dir) if env_dir else Path(self.server.data_dir)
        if not raw_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / raw_path
        else:
            path = raw_path
        path.mkdir(parents=True, exist_ok=True)
        for sub in ("datasets", "adapters", "eval", "logs"):
            (path / sub).mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "MPDCConfig":
        repo_root = Path(__file__).resolve().parents[2]
        path_str = config_path or os.environ.get("MPDC_CONFIG")
        if path_str:
            path = Path(path_str)
        else:
            path = repo_root / "config" / "default.yaml"
        data = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        cfg = cls(**data)
        if os.environ.get("MPDC_BACKEND_URL"):
            cfg.inference.backend_url = os.environ["MPDC_BACKEND_URL"]
        if os.environ.get("MPDC_BASE_MODEL"):
            cfg.inference.base_model = os.environ["MPDC_BASE_MODEL"]
        if os.environ.get("MPDC_THINKING_TOGGLE"):
            cfg.inference.thinking_toggle = os.environ["MPDC_THINKING_TOGGLE"].lower() in ("1", "true", "yes")
        return cfg
