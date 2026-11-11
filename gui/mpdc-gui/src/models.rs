//! Domain models, telemetry payloads, and deterministic demo state shared across TUI and GUI.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerInfo {
    pub engine: String,
    pub base_model: String,
    pub active_model: String,
    pub backend_url: String,
    pub port: u16,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTelemetry {
    pub name: String,
    pub vram_used_mb: f64,
    pub vram_total_mb: f64,
    pub vram_pct: f64,
    pub gpu_util_pct: f64,
    pub temp_c: f64,
    pub power_w: f64,
    pub power_cap_w: f64,
    pub kv_cache_pct: f64,
}

impl Default for GpuTelemetry {
    fn default() -> Self {
        Self {
            name: "NVIDIA RTX 5070".into(),
            vram_used_mb: 5120.0,
            vram_total_mb: 8192.0,
            vram_pct: 0.625,
            gpu_util_pct: 0.74,
            temp_c: 58.0,
            power_w: 62.4,
            power_cap_w: 115.0,
            kv_cache_pct: 0.48,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Scorecard {
    pub total_tasks: u64,
    pub pass_at_1: f64,
    pub mean_iterations_to_green: f64,
    pub mean_patch_survival_rate: f64,
    pub skill_attribution_lift_pass_at_1: f64,
    pub active_skills: u64,
    pub pruned_poison_skills: u64,
    pub train_trajectories: u64,
    pub golden_replay_holdout: u64,
    pub active_adapter: String,
    pub trend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientNode {
    pub node_id: String,
    pub hostname: String,
    pub ip_address: String,
    pub status: String,
    pub requests_count: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub last_tok_per_sec: f64,
    pub active_model: String,
    #[serde(default = "default_rate_cap")]
    pub rate_cap_tps: f64,
    #[serde(default = "default_priority")]
    pub priority: String,
}

fn default_rate_cap() -> f64 {
    120.0
}

fn default_priority() -> String {
    "NORM".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillItem {
    pub id: i64,
    pub title: String,
    pub domain: String,
    pub trigger_signature: String,
    pub anti_pattern: String,
    pub verified_pattern: String,
    pub times_retrieved: u64,
    #[serde(default)]
    pub success_count: u64,
    #[serde(default)]
    pub failure_count: u64,
    pub utility_score: f64,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterItem {
    pub version: String,
    pub base_model: String,
    pub replay_pass_rate: f64,
    pub holdout_perplexity: f64,
    pub sanity_score: f64,
    pub promoted: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OverviewPayload {
    pub server: ServerInfo,
    #[serde(default)]
    pub gpu: GpuTelemetry,
    pub scorecard: Scorecard,
    pub nodes: Vec<ClientNode>,
    pub skills: Vec<SkillItem>,
    pub adapters: Vec<AdapterItem>,
}

#[derive(Debug, Clone)]
pub struct AgentRoleParams {
    pub role: &'static str,
    pub thinking: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub retries: u8,
}

impl OverviewPayload {
    pub fn demo_seed() -> Self {
        Self {
            server: ServerInfo {
                engine: "vllm".into(),
                base_model: "Qwen/Qwen3-8B".into(),
                active_model: "qwen3-lora-v001".into(),
                backend_url: "http://127.0.0.1:8001/v1".into(),
                port: 8000,
                total_prompt_tokens: 24800,
                total_completion_tokens: 11420,
            },
            gpu: GpuTelemetry::default(),
            scorecard: Scorecard {
                total_tasks: 28,
                pass_at_1: 0.786,
                mean_iterations_to_green: 1.18,
                mean_patch_survival_rate: 0.924,
                skill_attribution_lift_pass_at_1: 0.265,
                active_skills: 3,
                pruned_poison_skills: 1,
                train_trajectories: 22,
                golden_replay_holdout: 6,
                active_adapter: "qwen3-lora-v001".into(),
                trend: "improving".into(),
            },
            nodes: vec![
                ClientNode {
                    node_id: "rtx5070-laptop".into(),
                    hostname: "sgk-nixos-laptop".into(),
                    ip_address: "192.168.1.42".into(),
                    status: "online".into(),
                    requests_count: 42,
                    prompt_tokens: 8400,
                    completion_tokens: 3920,
                    last_tok_per_sec: 68.4,
                    active_model: "qwen3-lora-v001".into(),
                    rate_cap_tps: 100.0,
                    priority: "HIGH".into(),
                },
                ClientNode {
                    node_id: "ros2-field-deck".into(),
                    hostname: "ros2-humble-deck".into(),
                    ip_address: "100.94.18.12".into(),
                    status: "online".into(),
                    requests_count: 29,
                    prompt_tokens: 5200,
                    completion_tokens: 2310,
                    last_tok_per_sec: 64.1,
                    active_model: "qwen3-lora-v001".into(),
                    rate_cap_tps: 80.0,
                    priority: "NORM".into(),
                },
                ClientNode {
                    node_id: "workstation-sim".into(),
                    hostname: "isaac-sim-rig".into(),
                    ip_address: "192.168.1.88".into(),
                    status: "online".into(),
                    requests_count: 54,
                    prompt_tokens: 11200,
                    completion_tokens: 5190,
                    last_tok_per_sec: 71.2,
                    active_model: "qwen3-mpdc-fast".into(),
                    rate_cap_tps: 120.0,
                    priority: "HIGH".into(),
                },
            ],
            skills: vec![
                SkillItem {
                    id: 1,
                    title: "ROS2 Humble LaserScan subscriber QoS mismatch".into(),
                    domain: "robotics".into(),
                    trigger_signature: "New publisher discovered on '/scan', incompatible QoS".into(),
                    anti_pattern: "QoSProfile(depth=10) # RELIABLE drops BestEffort".into(),
                    verified_pattern: "QoSProfile(depth=10, reliability=BEST_EFFORT)".into(),
                    times_retrieved: 14,
                    success_count: 12,
                    failure_count: 1,
                    utility_score: 0.87,
                    active: true,
                },
                SkillItem {
                    id: 2,
                    title: "VLA action chunk transformer output shape mismatch".into(),
                    domain: "robotics".into(),
                    trigger_signature: "RuntimeError: shape '[16, 7]' invalid for 896".into(),
                    anti_pattern: "actions = head(latent).reshape(batch_size, -1)".into(),
                    verified_pattern: "actions = head(latent).view(B, horizon, act_dim)".into(),
                    times_retrieved: 9,
                    success_count: 8,
                    failure_count: 0,
                    utility_score: 0.90,
                    active: true,
                },
                SkillItem {
                    id: 3,
                    title: "NixOS CUDA systemd service libcuda.so.1 missing".into(),
                    domain: "nixos".into(),
                    trigger_signature: "ImportError: libcuda.so.1: cannot open shared object".into(),
                    anti_pattern: "ExecStart = vllm serve (unwrapped)".into(),
                    verified_pattern: "Bind /run/opengl-driver/lib in systemd serviceConfig".into(),
                    times_retrieved: 6,
                    success_count: 5,
                    failure_count: 0,
                    utility_score: 0.86,
                    active: true,
                },
            ],
            adapters: vec![
                AdapterItem {
                    version: "qwen3-lora-v001".into(),
                    base_model: "Qwen/Qwen3-8B".into(),
                    replay_pass_rate: 0.68,
                    holdout_perplexity: 2.31,
                    sanity_score: 1.0,
                    promoted: 1,
                },
                AdapterItem {
                    version: "qwen3-lora-v000".into(),
                    base_model: "Qwen/Qwen3-8B".into(),
                    replay_pass_rate: 0.25,
                    holdout_perplexity: 4.50,
                    sanity_score: 1.0,
                    promoted: 1,
                },
            ],
        }
    }
}
