//! `MpdcWorkbenchApp` state, simulation engine, and top/bottom transport bars.

pub mod quadrants;
pub mod rack;
pub mod scopes;
pub mod theme;

use crate::{
    client::{
        format_client_opencode_json, spawn_poll_overview, spawn_post_action, BgResult,
    },
    gui::theme::{
        apply_workbench_style, draw_led, ACCENT_CYAN, STATUS_OK, STATUS_WARN, TEXT_MUTED,
        TEXT_PRIMARY,
    },
    models::{AdapterItem, AgentRoleParams, OverviewPayload},
};
use eframe::egui::{self, RichText, Vec2};
use std::{
    sync::mpsc::{self, Receiver, Sender},
    time::{Duration, Instant},
};

pub struct MpdcWorkbenchApp {
    pub gateway_url: String,
    pub datacenter_host: String,
    pub poll_interval_ms: u64,
    pub auto_poll: bool,
    pub master_tx_enabled: bool,
    pub scope_frozen: bool,
    pub scope_window: usize,
    pub connected: bool,
    pub status_line: String,
    pub last_poll: Instant,
    pub tick: f64,

    pub show_trace_cluster_tps: bool,
    pub show_trace_node0_tps: bool,
    pub show_trace_node1_tps: bool,
    pub show_trace_node2_tps: bool,
    pub show_trace_pass1: bool,
    pub show_trace_psr: bool,
    pub show_trace_lift: bool,
    pub show_trace_ppl: bool,
    pub show_trace_ttft: bool,
    pub show_trace_itl: bool,

    pub tps_cluster_hist: Vec<[f64; 2]>,
    pub tps_node0_hist: Vec<[f64; 2]>,
    pub tps_node1_hist: Vec<[f64; 2]>,
    pub tps_node2_hist: Vec<[f64; 2]>,
    pub pass1_hist: Vec<[f64; 2]>,
    pub psr_hist: Vec<[f64; 2]>,
    pub lift_hist: Vec<[f64; 2]>,
    pub ppl_hist: Vec<[f64; 2]>,
    pub ttft_hist: Vec<[f64; 2]>,
    pub itl_hist: Vec<[f64; 2]>,

    pub selected_engine: String,
    pub kv_quant_mode: String,
    pub vram_fraction: f32,
    pub max_num_seqs: u32,
    pub power_limit_w: f32,
    pub enable_prefix_cache: bool,
    pub enable_chunked_prefill: bool,
    pub enable_speculative: bool,
    pub speculative_draft_toks: u8,

    pub agent_roles: Vec<AgentRoleParams>,
    pub skill_top_k: usize,
    pub skill_min_sim: f32,
    pub skill_prune_thresh: f32,
    pub inject_ast_graph: bool,
    pub auto_distill_on_fail: bool,

    pub lora_rank: u32,
    pub lora_alpha: u32,
    pub lora_lr: f64,
    pub lora_epochs: u32,
    pub lora_seq_len: u32,
    pub train_split_ratio: f32,
    pub gate_min_replay_delta: f32,
    pub gate_max_sanity_drop: f32,
    pub gate_require_ppl: bool,
    pub gate_auto_hotswap: bool,

    pub node_filter: String,
    pub new_node_id: String,
    pub new_node_hostname: String,
    pub new_node_ip: String,
    pub new_node_rate_cap: f64,
    pub onboard_remote_mcp: bool,
    pub onboard_git_harvester: bool,
    pub generated_opencode_json: String,

    pub skill_filter: String,
    pub skill_domain_filter: String,
    pub new_skill_domain: String,
    pub new_skill_title: String,
    pub new_skill_anti: String,
    pub new_skill_verified: String,
    pub log_filter: String,
    pub test_prompt_input: String,
    pub event_feed: Vec<String>,

    pub overview: OverviewPayload,
    pub tx: Sender<BgResult>,
    pub rx: Receiver<BgResult>,
}

impl MpdcWorkbenchApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        apply_workbench_style(&cc.egui_ctx);

        let (tx, rx) = mpsc::channel();
        let gateway_url = std::env::var("MPDC_GATEWAY_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());

        let mut app = Self {
            gateway_url,
            datacenter_host: "mpdc.local".to_string(),
            poll_interval_ms: 1000,
            auto_poll: true,
            master_tx_enabled: true,
            scope_frozen: false,
            scope_window: 60,
            connected: false,
            status_line: "SIMULATION MODE // ALL CONTROLS ACTIVE".to_string(),
            last_poll: Instant::now() - Duration::from_secs(5),
            tick: 0.0,

            show_trace_cluster_tps: true,
            show_trace_node0_tps: true,
            show_trace_node1_tps: true,
            show_trace_node2_tps: true,
            show_trace_pass1: true,
            show_trace_psr: true,
            show_trace_lift: true,
            show_trace_ppl: true,
            show_trace_ttft: true,
            show_trace_itl: true,

            tps_cluster_hist: Vec::new(),
            tps_node0_hist: Vec::new(),
            tps_node1_hist: Vec::new(),
            tps_node2_hist: Vec::new(),
            pass1_hist: Vec::new(),
            psr_hist: Vec::new(),
            lift_hist: Vec::new(),
            ppl_hist: Vec::new(),
            ttft_hist: Vec::new(),
            itl_hist: Vec::new(),

            selected_engine: "vllm".into(),
            kv_quant_mode: "FP8_E4M3".into(),
            vram_fraction: 0.88,
            max_num_seqs: 16,
            power_limit_w: 115.0,
            enable_prefix_cache: true,
            enable_chunked_prefill: true,
            enable_speculative: true,
            speculative_draft_toks: 4,

            agent_roles: vec![
                AgentRoleParams {
                    role: "PLANNER",
                    thinking: true,
                    temperature: 0.20,
                    top_p: 0.90,
                    max_tokens: 4096,
                    retries: 2,
                },
                AgentRoleParams {
                    role: "CODER",
                    thinking: false,
                    temperature: 0.10,
                    top_p: 0.85,
                    max_tokens: 8192,
                    retries: 3,
                },
                AgentRoleParams {
                    role: "VERIFIER",
                    thinking: true,
                    temperature: 0.05,
                    top_p: 0.80,
                    max_tokens: 2048,
                    retries: 3,
                },
                AgentRoleParams {
                    role: "DISTILLER",
                    thinking: true,
                    temperature: 0.15,
                    top_p: 0.90,
                    max_tokens: 2048,
                    retries: 1,
                },
            ],
            skill_top_k: 4,
            skill_min_sim: 0.25,
            skill_prune_thresh: 0.30,
            inject_ast_graph: true,
            auto_distill_on_fail: true,

            lora_rank: 16,
            lora_alpha: 32,
            lora_lr: 2.0e-4,
            lora_epochs: 2,
            lora_seq_len: 4096,
            train_split_ratio: 0.80,
            gate_min_replay_delta: 0.00,
            gate_max_sanity_drop: 0.02,
            gate_require_ppl: true,
            gate_auto_hotswap: true,

            node_filter: String::new(),
            new_node_id: "thinkpad-x1".into(),
            new_node_hostname: "field-laptop-02".into(),
            new_node_ip: "192.168.1.64".into(),
            new_node_rate_cap: 90.0,
            onboard_remote_mcp: true,
            onboard_git_harvester: true,
            generated_opencode_json: format_client_opencode_json(
                "mpdc.local",
                8000,
                "rtx5070-laptop",
                true,
            ),

            skill_filter: String::new(),
            skill_domain_filter: "ALL".into(),
            new_skill_domain: "robotics".into(),
            new_skill_title: "MuJoCo contact solver instability".into(),
            new_skill_anti: "impratio=1.0 with high condim=6 slip".into(),
            new_skill_verified: "solref='0.004 1' impratio=10 noslip_iterations=3".into(),
            log_filter: String::new(),
            test_prompt_input: "Fix ROS2 Humble lifecycle sensor QoS publisher".into(),
            event_feed: vec![
                "00:50:01.104 [STREAM] node=rtx5070-laptop model=qwen3-lora-v001 mode=/think skills=[#1,#2] rate=68.4t/s".into(),
                "00:50:02.418 [STREAM] node=workstation-sim model=qwen3-mpdc-fast mode=/no_think skills=[#2] rate=71.2t/s".into(),
                "00:50:04.009 [HARVEST] task=verify-89a1c2 psr=0.962 itg=1 split=train dpo_pair=false".into(),
                "00:50:06.882 [MCP_RPC] tool=mpdc_plan node=ros2-field-deck latency=412ms skills_hit=2".into(),
                "00:50:09.120 [GATE] adapter=qwen3-lora-v001 replay@1=68.0% ppl=2.31 sanity=100% status=PROMOTED".into(),
            ],

            overview: OverviewPayload::demo_seed(),
            tx,
            rx,
        };

        for _ in 0..35 {
            app.advance_simulation_step();
        }
        app.poll_gateway();
        app
    }

    pub fn advance_simulation_step(&mut self) {
        if self.scope_frozen {
            return;
        }
        self.tick += 1.0;
        let t = self.tick;

        let n0 = if self
            .overview
            .nodes
            .first()
            .map(|n| n.status != "paused" && self.master_tx_enabled)
            .unwrap_or(false)
        {
            let cap = self.overview.nodes[0].rate_cap_tps;
            (68.4 + (t * 0.45).sin() * 4.8 + (t * 1.3).cos() * 1.9).min(cap)
        } else {
            0.0
        };
        let n1 = if self
            .overview
            .nodes
            .get(1)
            .map(|n| n.status != "paused" && self.master_tx_enabled)
            .unwrap_or(false)
        {
            let cap = self.overview.nodes[1].rate_cap_tps;
            (64.1 + (t * 0.35 + 1.0).sin() * 3.9).min(cap)
        } else {
            0.0
        };
        let n2 = if self
            .overview
            .nodes
            .get(2)
            .map(|n| n.status != "paused" && self.master_tx_enabled)
            .unwrap_or(false)
        {
            let cap = self.overview.nodes[2].rate_cap_tps;
            (71.2 + (t * 0.60 + 2.1).cos() * 5.2).min(cap)
        } else {
            0.0
        };

        if let Some(node) = self.overview.nodes.get_mut(0) {
            node.last_tok_per_sec = n0;
        }
        if let Some(node) = self.overview.nodes.get_mut(1) {
            node.last_tok_per_sec = n1;
        }
        if let Some(node) = self.overview.nodes.get_mut(2) {
            node.last_tok_per_sec = n2;
        }

        let cluster = n0 + n1 + n2;
        self.tps_node0_hist.push([t, n0]);
        self.tps_node1_hist.push([t, n1]);
        self.tps_node2_hist.push([t, n2]);
        self.tps_cluster_hist.push([t, cluster]);

        let p1 = self.overview.scorecard.pass_at_1 * 100.0 + (t * 0.2).sin() * 0.6;
        let psr = self.overview.scorecard.mean_patch_survival_rate * 100.0 + (t * 0.25).cos() * 0.4;
        let lift = self.overview.scorecard.skill_attribution_lift_pass_at_1 * 100.0;
        self.pass1_hist.push([t, p1]);
        self.psr_hist.push([t, psr]);
        self.lift_hist.push([t, lift]);

        let latest_ppl = self
            .overview
            .adapters
            .first()
            .map(|a| a.holdout_perplexity)
            .unwrap_or(2.31);
        let ttft = 17.5 + (t * 0.7).sin() * 2.1;
        let itl = 14.1 + (t * 0.5).cos() * 1.1;
        self.ppl_hist.push([t, latest_ppl * 10.0]);
        self.ttft_hist.push([t, ttft]);
        self.itl_hist.push([t, itl]);

        let w = self.scope_window;
        for buf in [
            &mut self.tps_cluster_hist,
            &mut self.tps_node0_hist,
            &mut self.tps_node1_hist,
            &mut self.tps_node2_hist,
            &mut self.pass1_hist,
            &mut self.psr_hist,
            &mut self.lift_hist,
            &mut self.ppl_hist,
            &mut self.ttft_hist,
            &mut self.itl_hist,
        ] {
            while buf.len() > w {
                buf.remove(0);
            }
        }
    }

    pub fn poll_gateway(&mut self) {
        self.last_poll = Instant::now();
        spawn_poll_overview(&self.gateway_url, self.tx.clone());
    }

    pub fn trigger_post(&self, path: &str, body: serde_json::Value) {
        spawn_post_action(&self.gateway_url, path, body, self.tx.clone());
    }

    pub fn run_qlora_action(&mut self) {
        if self.connected {
            self.trigger_post("/v1/mpdc/train?dry_run=true", serde_json::json!({}));
        } else {
            let next_num = self.overview.adapters.len();
            let ver = format!("qwen3-lora-v{:03}", next_num);
            let cand_replay = (self.overview.scorecard.pass_at_1 + 0.032).min(0.97);
            let cand_ppl = (2.31 - (next_num as f64 * 0.14)).max(1.18);
            let delta = (cand_replay - self.overview.scorecard.pass_at_1) as f32;
            let promoted = delta >= self.gate_min_replay_delta;

            if promoted {
                self.overview.scorecard.pass_at_1 = cand_replay;
                if self.gate_auto_hotswap {
                    self.overview.server.active_model = ver.clone();
                }
            }
            self.overview.adapters.insert(
                0,
                AdapterItem {
                    version: ver.clone(),
                    base_model: self.overview.server.base_model.clone(),
                    replay_pass_rate: cand_replay,
                    holdout_perplexity: cand_ppl,
                    sanity_score: 1.0,
                    promoted: if promoted { 1 } else { 0 },
                },
            );
            self.event_feed.insert(
                0,
                format!(
                    "[QLORA_GATE] ver={} r={} alpha={} lr={:.1e} replay@1={:.1}% ppl={:.2} -> {}",
                    ver,
                    self.lora_rank,
                    self.lora_alpha,
                    self.lora_lr,
                    cand_replay * 100.0,
                    cand_ppl,
                    if promoted { "PROMOTED" } else { "ROLLBACK" }
                ),
            );
            self.advance_simulation_step();
        }
    }
}

impl eframe::App for MpdcWorkbenchApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                BgResult::Overview(Ok(payload)) => {
                    self.connected = true;
                    self.overview = payload;
                    self.advance_simulation_step();
                    self.status_line = format!("GATEWAY LINK OK // {}", self.gateway_url);
                }
                BgResult::Overview(Err(_)) => {
                    self.connected = false;
                    self.advance_simulation_step();
                    self.status_line =
                        "STANDALONE WORKBENCH SIMULATOR // ALL CONTROLS INTERACTIVE".to_string();
                }
                BgResult::ActionLog(log) => {
                    self.event_feed.insert(0, format!("[RPC] {log}"));
                    self.status_line = log;
                    self.poll_gateway();
                }
                BgResult::GeneratedConfig(cfg) => {
                    self.generated_opencode_json = cfg;
                    self.poll_gateway();
                }
            }
        }

        if self.auto_poll
            && self.last_poll.elapsed() >= Duration::from_millis(self.poll_interval_ms)
        {
            self.poll_gateway();
        }
        ctx.request_repaint_after(Duration::from_millis(250));

        egui::TopBottomPanel::top("wb_top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                draw_led(ui, if self.connected { STATUS_OK } else { STATUS_WARN });
                ui.label(
                    RichText::new("MPDC::WORKBENCH v0.2")
                        .color(TEXT_PRIMARY)
                        .strong(),
                );
                ui.separator();

                ui.label(RichText::new("URI").color(TEXT_MUTED));
                ui.add(egui::TextEdit::singleline(&mut self.gateway_url).desired_width(145.0));

                ui.label(RichText::new("POLL(ms)").color(TEXT_MUTED));
                ui.add(
                    egui::DragValue::new(&mut self.poll_interval_ms)
                        .range(100..=5000)
                        .speed(25.0),
                );
                ui.checkbox(&mut self.auto_poll, "AUTO");
                if ui.button("[POLL]").clicked() {
                    self.poll_gateway();
                }
                if ui.button("[STEP]").clicked() {
                    let prev = self.scope_frozen;
                    self.scope_frozen = false;
                    self.advance_simulation_step();
                    self.scope_frozen = prev;
                }

                ui.separator();
                ui.checkbox(&mut self.master_tx_enabled, "MASTER_TX");
                ui.checkbox(&mut self.scope_frozen, "FREEZE_SCOPE");

                if ui.button("[+BURST TRAFFIC]").clicked() {
                    for n in &mut self.overview.nodes {
                        n.requests_count += 5;
                        n.prompt_tokens += 1400;
                        n.completion_tokens += 680;
                    }
                    self.overview.server.total_prompt_tokens += 4200;
                    self.overview.server.total_completion_tokens += 2040;
                    self.event_feed.insert(
                        0,
                        format!(
                            "00:50:{:02.0} [BURST] Injected +2040 output tokens across {} nodes",
                            self.tick % 60.0,
                            self.overview.nodes.len()
                        ),
                    );
                    self.advance_simulation_step();
                }

                if ui.button("[RUN QLORA SLEEP-CYCLE]").clicked() {
                    self.run_qlora_action();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(&self.status_line).color(TEXT_MUTED));
                });
            });
        });

        egui::TopBottomPanel::bottom("wb_bottom_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let cluster_tps: f64 = self
                    .overview
                    .nodes
                    .iter()
                    .filter(|n| n.status != "paused" && self.master_tx_enabled)
                    .map(|n| n.last_tok_per_sec)
                    .sum();
                ui.label(
                    RichText::new(format!(
                        "ENG={} | BACKBONE={} | ACTIVE_LORA={} | CLUSTER_RATE={:.1} tok/s | TOTAL_OUT={} | TOTAL_IN={} | PASS@1={:.1}% | PSR={:.1}%",
                        self.selected_engine.to_uppercase(),
                        self.overview.server.base_model,
                        self.overview.server.active_model,
                        cluster_tps,
                        self.overview.server.total_completion_tokens,
                        self.overview.server.total_prompt_tokens,
                        self.overview.scorecard.pass_at_1 * 100.0,
                        self.overview.scorecard.mean_patch_survival_rate * 100.0,
                    ))
                    .color(ACCENT_CYAN),
                );
            });
        });

        egui::SidePanel::left("wb_left_rack")
            .resizable(true)
            .default_width(310.0)
            .min_width(275.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("left_rack_scroll")
                    .show(ui, |ui| {
                        self.render_left_hardware_and_engine_rack(ui);
                        ui.add_space(4.0);
                        self.render_left_agent_routing_rack(ui);
                        ui.add_space(4.0);
                        self.render_left_qlora_trainer_rack(ui);
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let total_h = ui.available_height();
            let scope_h = (total_h * 0.28).clamp(155.0, 230.0);
            let mid_h = (total_h - scope_h) * 0.52;
            let bot_h = (total_h - scope_h) * 0.46;

            ui.allocate_ui(Vec2::new(ui.available_width(), scope_h), |ui| {
                self.render_oscilloscope_bank(ui, scope_h);
            });

            ui.allocate_ui(Vec2::new(ui.available_width(), mid_h), |ui| {
                ui.columns(2, |cols| {
                    self.render_node_switchboard(&mut cols[0], mid_h);
                    self.render_onboarder_and_packet_bus(&mut cols[1], ctx, mid_h);
                });
            });

            ui.allocate_ui(Vec2::new(ui.available_width(), bot_h), |ui| {
                ui.columns(2, |cols| {
                    self.render_skill_editor_matrix(&mut cols[0], bot_h);
                    self.render_adapter_promotion_matrix(&mut cols[1], bot_h);
                });
            });
        });
    }
}

pub fn run_gui_entry() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(Vec2::new(1480.0, 880.0))
            .with_min_inner_size(Vec2::new(1150.0, 680.0))
            .with_title(
                "MPDC::WORKBENCH // Multi-Node Token Switchboard, Oscilloscope & QLoRA Control Plane",
            ),
        ..Default::default()
    };
    eframe::run_native(
        "mpdc-gui",
        native_options,
        Box::new(|cc| Ok(Box::new(MpdcWorkbenchApp::new(cc)))),
    )
}
