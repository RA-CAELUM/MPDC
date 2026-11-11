//! Central workbench quadrants (`[SWB.01]` Switchboard, `[BUS.02]` Onboarder/Bus, `[MEM.03]` Skill Matrix, `[GAT.04]` Adapter Gate).

use crate::{
    client::format_client_opencode_json,
    gui::{
        theme::{
            draw_led, draw_segmented_meter, section_header, ACCENT_CYAN, ACCENT_STEEL, BG_PANEL,
            STATUS_ERR, STATUS_OK, STATUS_WARN, TEXT_MUTED, TEXT_PRIMARY,
        },
        MpdcWorkbenchApp,
    },
    models::{ClientNode, SkillItem},
};
use eframe::egui::{self, Color32, RichText};
use egui_plot::{Bar, BarChart, Plot};

impl MpdcWorkbenchApp {
    pub fn render_node_switchboard(&mut self, ui: &mut egui::Ui, height: f32) {
        section_header(ui, "SWB.01", "OUTPUT COMPUTERS TOKEN SWITCHBOARD");
        egui::Frame::group(ui.style())
            .fill(BG_PANEL)
            .show(ui, |ui| {
                ui.set_min_height(height - 28.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("FLT:").color(TEXT_MUTED));
                    ui.add(egui::TextEdit::singleline(&mut self.node_filter).desired_width(90.0));
                    if ui.small_button("[ALL TX ON]").clicked() {
                        for n in &mut self.overview.nodes {
                            n.status = "online".into();
                        }
                    }
                    if ui.small_button("[HALT ALL]").clicked() {
                        for n in &mut self.overview.nodes {
                            n.status = "paused".into();
                        }
                    }
                    if ui.small_button("[RST COUNTERS]").clicked() {
                        for n in &mut self.overview.nodes {
                            n.prompt_tokens = 0;
                            n.completion_tokens = 0;
                            n.requests_count = 0;
                        }
                    }
                });
                ui.separator();

                let total_out = self
                    .overview
                    .nodes
                    .iter()
                    .map(|n| n.completion_tokens)
                    .sum::<u64>()
                    .max(1);

                let filter_lc = self.node_filter.to_lowercase();
                let mut cfg_target: Option<String> = None;
                let mut drop_idx: Option<usize> = None;

                egui::ScrollArea::vertical()
                    .id_salt("swb_nodes_scroll")
                    .max_height(height - 65.0)
                    .show(ui, |ui| {
                        egui::Grid::new("swb_nodes_grid")
                            .striped(true)
                            .num_columns(10)
                            .spacing([5.0, 3.0])
                            .show(ui, |ui| {
                                for h in [
                                    "TX", "NODE_ID", "IP_ADDR", "PRI", "CAP", "TOK/S", "SHARE",
                                    "OUT_TOK", "MODEL", "OPS",
                                ] {
                                    ui.label(RichText::new(h).color(ACCENT_CYAN).strong());
                                }
                                ui.end_row();

                                for (idx, node) in self.overview.nodes.iter_mut().enumerate() {
                                    if !filter_lc.is_empty()
                                        && !node.node_id.to_lowercase().contains(&filter_lc)
                                        && !node.ip_address.contains(&filter_lc)
                                    {
                                        continue;
                                    }

                                    let mut tx_on = node.status != "paused";
                                    ui.horizontal(|ui| {
                                        draw_led(
                                            ui,
                                            if tx_on && self.master_tx_enabled {
                                                STATUS_OK
                                            } else {
                                                STATUS_ERR
                                            },
                                        );
                                        if ui.checkbox(&mut tx_on, "").changed() {
                                            node.status =
                                                if tx_on { "online".into() } else { "paused".into() };
                                        }
                                    });

                                    ui.label(RichText::new(&node.node_id).color(TEXT_PRIMARY).strong());
                                    ui.label(RichText::new(&node.ip_address).color(TEXT_MUTED));

                                    egui::ComboBox::from_id_salt(format!("pri_{idx}"))
                                        .width(52.0)
                                        .selected_text(&node.priority)
                                        .show_ui(ui, |ui| {
                                            for p in ["HIGH", "NORM", "LOW"] {
                                                ui.selectable_value(&mut node.priority, p.into(), p);
                                            }
                                        });

                                    ui.add(
                                        egui::DragValue::new(&mut node.rate_cap_tps)
                                            .range(10.0..=250.0)
                                            .speed(1.0)
                                            .fixed_decimals(0),
                                    );

                                    let tps = if tx_on && self.master_tx_enabled {
                                        node.last_tok_per_sec
                                    } else {
                                        0.0
                                    };
                                    draw_segmented_meter(
                                        ui,
                                        (tps / 100.0) as f32,
                                        58.0,
                                        12.0,
                                        Color32::from_rgb(40, 140, 80),
                                        Color32::from_rgb(80, 210, 120),
                                        Color32::from_rgb(220, 190, 50),
                                        &format!("{tps:4.1}"),
                                    );

                                    let share =
                                        (node.completion_tokens as f64 / total_out as f64) as f32;
                                    draw_segmented_meter(
                                        ui,
                                        share,
                                        46.0,
                                        12.0,
                                        Color32::from_rgb(40, 95, 180),
                                        Color32::from_rgb(70, 150, 235),
                                        Color32::from_rgb(150, 215, 255),
                                        &format!("{:2.0}%", share * 100.0),
                                    );

                                    ui.label(
                                        RichText::new(format!("{}", node.completion_tokens))
                                            .color(STATUS_OK),
                                    );

                                    egui::ComboBox::from_id_salt(format!("mdl_{idx}"))
                                        .width(105.0)
                                        .selected_text(&node.active_model)
                                        .show_ui(ui, |ui| {
                                            for m in [
                                                "qwen3-lora-v001",
                                                "qwen3-mpdc",
                                                "qwen3-mpdc-fast",
                                                "Qwen/Qwen3-8B",
                                            ] {
                                                ui.selectable_value(
                                                    &mut node.active_model,
                                                    m.into(),
                                                    m,
                                                );
                                            }
                                        });

                                    ui.horizontal(|ui| {
                                        if ui.small_button("CFG").clicked() {
                                            cfg_target = Some(node.node_id.clone());
                                        }
                                        if ui.small_button("X").clicked() {
                                            drop_idx = Some(idx);
                                        }
                                    });
                                    ui.end_row();
                                }
                            });
                    });

                if let Some(nid) = cfg_target {
                    self.generated_opencode_json = format_client_opencode_json(
                        &self.datacenter_host,
                        self.overview.server.port,
                        &nid,
                        self.onboard_remote_mcp,
                    );
                }
                if let Some(i) = drop_idx {
                    let removed = self.overview.nodes.remove(i);
                    self.event_feed
                        .insert(0, format!("[SWB] Dropped output node '{}'", removed.node_id));
                }
            });
    }

    pub fn render_onboarder_and_packet_bus(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        height: f32,
    ) {
        section_header(ui, "BUS.02", "NODE ONBOARDER, CONFIG GENERATOR & RPC BUS");
        egui::Frame::group(ui.style())
            .fill(BG_PANEL)
            .show(ui, |ui| {
                ui.set_min_height(height - 28.0);

                ui.horizontal(|ui| {
                    ui.label(RichText::new("ID").color(TEXT_MUTED));
                    ui.add(egui::TextEdit::singleline(&mut self.new_node_id).desired_width(82.0));
                    ui.label(RichText::new("IP").color(TEXT_MUTED));
                    ui.add(egui::TextEdit::singleline(&mut self.new_node_ip).desired_width(82.0));
                    ui.label(RichText::new("CAP").color(TEXT_MUTED));
                    ui.add(
                        egui::DragValue::new(&mut self.new_node_rate_cap)
                            .range(10.0..=250.0)
                            .fixed_decimals(0),
                    );
                    ui.checkbox(&mut self.onboard_remote_mcp, "MCP");
                    ui.checkbox(&mut self.onboard_git_harvester, "DIFF_HARVEST");

                    if ui.button("[+PROVISION NODE]").clicked() {
                        self.generated_opencode_json = format_client_opencode_json(
                            &self.datacenter_host,
                            self.overview.server.port,
                            &self.new_node_id,
                            self.onboard_remote_mcp,
                        );
                        self.overview.nodes.push(ClientNode {
                            node_id: self.new_node_id.clone(),
                            hostname: self.new_node_hostname.clone(),
                            ip_address: self.new_node_ip.clone(),
                            status: "online".into(),
                            requests_count: 1,
                            prompt_tokens: 256,
                            completion_tokens: 128,
                            last_tok_per_sec: 62.5,
                            active_model: self.overview.server.active_model.clone(),
                            rate_cap_tps: self.new_node_rate_cap,
                            priority: "NORM".into(),
                        });
                        self.event_feed.insert(
                            0,
                            format!(
                                "[PROVISION] Added node={} ip={} cap={:.0}t/s",
                                self.new_node_id, self.new_node_ip, self.new_node_rate_cap
                            ),
                        );
                    }
                    if ui.button("[COPY JSON]").clicked() {
                        ctx.copy_text(self.generated_opencode_json.clone());
                        self.status_line = "COPIED opencode.json TO CLIPBOARD".into();
                    }
                });

                ui.horizontal(|ui| {
                    ui.label(RichText::new("TEST_PROMPT").color(TEXT_MUTED));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.test_prompt_input).desired_width(220.0),
                    );
                    if ui.button("[DISPATCH SSE STREAM]").clicked() {
                        self.overview.server.total_prompt_tokens += 512;
                        self.overview.server.total_completion_tokens += 240;
                        self.event_feed.insert(
                            0,
                            format!(
                                "[DISPATCH] prompt='{}' -> routed={} (+2 skills injected, 70.4 tok/s)",
                                self.test_prompt_input, self.overview.server.active_model
                            ),
                        );
                        self.advance_simulation_step();
                    }
                    ui.label(RichText::new("LOG_FLT").color(TEXT_MUTED));
                    ui.add(egui::TextEdit::singleline(&mut self.log_filter).desired_width(65.0));
                });
                ui.separator();

                let body_h = (height - 86.0).max(60.0);
                ui.columns(2, |sub| {
                    egui::ScrollArea::vertical()
                        .id_salt("cfg_code_scroll")
                        .max_height(body_h)
                        .show(&mut sub[0], |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.generated_opencode_json)
                                    .code_editor()
                                    .desired_width(f32::INFINITY),
                            );
                        });

                    let flt = self.log_filter.to_lowercase();
                    egui::ScrollArea::vertical()
                        .id_salt("rpc_bus_scroll")
                        .max_height(body_h)
                        .show(&mut sub[1], |ui| {
                            for line in &self.event_feed {
                                if !flt.is_empty() && !line.to_lowercase().contains(&flt) {
                                    continue;
                                }
                                ui.label(RichText::new(line).color(TEXT_PRIMARY));
                            }
                        });
                });
            });
    }

    pub fn render_skill_editor_matrix(&mut self, ui: &mut egui::Ui, height: f32) {
        section_header(ui, "MEM.03", "TIER 1 EPISODIC SKILL MATRIX & DISTILLATION INJECTOR");
        egui::Frame::group(ui.style())
            .fill(BG_PANEL)
            .show(ui, |ui| {
                ui.set_min_height(height - 28.0);

                ui.horizontal(|ui| {
                    ui.label(RichText::new("DOM").color(TEXT_MUTED));
                    egui::ComboBox::from_id_salt("dom_flt")
                        .width(68.0)
                        .selected_text(&self.skill_domain_filter)
                        .show_ui(ui, |ui| {
                            for d in ["ALL", "robotics", "nixos", "cpp"] {
                                ui.selectable_value(&mut self.skill_domain_filter, d.into(), d);
                            }
                        });
                    ui.label(RichText::new("FLT").color(TEXT_MUTED));
                    ui.add(egui::TextEdit::singleline(&mut self.skill_filter).desired_width(62.0));
                    ui.label(RichText::new("ANTI").color(TEXT_MUTED));
                    ui.add(egui::TextEdit::singleline(&mut self.new_skill_anti).desired_width(90.0));
                    ui.label(RichText::new("FIX").color(TEXT_MUTED));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_skill_verified).desired_width(95.0),
                    );
                    if ui.button("[+DISTILL]").clicked() {
                        let next_id =
                            self.overview.skills.iter().map(|s| s.id).max().unwrap_or(0) + 1;
                        self.overview.skills.insert(
                            0,
                            SkillItem {
                                id: next_id,
                                title: self.new_skill_title.clone(),
                                domain: self.new_skill_domain.clone(),
                                trigger_signature: self.new_skill_title.clone(),
                                anti_pattern: self.new_skill_anti.clone(),
                                verified_pattern: self.new_skill_verified.clone(),
                                times_retrieved: 1,
                                success_count: 2,
                                failure_count: 0,
                                utility_score: 0.75,
                                active: true,
                            },
                        );
                        self.overview.scorecard.active_skills += 1;
                        self.event_feed
                            .insert(0, format!("[DISTILLER] Added Episodic Skill #{next_id}"));
                    }
                    if ui.small_button("[PRUNE]").clicked() {
                        for s in &mut self.overview.skills {
                            if (s.utility_score as f32) < self.skill_prune_thresh {
                                s.active = false;
                            }
                        }
                    }
                });
                ui.separator();

                let dom_flt = self.skill_domain_filter.clone();
                let kw_flt = self.skill_filter.to_lowercase();
                let mut del_skill_idx: Option<usize> = None;

                egui::ScrollArea::vertical()
                    .id_salt("skills_matrix_scroll")
                    .max_height(height - 62.0)
                    .show(ui, |ui| {
                        egui::Grid::new("skills_matrix_grid")
                            .striped(true)
                            .num_columns(7)
                            .spacing([5.0, 3.0])
                            .show(ui, |ui| {
                                for h in [
                                    "ACT", "ID", "UTILITY", "DOM", "W/L",
                                    "ANTI-PATTERN -> VERIFIED FIX", "OPS",
                                ] {
                                    ui.label(RichText::new(h).color(ACCENT_CYAN).strong());
                                }
                                ui.end_row();

                                for (idx, s) in self.overview.skills.iter_mut().enumerate() {
                                    if dom_flt != "ALL" && s.domain != dom_flt {
                                        continue;
                                    }
                                    if !kw_flt.is_empty()
                                        && !s.anti_pattern.to_lowercase().contains(&kw_flt)
                                        && !s.verified_pattern.to_lowercase().contains(&kw_flt)
                                    {
                                        continue;
                                    }
                                    ui.checkbox(&mut s.active, "");
                                    ui.label(format!("#{}", s.id));
                                    draw_segmented_meter(
                                        ui,
                                        s.utility_score as f32,
                                        52.0,
                                        12.0,
                                        STATUS_ERR,
                                        STATUS_WARN,
                                        STATUS_OK,
                                        &format!("{:.2}", s.utility_score),
                                    );
                                    ui.label(RichText::new(&s.domain).color(TEXT_MUTED));
                                    ui.label(format!("{}/{}", s.success_count, s.failure_count));
                                    ui.label(
                                        RichText::new(format!(
                                            "[-] {}  =>  [+] {}",
                                            s.anti_pattern, s.verified_pattern
                                        ))
                                        .color(TEXT_PRIMARY),
                                    );

                                    ui.horizontal(|ui| {
                                        if ui.small_button("+W").clicked() {
                                            s.success_count += 1;
                                            s.times_retrieved += 1;
                                            s.utility_score = (s.success_count as f64 + 1.0)
                                                / (s.success_count as f64
                                                    + s.failure_count as f64
                                                    + 2.0);
                                        }
                                        if ui.small_button("+L").clicked() {
                                            s.failure_count += 1;
                                            s.times_retrieved += 1;
                                            s.utility_score = (s.success_count as f64 + 1.0)
                                                / (s.success_count as f64
                                                    + s.failure_count as f64
                                                    + 2.0);
                                            if (s.utility_score as f32) < self.skill_prune_thresh {
                                                s.active = false;
                                            }
                                        }
                                        if ui.small_button("X").clicked() {
                                            del_skill_idx = Some(idx);
                                        }
                                    });
                                    ui.end_row();
                                }
                            });
                    });

                if let Some(i) = del_skill_idx {
                    self.overview.skills.remove(i);
                }
            });
    }

    pub fn render_adapter_promotion_matrix(&mut self, ui: &mut egui::Ui, height: f32) {
        section_header(
            ui,
            "GAT.04",
            "TIER 3 PEFT LORA REGISTRY & GOLDEN REPLAY BENCHMARK",
        );
        egui::Frame::group(ui.style())
            .fill(BG_PANEL)
            .show(ui, |ui| {
                ui.set_min_height(height - 28.0);

                ui.columns(2, |cols| {
                    let active_mdl = self.overview.server.active_model.clone();
                    let mut hotload_target: Option<String> = None;

                    egui::ScrollArea::vertical()
                        .id_salt("adapter_reg_scroll")
                        .max_height(height - 38.0)
                        .show(&mut cols[0], |ui| {
                            egui::Grid::new("adapter_reg_grid")
                                .striped(true)
                                .num_columns(6)
                                .spacing([5.0, 3.0])
                                .show(ui, |ui| {
                                    for h in ["GATE", "VERSION", "REPLAY@1", "PPL", "SAN", "LOAD"] {
                                        ui.label(RichText::new(h).color(ACCENT_CYAN).strong());
                                    }
                                    ui.end_row();

                                    for a in &self.overview.adapters {
                                        let prom = a.promoted != 0;
                                        ui.label(
                                            RichText::new(if prom { "PROMOTED" } else { "ROLLBACK" })
                                                .color(if prom { STATUS_OK } else { STATUS_ERR })
                                                .strong(),
                                        );
                                        let is_act = active_mdl == a.version;
                                        ui.label(
                                            RichText::new(&a.version)
                                                .color(if is_act {
                                                    ACCENT_CYAN
                                                } else {
                                                    TEXT_PRIMARY
                                                })
                                                .strong(),
                                        );
                                        draw_segmented_meter(
                                            ui,
                                            a.replay_pass_rate as f32,
                                            52.0,
                                            12.0,
                                            STATUS_ERR,
                                            STATUS_WARN,
                                            STATUS_OK,
                                            &format!("{:.0}%", a.replay_pass_rate * 100.0),
                                        );
                                        ui.label(format!("{:.2}", a.holdout_perplexity));
                                        ui.label(format!("{:.0}%", a.sanity_score * 100.0));
                                        if ui
                                            .small_button(if is_act { "[ACTIVE]" } else { "[MOUNT]" })
                                            .clicked()
                                        {
                                            hotload_target = Some(a.version.clone());
                                        }
                                        ui.end_row();
                                    }
                                });
                        });

                    if let Some(ver) = hotload_target {
                        self.overview.server.active_model = ver.clone();
                        self.event_feed.insert(
                            0,
                            format!("[VLLM_LORA] Hot-loaded adapter '{ver}' via /v1/load_lora_adapter"),
                        );
                    }

                    let bars: Vec<Bar> = self
                        .overview
                        .adapters
                        .iter()
                        .rev()
                        .enumerate()
                        .map(|(i, a)| {
                            Bar::new(i as f64, a.replay_pass_rate * 100.0)
                                .name(&a.version)
                                .width(0.55)
                                .fill(if a.promoted != 0 {
                                    ACCENT_STEEL
                                } else {
                                    STATUS_ERR
                                })
                        })
                        .collect();

                    Plot::new("adapter_bar_plot")
                        .height(height - 42.0)
                        .allow_zoom(false)
                        .allow_drag(false)
                        .show(&mut cols[1], |plot_ui| {
                            plot_ui.bar_chart(BarChart::new(bars).name("Replay Pass@1 %"));
                        });
                });
            });
    }
}
