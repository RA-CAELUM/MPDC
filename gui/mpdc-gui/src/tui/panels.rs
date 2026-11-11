//! Box rendering methods (`1gpu`, `2kpi`, `3net`, `4nodes`, `5skills`, `6qlora`, and `stream bus`).

use crate::tui::{
    theme::{
        btop_key_label, btop_meter_spans, btop_spark_spans, sample_grad, BTOP_CPU_BOX,
        BTOP_GRAPH_TEXT, BTOP_HI_FG, BTOP_MAIN_FG, BTOP_MEM_BOX, BTOP_NET_BOX, BTOP_PROC_BOX,
        BTOP_PROC_MISC, BTOP_SELECTED_BG, BTOP_SELECTED_FG, BTOP_TITLE, GRAD_CPU, GRAD_DL,
        GRAD_MEM, GRAD_TEMP, GRAD_UL,
    },
    BtopTuiApp,
};
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

impl BtopTuiApp {
    pub fn draw_gpu_box(&self, f: &mut Frame, area: Rect) {
        let g = &self.overview.gpu;
        let title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_CPU_BOX)),
            Span::styled("1", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("gpu [{}]─", g.name),
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let bar_w = area.width.saturating_sub(21).max(6) as usize;

        let mut l1 = vec![Span::styled(" CORE ", Style::default().fg(BTOP_MAIN_FG))];
        l1.extend(btop_meter_spans(g.gpu_util_pct, bar_w, GRAD_CPU));
        l1.push(Span::styled(
            format!(" {:3.0}% {:2.0}°C", g.gpu_util_pct * 100.0, g.temp_c),
            Style::default().fg(sample_grad(GRAD_TEMP, (g.temp_c as f32 - 30.0) / 60.0)),
        ));

        let mut l2 = vec![Span::styled(" VRAM ", Style::default().fg(BTOP_MAIN_FG))];
        l2.extend(btop_meter_spans(g.vram_pct, bar_w, GRAD_MEM));
        l2.push(Span::styled(
            format!(" {:.1}/{:.0}G", g.vram_used_mb / 1024.0, g.vram_total_mb / 1024.0),
            Style::default().fg(BTOP_TITLE),
        ));

        let mut l3 = vec![Span::styled(" KV$  ", Style::default().fg(BTOP_MAIN_FG))];
        l3.extend(btop_meter_spans(g.kv_cache_pct, bar_w, GRAD_DL));
        l3.push(Span::styled(
            format!(" {:3.0}% Paged", g.kv_cache_pct * 100.0),
            Style::default().fg(BTOP_MAIN_FG),
        ));

        let pwr_ratio = g.power_w / g.power_cap_w.max(1.0);
        let mut l4 = vec![Span::styled(" PWR  ", Style::default().fg(BTOP_MAIN_FG))];
        l4.extend(btop_meter_spans(pwr_ratio, bar_w, GRAD_CPU));
        l4.push(Span::styled(
            format!(" {:4.1}/{:.0}W", g.power_w, g.power_cap_w),
            Style::default().fg(BTOP_MAIN_FG),
        ));

        let l5 = Line::from(vec![
            Span::styled(" ENG  ", Style::default().fg(BTOP_GRAPH_TEXT)),
            Span::styled(
                self.overview.server.engine.to_uppercase(),
                Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" | LORA ", Style::default().fg(BTOP_GRAPH_TEXT)),
            Span::styled(
                self.overview.server.active_model.clone(),
                Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
            ),
        ]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_CPU_BOX))
            .title(title);

        f.render_widget(
            Paragraph::new(vec![
                Line::from(l1),
                Line::from(l2),
                Line::from(l3),
                Line::from(l4),
                l5,
            ])
            .block(block),
            area,
        );
    }

    pub fn draw_mem_kpi_box(&self, f: &mut Frame, area: Rect) {
        let sc = &self.overview.scorecard;
        let title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_MEM_BOX)),
            Span::styled("2", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
            Span::styled(
                "self-improvement kpi─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let bar_w = area.width.saturating_sub(24).max(6) as usize;

        let mut l1 = vec![Span::styled(" PASS@1 ", Style::default().fg(BTOP_MAIN_FG))];
        l1.extend(btop_meter_spans(sc.pass_at_1, bar_w, GRAD_CPU));
        l1.push(Span::styled(
            format!(" {:5.1}% ({:.2}x)", sc.pass_at_1 * 100.0, sc.mean_iterations_to_green),
            Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
        ));

        let mut l2 = vec![Span::styled(" SURVIV ", Style::default().fg(BTOP_MAIN_FG))];
        l2.extend(btop_meter_spans(sc.mean_patch_survival_rate, bar_w, GRAD_MEM));
        l2.push(Span::styled(
            format!(" {:5.1}% git", sc.mean_patch_survival_rate * 100.0),
            Style::default().fg(BTOP_TITLE),
        ));

        let lift_norm = (sc.skill_attribution_lift_pass_at_1 + 0.5).clamp(0.0, 1.0);
        let mut l3 = vec![Span::styled(" LIFT   ", Style::default().fg(BTOP_MAIN_FG))];
        l3.extend(btop_meter_spans(lift_norm, bar_w, GRAD_UL));
        l3.push(Span::styled(
            format!(" {:+5.1}% Pass@1", sc.skill_attribution_lift_pass_at_1 * 100.0),
            Style::default().fg(Color::Rgb(255, 118, 238)),
        ));

        let l4 = Line::from(vec![
            Span::styled(" SKILLS ", Style::default().fg(BTOP_MAIN_FG)),
            Span::styled(
                format!("{} active", sc.active_skills),
                Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" / ", Style::default().fg(BTOP_GRAPH_TEXT)),
            Span::styled(
                format!("{} auto-pruned (<0.30)", sc.pruned_poison_skills),
                Style::default().fg(BTOP_HI_FG),
            ),
        ]);

        let l5 = Line::from(vec![
            Span::styled(" SPLIT  ", Style::default().fg(BTOP_MAIN_FG)),
            Span::styled(
                format!("{} SFT/DPO train (80%)", sc.train_trajectories),
                Style::default().fg(Color::Rgb(124, 159, 254)),
            ),
            Span::styled(" | ", Style::default().fg(BTOP_GRAPH_TEXT)),
            Span::styled(
                format!("{} Golden Replay (20%)", sc.golden_replay_holdout),
                Style::default().fg(Color::Rgb(255, 118, 238)),
            ),
        ]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_MEM_BOX))
            .title(title);

        f.render_widget(
            Paragraph::new(vec![
                Line::from(l1),
                Line::from(l2),
                Line::from(l3),
                l4,
                l5,
            ])
            .block(block),
            area,
        );
    }

    pub fn draw_net_stream_box(&self, f: &mut Frame, area: Rect) {
        let title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_NET_BOX)),
            Span::styled("3", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
            Span::styled(
                "net // cluster token stream─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let cur_tps = self.tps_history.last().copied().unwrap_or(0.0);
        let peak_tps = self.tps_history.iter().copied().fold(0.0_f64, f64::max);
        let spark_w = area.width.saturating_sub(10).max(8) as usize;

        let mut l1 = vec![Span::styled(" WAVE ", Style::default().fg(BTOP_MAIN_FG))];
        l1.extend(btop_spark_spans(&self.tps_history, spark_w, GRAD_DL));

        let mut l2 = vec![Span::styled(" RATE ", Style::default().fg(BTOP_MAIN_FG))];
        l2.push(Span::styled(
            format!("{:5.1} tok/s ", cur_tps),
            Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
        ));
        l2.push(Span::styled(
            format!("(top {:5.1} tok/s)", peak_tps),
            Style::default().fg(BTOP_GRAPH_TEXT),
        ));

        let online_cnt = self
            .overview
            .nodes
            .iter()
            .filter(|n| n.status != "paused")
            .count();

        let l3 = Line::from(vec![
            Span::styled(" SENT ", Style::default().fg(BTOP_MAIN_FG)),
            Span::styled(
                format!("{} tok out", self.overview.server.total_completion_tokens),
                Style::default()
                    .fg(Color::Rgb(163, 216, 255))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" | RECV ", Style::default().fg(BTOP_GRAPH_TEXT)),
            Span::styled(
                format!("{} tok in", self.overview.server.total_prompt_tokens),
                Style::default().fg(Color::Rgb(255, 118, 238)),
            ),
        ]);

        let l4 = Line::from(vec![
            Span::styled(" LINK ", Style::default().fg(BTOP_MAIN_FG)),
            Span::styled(
                format!("{}/{} computers active", online_cnt, self.overview.nodes.len()),
                Style::default().fg(BTOP_TITLE),
            ),
            Span::styled(" | TTFT 18ms", Style::default().fg(BTOP_GRAPH_TEXT)),
        ]);

        let mode_lbl = if self.connected {
            "LIVE SOCKET (:8000)"
        } else {
            "STANDALONE SIMULATOR"
        };
        let l5 = Line::from(vec![
            Span::styled(" MODE ", Style::default().fg(BTOP_MAIN_FG)),
            Span::styled(
                mode_lbl,
                Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
            ),
        ]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_NET_BOX))
            .title(title);

        f.render_widget(
            Paragraph::new(vec![
                Line::from(l1),
                Line::from(l2),
                l3,
                l4,
                l5,
            ])
            .block(block),
            area,
        );
    }

    pub fn draw_nodes_box(&mut self, f: &mut Frame, area: Rect) {
        let mut title_spans = vec![
            Span::styled("─", Style::default().fg(BTOP_PROC_BOX)),
            Span::styled("4", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
            Span::styled(
                "nodes (output computers)─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ];
        title_spans.extend(btop_key_label("─[", "p", "]ause "));
        title_spans.extend(btop_key_label("[", "o", "]pencode.json─"));

        let total_out = self
            .overview
            .nodes
            .iter()
            .map(|n| n.completion_tokens)
            .sum::<u64>()
            .max(1);

        let header = Row::new(vec![
            Cell::from("ST"),
            Cell::from("NODE_ID"),
            Cell::from("HOST@IP"),
            Cell::from("TOK/S METER"),
            Cell::from("SHARE"),
            Cell::from("IN_TOK"),
            Cell::from("OUT_TOK"),
            Cell::from("MODEL"),
        ])
        .style(Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD));

        let rows: Vec<Row> = self
            .overview
            .nodes
            .iter()
            .map(|n| {
                let paused = n.status == "paused";
                let st_cell = if paused {
                    Cell::from(Span::styled(
                        "PAU",
                        Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Cell::from(Span::styled(
                        "ONL",
                        Style::default()
                            .fg(BTOP_PROC_MISC)
                            .add_modifier(Modifier::BOLD),
                    ))
                };

                let tps = if paused { 0.0 } else { n.last_tok_per_sec };
                let mut tps_spans = btop_meter_spans(tps / 100.0, 7, GRAD_CPU);
                tps_spans.push(Span::styled(
                    format!(" {:4.1}", tps),
                    Style::default().fg(BTOP_MAIN_FG),
                ));

                let share = n.completion_tokens as f64 / total_out as f64;
                let mut sh_spans = btop_meter_spans(share, 5, GRAD_DL);
                sh_spans.push(Span::styled(
                    format!(" {:2.0}%", share * 100.0),
                    Style::default().fg(BTOP_MAIN_FG),
                ));

                Row::new(vec![
                    st_cell,
                    Cell::from(Span::styled(
                        n.node_id.clone(),
                        Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
                    )),
                    Cell::from(Span::styled(
                        format!("{}@{}", n.hostname, n.ip_address),
                        Style::default().fg(BTOP_GRAPH_TEXT),
                    )),
                    Cell::from(Line::from(tps_spans)),
                    Cell::from(Line::from(sh_spans)),
                    Cell::from(format!("{}", n.prompt_tokens)),
                    Cell::from(Span::styled(
                        format!("{}", n.completion_tokens),
                        Style::default().fg(BTOP_PROC_MISC),
                    )),
                    Cell::from(Span::styled(
                        n.active_model.clone(),
                        Style::default().fg(Color::Rgb(163, 216, 255)),
                    )),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(4),
                Constraint::Length(15),
                Constraint::Length(22),
                Constraint::Length(14),
                Constraint::Length(10),
                Constraint::Length(7),
                Constraint::Length(8),
                Constraint::Min(10),
            ],
        )
        .header(header)
        .row_highlight_style(
            Style::default()
                .bg(BTOP_SELECTED_BG)
                .fg(BTOP_SELECTED_FG)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(BTOP_PROC_BOX))
                .title(Line::from(title_spans)),
        );

        f.render_stateful_widget(table, area, &mut self.node_table_state);
    }

    pub fn draw_bus_box(&self, f: &mut Frame, area: Rect) {
        let title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_NET_BOX)),
            Span::styled(
                "stream bus & inspector─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let max_lines = area.height.saturating_sub(2) as usize;
        let start = self.events.len().saturating_sub(max_lines);
        let lines: Vec<Line> = self.events[start..]
            .iter()
            .map(|e| Line::from(Span::styled(e.clone(), Style::default().fg(BTOP_MAIN_FG))))
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_NET_BOX))
            .title(title);

        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    pub fn draw_skills_box(&self, f: &mut Frame, area: Rect) {
        let mut title_spans = vec![
            Span::styled("─", Style::default().fg(BTOP_MEM_BOX)),
            Span::styled("5", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
            Span::styled(
                "episodic skills (tier 1)─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ];
        title_spans.extend(btop_key_label("─[", "s", "]kill toggle─"));

        let header = Row::new(vec![
            Cell::from("ID"),
            Cell::from("ST"),
            Cell::from("UTILITY"),
            Cell::from("DOM"),
            Cell::from("W/L"),
            Cell::from("ANTI-PATTERN -> VERIFIED FIX"),
        ])
        .style(Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD));

        let rows: Vec<Row> = self
            .overview
            .skills
            .iter()
            .map(|s| {
                let st = if s.active {
                    Span::styled(
                        "ACT",
                        Style::default()
                            .fg(BTOP_PROC_MISC)
                            .add_modifier(Modifier::BOLD),
                    )
                } else {
                    Span::styled(
                        "PRN",
                        Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                    )
                };
                let mut u_spans = btop_meter_spans(s.utility_score, 6, GRAD_CPU);
                u_spans.push(Span::styled(
                    format!(" {:.2}", s.utility_score),
                    Style::default().fg(BTOP_MAIN_FG),
                ));

                let fix_line = Line::from(vec![
                    Span::styled(
                        format!("- {} ", s.anti_pattern),
                        Style::default().fg(BTOP_HI_FG),
                    ),
                    Span::styled("-> ", Style::default().fg(BTOP_GRAPH_TEXT)),
                    Span::styled(
                        format!("+ {}", s.verified_pattern),
                        Style::default().fg(BTOP_PROC_MISC),
                    ),
                ]);

                Row::new(vec![
                    Cell::from(format!("#{}", s.id)),
                    Cell::from(st),
                    Cell::from(Line::from(u_spans)),
                    Cell::from(Span::styled(
                        s.domain.clone(),
                        Style::default().fg(BTOP_MAIN_FG),
                    )),
                    Cell::from(format!("{}/{}", s.success_count, s.failure_count)),
                    Cell::from(fix_line),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(3),
                Constraint::Length(4),
                Constraint::Length(12),
                Constraint::Length(8),
                Constraint::Length(5),
                Constraint::Min(20),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(BTOP_MEM_BOX))
                .title(Line::from(title_spans)),
        );

        f.render_widget(table, area);
    }

    pub fn draw_adapters_box(&self, f: &mut Frame, area: Rect) {
        let mut title_spans = vec![
            Span::styled("─", Style::default().fg(BTOP_CPU_BOX)),
            Span::styled("6", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
            Span::styled(
                "qlora promotion gate (tier 3)─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ];
        title_spans.extend(btop_key_label("─[", "t", "]rain─"));

        let header = Row::new(vec![
            Cell::from("GATE"),
            Cell::from("ADAPTER_VER"),
            Cell::from("REPLAY@1"),
            Cell::from("PPL"),
            Cell::from("SAN"),
        ])
        .style(Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD));

        let rows: Vec<Row> = self
            .overview
            .adapters
            .iter()
            .map(|a| {
                let gate = if a.promoted != 0 {
                    Span::styled(
                        "+ PROMOTED",
                        Style::default()
                            .fg(BTOP_PROC_MISC)
                            .add_modifier(Modifier::BOLD),
                    )
                } else {
                    Span::styled(
                        "x ROLLBACK",
                        Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                    )
                };
                let mut r_spans = btop_meter_spans(a.replay_pass_rate, 6, GRAD_MEM);
                r_spans.push(Span::styled(
                    format!(" {:3.0}%", a.replay_pass_rate * 100.0),
                    Style::default().fg(BTOP_MAIN_FG),
                ));

                Row::new(vec![
                    Cell::from(gate),
                    Cell::from(Span::styled(
                        a.version.clone(),
                        Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
                    )),
                    Cell::from(Line::from(r_spans)),
                    Cell::from(format!("{:.2}", a.holdout_perplexity)),
                    Cell::from(format!("{:.0}%", a.sanity_score * 100.0)),
                ])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(11),
                Constraint::Length(16),
                Constraint::Length(12),
                Constraint::Length(6),
                Constraint::Min(5),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(BTOP_CPU_BOX))
                .title(Line::from(title_spans)),
        );

        f.render_widget(table, area);
    }
}
