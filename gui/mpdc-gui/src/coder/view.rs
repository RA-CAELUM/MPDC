//! Rendering for the Claude-Code-style Interactive Coding, Planning & Spatial Swarm TUI (`mpdc-code`).

use crate::{
    coder::{CoderMode, MpdcCoderApp, TurnBlock, SLASH_COMMANDS},
    tui::theme::{
        btop_key_label, btop_meter_spans, sample_grad, BTOP_CPU_BOX, BTOP_DIV_LINE,
        BTOP_GRAPH_TEXT, BTOP_HI_FG, BTOP_MAIN_BG, BTOP_MAIN_FG, BTOP_MEM_BOX, BTOP_NET_BOX,
        BTOP_PROC_MISC, BTOP_SELECTED_BG, BTOP_TITLE, GRAD_MPDC,
    },
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
    Frame,
};

fn gradient_text_spans(text: &str) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len().saturating_sub(1).max(1) as f32;
    chars
        .into_iter()
        .enumerate()
        .map(|(i, ch)| {
            let col = sample_grad(GRAD_MPDC, i as f32 / len);
            Span::styled(
                ch.to_string(),
                Style::default().fg(col).add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

impl MpdcCoderApp {
    pub fn draw(&mut self, f: &mut Frame) {
        let size = f.area();
        f.render_widget(Block::default().style(Style::default().bg(BTOP_MAIN_BG)), size);

        let root = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Min(12),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(size);

        self.draw_header_banner(f, root[0]);

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(66), Constraint::Percentage(34)])
            .split(root[1]);

        self.draw_conversation_stream(f, body[0]);
        self.draw_plan_and_pools_sidebar(f, body[1]);
        self.draw_prompt_box(f, root[2]);
        self.draw_footer_bar(f, root[3]);

        if self.input.starts_with('/') && !self.input.contains(' ') {
            self.draw_slash_popup(f, root[1]);
        }
        if self.show_model_modal {
            self.draw_model_pool_modal(f, size);
        }
    }

    fn draw_header_banner(&self, f: &mut Frame, area: Rect) {
        let c_models: Vec<String> = self
            .active_coding_models()
            .iter()
            .map(|m| m.id.clone())
            .collect();
        let s_models: Vec<String> = self
            .active_spatial_models()
            .iter()
            .map(|m| m.id.clone())
            .collect();

        let mode_color = match self.mode {
            CoderMode::Plan => Color::Rgb(167, 139, 250),
            CoderMode::AutoEdit => Color::Rgb(56, 189, 248),
            CoderMode::SpatialSwarm => Color::Rgb(244, 114, 182),
        };

        let mut line1 = vec![Span::raw(" ")];
        line1.extend(gradient_text_spans("███╗   ███╗██████╗ ██████╗  ██████╗"));
        line1.push(Span::raw("   "));
        line1.push(Span::styled(
            format!(" {} ", self.mode.label()),
            Style::default()
                .bg(mode_color)
                .fg(Color::Rgb(11, 13, 20))
                .add_modifier(Modifier::BOLD),
        ));
        line1.push(Span::styled(
            format!("  cwd: {} ", self.cwd),
            Style::default().fg(BTOP_GRAPH_TEXT),
        ));

        let mut line2 = vec![Span::raw(" ")];
        line2.extend(gradient_text_spans("████╗ ████║██╔══██╗██╔══██╗██╔════╝"));
        line2.push(Span::styled("   CODE POOL   : ", Style::default().fg(BTOP_GRAPH_TEXT)));
        line2.push(Span::styled(
            c_models.join(" + "),
            Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
        ));

        let mut line3 = vec![Span::raw(" ")];
        line3.extend(gradient_text_spans("██║╚██╔╝██║██████╔╝██║  ██║██║     "));
        line3.push(Span::styled("   SPATIAL POOL: ", Style::default().fg(BTOP_GRAPH_TEXT)));
        line3.push(Span::styled(
            s_models.join(" + "),
            Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
        ));

        let ctx_ratio = self.context_tokens_used as f64 / self.context_tokens_max.max(1) as f64;
        let mut line4 = vec![Span::raw(" ")];
        line4.extend(gradient_text_spans("╚═╝     ╚═╝╚═╝     ╚═════╝  ╚═════╝"));
        line4.push(Span::styled("   CONTEXT     : ", Style::default().fg(BTOP_GRAPH_TEXT)));
        line4.extend(btop_meter_spans(ctx_ratio, 12, GRAD_MPDC));
        line4.push(Span::styled(
            format!(
                " {}/{} tok ({:.1} tok/s) | ATTS: {}",
                self.context_tokens_used,
                self.context_tokens_max,
                self.last_tps,
                self.attachments.join(" ")
            ),
            Style::default().fg(BTOP_MAIN_FG),
        ));

        let block = Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(BTOP_MEM_BOX));

        f.render_widget(
            Paragraph::new(vec![
                Line::from(line1),
                Line::from(line2),
                Line::from(line3),
                Line::from(line4),
            ])
            .block(block),
            area,
        );
    }

    fn draw_conversation_stream(&self, f: &mut Frame, area: Rect) {
        let title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_CPU_BOX)),
            Span::styled(
                "session transcript // code, plan & 3d spatial grounding─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let mut lines: Vec<Line> = Vec::new();
        for turn in &self.turns {
            match turn {
                TurnBlock::UserPrompt {
                    text,
                    mode,
                    attachments,
                } => {
                    lines.push(Line::from(vec![
                        Span::styled("❯ ", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
                        Span::styled(
                            text.clone(),
                            Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("  [{}]", mode.label()),
                            Style::default().fg(Color::Rgb(167, 139, 250)),
                        ),
                    ]));
                    if !attachments.is_empty() {
                        lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(
                                format!("📎 {}", attachments.join("  ")),
                                Style::default().fg(BTOP_GRAPH_TEXT),
                            ),
                        ]));
                    }
                    lines.push(Line::from(""));
                }
                TurnBlock::SpatialGrounding {
                    models,
                    scene_graph,
                } => {
                    lines.push(Line::from(vec![
                        Span::styled("  ◈ ", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
                        Span::styled(
                            format!("Spatial & Visual Grounding [{}]", models.join(" + ")),
                            Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                        ),
                    ]));
                    for sg in scene_graph {
                        lines.push(Line::from(vec![
                            Span::styled("    │ ", Style::default().fg(BTOP_NET_BOX)),
                            Span::styled(sg.clone(), Style::default().fg(Color::Rgb(218, 200, 255))),
                        ]));
                    }
                    lines.push(Line::from(""));
                }
                TurnBlock::Thinking {
                    model,
                    duration_s,
                    summary,
                } => {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "  ✻ ",
                            Style::default()
                                .fg(Color::Rgb(167, 139, 250))
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("Thought for {:.1}s ({})", duration_s, model),
                            Style::default()
                                .fg(Color::Rgb(167, 139, 250))
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                    lines.push(Line::from(vec![
                        Span::styled("    │ ", Style::default().fg(BTOP_MEM_BOX)),
                        Span::styled(summary.clone(), Style::default().fg(BTOP_GRAPH_TEXT)),
                    ]));
                    lines.push(Line::from(""));
                }
                TurnBlock::ToolCall {
                    tool,
                    target,
                    status,
                    diff_lines,
                } => {
                    lines.push(Line::from(vec![
                        Span::styled("  ● ", Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD)),
                        Span::styled(
                            format!("{tool}({target})"),
                            Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!(" — {status}"),
                            Style::default().fg(BTOP_PROC_MISC),
                        ),
                    ]));
                    for dl in diff_lines {
                        let col = if dl.starts_with('+') {
                            Color::Rgb(56, 189, 248)
                        } else if dl.starts_with('-') {
                            Color::Rgb(244, 114, 182)
                        } else {
                            Color::Rgb(167, 139, 250)
                        };
                        lines.push(Line::from(vec![
                            Span::styled("    │ ", Style::default().fg(BTOP_CPU_BOX)),
                            Span::styled(dl.clone(), Style::default().fg(col)),
                        ]));
                    }
                    lines.push(Line::from(""));
                }
                TurnBlock::AssistantReply { model, lines: rep } => {
                    lines.push(Line::from(vec![
                        Span::styled("  ◆ ", Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD)),
                        Span::styled(
                            format!("Response [{model}]"),
                            Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
                        ),
                    ]));
                    for r in rep {
                        lines.push(Line::from(vec![
                            Span::raw("    "),
                            Span::styled(r.clone(), Style::default().fg(BTOP_MAIN_FG)),
                        ]));
                    }
                    lines.push(Line::from(""));
                }
            }
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_CPU_BOX))
            .title(title);

        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((self.scroll_offset, 0))
                .block(block),
            area,
        );
    }

    fn draw_plan_and_pools_sidebar(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
            .split(area);

        // Top: Step-by-Step Plan Checklist (like Claude Code's TodoWrite / Plan panel)
        let plan_title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_MEM_BOX)),
            Span::styled(
                "execution plan & verification─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let mut plan_lines: Vec<Line> = Vec::new();
        for step in &self.plan_steps {
            let (badge, col) = match step.status {
                "done" => ("[✓]", BTOP_PROC_MISC),
                "active" => ("[▶]", BTOP_HI_FG),
                _ => ("[ ]", BTOP_GRAPH_TEXT),
            };
            plan_lines.push(Line::from(vec![
                Span::styled(
                    format!(" {badge} "),
                    Style::default().fg(col).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{}. {}", step.id, step.title),
                    Style::default().fg(BTOP_MAIN_FG),
                ),
            ]));
            plan_lines.push(Line::from(vec![
                Span::raw("      ↳ agent: "),
                Span::styled(
                    step.assigned_agent.clone(),
                    Style::default().fg(Color::Rgb(167, 139, 250)),
                ),
            ]));
        }

        let plan_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_MEM_BOX))
            .title(plan_title);
        f.render_widget(Paragraph::new(plan_lines).block(plan_block), chunks[0]);

        // Bottom: Active Programming + Visual/Spatial Model Pools & Episodic Skills
        let pool_title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_NET_BOX)),
            Span::styled(
                "active model pools (Ctrl+P)─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let mut pool_lines: Vec<Line> = Vec::new();
        pool_lines.push(Line::from(Span::styled(
            " PROGRAMMING & REASONING POOL:",
            Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
        )));
        for m in &self.overview.model_pools.coding_models {
            let icon = if m.active { "●" } else { "○" };
            let col = if m.active { BTOP_PROC_MISC } else { BTOP_GRAPH_TEXT };
            pool_lines.push(Line::from(vec![
                Span::styled(format!("  {icon} "), Style::default().fg(col)),
                Span::styled(format!("{:<20}", m.id), Style::default().fg(BTOP_MAIN_FG)),
                Span::styled(format!("[{}]", m.role), Style::default().fg(BTOP_GRAPH_TEXT)),
            ]));
        }

        pool_lines.push(Line::from(Span::styled(
            " VISUAL & SPATIAL INTELLIGENCE POOL:",
            Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
        )));
        for m in &self.overview.model_pools.spatial_visual_models {
            let icon = if m.active { "◈" } else { "◇" };
            let col = if m.active { BTOP_HI_FG } else { BTOP_GRAPH_TEXT };
            pool_lines.push(Line::from(vec![
                Span::styled(format!("  {icon} "), Style::default().fg(col)),
                Span::styled(format!("{:<20}", m.id), Style::default().fg(BTOP_MAIN_FG)),
                Span::styled(format!("[{}]", m.category), Style::default().fg(Color::Rgb(167, 139, 250))),
            ]));
        }

        let pool_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_NET_BOX))
            .title(pool_title);
        f.render_widget(Paragraph::new(pool_lines).block(pool_block), chunks[1]);
    }

    fn draw_prompt_box(&self, f: &mut Frame, area: Rect) {
        let title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_HI_FG)),
            Span::styled(
                format!("prompt [{}] — type `/` for commands or press Tab to switch mode─", self.mode.label()),
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let prompt_line = if self.input.is_empty() {
            Line::from(vec![
                Span::styled(" ❯ ", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
                Span::styled(
                    "Ask MPDC to plan, write code, or analyze 3D/visual scenes (/plan, /code, /spatial, /models, /apply)...",
                    Style::default().fg(BTOP_GRAPH_TEXT),
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled(" ❯ ", Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD)),
                Span::styled(self.input.clone(), Style::default().fg(BTOP_TITLE)),
                Span::styled("█", Style::default().fg(BTOP_PROC_MISC)),
            ])
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_HI_FG))
            .title(title);

        f.render_widget(Paragraph::new(prompt_line).block(block), area);
    }

    fn draw_footer_bar(&self, f: &mut Frame, area: Rect) {
        let mut spans = vec![
            Span::styled(
                " MPDC CODE ",
                Style::default()
                    .bg(Color::Rgb(56, 189, 248))
                    .fg(Color::Rgb(11, 13, 20))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
        ];
        spans.extend(btop_key_label("[", "Tab", "] Cycle Mode (Plan/Code/Swarm)  "));
        spans.extend(btop_key_label("[", "Ctrl+Y", "] Apply Diff  "));
        spans.extend(btop_key_label("[", "Ctrl+P", "] Toggle Models  "));
        spans.extend(btop_key_label("[", "/", "] Slash Commands  "));
        spans.extend(btop_key_label("[", "↑/↓", "] Scroll  "));
        spans.extend(btop_key_label("[", "Esc", "] Quit"));

        f.render_widget(
            Paragraph::new(Line::from(spans)).style(Style::default().bg(BTOP_DIV_LINE)),
            area,
        );
    }

    fn draw_slash_popup(&self, f: &mut Frame, body_area: Rect) {
        let popup_h = (SLASH_COMMANDS.len() as u16 + 2).min(body_area.height);
        let popup_w = 88u16.min(body_area.width.saturating_sub(4));
        let rect = Rect::new(
            body_area.x + 2,
            body_area.bottom().saturating_sub(popup_h),
            popup_w,
            popup_h,
        );
        f.render_widget(Clear, rect);

        let lines: Vec<Line> = SLASH_COMMANDS
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(&self.input))
            .map(|(cmd, desc)| {
                Line::from(vec![
                    Span::styled(
                        format!(" {:<10}", cmd),
                        Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(*desc, Style::default().fg(BTOP_MAIN_FG)),
                ])
            })
            .collect();

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .style(Style::default().bg(Color::Rgb(17, 20, 30)))
            .border_style(Style::default().fg(Color::Rgb(167, 139, 250)))
            .title(Span::styled(
                "─slash commands─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ));
        f.render_widget(Paragraph::new(lines).block(block), rect);
    }

    fn draw_model_pool_modal(&self, f: &mut Frame, screen: Rect) {
        let w = 78u16.min(screen.width.saturating_sub(6));
        let h = 16u16.min(screen.height.saturating_sub(4));
        let rect = Rect::new(
            screen.x + (screen.width.saturating_sub(w)) / 2,
            screen.y + (screen.height.saturating_sub(h)) / 2,
            w,
            h,
        );
        f.render_widget(Clear, rect);

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(Span::styled(
            " Use ↑/↓ (or j/k) and Space/Enter to toggle models in the swarm (Esc to close):",
            Style::default().fg(BTOP_GRAPH_TEXT),
        )));
        lines.push(Line::from(""));

        let c_len = self.overview.model_pools.coding_models.len();
        for (i, m) in self.overview.model_pools.coding_models.iter().enumerate() {
            let sel = i == self.selected_pool_idx;
            let chk = if m.active { "[x]" } else { "[ ]" };
            let bg = if sel { BTOP_SELECTED_BG } else { Color::Rgb(17, 20, 30) };
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {chk} [CODE]    {:<22} ({})", m.id, m.name),
                    Style::default().bg(bg).fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
                ),
            ]));
        }
        for (j, m) in self.overview.model_pools.spatial_visual_models.iter().enumerate() {
            let idx = c_len + j;
            let sel = idx == self.selected_pool_idx;
            let chk = if m.active { "[x]" } else { "[ ]" };
            let bg = if sel { BTOP_SELECTED_BG } else { Color::Rgb(17, 20, 30) };
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {chk} [SPATIAL] {:<22} ({})", m.id, m.name),
                    Style::default().bg(bg).fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                ),
            ]));
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .style(Style::default().bg(Color::Rgb(17, 20, 30)))
            .border_style(Style::default().fg(BTOP_HI_FG))
            .title(Span::styled(
                "─multi-agent model pool selector (N Coding + M Spatial/Visual)─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ));
        f.render_widget(Paragraph::new(lines).block(block), rect);
    }
}
