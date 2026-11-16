//! Rendering for `mpdc-code`: Full intact MPDC SVG banner, clean prompt-first UI, clickable Plans, and live Files Changed + Diff view.

use crate::{
    coder::{CoderMode, HitAction, HitBox, MpdcCoderApp, TurnBlock, SLASH_COMMANDS},
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

const MPDC_LOGO_LINES: [&str; 6] = [
    "███╗   ███╗██████╗ ██████╗  ██████╗",
    "████╗ ████║██╔══██╗██╔══██╗██╔════╝",
    "██╔████╔██║██████╔╝██║  ██║██║     ",
    "██║╚██╔╝██║██╔═══╝ ██║  ██║██║     ",
    "██║ ╚═╝ ██║██║     ██████╔╝╚██████╗",
    "╚═╝     ╚═╝╚═╝     ╚═════╝  ╚═════╝",
];
const MPDC_SUBTITLE: &str = " M a n   P o r t a b l e   D a t a   C e n t e r";

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
        self.hitboxes.clear();
        let size = f.area();
        f.render_widget(Block::default().style(Style::default().bg(BTOP_MAIN_BG)), size);

        let root = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(8), // Full 6-line MPDC logo + 1-line subtitle + bottom border
                Constraint::Min(8),    // Main workspace area
                Constraint::Length(3), // Prompt input bar
                Constraint::Length(1), // Footer keybind bar
            ])
            .split(size);

        self.draw_intact_header_banner(f, root[0]);

        // Only split the workspace when code changes exist; otherwise keep a clean single column!
        if self.changed_files.is_empty() {
            self.diff_area_rect = Rect::default();
            self.draw_main_column(f, root[1]);
        } else {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
                .split(root[1]);
            self.draw_main_column(f, cols[0]);
            self.draw_files_changed_and_diff_pane(f, cols[1]);
        }

        self.draw_prompt_box(f, root[2]);
        self.draw_footer_bar(f, root[3]);

        if self.input.starts_with('/') && !self.input.contains(' ') {
            self.draw_slash_popup(f, root[1]);
        }
        if self.show_model_modal {
            self.draw_model_pool_modal(f, size);
        }
    }

    fn draw_intact_header_banner(&mut self, f: &mut Frame, area: Rect) {
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

        let mut header_lines: Vec<Line> = Vec::with_capacity(7);

        for (idx, logo_row) in MPDC_LOGO_LINES.iter().enumerate() {
            let mut spans = vec![Span::raw("  ")];
            spans.extend(gradient_text_spans(logo_row));
            spans.push(Span::raw("   "));

            match idx {
                0 => {
                    spans.push(Span::styled(
                        format!(" {} ", self.mode.label()),
                        Style::default()
                            .bg(mode_color)
                            .fg(Color::Rgb(11, 13, 20))
                            .add_modifier(Modifier::BOLD),
                    ));
                    spans.push(Span::styled(
                        " (click or press Tab)",
                        Style::default().fg(BTOP_GRAPH_TEXT),
                    ));
                }
                1 => {
                    spans.push(Span::styled("WORKDIR     : ", Style::default().fg(BTOP_GRAPH_TEXT)));
                    spans.push(Span::styled(
                        self.cwd.clone(),
                        Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
                    ));
                }
                2 => {
                    spans.push(Span::styled("CODING POOL : ", Style::default().fg(BTOP_GRAPH_TEXT)));
                    spans.push(Span::styled(
                        c_models.join(" + "),
                        Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
                    ));
                }
                3 => {
                    spans.push(Span::styled("SPATIAL POOL: ", Style::default().fg(BTOP_GRAPH_TEXT)));
                    spans.push(Span::styled(
                        s_models.join(" + "),
                        Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                    ));
                    spans.push(Span::styled(
                        "  [Ctrl+P Models]",
                        Style::default().fg(Color::Rgb(167, 139, 250)),
                    ));
                }
                4 => {
                    let g = &self.overview.gpu;
                    spans.push(Span::styled("GPU         : ", Style::default().fg(BTOP_GRAPH_TEXT)));
                    spans.extend(btop_meter_spans(g.gpu_util_pct, 10, GRAD_MPDC));
                    spans.push(Span::styled(
                        format!(
                            " {:3.0}% ({:.0}°C)  VRAM ",
                            g.gpu_util_pct * 100.0,
                            g.temp_c
                        ),
                        Style::default().fg(BTOP_MAIN_FG),
                    ));
                    spans.extend(btop_meter_spans(g.vram_pct, 10, GRAD_MPDC));
                    spans.push(Span::styled(
                        format!(
                            " {:.1}/{:.0} GB",
                            g.vram_used_mb / 1024.0,
                            g.vram_total_mb / 1024.0
                        ),
                        Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
                    ));
                }
                5 => {
                    spans.push(Span::styled("CPU         : ", Style::default().fg(BTOP_GRAPH_TEXT)));
                    spans.extend(btop_meter_spans(self.cpu_pct, 10, GRAD_MPDC));
                    spans.push(Span::styled(
                        format!(" {:3.0}%         RAM  ", self.cpu_pct * 100.0),
                        Style::default().fg(BTOP_MAIN_FG),
                    ));
                    spans.extend(btop_meter_spans(self.ram_pct, 10, GRAD_MPDC));
                    spans.push(Span::styled(
                        format!(
                            " {:.1}/{:.1} GB ({:.0}%)",
                            self.ram_used_gb,
                            self.ram_total_gb,
                            self.ram_pct * 100.0
                        ),
                        Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
                    ));
                }
                _ => {}
            }
            header_lines.push(Line::from(spans));
        }

        // Row 7: Subtitle from assets/mpdc-banner.svg
        let mut sub_spans = vec![Span::raw("  ")];
        sub_spans.extend(gradient_text_spans(MPDC_SUBTITLE));
        header_lines.push(Line::from(sub_spans));

        // Register clickable hitboxes in header for Mode toggle (row 0) and Model Pool modal (row 3)
        if area.width > 42 && area.height >= 6 {
            self.hitboxes.push(HitBox {
                rect: Rect::new(area.x + 40, area.y, area.width.saturating_sub(40), 1),
                action: HitAction::CycleMode,
            });
            self.hitboxes.push(HitBox {
                rect: Rect::new(area.x + 40, area.y + 3, area.width.saturating_sub(40), 1),
                action: HitAction::ToggleModelModal,
            });
        }

        let block = Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(BTOP_MEM_BOX));

        f.render_widget(Paragraph::new(header_lines).block(block), area);
    }

    fn draw_main_column(&mut self, f: &mut Frame, area: Rect) {
        if self.active_plans.is_empty() {
            self.draw_chat_stream(f, area);
        } else {
            let plan_h = (self.active_plans.len() as u16 + 5).min(area.height.saturating_sub(5));
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(5), Constraint::Length(plan_h)])
                .split(area);
            self.draw_chat_stream(f, chunks[0]);
            self.draw_selectable_plan_box(f, chunks[1]);
        }
    }

    fn draw_chat_stream(&mut self, f: &mut Frame, area: Rect) {
        self.chat_area_rect = area;

        let mut lines: Vec<Line> = Vec::new();
        if self.turns.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    "Type a prompt below and press Enter to generate an execution plan and edit code.",
                    Style::default().fg(BTOP_GRAPH_TEXT),
                ),
            ]));
        } else {
            for turn in &self.turns {
                match turn {
                    TurnBlock::UserPrompt { text, mode } => {
                        lines.push(Line::from(vec![
                            Span::styled(
                                "❯ ",
                                Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                text.clone(),
                                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                format!("  [{}]", mode.label()),
                                Style::default().fg(Color::Rgb(167, 139, 250)),
                            ),
                        ]));
                        lines.push(Line::from(""));
                    }
                    TurnBlock::Thinking { model, summary } => {
                        lines.push(Line::from(vec![
                            Span::styled(
                                "  ✻ ",
                                Style::default()
                                    .fg(Color::Rgb(167, 139, 250))
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                format!("Thinking ({model})"),
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
                    TurnBlock::ProposedPlan {
                        prompt: _,
                        explanation,
                    } => {
                        lines.push(Line::from(vec![
                            Span::styled(
                                "  ◆ Plan Proposed: ",
                                Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(explanation.clone(), Style::default().fg(BTOP_MAIN_FG)),
                        ]));
                        lines.push(Line::from(""));
                    }
                    TurnBlock::CodeChangeSummary { summary, files } => {
                        lines.push(Line::from(vec![
                            Span::styled(
                                "  ● Code Changes Applied: ",
                                Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(summary.clone(), Style::default().fg(BTOP_TITLE)),
                        ]));
                        for file in files {
                            lines.push(Line::from(vec![
                                Span::styled("    │ ", Style::default().fg(BTOP_CPU_BOX)),
                                Span::styled(
                                    format!("M {file} (click in Files Changed pane to inspect diff)"),
                                    Style::default().fg(BTOP_PROC_MISC),
                                ),
                            ]));
                        }
                        lines.push(Line::from(""));
                    }
                    TurnBlock::AssistantReply { model, lines: rep } => {
                        lines.push(Line::from(vec![
                            Span::styled(
                                "  ◆ ",
                                Style::default().fg(BTOP_PROC_MISC).add_modifier(Modifier::BOLD),
                            ),
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
        }

        // Calculate wrapped line count so scrolling up and down works accurately
        let inner_w = area.width.saturating_sub(2).max(1) as usize;
        let inner_h = area.height.saturating_sub(2);
        let mut total_wrapped_lines: u16 = 0;
        for l in &lines {
            let char_len = l.width();
            let rows = ((char_len.max(1) + inner_w - 1) / inner_w) as u16;
            total_wrapped_lines = total_wrapped_lines.saturating_add(rows);
        }

        self.max_scroll = total_wrapped_lines.saturating_sub(inner_h);
        if self.scroll_to_bottom {
            self.scroll_offset = self.max_scroll;
            self.scroll_to_bottom = false;
        } else if self.scroll_offset > self.max_scroll {
            self.scroll_offset = self.max_scroll;
        }

        let scroll_badge = if self.max_scroll > 0 {
            format!("─chat (scroll {}/{} ↑/↓ or mouse wheel)─", self.scroll_offset, self.max_scroll)
        } else {
            "─chat─".into()
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_CPU_BOX))
            .title(Span::styled(
                scroll_badge,
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ));

        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((self.scroll_offset, 0))
                .block(block),
            area,
        );
    }

    fn draw_selectable_plan_box(&mut self, f: &mut Frame, area: Rect) {
        let title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_MEM_BOX)),
            Span::styled(
                "proposed plan (click steps to select/deselect, click Execute to apply code changes)─",
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let mut lines: Vec<Line> = Vec::new();
        let inner_y = area.y + 1;
        let inner_x = area.x + 1;
        let inner_w = area.width.saturating_sub(2);

        for (idx, plan) in self.active_plans.iter().enumerate() {
            let chk = if plan.executed {
                "[✓ DONE]"
            } else if plan.selected {
                "[x]"
            } else {
                "[ ]"
            };
            let chk_col = if plan.executed || plan.selected {
                BTOP_PROC_MISC
            } else {
                BTOP_GRAPH_TEXT
            };

            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {chk} "),
                    Style::default().fg(chk_col).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{}. {} ", plan.id, plan.title),
                    Style::default().fg(BTOP_MAIN_FG).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("({})", plan.agent),
                    Style::default().fg(Color::Rgb(167, 139, 250)),
                ),
            ]));

            let row_y = inner_y + idx as u16;
            if row_y < area.bottom().saturating_sub(1) {
                self.hitboxes.push(HitBox {
                    rect: Rect::new(inner_x, row_y, inner_w, 1),
                    action: HitAction::TogglePlanOption(idx),
                });
            }
        }

        lines.push(Line::from(""));
        let btn_row_y = inner_y + (self.active_plans.len() as u16) + 1;
        lines.push(Line::from(vec![
            Span::raw(" "),
            Span::styled(
                " [ ▶ Execute Selected Plan & Apply Code Changes (Click or press Ctrl+Y) ] ",
                Style::default()
                    .bg(Color::Rgb(56, 189, 248))
                    .fg(Color::Rgb(11, 13, 20))
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        if btn_row_y < area.bottom() {
            self.hitboxes.push(HitBox {
                rect: Rect::new(inner_x, btn_row_y, inner_w, 1),
                action: HitAction::ExecuteSelectedPlans,
            });
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_MEM_BOX))
            .title(title);

        f.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_files_changed_and_diff_pane(&mut self, f: &mut Frame, area: Rect) {
        self.diff_area_rect = area;
        let files_h = (self.changed_files.len() as u16 + 3).clamp(4, 9);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(files_h), Constraint::Min(5)])
            .split(area);

        // Top: Files Changed list (clickable!)
        let files_title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_PROC_MISC)),
            Span::styled(
                format!("files changed ({}) — click file to inspect diff─", self.changed_files.len()),
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let mut file_lines: Vec<Line> = Vec::new();
        let inner_y = chunks[0].y + 1;
        let inner_x = chunks[0].x + 1;
        let inner_w = chunks[0].width.saturating_sub(2);

        for (idx, fc) in self.changed_files.iter().enumerate() {
            let is_sel = idx == self.selected_file_idx;
            let bg = if is_sel {
                BTOP_SELECTED_BG
            } else {
                BTOP_MAIN_BG
            };
            let marker = if is_sel { "▸ M " } else { "  M " };
            file_lines.push(Line::from(vec![
                Span::styled(
                    marker,
                    Style::default()
                        .bg(bg)
                        .fg(BTOP_PROC_MISC)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{:<28} ", fc.path),
                    Style::default()
                        .bg(bg)
                        .fg(BTOP_TITLE)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("+{}", fc.added),
                    Style::default()
                        .bg(bg)
                        .fg(Color::Rgb(56, 189, 248))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ", Style::default().bg(bg)),
                Span::styled(
                    format!("-{}", fc.removed),
                    Style::default()
                        .bg(bg)
                        .fg(Color::Rgb(244, 114, 182))
                        .add_modifier(Modifier::BOLD),
                ),
            ]));

            let row_y = inner_y + idx as u16;
            if row_y < chunks[0].bottom().saturating_sub(1) {
                self.hitboxes.push(HitBox {
                    rect: Rect::new(inner_x, row_y, inner_w, 1),
                    action: HitAction::SelectChangedFile(idx),
                });
            }
        }

        let files_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_PROC_MISC))
            .title(files_title);
        f.render_widget(Paragraph::new(file_lines).block(files_block), chunks[0]);

        // Bottom: Unified Diff of what changed in the selected file
        let current_file = self.changed_files.get(self.selected_file_idx);
        let diff_title_str = current_file
            .map(|c| format!("what changed // {}", c.path))
            .unwrap_or_else(|| "what changed".into());

        let diff_title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_NET_BOX)),
            Span::styled(
                format!("{diff_title_str}─"),
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let mut diff_lines: Vec<Line> = Vec::new();
        if let Some(fc) = current_file {
            for dl in &fc.diff_lines {
                let style = if dl.starts_with("+++") || dl.starts_with("---") {
                    Style::default()
                        .fg(BTOP_TITLE)
                        .add_modifier(Modifier::BOLD)
                } else if dl.starts_with("@@") {
                    Style::default().fg(Color::Rgb(167, 139, 250))
                } else if dl.starts_with('+') {
                    Style::default().fg(Color::Rgb(56, 189, 248))
                } else if dl.starts_with('-') {
                    Style::default().fg(Color::Rgb(244, 114, 182))
                } else {
                    Style::default().fg(BTOP_GRAPH_TEXT)
                };
                diff_lines.push(Line::from(Span::styled(format!(" {dl}"), style)));
            }
        }

        let diff_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(BTOP_NET_BOX))
            .title(diff_title);

        f.render_widget(
            Paragraph::new(diff_lines)
                .wrap(Wrap { trim: false })
                .scroll((self.diff_scroll_offset, 0))
                .block(diff_block),
            chunks[1],
        );
    }

    fn draw_prompt_box(&self, f: &mut Frame, area: Rect) {
        let title = Line::from(vec![
            Span::styled("─", Style::default().fg(BTOP_HI_FG)),
            Span::styled(
                format!("prompt [{}]─", self.mode.label()),
                Style::default().fg(BTOP_TITLE).add_modifier(Modifier::BOLD),
            ),
        ]);

        let prompt_line = if self.input.is_empty() {
            Line::from(vec![
                Span::styled(
                    " ❯ ",
                    Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Enter a coding or spatial task (or type `/` for commands)...",
                    Style::default().fg(BTOP_GRAPH_TEXT),
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled(
                    " ❯ ",
                    Style::default().fg(BTOP_HI_FG).add_modifier(Modifier::BOLD),
                ),
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
        spans.extend(btop_key_label("[", "Mouse Click", "] Select Plan/File  "));
        spans.extend(btop_key_label("[", "Ctrl+Y", "] Execute Plan  "));
        spans.extend(btop_key_label("[", "Tab", "] Mode  "));
        spans.extend(btop_key_label("[", "Ctrl+P", "] Models  "));
        spans.extend(btop_key_label("[", "↑/↓/Wheel", "] Scroll  "));
        spans.extend(btop_key_label("[", "Esc", "] Quit"));

        f.render_widget(
            Paragraph::new(Line::from(spans)).style(Style::default().bg(BTOP_DIV_LINE)),
            area,
        );
    }

    fn draw_slash_popup(&self, f: &mut Frame, body_area: Rect) {
        let popup_h = (SLASH_COMMANDS.len() as u16 + 2).min(body_area.height);
        let popup_w = 84u16.min(body_area.width.saturating_sub(4));
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
                        Style::default()
                            .fg(BTOP_PROC_MISC)
                            .add_modifier(Modifier::BOLD),
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
            let bg = if sel {
                BTOP_SELECTED_BG
            } else {
                Color::Rgb(17, 20, 30)
            };
            lines.push(Line::from(vec![Span::styled(
                format!(" {chk} [CODE]    {:<22} ({})", m.id, m.name),
                Style::default()
                    .bg(bg)
                    .fg(BTOP_PROC_MISC)
                    .add_modifier(Modifier::BOLD),
            )]));
        }
        for (j, m) in self
            .overview
            .model_pools
            .spatial_visual_models
            .iter()
            .enumerate()
        {
            let idx = c_len + j;
            let sel = idx == self.selected_pool_idx;
            let chk = if m.active { "[x]" } else { "[ ]" };
            let bg = if sel {
                BTOP_SELECTED_BG
            } else {
                Color::Rgb(17, 20, 30)
            };
            lines.push(Line::from(vec![Span::styled(
                format!(" {chk} [SPATIAL] {:<22} ({})", m.id, m.name),
                Style::default()
                    .bg(bg)
                    .fg(BTOP_HI_FG)
                    .add_modifier(Modifier::BOLD),
            )]));
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
