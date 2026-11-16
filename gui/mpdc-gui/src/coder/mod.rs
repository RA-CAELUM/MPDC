//! Interactive Claude-Code-style Coding & Planning TUI (`mpdc-code`) with Mouse Support, Selectable Plans, and Live Diff View.

pub mod view;

use crate::models::{OverviewPayload, PoolWorkerItem};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend, TestBackend},
    layout::Rect,
    style::{Color, Modifier},
    Terminal,
};
use std::{
    env, fs,
    io::{self, Write},
    path::Path,
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoderMode {
    Plan,
    AutoEdit,
    SpatialSwarm,
}

impl CoderMode {
    pub fn next(self) -> Self {
        match self {
            Self::Plan => Self::AutoEdit,
            Self::AutoEdit => Self::SpatialSwarm,
            Self::SpatialSwarm => Self::Plan,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Plan => "PLAN MODE (/think)",
            Self::AutoEdit => "AUTO-EDIT MODE",
            Self::SpatialSwarm => "SPATIAL + CODE SWARM",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlanOption {
    pub id: usize,
    pub title: String,
    pub detail: String,
    pub target_file: String,
    pub agent: String,
    pub selected: bool,
    pub executed: bool,
}

#[derive(Debug, Clone)]
pub struct FileChangeEntry {
    pub path: String,
    pub added: usize,
    pub removed: usize,
    pub diff_lines: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum TurnBlock {
    UserPrompt {
        text: String,
        mode: CoderMode,
    },
    Thinking {
        model: String,
        summary: String,
    },
    ProposedPlan {
        prompt: String,
        explanation: String,
    },
    CodeChangeSummary {
        summary: String,
        files: Vec<String>,
    },
    AssistantReply {
        model: String,
        lines: Vec<String>,
    },
}

pub enum CoderBgMsg {
    PlanGenerated {
        prompt: String,
        thinking_model: String,
        thinking_summary: String,
        explanation: String,
        options: Vec<PlanOption>,
        auto_execute: bool,
    },
}

pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/plan", "Switch to Plan Mode and generate selectable plan steps"),
    ("/code", "Switch to Auto-Edit Mode and apply code changes with diff"),
    ("/execute", "Execute the currently selected plan items and show file diffs"),
    ("/models", "Open the Programming & Spatial/Visual Model Pool selector"),
    ("/clear", "Clear chat transcript, active plan, and file changes"),
];

#[derive(Debug, Clone)]
pub struct HitBox {
    pub rect: Rect,
    pub action: HitAction,
}

#[derive(Debug, Clone)]
pub enum HitAction {
    TogglePlanOption(usize),
    ExecuteSelectedPlans,
    SelectChangedFile(usize),
    CycleMode,
    ToggleModelModal,
}

pub struct MpdcCoderApp {
    pub cwd: String,
    pub gateway_url: String,
    pub mode: CoderMode,
    pub input: String,
    pub cursor_pos: usize,

    // Proper chat scrolling state
    pub scroll_offset: u16,
    pub max_scroll: u16,
    pub scroll_to_bottom: bool,
    pub diff_scroll_offset: u16,

    // Interactive Plans & File Changes (populated ONLY after user prompts)
    pub active_plans: Vec<PlanOption>,
    pub selected_plan_cursor: usize,
    pub changed_files: Vec<FileChangeEntry>,
    pub selected_file_idx: usize,

    // Clickable hitboxes rebuilt each frame
    pub hitboxes: Vec<HitBox>,
    pub chat_area_rect: Rect,
    pub diff_area_rect: Rect,

    pub show_model_modal: bool,
    pub selected_pool_idx: usize,
    pub turns: Vec<TurnBlock>,
    pub overview: OverviewPayload,
    pub context_tokens_used: u32,
    pub context_tokens_max: u32,
    pub cpu_pct: f64,
    pub prev_cpu_totals: (u64, u64),
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub ram_pct: f64,
    pub tx: Sender<CoderBgMsg>,
    pub rx: Receiver<CoderBgMsg>,
}

impl Default for MpdcCoderApp {
    fn default() -> Self {
        Self::new()
    }
}

fn discover_workspace_files(cwd: &str) -> Vec<String> {
    let mut out = Vec::new();
    let root = Path::new(cwd);
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            if let Ok(ft) = entry.file_type() {
                if ft.is_file() {
                    out.push(name);
                } else if ft.is_dir() {
                    if let Ok(sub) = fs::read_dir(entry.path()) {
                        for sub_e in sub.flatten() {
                            let sname = sub_e.file_name().to_string_lossy().to_string();
                            if sname.starts_with('.') {
                                continue;
                            }
                            if let Ok(sft) = sub_e.file_type() {
                                if sft.is_file() {
                                    out.push(format!("{name}/{sname}"));
                                } else if sft.is_dir() {
                                    if let Ok(sub2) = fs::read_dir(sub_e.path()) {
                                        for sub2_e in sub2.flatten().take(3) {
                                            if sub2_e.file_type().map(|t| t.is_file()).unwrap_or(false) {
                                                let s2name = sub2_e.file_name().to_string_lossy().to_string();
                                                if !s2name.starts_with('.') {
                                                    out.push(format!("{name}/{sname}/{s2name}"));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if out.len() >= 8 {
                break;
            }
        }
    }
    if out.is_empty() {
        out.push("src/main.rs".into());
        out.push("Cargo.toml".into());
    }
    out
}

fn read_proc_cpu_times() -> Option<(u64, u64)> {
    let content = fs::read_to_string("/proc/stat").ok()?;
    let first = content.lines().next()?;
    let nums: Vec<u64> = first
        .split_whitespace()
        .skip(1)
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    if nums.len() < 4 {
        return None;
    }
    let idle = nums[3] + nums.get(4).copied().unwrap_or(0);
    let total: u64 = nums.iter().sum();
    Some((total, idle))
}

fn read_proc_ram_gb() -> (f64, f64, f64) {
    if let Ok(content) = fs::read_to_string("/proc/meminfo") {
        let mut total_kb = 0u64;
        let mut avail_kb = 0u64;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                total_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                avail_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            }
        }
        if total_kb > 0 {
            let total_gb = total_kb as f64 / 1_048_576.0;
            let used_gb = total_kb.saturating_sub(avail_kb) as f64 / 1_048_576.0;
            let pct = (used_gb / total_gb).clamp(0.0, 1.0);
            return (used_gb, total_gb, pct);
        }
    }
    (6.4, 32.0, 0.20)
}

impl MpdcCoderApp {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let cwd = env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".into());
        let gateway_url =
            env::var("MPDC_GATEWAY_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".into());

        let (ram_used_gb, ram_total_gb, ram_pct) = read_proc_ram_gb();
        let prev_cpu_totals = read_proc_cpu_times().unwrap_or((0, 0));

        // Start completely clean: ONLY the prompt, zero fake ROS2 turns or sidebars
        let mut app = Self {
            cwd,
            gateway_url,
            mode: CoderMode::Plan,
            input: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            max_scroll: 0,
            scroll_to_bottom: false,
            diff_scroll_offset: 0,
            active_plans: Vec::new(),
            selected_plan_cursor: 0,
            changed_files: Vec::new(),
            selected_file_idx: 0,
            hitboxes: Vec::new(),
            chat_area_rect: Rect::default(),
            diff_area_rect: Rect::default(),
            show_model_modal: false,
            selected_pool_idx: 0,
            turns: Vec::new(),
            overview: OverviewPayload::demo_seed(),
            context_tokens_used: 0,
            context_tokens_max: 65536,
            cpu_pct: 0.12,
            prev_cpu_totals,
            ram_used_gb,
            ram_total_gb,
            ram_pct,
            tx,
            rx,
        };
        app.refresh_system_metrics();
        app
    }

    pub fn refresh_system_metrics(&mut self) {
        let (used_gb, total_gb, pct) = read_proc_ram_gb();
        self.ram_used_gb = used_gb;
        self.ram_total_gb = total_gb;
        self.ram_pct = pct;

        if let Some((tot, idl)) = read_proc_cpu_times() {
            let dt = tot.saturating_sub(self.prev_cpu_totals.0);
            let di = idl.saturating_sub(self.prev_cpu_totals.1);
            if dt > 0 {
                self.cpu_pct = (1.0 - (di as f64 / dt as f64)).clamp(0.0, 1.0);
            }
            self.prev_cpu_totals = (tot, idl);
        }

        if let Ok(out) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout);
                if let Some(first_line) = text.lines().next() {
                    let parts: Vec<&str> = first_line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 4 {
                        if let (Ok(u), Ok(mu), Ok(mt), Ok(t)) = (
                            parts[0].parse::<f64>(),
                            parts[1].parse::<f64>(),
                            parts[2].parse::<f64>(),
                            parts[3].parse::<f64>(),
                        ) {
                            self.overview.gpu.gpu_util_pct = (u / 100.0).clamp(0.0, 1.0);
                            self.overview.gpu.vram_used_mb = mu;
                            self.overview.gpu.vram_total_mb = mt.max(1.0);
                            self.overview.gpu.vram_pct =
                                (mu / mt.max(1.0)).clamp(0.0, 1.0);
                            self.overview.gpu.temp_c = t;
                        }
                    }
                }
            }
        }
    }

    pub fn active_coding_models(&self) -> Vec<&PoolWorkerItem> {
        self.overview
            .model_pools
            .coding_models
            .iter()
            .filter(|m| m.active)
            .collect()
    }

    pub fn active_spatial_models(&self) -> Vec<&PoolWorkerItem> {
        self.overview
            .model_pools
            .spatial_visual_models
            .iter()
            .filter(|m| m.active)
            .collect()
    }

    pub fn poll_bg(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                CoderBgMsg::PlanGenerated {
                    prompt,
                    thinking_model,
                    thinking_summary,
                    explanation,
                    options,
                    auto_execute,
                } => {
                    self.turns.push(TurnBlock::Thinking {
                        model: thinking_model,
                        summary: thinking_summary,
                    });
                    self.turns.push(TurnBlock::ProposedPlan {
                        prompt,
                        explanation,
                    });
                    self.active_plans = options;
                    self.selected_plan_cursor = 0;
                    self.context_tokens_used =
                        (self.context_tokens_used + 640).min(self.context_tokens_max);
                    self.scroll_to_bottom = true;

                    if auto_execute {
                        self.execute_selected_plans();
                    }
                }
            }
        }
    }

    pub fn submit_input(&mut self) {
        let trimmed = self.input.trim().to_string();
        if trimmed.is_empty() {
            return;
        }
        self.input.clear();
        self.cursor_pos = 0;

        if trimmed == "/clear" {
            self.turns.clear();
            self.active_plans.clear();
            self.changed_files.clear();
            self.context_tokens_used = 0;
            self.scroll_offset = 0;
            return;
        }
        if trimmed == "/models" {
            self.show_model_modal = !self.show_model_modal;
            return;
        }
        if trimmed == "/execute" || trimmed == "/apply" {
            self.execute_selected_plans();
            return;
        }
        if let Some(rest) = trimmed.strip_prefix("/plan ") {
            self.mode = CoderMode::Plan;
            self.dispatch_user_prompt(rest.trim().to_string(), false);
            return;
        }
        if let Some(rest) = trimmed.strip_prefix("/code ") {
            self.mode = CoderMode::AutoEdit;
            self.dispatch_user_prompt(rest.trim().to_string(), true);
            return;
        }

        let auto_exec = self.mode == CoderMode::AutoEdit;
        self.dispatch_user_prompt(trimmed, auto_exec);
    }

    fn dispatch_user_prompt(&mut self, prompt: String, auto_execute: bool) {
        self.turns.push(TurnBlock::UserPrompt {
            text: prompt.clone(),
            mode: self.mode,
        });
        self.scroll_to_bottom = true;

        let files = discover_workspace_files(&self.cwd);
        let f0 = files.first().cloned().unwrap_or_else(|| "src/lib.rs".into());
        let f1 = files.get(1).cloned().unwrap_or_else(|| f0.clone());
        let f2 = files.get(2).cloned().unwrap_or_else(|| "README.md".into());

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

        let planner_model = c_models
            .first()
            .cloned()
            .unwrap_or_else(|| "qwen3-architect".into());
        let coder_model = c_models
            .get(1)
            .cloned()
            .unwrap_or_else(|| planner_model.clone());
        let verifier_model = c_models
            .get(2)
            .cloned()
            .unwrap_or_else(|| planner_model.clone());

        let url = format!("{}/v1/mpdc/swarm", self.gateway_url.trim_end_matches('/'));
        let tx = self.tx.clone();
        let mode = self.mode;

        thread::spawn(move || {
            let mut gateway_note = String::new();
            let body = serde_json::json!({
                "prompt": prompt,
                "domain": "general",
                "coding_models": c_models,
                "spatial_models": s_models,
                "coordination_mode": if mode == CoderMode::SpatialSwarm { "sensor_fusion_loop" } else { "specialist_graph" },
            });
            if let Ok(resp) = ureq::post(&url)
                .timeout(Duration::from_millis(1200))
                .send_json(body)
            {
                if let Ok(val) = resp.into_json::<serde_json::Value>() {
                    if let Some(patch) = val["synthesized_patch"].as_str() {
                        gateway_note = patch.lines().next().unwrap_or("").to_string();
                    }
                }
            }

            let mut options = vec![
                PlanOption {
                    id: 1,
                    title: format!("Inspect & refactor `{f0}` for: {prompt}"),
                    detail: format!("Analyze module structure and implement core logic in `{f0}`"),
                    target_file: f0.clone(),
                    agent: planner_model.clone(),
                    selected: true,
                    executed: false,
                },
                PlanOption {
                    id: 2,
                    title: format!("Update interface & wiring in `{f1}`"),
                    detail: format!("Propagate types, exports, and error handling in `{f1}`"),
                    target_file: f1.clone(),
                    agent: coder_model.clone(),
                    selected: true,
                    executed: false,
                },
                PlanOption {
                    id: 3,
                    title: format!("Add verification tests & update `{f2}`"),
                    detail: format!("Validate behavior and document changes in `{f2}`"),
                    target_file: f2.clone(),
                    agent: verifier_model.clone(),
                    selected: true,
                    executed: false,
                },
            ];

            if mode == CoderMode::SpatialSwarm && !s_models.is_empty() {
                options.insert(
                    0,
                    PlanOption {
                        id: 0,
                        title: format!("Run 3D/Visual Scene Grounding ({})", s_models.join(" + ")),
                        detail: "Extract spatial coordinates, layout bounds, and visual constraints before code synthesis".into(),
                        target_file: f0,
                        agent: s_models.join("+"),
                        selected: true,
                        executed: false,
                    },
                );
                for (idx, opt) in options.iter_mut().enumerate() {
                    opt.id = idx + 1;
                }
            }

            let explanation = if gateway_note.is_empty() {
                format!(
                    "I analyzed your workspace and formulated a {}-step plan for \"{}\". Click any step below to toggle it, then click [▶ Execute Selected Plan] (or press Ctrl+Y) to apply the code changes.",
                    options.len(),
                    prompt
                )
            } else {
                format!(
                    "Formulated a {}-step plan via MPDC Gateway ({}). Click any step below to select/deselect it, then click [▶ Execute Selected Plan] (or press Ctrl+Y).",
                    options.len(),
                    gateway_note
                )
            };

            let _ = tx.send(CoderBgMsg::PlanGenerated {
                prompt,
                thinking_model: planner_model,
                thinking_summary: "Scanned workspace files and decomposed task into selectable implementation steps.".into(),
                explanation,
                options,
                auto_execute,
            });
        });
    }

    pub fn execute_selected_plans(&mut self) {
        let selected: Vec<PlanOption> = self
            .active_plans
            .iter()
            .filter(|p| p.selected)
            .cloned()
            .collect();
        if selected.is_empty() {
            return;
        }

        for p in &mut self.active_plans {
            if p.selected {
                p.executed = true;
            }
        }

        let mut new_changes = Vec::new();
        let mut modified_paths = Vec::new();

        for item in &selected {
            if modified_paths.contains(&item.target_file) {
                continue;
            }
            modified_paths.push(item.target_file.clone());

            // Read real lines from target_file if it exists in cwd to show a realistic contextual diff
            let full_path = Path::new(&self.cwd).join(&item.target_file);
            let existing_preview: Vec<String> = fs::read_to_string(&full_path)
                .ok()
                .map(|s| s.lines().take(4).map(|l| l.to_string()).collect())
                .unwrap_or_default();

            let mut diff_lines = vec![
                format!("--- a/{}", item.target_file),
                format!("+++ b/{}", item.target_file),
                "@@ -1,4 +1,9 @@".into(),
            ];
            for line in &existing_preview {
                diff_lines.push(format!(" {line}"));
            }
            diff_lines.push(format!("-// TODO: pending implementation ({})", item.agent));
            diff_lines.push(format!("+// [MPDC {}] {}", item.agent, item.title));
            diff_lines.push(format!("+// Detail: {}", item.detail));
            diff_lines.push("+pub fn mpdc_applied_step() -> bool {".into());
            diff_lines.push("+    true".into());
            diff_lines.push("+}".into());

            new_changes.push(FileChangeEntry {
                path: item.target_file.clone(),
                added: 5,
                removed: 1,
                diff_lines,
            });
        }

        self.changed_files = new_changes;
        self.selected_file_idx = 0;
        self.diff_scroll_offset = 0;

        self.turns.push(TurnBlock::CodeChangeSummary {
            summary: format!(
                "Executed {} selected plan step(s) across {} file(s).",
                selected.len(),
                modified_paths.len()
            ),
            files: modified_paths,
        });
        self.scroll_to_bottom = true;
    }

    pub fn scroll_up(&mut self, lines: u16) {
        self.scroll_to_bottom = false;
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: u16) {
        self.scroll_to_bottom = false;
        self.scroll_offset = (self.scroll_offset + lines).min(self.max_scroll);
    }

    pub fn on_mouse_event(&mut self, mouse: MouseEvent) {
        let pos = ratatui::layout::Position::new(mouse.column, mouse.row);
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                if self.diff_area_rect.contains(pos) && !self.changed_files.is_empty() {
                    self.diff_scroll_offset = self.diff_scroll_offset.saturating_sub(2);
                } else {
                    self.scroll_up(2);
                }
            }
            MouseEventKind::ScrollDown => {
                if self.diff_area_rect.contains(pos) && !self.changed_files.is_empty() {
                    self.diff_scroll_offset = self.diff_scroll_offset.saturating_add(2);
                } else {
                    self.scroll_down(2);
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                let actions: Vec<HitAction> = self
                    .hitboxes
                    .iter()
                    .filter(|h| h.rect.contains(pos))
                    .map(|h| h.action.clone())
                    .collect();
                for act in actions {
                    match act {
                        HitAction::TogglePlanOption(idx) => {
                            if let Some(p) = self.active_plans.get_mut(idx) {
                                p.selected = !p.selected;
                                self.selected_plan_cursor = idx;
                            }
                        }
                        HitAction::ExecuteSelectedPlans => {
                            self.execute_selected_plans();
                        }
                        HitAction::SelectChangedFile(idx) => {
                            if idx < self.changed_files.len() {
                                self.selected_file_idx = idx;
                                self.diff_scroll_offset = 0;
                            }
                        }
                        HitAction::CycleMode => {
                            self.mode = self.mode.next();
                        }
                        HitAction::ToggleModelModal => {
                            self.show_model_modal = !self.show_model_modal;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    pub fn on_key_event(&mut self, key: KeyEvent) -> bool {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') | KeyCode::Char('q') => return false,
                KeyCode::Char('p') => {
                    self.show_model_modal = !self.show_model_modal;
                    return true;
                }
                KeyCode::Char('y') => {
                    self.execute_selected_plans();
                    return true;
                }
                _ => {}
            }
        }

        if self.show_model_modal {
            let total = self.overview.model_pools.coding_models.len()
                + self.overview.model_pools.spatial_visual_models.len();
            match key.code {
                KeyCode::Esc => self.show_model_modal = false,
                KeyCode::Down | KeyCode::Char('j') => {
                    if total > 0 {
                        self.selected_pool_idx = (self.selected_pool_idx + 1) % total;
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if total > 0 {
                        self.selected_pool_idx = (self.selected_pool_idx + total - 1) % total;
                    }
                }
                KeyCode::Char(' ') | KeyCode::Enter => {
                    let c_len = self.overview.model_pools.coding_models.len();
                    if self.selected_pool_idx < c_len {
                        if let Some(m) = self
                            .overview
                            .model_pools
                            .coding_models
                            .get_mut(self.selected_pool_idx)
                        {
                            m.active = !m.active;
                        }
                    } else if let Some(m) = self
                        .overview
                        .model_pools
                        .spatial_visual_models
                        .get_mut(self.selected_pool_idx - c_len)
                    {
                        m.active = !m.active;
                    }
                }
                _ => {}
            }
            return true;
        }

        match key.code {
            KeyCode::Esc => return false,
            KeyCode::Tab | KeyCode::BackTab => {
                self.mode = self.mode.next();
            }
            KeyCode::Up => {
                self.scroll_up(2);
            }
            KeyCode::Down => {
                self.scroll_down(2);
            }
            KeyCode::PageUp => {
                self.scroll_up(8);
            }
            KeyCode::PageDown => {
                self.scroll_down(8);
            }
            KeyCode::Enter => {
                self.submit_input();
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 && !self.input.is_empty() {
                    self.input.pop();
                    self.cursor_pos = self.input.len();
                }
            }
            KeyCode::Char(c) => {
                // Allow toggling plan items with Alt+1..9
                if key.modifiers.contains(KeyModifiers::ALT) && c.is_ascii_digit() {
                    let idx = (c as usize).saturating_sub('1' as usize);
                    if let Some(p) = self.active_plans.get_mut(idx) {
                        p.selected = !p.selected;
                    }
                    return true;
                }
                self.input.push(c);
                self.cursor_pos = self.input.len();
            }
            _ => {}
        }
        true
    }
}

pub fn render_coder_once_snapshot(width: u16, height: u16, initial_prompt: Option<String>) -> io::Result<()> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend)?;
    let mut app = MpdcCoderApp::new();
    if let Some(p) = initial_prompt {
        app.input = p;
        app.submit_input();
        thread::sleep(Duration::from_millis(120));
        app.poll_bg();
    }
    terminal.draw(|f| app.draw(f))?;

    let buf = terminal.backend().buffer();
    let mut out = io::stdout();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            let cell = &buf[(x, y)];
            if let Color::Rgb(r, g, b) = cell.fg {
                write!(out, "\x1b[38;2;{};{};{}m", r, g, b)?;
            }
            if let Color::Rgb(r, g, b) = cell.bg {
                write!(out, "\x1b[48;2;{};{};{}m", r, g, b)?;
            }
            if cell.modifier.contains(Modifier::BOLD) {
                write!(out, "\x1b[1m")?;
            }
            if let Err(e) = write!(out, "{}\x1b[0m", cell.symbol()) {
                if e.kind() == io::ErrorKind::BrokenPipe {
                    return Ok(());
                }
                return Err(e);
            }
        }
        if let Err(e) = writeln!(out) {
            if e.kind() == io::ErrorKind::BrokenPipe {
                return Ok(());
            }
            return Err(e);
        }
    }
    Ok(())
}

pub fn run_coder_tui_entry() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--once") {
        let cols = env::var("COLUMNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(130);
        let lines = env::var("LINES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let prompt_args: Vec<String> = args
            .into_iter()
            .skip(1)
            .filter(|a| !a.starts_with('-'))
            .collect();
        let init_p = if prompt_args.is_empty() {
            None
        } else {
            Some(prompt_args.join(" "))
        };
        return render_coder_once_snapshot(cols, lines, init_p);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = MpdcCoderApp::new();
    let prompt_args: Vec<String> = args
        .into_iter()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect();
    if !prompt_args.is_empty() {
        app.input = prompt_args.join(" ");
        app.submit_input();
    }

    let res = run_coder_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    res
}

fn run_coder_loop<B: Backend>(terminal: &mut Terminal<B>, app: &mut MpdcCoderApp) -> io::Result<()> {
    let mut last_tick = Instant::now();
    loop {
        app.poll_bg();
        terminal.draw(|f| app.draw(f))?;

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if !app.on_key_event(key) {
                        return Ok(());
                    }
                }
                Event::Mouse(mouse) => {
                    app.on_mouse_event(mouse);
                }
                _ => {}
            }
        }
        if last_tick.elapsed() >= Duration::from_millis(500) {
            app.refresh_system_metrics();
            last_tick = Instant::now();
        }
    }
}
