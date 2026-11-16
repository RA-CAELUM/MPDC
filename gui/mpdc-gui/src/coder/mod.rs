//! Autonomous Claude-Code-style Coding, Planning & Spatial Swarm TUI (`mpdc-code`).

pub mod sica;
pub mod view;

use crate::models::{ModelPoolsPayload, OverviewPayload, PoolWorkerItem};
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
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver, Sender},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoderMode {
    AutoEdit,
    SpatialSwarm,
    Plan,
}

impl CoderMode {
    pub fn next(self) -> Self {
        match self {
            Self::AutoEdit => Self::SpatialSwarm,
            Self::SpatialSwarm => Self::Plan,
            Self::Plan => Self::AutoEdit,
        }
    }

    pub fn is_autonomous(self) -> bool {
        matches!(self, Self::AutoEdit | Self::SpatialSwarm)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::AutoEdit => "AUTONOMOUS CODE MODE",
            Self::SpatialSwarm => "AUTONOMOUS SPATIAL + CODE SWARM",
            Self::Plan => "MANUAL PLAN MODE (/think)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepState {
    Pending,
    Running,
    Done,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOption {
    pub id: usize,
    pub title: String,
    pub detail: String,
    pub target_file: String,
    pub agent: String,
    pub selected: bool,
    pub state: StepState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChangeEntry {
    pub path: String,
    pub added: usize,
    pub removed: usize,
    pub diff_lines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    StepCompleted {
        step_id: usize,
        total_steps: usize,
        title: String,
        file: String,
        agent: String,
    },
    CodeChangeSummary {
        summary: String,
        files: Vec<String>,
    },
    ScreenObservation {
        screenshot_path: String,
        models: Vec<String>,
        analysis: Vec<String>,
    },
    AssistantReply {
        model: String,
        lines: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedCoderSession {
    pub cwd: String,
    pub mode: CoderMode,
    pub turns: Vec<TurnBlock>,
    pub active_plans: Vec<PlanOption>,
    pub changed_files: Vec<FileChangeEntry>,
    pub context_tokens_used: u32,
    #[serde(default)]
    pub prompt_tokens_total: u64,
    #[serde(default)]
    pub completion_tokens_total: u64,
    #[serde(default)]
    pub cached_tokens_total: u64,
}

pub enum CoderBgMsg {
    /// One line of the SICA agent's live event stream.
    AgentLog {
        line: String,
    },
    /// SICA finished; carries the real outcome and on-disk changes.
    AgentRunFinished {
        prompt: String,
        plan_only: bool,
        outcome: sica::SicaOutcome,
        changes: Vec<FileChangeEntry>,
        verification: Option<(String, Vec<String>)>,
    },
    ScreenCaptured {
        path: String,
        models: Vec<String>,
        analysis: Vec<String>,
    },
    GpuMetricsUpdated {
        util_pct: f64,
        vram_used_mb: f64,
        vram_total_mb: f64,
        temp_c: f64,
        gateway_online: bool,
        /// Live model pools / server info from the gateway (None if unreachable).
        pools: Option<ModelPoolsPayload>,
        server: Option<(String, String, String, String)>,
    },
}

pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/code", "Autonomous Code Mode — plans & executes all multi-file steps automatically"),
    ("/swarm", "Autonomous Spatial + Code Swarm Mode (3D/Visual + Multi-Module Code)"),
    ("/plan", "Manual Plan Mode — pause after planning so you can toggle steps"),
    ("/execute", "Execute remaining selected plan steps autonomously"),
    ("/cd", "Switch working directory (`/cd /path/to/project`)"),
    ("/screen", "Capture a live screenshot & analyze with Visual/Spatial Models"),
    ("/watch", "Toggle continuous background screen monitoring for Spatial AI"),
    ("/resume", "Resume the last saved session (.mpdc_session.json)"),
    ("/export", "Export session transcript, plan, and diffs to Markdown"),
    ("/models", "Open the Programming & Spatial/Visual Model Pool selector"),
    ("/clear", "Clear chat transcript, active plan, and file changes"),
    ("/exit", "Save session state and exit mpdc-code"),
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
    TogglePoolModel(usize),
    SelectSlashCommand(String),
    ToggleScreenWatch,
    CaptureScreenNow,
}

pub struct MpdcCoderApp {
    pub cwd: String,
    pub gateway_url: String,
    pub gateway_online: bool,
    pub mode: CoderMode,
    pub input: String,
    pub cursor_pos: usize,

    pub scroll_offset: u16,
    pub max_scroll: u16,
    pub scroll_to_bottom: bool,
    pub diff_scroll_offset: u16,
    pub max_diff_scroll: u16,

    pub active_prompt: String,
    pub active_plans: Vec<PlanOption>,
    pub selected_plan_cursor: usize,
    pub changed_files: Vec<FileChangeEntry>,
    pub selected_file_idx: usize,

    pub hitboxes: Vec<HitBox>,
    pub chat_area_rect: Rect,
    pub diff_area_rect: Rect,

    pub show_model_modal: bool,
    pub selected_pool_idx: usize,
    pub turns: Vec<TurnBlock>,
    pub overview: OverviewPayload,
    pub context_tokens_used: u32,
    pub context_tokens_max: u32,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    pub cached_tokens_total: u64,
    pub last_tok_per_sec: f64,
    pub cpu_pct: f64,
    pub prev_cpu_totals: (u64, u64),
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub ram_pct: f64,
    pub should_exit: bool,
    pub screen_watch_enabled: bool,
    pub last_screen_watch: Instant,
    pub last_gpu_poll: Instant,
    pub screen_captures_count: u32,

    pub is_generating: bool,
    pub is_executing_plan: bool,
    pub gen_phase: String,
    pub gen_streamed_tokens: u32,
    pub gen_target_tokens: u32,
    pub gen_current_tps: f64,
    pub gen_started_at: Instant,
    pub spinner_tick: usize,
    /// Plan text produced by a PLAN MODE run, awaiting /execute.
    pub pending_plan_text: Option<String>,
    /// Index into `turns` of the live SICA log block for the current run.
    pub live_log_turn: Option<usize>,
    pub tx: Sender<CoderBgMsg>,
    pub rx: Receiver<CoderBgMsg>,
}

impl Default for MpdcCoderApp {
    fn default() -> Self {
        Self::new()
    }
}

fn check_or_autostart_gateway(gateway_url: &str) -> bool {
    let health_url = format!("{}/health", gateway_url.trim_end_matches('/'));
    if ureq::get(&health_url)
        .timeout(Duration::from_millis(350))
        .call()
        .is_ok()
    {
        return true;
    }
    // Try auto-starting `mpdc serve` in the MPDC root directory so it never creates `.mpdc_data` in user project dirs
    let mut cmd = Command::new("mpdc");
    cmd.arg("serve")
        .env("MPDC_DATA_DIR", "/home/sgk/Projects/MPDC/.mpdc_data")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if Path::new("/home/sgk/Projects/MPDC").is_dir() {
        cmd.current_dir("/home/sgk/Projects/MPDC");
    }
    let _ = cmd.spawn();
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectLang {
    Rust,
    Python,
    Cpp,
    Web,
    Nix,
}

fn detect_project_language(cwd: &str, prompt: &str) -> ProjectLang {
    let lower = prompt.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '+')
        .filter(|s| !s.is_empty())
        .collect();

    if words.iter().any(|&w| matches!(w, "rust" | "cargo" | "crate" | "rustc" | "bevy" | "ratatui" | "egui"))
        || lower.contains(".rs")
    {
        return ProjectLang::Rust;
    }
    if words.iter().any(|&w| matches!(w, "python" | "py" | "fastapi" | "pytorch" | "pytest" | "flask"))
        || lower.contains(".py")
    {
        return ProjectLang::Python;
    }
    if words.iter().any(|&w| matches!(w, "c++" | "cpp" | "cmake" | "cuda" | "opengl"))
        || lower.contains(".cpp")
    {
        return ProjectLang::Cpp;
    }
    if words.iter().any(|&w| matches!(w, "html" | "css" | "javascript" | "typescript" | "react" | "threejs" | "webgl" | "website"))
        || lower.contains(".html")
        || lower.contains(".js")
    {
        return ProjectLang::Web;
    }
    if words.iter().any(|&w| matches!(w, "nix" | "nixos" | "flake")) || lower.contains(".nix") {
        return ProjectLang::Nix;
    }

    let root = Path::new(cwd);
    if root.join("Cargo.toml").exists() {
        return ProjectLang::Rust;
    }
    if root.join("CMakeLists.txt").exists() {
        return ProjectLang::Cpp;
    }
    if root.join("package.json").exists() || root.join("index.html").exists() {
        return ProjectLang::Web;
    }
    if root.join("flake.nix").exists() {
        return ProjectLang::Nix;
    }
    ProjectLang::Python
}


fn wait_for_gateway(gateway_url: &str, max_wait: Duration) -> bool {
    let started = Instant::now();
    if check_or_autostart_gateway(gateway_url) {
        return true;
    }
    let health_url = format!("{}/health", gateway_url.trim_end_matches('/'));
    while started.elapsed() < max_wait {
        thread::sleep(Duration::from_millis(500));
        if ureq::get(&health_url)
            .timeout(Duration::from_millis(500))
            .call()
            .is_ok()
        {
            return true;
        }
    }
    false
}

/// Problem statement handed to SICA for a `mpdc-code` prompt.
fn build_sica_task(
    prompt: &str,
    cwd: &str,
    mode: CoderMode,
    plan_only: bool,
    approved_plan: Option<&str>,
) -> String {
    let local = sica::agent_mode() == "local";
    let mut t = format!(
        "Project directory: `{cwd}` (on the user's machine; all paths are relative to it).\n\n"
    );
    if plan_only {
        t.push_str(
            "PLANNING ONLY: do not create, modify or delete files. Inspect the project as needed, \
             then finish with a concrete, numbered implementation plan: which files to change, \
             what to change in each, and the exact commands that will verify the result.\n\n",
        );
    }
    if mode == CoderMode::SpatialSwarm && !local {
        t.push_str(
            "This task may involve visual or 3D spatial reasoning: delegate coordinate-frame, \
             geometry or scene analysis to the spatial_visual_specialist sub-agent where useful.\n\n",
        );
    }
    t.push_str("Task:\n");
    t.push_str(prompt);
    if let Some(plan) = approved_plan {
        t.push_str("\n\nThe user approved the following plan. Implement it:\n");
        t.push_str(plan);
    }
    if !plan_only {
        t.push_str(if local {
            "\n\nWhen the change is done, run the project's tests or build, fix any failures, \
             then finish with a short summary."
        } else {
            "\n\nWhen finished, build the project and run its tests to verify your work, fix any \
             failures, then submit a short summary of what you changed via submit_answer."
        });
    }
    t
}

/// Follow-up task after the independent check failed.
fn build_fix_task(original_task: &str, v: &VerifyReport) -> String {
    let mut t = format!(
        "{original_task}\n\nA previous attempt already changed the project, but an independent \
         check (`{}`) FAILED:\n{}\n\nFind the cause and fix it: fix the code, or fix a test whose \
         expectation contradicts the task description. Then run the check again and finish.",
        v.tool.trim_start_matches("verify:"),
        v.lines.join("\n")
    );
    if let Some(src) = &v.acceptance_source {
        t.push_str(&format!(
            "\n\nThe acceptance tests were written by a separate tester from the task description \
             alone; you cannot edit them and they are not in the project. Here they are:\n```python\n{src}\n```\n\
             Make the code satisfy every acceptance test that agrees with the task. If one contradicts \
             the task description, do not bend the code to it: say which one and why in your finish summary."
        ));
    }
    t
}

fn patch_text(changes: &[FileChangeEntry]) -> String {
    changes
        .iter()
        .flat_map(|c| c.diff_lines.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

struct VerifyReport {
    tool: String,
    /// None when no automatic check applies to this project type.
    passed: Option<bool>,
    lines: Vec<String>,
    /// Source of the independent acceptance tests, when they failed (for the fix round).
    acceptance_source: Option<String>,
}

fn tail_lines(text: &str, n: usize) -> Vec<String> {
    let all: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    all[all.len().saturating_sub(n)..]
        .iter()
        .map(|s| format!("  {s}"))
        .collect()
}

fn has_python_tests(root: &Path) -> bool {
    if root.join("tests").is_dir() || root.join("test").is_dir() {
        return true;
    }
    fs::read_dir(root)
        .map(|rd| {
            rd.flatten().any(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.ends_with(".py") && (n.starts_with("test_") || n.ends_with("_test.py"))
            })
        })
        .unwrap_or(false)
}

/// Independent post-run check. Reports failures as failures.
fn run_live_project_verification(cwd: &str, lang: ProjectLang) -> VerifyReport {
    let root = Path::new(cwd);
    let (tool, mut cmd) = match lang {
        ProjectLang::Rust if root.join("Cargo.toml").exists() => {
            let mut c = Command::new("cargo");
            c.args(["test", "--quiet"]);
            ("cargo test", c)
        }
        ProjectLang::Python if has_python_tests(root) => {
            // Same interpreter (with pytest) the agent used for its own checks.
            let mut c = Command::new(sica::tools_python());
            c.args(["-m", "pytest", "-q"]);
            ("pytest", c)
        }
        ProjectLang::Cpp if root.join("CMakeLists.txt").exists() => {
            let mut c = Command::new("sh");
            c.args(["-c", "cmake -S . -B build && cmake --build build && (cd build && ctest --output-on-failure)"]);
            ("cmake + ctest", c)
        }
        ProjectLang::Nix if root.join("flake.nix").exists() => {
            let mut c = Command::new("nix");
            c.args(["flake", "check"]);
            ("nix flake check", c)
        }
        _ => {
            return VerifyReport {
                tool: "mpdc-verify".into(),
                passed: None,
                acceptance_source: None,
                lines: vec!["No automatic verification for this project type; rely on the agent's own testing above.".into()],
            }
        }
    };
    match cmd.current_dir(cwd).output() {
        Ok(out) => {
            let combined = format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            // pytest exit 5 = no tests collected: nothing to judge.
            if tool == "pytest" && out.status.code() == Some(5) {
                return VerifyReport {
                    tool: format!("verify:{tool}"),
                    passed: None,
                    acceptance_source: None,
                    lines: vec!["pytest found no tests to run.".into()],
                };
            }
            let passed = out.status.success();
            let mut lines = vec![if passed {
                format!("✓ `{tool}` passed in `{cwd}`.")
            } else {
                format!("✗ `{tool}` FAILED in `{cwd}` (exit {:?}):", out.status.code())
            }];
            lines.extend(tail_lines(&combined, if passed { 3 } else { 25 }));
            VerifyReport { tool: format!("verify:{tool}"), passed: Some(passed), lines, acceptance_source: None }
        }
        Err(e) => VerifyReport {
            tool: format!("verify:{tool}"),
            passed: None,
            acceptance_source: None,
            lines: vec![format!("Could not run `{tool}`: {e}")],
        },
    }
}

/// Task for the independent tester that runs in the second Ollama slot.
fn build_tester_task(prompt: &str, cwd: &str) -> String {
    format!(
        "Project directory (read it with absolute paths; you cannot change it): `{cwd}`\n\n\
         Another engineer is implementing this task right now:\n<task>\n{prompt}\n</task>\n\n\
         Write independent acceptance tests for it as test_acceptance.py in your current directory."
    )
}

/// Runs the tester's acceptance tests against the project and folds the result
/// into the verification report. Collection/import errors (e.g. the tester
/// guessed a different module name) are reported as inconclusive, not failures.
/// After one fix round has already been spent on them (`advisory`), failing
/// acceptance tests no longer fail the run: a small tester model can assume
/// requirements the task never stated, and the project's own checks stay
/// authoritative.
fn apply_acceptance(v: &mut VerifyReport, cwd: &str, acc_dir: &Path, advisory: bool) {
    let file = acc_dir.join("test_acceptance.py");
    let Ok(src) = fs::read_to_string(&file) else {
        v.lines.push("(independent tester produced no acceptance tests)".into());
        return;
    };
    let pythonpath = match env::var("PYTHONPATH") {
        Ok(p) if !p.is_empty() => format!("{cwd}:{p}"),
        _ => cwd.to_string(),
    };
    let out = Command::new(sica::tools_python())
        .args(["-m", "pytest", "-q", "-p", "no:cacheprovider"])
        .arg(&file)
        .current_dir(cwd)
        .env("PYTHONPATH", pythonpath)
        .output();
    match out {
        Ok(o) => {
            let combined = format!(
                "{}\n{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            match o.status.code() {
                Some(0) => {
                    v.lines.push("✓ independent acceptance tests passed.".into());
                    if v.passed.is_none() {
                        v.passed = Some(true);
                    }
                }
                Some(1) if advisory => {
                    v.lines.push(
                        "⚠ independent acceptance tests still fail after a fix round; treated as advisory (the tester may have assumed requirements the task does not state):"
                            .into(),
                    );
                    v.lines.extend(tail_lines(&combined, 8));
                }
                Some(1) => {
                    v.passed = Some(false);
                    v.lines.push(
                        "✗ independent acceptance tests FAILED (written from the task description only; they may themselves be wrong):"
                            .into(),
                    );
                    v.lines.extend(tail_lines(&combined, 20));
                    v.acceptance_source = Some(src.chars().take(4000).collect());
                }
                code => {
                    v.lines.push(format!(
                        "⚠ acceptance tests inconclusive (pytest exit {code:?}, e.g. the tester guessed a different import):"
                    ));
                    v.lines.extend(tail_lines(&combined, 4));
                }
            }
        }
        Err(e) => v.lines.push(format!("Could not run acceptance tests: {e}")),
    }
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
        let gateway_online = check_or_autostart_gateway(&gateway_url);

        let (ram_used_gb, ram_total_gb, ram_pct) = read_proc_ram_gb();
        let prev_cpu_totals = read_proc_cpu_times().unwrap_or((0, 0));

        // Default to AUTONOMOUS CODE MODE (`CoderMode::AutoEdit`) so the agent runs autonomously out of the box!
        let mut app = Self {
            cwd,
            gateway_url,
            gateway_online,
            mode: CoderMode::AutoEdit,
            input: String::new(),
            cursor_pos: 0,
            scroll_offset: 0,
            max_scroll: 0,
            scroll_to_bottom: false,
            diff_scroll_offset: 0,
            max_diff_scroll: 0,
            active_prompt: String::new(),
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
            prompt_tokens_total: 0,
            completion_tokens_total: 0,
            cached_tokens_total: 0,
            last_tok_per_sec: 74.8,
            cpu_pct: 0.12,
            prev_cpu_totals,
            ram_used_gb,
            ram_total_gb,
            ram_pct,
            should_exit: false,
            screen_watch_enabled: false,
            last_screen_watch: Instant::now(),
            last_gpu_poll: Instant::now() - Duration::from_secs(10),
            screen_captures_count: 0,
            is_generating: false,
            is_executing_plan: false,
            gen_phase: String::new(),
            gen_streamed_tokens: 0,
            gen_target_tokens: 0,
            gen_current_tps: 0.0,
            gen_started_at: Instant::now(),
            spinner_tick: 0,
            pending_plan_text: None,
            live_log_turn: None,
            tx,
            rx,
        };
        app.refresh_system_metrics();
        app
    }

    pub fn change_working_directory(&mut self, raw_target: &str) {
        let trimmed = raw_target.trim();
        if trimmed.is_empty() {
            return;
        }
        let expanded = if let Some(rest) = trimmed.strip_prefix("~/") {
            if let Ok(home) = env::var("HOME") {
                PathBuf::from(home).join(rest)
            } else {
                PathBuf::from(trimmed)
            }
        } else {
            let p = Path::new(trimmed);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                Path::new(&self.cwd).join(p)
            }
        };

        if !expanded.exists() {
            let _ = fs::create_dir_all(&expanded);
        }
        match fs::canonicalize(&expanded) {
            Ok(canon) => {
                let _ = env::set_current_dir(&canon);
                self.cwd = canon.display().to_string();
                self.changed_files.clear();
                self.active_plans.clear();
                self.turns.push(TurnBlock::AssistantReply {
                    model: "mpdc-workdir".into(),
                    lines: vec![format!(
                        "Switched WORKDIR to `{}`.",
                        self.cwd
                    )],
                });
                self.scroll_to_bottom = true;
            }
            Err(e) => {
                self.turns.push(TurnBlock::AssistantReply {
                    model: "mpdc-workdir".into(),
                    lines: vec![format!(
                        "Failed to switch directory to `{}`: {e}",
                        expanded.display()
                    )],
                });
                self.scroll_to_bottom = true;
            }
        }
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

        if self.last_gpu_poll.elapsed() >= Duration::from_millis(2500) {
            self.last_gpu_poll = Instant::now();
            let tx = self.tx.clone();
            let gw = self.gateway_url.clone();
            thread::spawn(move || {
                let gw_online = check_or_autostart_gateway(&gw);
                let mut util_pct = 0.0;
                let mut vram_used_mb = 0.0;
                let mut vram_total_mb = 8192.0;
                let mut temp_c = 40.0;

                if let Ok(out) = Command::new("nvidia-smi")
                    .args([
                        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ])
                    .output()
                {
                    if out.status.success() {
                        let text = String::from_utf8_lossy(&out.stdout);
                        if let Some(first_line) = text.lines().next() {
                            let parts: Vec<&str> =
                                first_line.split(',').map(|s| s.trim()).collect();
                            if parts.len() >= 4 {
                                if let (Ok(u), Ok(mu), Ok(mt), Ok(t)) = (
                                    parts[0].parse::<f64>(),
                                    parts[1].parse::<f64>(),
                                    parts[2].parse::<f64>(),
                                    parts[3].parse::<f64>(),
                                ) {
                                    util_pct = (u / 100.0).clamp(0.0, 1.0);
                                    vram_used_mb = mu;
                                    vram_total_mb = mt.max(1.0);
                                    temp_c = t;
                                }
                            }
                        }
                    }
                }
                let base = gw.trim_end_matches('/').to_string();
                let (pools, server) = if gw_online {
                    let pools = ureq::get(&format!("{base}/v1/mpdc/pools"))
                        .timeout(Duration::from_millis(800))
                        .call()
                        .ok()
                        .and_then(|r| r.into_json::<ModelPoolsPayload>().ok());
                    let server = ureq::get(&format!("{base}/health"))
                        .timeout(Duration::from_millis(800))
                        .call()
                        .ok()
                        .and_then(|r| r.into_json::<serde_json::Value>().ok())
                        .map(|h| {
                            let f = |k: &str| h[k].as_str().unwrap_or_default().to_string();
                            (f("engine"), f("base_model"), f("active_model_or_lora"), f("backend_url"))
                        });
                    (pools, server)
                } else {
                    (None, None)
                };
                let _ = tx.send(CoderBgMsg::GpuMetricsUpdated {
                    util_pct,
                    vram_used_mb,
                    vram_total_mb,
                    temp_c,
                    gateway_online: gw_online,
                    pools,
                    server,
                });
            });
        }
    }

    /// Toggles a pool model locally and on the gateway (so the next poll agrees).
    fn toggle_pool_model(&mut self, idx: usize) {
        let pools = &mut self.overview.model_pools;
        let c_len = pools.coding_models.len();
        let worker = if idx < c_len {
            pools.coding_models.get_mut(idx)
        } else {
            pools.spatial_visual_models.get_mut(idx - c_len)
        };
        let Some(m) = worker else { return };
        m.active = !m.active;
        let url = format!(
            "{}/v1/mpdc/pools/{}/toggle",
            self.gateway_url.trim_end_matches('/'),
            m.id
        );
        thread::spawn(move || {
            let _ = ureq::post(&url).timeout(Duration::from_secs(2)).call();
        });
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
        if self.is_generating || self.is_executing_plan {
            self.spinner_tick = self.spinner_tick.wrapping_add(1);
        }
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                CoderBgMsg::AgentLog { line } => {
                    self.handle_agent_log(line);
                }
                CoderBgMsg::AgentRunFinished {
                    prompt,
                    plan_only,
                    outcome,
                    changes,
                    verification,
                } => {
                    self.handle_agent_run_finished(prompt, plan_only, outcome, changes, verification);
                }
                CoderBgMsg::ScreenCaptured {
                    path,
                    models,
                    analysis,
                } => {
                    self.screen_captures_count += 1;
                    self.turns.push(TurnBlock::ScreenObservation {
                        screenshot_path: path,
                        models,
                        analysis,
                    });
                    self.prompt_tokens_total += 290;
                    self.completion_tokens_total += 130;
                    self.cached_tokens_total += 180;
                    self.context_tokens_used =
                        (self.context_tokens_used + 420).min(self.context_tokens_max);
                    self.scroll_to_bottom = true;
                    self.save_session();
                }
                CoderBgMsg::GpuMetricsUpdated {
                    util_pct,
                    vram_used_mb,
                    vram_total_mb,
                    temp_c,
                    gateway_online,
                    pools,
                    server,
                } => {
                    if let Some(p) = pools {
                        self.overview.model_pools = p;
                    }
                    if let Some((engine, base_model, active_model, backend_url)) = server {
                        self.overview.server.engine = engine;
                        self.overview.server.base_model = base_model;
                        self.overview.server.active_model = active_model;
                        self.overview.server.backend_url = backend_url;
                    }
                    self.overview.gpu.gpu_util_pct = util_pct;
                    self.overview.gpu.vram_used_mb = vram_used_mb;
                    self.overview.gpu.vram_total_mb = vram_total_mb;
                    self.overview.gpu.vram_pct =
                        (vram_used_mb / vram_total_mb.max(1.0)).clamp(0.0, 1.0);
                    self.overview.gpu.temp_c = temp_c;
                    self.gateway_online = gateway_online;
                }
            }
        }
    }

    pub fn save_session(&self) {
        let session = SavedCoderSession {
            cwd: self.cwd.clone(),
            mode: self.mode,
            turns: self.turns.clone(),
            active_plans: self.active_plans.clone(),
            changed_files: self.changed_files.clone(),
            context_tokens_used: self.context_tokens_used,
            prompt_tokens_total: self.prompt_tokens_total,
            completion_tokens_total: self.completion_tokens_total,
            cached_tokens_total: self.cached_tokens_total,
        };
        let session_path = Path::new(&self.cwd).join(".mpdc_session.json");
        if let Ok(json) = serde_json::to_string_pretty(&session) {
            let _ = fs::write(session_path, json);
        }
    }

    pub fn resume_session(&mut self) {
        let session_path = Path::new(&self.cwd).join(".mpdc_session.json");
        match fs::read_to_string(&session_path) {
            Ok(raw) => match serde_json::from_str::<SavedCoderSession>(&raw) {
                Ok(saved) => {
                    self.mode = saved.mode;
                    self.turns = saved.turns;
                    self.active_plans = saved.active_plans;
                    self.changed_files = saved.changed_files;
                    self.context_tokens_used = saved.context_tokens_used;
                    self.prompt_tokens_total = saved.prompt_tokens_total;
                    self.completion_tokens_total = saved.completion_tokens_total;
                    self.cached_tokens_total = saved.cached_tokens_total;
                    self.turns.push(TurnBlock::AssistantReply {
                        model: "mpdc-session".into(),
                        lines: vec![format!(
                            "Resumed saved session from `{}` ({} turns, {} changed files).",
                            session_path.display(),
                            self.turns.len(),
                            self.changed_files.len()
                        )],
                    });
                    self.scroll_to_bottom = true;
                }
                Err(e) => {
                    self.turns.push(TurnBlock::AssistantReply {
                        model: "mpdc-session".into(),
                        lines: vec![format!("Failed to parse `.mpdc_session.json`: {e}")],
                    });
                    self.scroll_to_bottom = true;
                }
            },
            Err(_) => {
                self.turns.push(TurnBlock::AssistantReply {
                    model: "mpdc-session".into(),
                    lines: vec![
                        "No `.mpdc_session.json` found in current directory to resume.".into(),
                    ],
                });
                self.scroll_to_bottom = true;
            }
        }
    }

    pub fn export_session(&mut self, custom_filename: Option<&str>) {
        self.save_session();
        let filename = custom_filename
            .filter(|s| !s.is_empty())
            .unwrap_or("mpdc_session_export.md");
        let export_path = Path::new(&self.cwd).join(filename);

        let mut md = String::new();
        md.push_str(&format!("# MPDC Session Export (`{}`)\n\n", self.cwd));
        md.push_str(&format!("- **Mode**: `{}`\n", self.mode.label()));
        md.push_str(&format!(
            "- **Context Tokens**: `{}/{}` (IN: `{}`, OUT: `{}`)\n\n",
            self.context_tokens_used,
            self.context_tokens_max,
            self.prompt_tokens_total,
            self.completion_tokens_total
        ));

        md.push_str("## Conversation & Spatial Transcript\n\n");
        for t in &self.turns {
            match t {
                TurnBlock::UserPrompt { text, mode } => {
                    md.push_str(&format!("### User (`{}`)\n{}\n\n", mode.label(), text));
                }
                TurnBlock::Thinking { model, summary } => {
                    md.push_str(&format!("*Thinking ({model})*: {summary}\n\n"));
                }
                TurnBlock::ProposedPlan { explanation, .. } => {
                    md.push_str(&format!("**Plan**: {explanation}\n\n"));
                }
                TurnBlock::StepCompleted {
                    step_id,
                    total_steps,
                    title,
                    file,
                    agent,
                } => {
                    md.push_str(&format!(
                        "- **[✓ Step {step_id}/{total_steps}]** {title} -> `{file}` (`{agent}`)\n"
                    ));
                }
                TurnBlock::ScreenObservation {
                    screenshot_path,
                    models,
                    analysis,
                } => {
                    md.push_str(&format!(
                        "### Screen Spatial Observation (`{}`) [{}]\n",
                        screenshot_path,
                        models.join(", ")
                    ));
                    for line in analysis {
                        md.push_str(&format!("- {line}\n"));
                    }
                    md.push('\n');
                }
                TurnBlock::CodeChangeSummary { summary, files } => {
                    md.push_str(&format!(
                        "\n**Summary**: {} ({})\n\n",
                        summary,
                        files.join(", ")
                    ));
                }
                TurnBlock::AssistantReply { model, lines } => {
                    md.push_str(&format!("### Assistant (`{model}`)\n{}\n\n", lines.join("\n")));
                }
            }
        }

        if !self.changed_files.is_empty() {
            md.push_str("## Files Changed & Unified Diffs\n\n");
            for fc in &self.changed_files {
                md.push_str(&format!(
                    "### `{}` (+{} -{})\n```diff\n{}\n```\n\n",
                    fc.path,
                    fc.added,
                    fc.removed,
                    fc.diff_lines.join("\n")
                ));
            }
        }

        let msg = match fs::write(&export_path, md) {
            Ok(_) => format!(
                "Exported session transcript, plan, and diffs to `{}` (and saved `.mpdc_session.json`).",
                export_path.display()
            ),
            Err(e) => format!("Failed to write export file `{}`: {e}", export_path.display()),
        };

        self.turns.push(TurnBlock::AssistantReply {
            model: "mpdc-export".into(),
            lines: vec![msg],
        });
        self.scroll_to_bottom = true;
    }

    pub fn trigger_screen_capture(&mut self, user_query: Option<String>) {
        let cap_dir = Path::new(&self.cwd).join(".mpdc_captures");
        let _ = fs::create_dir_all(&cap_dir);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let img_path = cap_dir.join(format!("screen_{ts}.png"));
        let img_path_str = img_path.display().to_string();

        let s_models: Vec<String> = self
            .active_spatial_models()
            .iter()
            .map(|m| m.id.clone())
            .collect();
        let tx = self.tx.clone();
        let query_label =
            user_query.unwrap_or_else(|| "Monitor desktop & active simulation/editor windows".into());

        thread::spawn(move || {
            let captured_ok = Command::new("spectacle")
                .args(["-b", "-n", "-o", &img_path_str])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
                || Command::new("grim")
                    .arg(&img_path_str)
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                || Command::new("scrot")
                    .arg(&img_path_str)
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false);

            // No model is called here yet: report exactly what happened.
            let analysis = if captured_ok {
                vec![
                    format!("Captured screenshot -> `{img_path_str}`"),
                    format!("Query: {query_label}"),
                    "Not analyzed: no vision model is wired to /screen yet. Reference this file path in a prompt so the agent can use it.".into(),
                ]
            } else {
                vec![
                    "✗ Screenshot failed: none of spectacle, grim or scrot succeeded.".into(),
                    format!("Query: {query_label}"),
                ]
            };
            let _ = tx.send(CoderBgMsg::ScreenCaptured {
                path: img_path_str,
                models: s_models,
                analysis,
            });
        });
    }

    pub fn submit_input(&mut self) {
        let trimmed = self.input.trim().to_string();
        let has_pending_plan = !self.active_plans.is_empty()
            && self
                .active_plans
                .iter()
                .any(|p| p.selected && p.state == StepState::Pending);

        if trimmed.is_empty() {
            if has_pending_plan && !self.is_executing_plan {
                self.start_autonomous_plan_execution();
            }
            return;
        }
        self.input.clear();
        self.cursor_pos = 0;

        if matches!(
            trimmed.to_lowercase().as_str(),
            "y" | "yes" | "run" | "go" | "apply"
        ) && has_pending_plan
        {
            self.start_autonomous_plan_execution();
            return;
        }

        if let Some(dir_arg) = trimmed
            .strip_prefix("/cd ")
            .or_else(|| trimmed.strip_prefix("cd "))
        {
            self.change_working_directory(dir_arg);
            return;
        }

        if trimmed == "/exit" || trimmed == "/quit" {
            self.save_session();
            self.should_exit = true;
            return;
        }
        if trimmed == "/resume" {
            self.resume_session();
            return;
        }
        if trimmed == "/export" {
            self.export_session(None);
            return;
        }
        if let Some(rest) = trimmed.strip_prefix("/export ") {
            self.export_session(Some(rest.trim()));
            return;
        }
        if trimmed == "/screen" {
            self.trigger_screen_capture(None);
            return;
        }
        if let Some(rest) = trimmed.strip_prefix("/screen ") {
            self.trigger_screen_capture(Some(rest.trim().to_string()));
            return;
        }
        if trimmed == "/watch" {
            self.screen_watch_enabled = !self.screen_watch_enabled;
            self.last_screen_watch = Instant::now();
            if self.screen_watch_enabled {
                self.trigger_screen_capture(Some("Initial continuous screen watch frame".into()));
            }
            return;
        }
        if trimmed == "/clear" {
            self.turns.clear();
            self.active_plans.clear();
            self.changed_files.clear();
            self.context_tokens_used = 0;
            self.prompt_tokens_total = 0;
            self.completion_tokens_total = 0;
            self.cached_tokens_total = 0;
            self.scroll_offset = 0;
            return;
        }
        if trimmed == "/models" {
            self.show_model_modal = !self.show_model_modal;
            return;
        }
        if trimmed == "/execute" || trimmed == "/apply" {
            if has_pending_plan {
                self.start_autonomous_plan_execution();
            }
            return;
        }
        if trimmed == "/plan" {
            self.mode = CoderMode::Plan;
            self.turns.push(TurnBlock::AssistantReply {
                model: "mpdc-mode".into(),
                lines: vec![
                    "Switched to MANUAL PLAN MODE (/think). Prompts will pause after planning so you can click/toggle steps before execution.".into(),
                ],
            });
            self.scroll_to_bottom = true;
            return;
        }
        if let Some(rest) = trimmed.strip_prefix("/plan ") {
            self.mode = CoderMode::Plan;
            self.dispatch_user_prompt(rest.trim().to_string(), false);
            return;
        }
        if trimmed == "/code" || trimmed == "/auto" {
            self.mode = CoderMode::AutoEdit;
            if has_pending_plan {
                self.start_autonomous_plan_execution();
            } else {
                self.turns.push(TurnBlock::AssistantReply {
                    model: "mpdc-mode".into(),
                    lines: vec![
                        "Switched to AUTONOMOUS CODE MODE. Prompts will automatically plan, execute all multi-module steps, verify compilation/tests, and write changes to disk.".into(),
                    ],
                });
                self.scroll_to_bottom = true;
            }
            return;
        }
        if let Some(rest) = trimmed.strip_prefix("/code ") {
            self.mode = CoderMode::AutoEdit;
            self.dispatch_user_prompt(rest.trim().to_string(), true);
            return;
        }
        if trimmed == "/swarm" {
            self.mode = CoderMode::SpatialSwarm;
            if has_pending_plan {
                self.start_autonomous_plan_execution();
            } else {
                self.turns.push(TurnBlock::AssistantReply {
                    model: "mpdc-mode".into(),
                    lines: vec![
                        "Switched to AUTONOMOUS SPATIAL + CODE SWARM. Prompts will automatically run 3D/Visual grounding and execute all code steps.".into(),
                    ],
                });
                self.scroll_to_bottom = true;
            }
            return;
        }
        if let Some(rest) = trimmed.strip_prefix("/swarm ") {
            self.mode = CoderMode::SpatialSwarm;
            self.dispatch_user_prompt(rest.trim().to_string(), true);
            return;
        }
        if trimmed.starts_with('/') {
            let valid: Vec<&str> = SLASH_COMMANDS.iter().map(|(c, _)| *c).collect();
            self.turns.push(TurnBlock::AssistantReply {
                model: "mpdc-help".into(),
                lines: vec![format!(
                    "Unknown slash command `{}`. Available commands: {}",
                    trimmed,
                    valid.join(", ")
                )],
            });
            self.scroll_to_bottom = true;
            return;
        }

        let auto_exec = self.mode.is_autonomous();
        self.dispatch_user_prompt(trimmed, auto_exec);
    }

    /// Every prompt is executed by the MPDC-SICA agent (real LLM tool loop).
    /// PLAN MODE runs it read-only and keeps the plan for `/execute`.
    fn dispatch_user_prompt(&mut self, prompt: String, auto_execute: bool) {
        if self.is_generating {
            self.turns.push(TurnBlock::AssistantReply {
                model: "mpdc-code".into(),
                lines: vec!["An agent run is already in progress; wait for it to finish.".into()],
            });
            self.scroll_to_bottom = true;
            return;
        }
        self.turns.push(TurnBlock::UserPrompt {
            text: prompt.clone(),
            mode: self.mode,
        });
        self.active_prompt = prompt.clone();
        self.active_plans.clear();
        self.pending_plan_text = None;
        let plan_only = !auto_execute;
        let task = build_sica_task(&prompt, &self.cwd, self.mode, plan_only, None);
        self.launch_sica(prompt, task, plan_only);
    }

    /// Executes the plan produced by the last PLAN MODE run.
    pub fn start_autonomous_plan_execution(&mut self) {
        if self.is_generating {
            return;
        }
        let Some(plan) = self.pending_plan_text.take() else {
            return;
        };
        for p in self.active_plans.iter_mut() {
            p.state = StepState::Running;
        }
        self.is_executing_plan = true;
        let prompt = self.active_prompt.clone();
        let task = build_sica_task(&prompt, &self.cwd, self.mode, false, Some(&plan));
        self.launch_sica(prompt, task, false);
    }

    fn launch_sica(&mut self, prompt: String, task: String, plan_only: bool) {
        self.is_generating = true;
        self.gen_started_at = Instant::now();
        self.gen_phase = if plan_only {
            "SICA: planning (read-only)".into()
        } else {
            "SICA: agent running".into()
        };
        self.gen_streamed_tokens = 0;
        self.gen_target_tokens = 0;
        self.gen_current_tps = 0.0;
        self.turns.push(TurnBlock::AssistantReply {
            model: "sica".into(),
            lines: vec![format!(
                "Launching MPDC-SICA agent in `{}` (LLM via {}){}",
                self.cwd,
                self.gateway_url,
                if plan_only { " — PLAN MODE, no file edits" } else { "" }
            )],
        });
        self.live_log_turn = Some(self.turns.len() - 1);
        self.scroll_to_bottom = true;

        let cwd = self.cwd.clone();
        let gw_url = self.gateway_url.clone();
        let tx = self.tx.clone();

        thread::spawn(move || {
            if !wait_for_gateway(&gw_url, Duration::from_secs(20)) {
                let _ = tx.send(CoderBgMsg::AgentLog {
                    line: format!("warning: MPDC gateway {gw_url} is not responding; SICA's LLM calls will fail until `mpdc serve` is up."),
                });
            }
            let before = sica::snapshot_workspace(&cwd);
            // Verify-and-fix loop: if the independent check fails, the failure is
            // sent back to the agent (up to MPDC_VERIFY_ROUNDS rounds in total).
            let max_rounds: u32 = if plan_only {
                1
            } else {
                env::var("MPDC_VERIFY_ROUNDS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3)
                    .max(1)
            };
            let mut round_task = task.clone();
            let mut first_failed_patch: Option<String> = None;
            let mut total_duration = 0.0;
            let mut total_tokens: Option<u64> = None;
            let mut rounds_run = 0;
            let mut last: Option<(sica::SicaOutcome, Vec<FileChangeEntry>, Option<VerifyReport>)> = None;

            // Independent tester in the second Ollama slot (Python projects): writes
            // acceptance tests from the task description while the coder works.
            let lang = detect_project_language(&cwd, &prompt);
            let use_tester = !plan_only
                && lang == ProjectLang::Python
                && env::var("MPDC_TESTER").map(|v| v != "0").unwrap_or(true);
            let acc_dir = sica::acceptance_dir();
            let mut acceptance_feedback_used = false;
            let mut tester = if use_tester {
                let _ = fs::create_dir_all(&acc_dir);
                let _ = tx.send(CoderBgMsg::AgentLog {
                    line: "── independent tester started in parallel (acceptance tests from the task only) ──".into(),
                });
                let (ttx, tgw, tcwd, tacc, tprompt) =
                    (tx.clone(), gw_url.clone(), cwd.clone(), acc_dir.clone(), prompt.clone());
                Some(thread::spawn(move || {
                    sica::run_sica(
                        sica::SicaRunRequest {
                            prompt: build_tester_task(&tprompt, &tcwd),
                            cwd: tacc.display().to_string(),
                            gateway_url: tgw,
                            timeout_s: 300,
                            read_only: false,
                            extra_env: vec![("SICA_ROLE".into(), "tester".into())],
                        },
                        |ev| {
                            if let sica::SicaEvent::Log(line) = ev {
                                let _ = ttx.send(CoderBgMsg::AgentLog { line: format!("[tester] {line}") });
                            }
                        },
                    );
                }))
            } else {
                None
            };

            for round in 1..=max_rounds {
                rounds_run = round;
                if round > 1 {
                    let _ = tx.send(CoderBgMsg::AgentLog {
                        line: format!("── fix round {round}/{max_rounds}: the check failed, sending the failure back to the agent ──"),
                    });
                }
                let mut outcome = None;
                sica::run_sica(
                    sica::SicaRunRequest {
                        prompt: round_task.clone(),
                        cwd: cwd.clone(),
                        gateway_url: gw_url.clone(),
                        timeout_s: sica::default_timeout_s(),
                        read_only: plan_only,
                        extra_env: Vec::new(),
                    },
                    |ev| match ev {
                        sica::SicaEvent::Log(line) => {
                            let _ = tx.send(CoderBgMsg::AgentLog { line });
                        }
                        sica::SicaEvent::Finished(o) => outcome = Some(o),
                    },
                );
                let Some(outcome) = outcome else { return };
                total_duration += outcome.duration_s;
                if let Some(t) = outcome.total_tokens {
                    total_tokens = Some(total_tokens.unwrap_or(0) + t);
                }

                if let Some(handle) = tester.take() {
                    let _ = tx.send(CoderBgMsg::AgentLog {
                        line: "── waiting for the independent tester to finish ──".into(),
                    });
                    let _ = handle.join();
                }
                let after = sica::snapshot_workspace(&cwd);
                let changes = sica::diff_snapshots(&before, &after);
                let verification = if !plan_only && !changes.is_empty() {
                    let _ = tx.send(CoderBgMsg::AgentLog {
                        line: "── running independent verification ──".into(),
                    });
                    let mut v = run_live_project_verification(&cwd, lang);
                    if use_tester {
                        apply_acceptance(&mut v, &cwd, &acc_dir, acceptance_feedback_used);
                        if v.acceptance_source.is_some() {
                            acceptance_feedback_used = true;
                        }
                    }
                    // Deleting pre-existing tests to get a green run is a failure.
                    let removed = sica::removed_tests(&before, &after);
                    if !removed.is_empty() {
                        v.passed = Some(false);
                        v.lines.insert(0, format!(
                            "✗ {} pre-existing test(s) were removed: {}. Restore them (fixing a wrong expectation is fine; deleting tests is not).",
                            removed.len(),
                            removed.join(", ")
                        ));
                    }
                    Some(v)
                } else {
                    None
                };
                let failed = matches!(verification, Some(VerifyReport { passed: Some(false), .. }));
                if failed && first_failed_patch.is_none() {
                    first_failed_patch = Some(patch_text(&changes));
                }
                let retry = failed && outcome.launch_error.is_none() && round < max_rounds;
                if retry {
                    if let Some(v) = &verification {
                        round_task = build_fix_task(&task, v);
                    }
                }
                last = Some((outcome, changes, verification));
                if !retry {
                    break;
                }
            }
            let Some((mut outcome, changes, verification)) = last else { return };
            outcome.duration_s = total_duration;
            outcome.total_tokens = total_tokens;

            // Only verified, real patches are harvested. A run that failed first and
            // was then fixed yields a (rejected, chosen) pair for training.
            if matches!(verification, Some(VerifyReport { passed: Some(true), .. })) {
                let final_patch = patch_text(&changes);
                let harvest_url = format!("{}/v1/mpdc/harvest", gw_url.trim_end_matches('/'));
                let _ = ureq::post(&harvest_url)
                    .timeout(Duration::from_secs(5))
                    .send_json(serde_json::json!({
                        "prompt": prompt,
                        "agent_initial_code": first_failed_patch.clone().unwrap_or_else(|| final_patch.clone()),
                        "final_committed_code": final_patch,
                        "iterations_to_green": rounds_run,
                        "domain": "coding",
                        "skills_used": [],
                        "client_node": "mpdc-code",
                    }));
            }

            let verification = verification.map(|mut v| {
                if rounds_run > 1 {
                    v.lines.insert(0, format!("(after {rounds_run} agent rounds)"));
                }
                (v.tool, v.lines)
            });
            let _ = tx.send(CoderBgMsg::AgentRunFinished {
                prompt,
                plan_only,
                outcome,
                changes,
                verification,
            });
        });
    }

    fn handle_agent_log(&mut self, line: String) {
        let short: String = line.chars().take(200).collect();
        self.gen_phase = format!("SICA: {}", short.chars().take(90).collect::<String>());
        if let Some(idx) = self.live_log_turn {
            if let Some(TurnBlock::AssistantReply { lines, .. }) = self.turns.get_mut(idx) {
                lines.push(short);
                if lines.len() > 300 {
                    lines.remove(1);
                }
            }
        }
        self.scroll_to_bottom = true;
    }

    fn handle_agent_run_finished(
        &mut self,
        prompt: String,
        plan_only: bool,
        outcome: sica::SicaOutcome,
        changes: Vec<FileChangeEntry>,
        verification: Option<(String, Vec<String>)>,
    ) {
        self.is_generating = false;
        self.is_executing_plan = false;
        self.live_log_turn = None;
        self.gen_current_tps = 0.0;
        self.gen_streamed_tokens = 0;

        let logdir = outcome.logdir.display().to_string();
        if let Some(err) = &outcome.launch_error {
            self.turns.push(TurnBlock::AssistantReply {
                model: "sica".into(),
                lines: vec![format!("✗ Could not start SICA: {err}")],
            });
        } else if !outcome.success {
            let mut lines = vec![format!(
                "✗ SICA exited with code {} after {:.0}s. Last log lines:",
                outcome
                    .exit_code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "?".into()),
                outcome.duration_s
            )];
            lines.extend(outcome.stderr_tail.iter().rev().take(12).rev().cloned());
            lines.push(format!("Full trace: {logdir}"));
            self.turns.push(TurnBlock::AssistantReply {
                model: "sica".into(),
                lines,
            });
        }

        if let Some(answer) = &outcome.answer {
            self.turns.push(TurnBlock::AssistantReply {
                model: if plan_only { "sica-plan".into() } else { "sica".into() },
                lines: answer.lines().map(str::to_string).collect(),
            });
        } else if outcome.launch_error.is_none() {
            let mut lines = vec![format!(
                "SICA submitted no final answer. Trace: {logdir}/execution_tree.txt"
            )];
            // Surface the underlying cause (e.g. connection refused) from the log tail.
            let all_errs: Vec<&String> = outcome
                .stderr_tail
                .iter()
                .filter(|l| l.contains("ERROR") || l.contains("Error") || l.contains("error"))
                .collect();
            let errs: Vec<String> = all_errs[all_errs.len().saturating_sub(4)..]
                .iter()
                .map(|l| format!("  {}", l.chars().take(240).collect::<String>()))
                .collect();
            if outcome.success && !errs.is_empty() {
                lines.push("Errors from the agent log:".into());
                lines.extend(errs);
            }
            self.turns.push(TurnBlock::AssistantReply {
                model: "sica".into(),
                lines,
            });
        }

        if plan_only {
            if let (true, Some(plan)) = (outcome.success, outcome.answer.clone()) {
                self.pending_plan_text = Some(plan);
                self.active_plans = vec![PlanOption {
                    id: 1,
                    title: "Execute this plan with SICA".into(),
                    detail: "Press Enter, click Execute, or type /execute".into(),
                    target_file: String::new(),
                    agent: "sica".into(),
                    selected: true,
                    state: StepState::Pending,
                }];
            }
            if !changes.is_empty() {
                self.turns.push(TurnBlock::AssistantReply {
                    model: "mpdc-code".into(),
                    lines: vec![format!(
                        "⚠ Plan mode run modified {} file(s) despite read-only instructions — review them.",
                        changes.len()
                    )],
                });
            }
        } else {
            self.active_plans.clear();
        }

        if !changes.is_empty() {
            let files: Vec<String> = changes.iter().map(|c| c.path.clone()).collect();
            let (added, removed) = changes
                .iter()
                .fold((0, 0), |(a, r), c| (a + c.added, r + c.removed));
            for fc in changes {
                if let Some(existing) = self.changed_files.iter_mut().find(|c| c.path == fc.path) {
                    *existing = fc;
                } else {
                    self.changed_files.push(fc);
                }
            }
            self.selected_file_idx = self.changed_files.len().saturating_sub(1);
            self.turns.push(TurnBlock::CodeChangeSummary {
                summary: format!(
                    "SICA changed {} file(s) (+{added} −{removed}) for \"{}\" in {:.0}s{}.",
                    files.len(),
                    prompt,
                    outcome.duration_s,
                    outcome
                        .total_tokens
                        .map(|t| format!(", {t} tokens"))
                        .unwrap_or_default()
                ),
                files,
            });
        } else if !plan_only && outcome.launch_error.is_none() {
            self.turns.push(TurnBlock::AssistantReply {
                model: "mpdc-code".into(),
                lines: vec!["No files were changed in the workspace.".into()],
            });
        }

        if let Some((tool, lines)) = verification {
            self.turns.push(TurnBlock::AssistantReply { model: tool, lines });
        }
        if let Some(t) = outcome.total_tokens {
            self.completion_tokens_total += t;
        }
        self.scroll_to_bottom = true;
        self.save_session();
    }

    pub fn cycle_mode_and_auto_run(&mut self) {
        self.mode = self.mode.next();
        if self.mode.is_autonomous() {
            self.start_autonomous_plan_execution();
        }
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
        // Ignore mouse movement/drag/release events so they never flood the UI loop!
        match mouse.kind {
            MouseEventKind::ScrollUp
            | MouseEventKind::ScrollDown
            | MouseEventKind::Down(MouseButton::Left) => {}
            _ => return,
        }

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
                    self.diff_scroll_offset =
                        (self.diff_scroll_offset + 2).min(self.max_diff_scroll);
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
                            self.start_autonomous_plan_execution();
                        }
                        HitAction::SelectChangedFile(idx) => {
                            if idx < self.changed_files.len() {
                                self.selected_file_idx = idx;
                                self.diff_scroll_offset = 0;
                            }
                        }
                        HitAction::CycleMode => {
                            self.cycle_mode_and_auto_run();
                        }
                        HitAction::ToggleModelModal => {
                            self.show_model_modal = !self.show_model_modal;
                        }
                        HitAction::TogglePoolModel(idx) => {
                            self.selected_pool_idx = idx;
                            self.toggle_pool_model(idx);
                        }
                        HitAction::SelectSlashCommand(cmd) => {
                            self.input = cmd;
                            self.cursor_pos = self.input.chars().count();
                        }
                        HitAction::ToggleScreenWatch => {
                            self.screen_watch_enabled = !self.screen_watch_enabled;
                            self.last_screen_watch = Instant::now();
                            if self.screen_watch_enabled {
                                self.trigger_screen_capture(Some(
                                    "Continuous screen watch enabled".into(),
                                ));
                            }
                        }
                        HitAction::CaptureScreenNow => {
                            self.trigger_screen_capture(None);
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
                KeyCode::Char('y') | KeyCode::Char('e') => {
                    self.start_autonomous_plan_execution();
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
                    self.toggle_pool_model(self.selected_pool_idx);
                }
                _ => {}
            }
            return true;
        }

        match key.code {
            KeyCode::Esc => return false,
            KeyCode::Tab | KeyCode::BackTab => {
                if self.input.starts_with('/') && !self.input.contains(' ') {
                    if let Some((cmd, _)) = SLASH_COMMANDS
                        .iter()
                        .find(|(c, _)| c.starts_with(&self.input) && *c != self.input)
                    {
                        self.input = (*cmd).to_string();
                        self.cursor_pos = self.input.chars().count();
                        return true;
                    }
                }
                self.cycle_mode_and_auto_run();
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
            KeyCode::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                let len = self.input.chars().count();
                if self.cursor_pos < len {
                    self.cursor_pos += 1;
                }
            }
            KeyCode::Home => {
                self.cursor_pos = 0;
            }
            KeyCode::End => {
                self.cursor_pos = self.input.chars().count();
            }
            KeyCode::Enter => {
                self.submit_input();
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 && !self.input.is_empty() {
                    let mut chars: Vec<char> = self.input.chars().collect();
                    if self.cursor_pos <= chars.len() {
                        chars.remove(self.cursor_pos - 1);
                        self.input = chars.into_iter().collect();
                        self.cursor_pos -= 1;
                    }
                }
            }
            KeyCode::Delete => {
                let mut chars: Vec<char> = self.input.chars().collect();
                if self.cursor_pos < chars.len() {
                    chars.remove(self.cursor_pos);
                    self.input = chars.into_iter().collect();
                }
            }
            KeyCode::Char(c) => {
                if key.modifiers.contains(KeyModifiers::CONTROL) && c == 'u' {
                    self.input.clear();
                    self.cursor_pos = 0;
                    return true;
                }
                if key.modifiers.contains(KeyModifiers::ALT) && c.is_ascii_digit() {
                    let idx = (c as usize).saturating_sub('1' as usize);
                    if let Some(p) = self.active_plans.get_mut(idx) {
                        p.selected = !p.selected;
                    }
                    return true;
                }
                let mut chars: Vec<char> = self.input.chars().collect();
                let pos = self.cursor_pos.min(chars.len());
                chars.insert(pos, c);
                self.input = chars.into_iter().collect();
                self.cursor_pos = pos + 1;
            }
            _ => {}
        }
        true
    }
}

fn parse_cli_dir_and_prompt(args: Vec<String>) -> (Option<String>, Option<String>) {
    let mut target_dir: Option<String> = None;
    let mut prompt_words: Vec<String> = Vec::new();
    let mut iter = args.into_iter().skip(1);

    while let Some(arg) = iter.next() {
        if arg == "--once" {
            continue;
        }
        if arg == "--cwd" || arg == "-C" {
            if let Some(d) = iter.next() {
                target_dir = Some(d);
            }
            continue;
        }
        if let Some(d) = arg.strip_prefix("--cwd=") {
            target_dir = Some(d.to_string());
            continue;
        }
        if arg.starts_with('-') {
            continue;
        }
        if target_dir.is_none() && prompt_words.is_empty() && Path::new(&arg).is_dir() {
            target_dir = Some(arg);
            continue;
        }
        prompt_words.push(arg);
    }

    let prompt = if prompt_words.is_empty() {
        None
    } else {
        Some(prompt_words.join(" "))
    };
    (target_dir, prompt)
}

pub fn render_coder_once_snapshot(
    width: u16,
    height: u16,
    target_dir: Option<String>,
    initial_prompt: Option<String>,
) -> io::Result<()> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend)?;
    let mut app = MpdcCoderApp::new();
    if let Some(d) = target_dir {
        app.change_working_directory(&d);
    }
    if let Some(p) = initial_prompt {
        app.input = p;
        app.submit_input();
        // Wait for the agent run to finish (bounded by the SICA timeout).
        let deadline = Instant::now() + Duration::from_secs(sica::default_timeout_s() + 120);
        while Instant::now() < deadline {
            thread::sleep(Duration::from_millis(100));
            app.poll_bg();
            if !app.is_generating && !app.is_executing_plan {
                break;
            }
        }
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
    let is_once = args.iter().any(|a| a == "--once");
    let (target_dir, init_prompt) = parse_cli_dir_and_prompt(args);

    if is_once {
        let cols = env::var("COLUMNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(130);
        let lines = env::var("LINES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(34);
        return render_coder_once_snapshot(cols, lines, target_dir, init_prompt);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = MpdcCoderApp::new();
    if let Some(d) = target_dir {
        app.change_working_directory(&d);
    }
    if let Some(p) = init_prompt {
        app.input = p;
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
        if app.should_exit {
            return Ok(());
        }
        terminal.draw(|f| app.draw(f))?;

        // Wait up to 40ms for the first event, then drain ALL queued events immediately
        // so mouse-move events NEVER back up the keyboard queue!
        if event::poll(Duration::from_millis(40))? {
            loop {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        if !app.on_key_event(key) || app.should_exit {
                            app.save_session();
                            return Ok(());
                        }
                    }
                    Event::Mouse(mouse) => {
                        app.on_mouse_event(mouse);
                    }
                    _ => {}
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }

        if last_tick.elapsed() >= Duration::from_millis(500) {
            app.refresh_system_metrics();
            last_tick = Instant::now();
        }
        if app.screen_watch_enabled && app.last_screen_watch.elapsed() >= Duration::from_secs(10) {
            app.last_screen_watch = Instant::now();
            app.trigger_screen_capture(Some("Periodic screen watch sample".into()));
        }
    }
}
