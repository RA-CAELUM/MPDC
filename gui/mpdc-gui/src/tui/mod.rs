//! `BtopTuiApp` state machine, keyboard handler, and interactive / `--once` snapshot runners.

pub mod panels;
pub mod theme;

use crate::{
    client::{spawn_poll_overview, BgResult},
    models::{AdapterItem, OverviewPayload},
    tui::theme::{btop_key_label, BTOP_DIV_LINE, BTOP_MAIN_BG},
};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend, TestBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, TableState},
    Frame, Terminal,
};
use std::{
    env,
    io::{self, Write},
    sync::mpsc::{self, Receiver, Sender},
    time::{Duration, Instant},
};

pub struct BtopTuiApp {
    pub gateway_url: String,
    pub overview: OverviewPayload,
    pub connected: bool,
    pub tps_history: Vec<f64>,
    pub tick: f64,
    pub selected_node: usize,
    pub node_table_state: TableState,
    pub events: Vec<String>,
    pub last_poll: Instant,
    pub tx: Sender<BgResult>,
    pub rx: Receiver<BgResult>,
}

impl Default for BtopTuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl BtopTuiApp {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let gateway_url =
            env::var("MPDC_GATEWAY_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
        let mut node_table_state = TableState::default();
        node_table_state.select(Some(0));

        Self {
            gateway_url,
            overview: OverviewPayload::demo_seed(),
            connected: false,
            tps_history: vec![
                142.0, 151.5, 164.0, 172.8, 168.4, 181.2, 189.5, 195.0, 191.2, 198.8, 203.7,
                201.4, 206.5, 203.7,
            ],
            tick: 0.0,
            selected_node: 0,
            node_table_state,
            events: vec![
                "23:45:01 [rtx5070-laptop] /think qwen3-lora-v001 (+2 skills) -> 68.4 tok/s | PSR 100%".into(),
                "23:45:04 [workstation-sim] /no_think qwen3-mpdc-fast (+1 skill) -> 71.2 tok/s | PSR 94%".into(),
                "23:45:09 [ros2-field-deck] /think qwen3-lora-v001 (+2 skills) -> 64.1 tok/s | Pass@1 OK".into(),
                "23:45:12 [DISTILLER] Extracted Skill #3 (NixOS CUDA libcuda.so.1) -> ACTIVE".into(),
            ],
            last_poll: Instant::now() - Duration::from_secs(5),
            tx,
            rx,
        }
    }

    pub fn poll_async(&mut self) {
        self.last_poll = Instant::now();
        spawn_poll_overview(&self.gateway_url, self.tx.clone());
    }

    pub fn step_tick(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                BgResult::Overview(Ok(payload)) => {
                    self.connected = true;
                    self.overview = payload;
                }
                BgResult::Overview(Err(_)) => {
                    self.connected = false;
                }
                _ => {}
            }
        }

        self.tick += 1.0;
        let base_tps: f64 = self
            .overview
            .nodes
            .iter()
            .filter(|n| n.status != "paused")
            .map(|n| n.last_tok_per_sec)
            .sum();
        let wave = if self.connected {
            base_tps
        } else {
            (base_tps + (self.tick * 0.8).sin() * 4.5).max(0.0)
        };
        self.tps_history.push(wave);
        if self.tps_history.len() > 60 {
            self.tps_history.remove(0);
        }
    }

    pub fn on_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => return false,
            KeyCode::Char('j') | KeyCode::Down => {
                if !self.overview.nodes.is_empty() {
                    self.selected_node = (self.selected_node + 1) % self.overview.nodes.len();
                    self.node_table_state.select(Some(self.selected_node));
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if !self.overview.nodes.is_empty() {
                    self.selected_node = (self.selected_node + self.overview.nodes.len() - 1)
                        % self.overview.nodes.len();
                    self.node_table_state.select(Some(self.selected_node));
                }
            }
            KeyCode::Char('p') => {
                if let Some(node) = self.overview.nodes.get_mut(self.selected_node) {
                    node.status = if node.status == "paused" {
                        "online".into()
                    } else {
                        "paused".into()
                    };
                    self.events.push(format!(
                        "[NODE] {} ({}) stream -> {}",
                        node.node_id,
                        node.ip_address,
                        node.status.to_uppercase()
                    ));
                }
            }
            KeyCode::Char('s') => {
                if let Some(sk) = self.overview.skills.first_mut() {
                    sk.active = !sk.active;
                    self.events.push(format!(
                        "[SKILL #{}] {} -> {}",
                        sk.id,
                        sk.title,
                        if sk.active { "ACTIVE" } else { "PRUNED" }
                    ));
                }
            }
            KeyCode::Char('o') => {
                if let Some(node) = self.overview.nodes.get(self.selected_node) {
                    self.events.push(format!(
                        "[CONFIG] opencode.json for '{}': baseURL=http://mpdc.local:{}/v1, X-MPDC-Client-Node={}",
                        node.node_id, self.overview.server.port, node.node_id
                    ));
                }
            }
            KeyCode::Char('t') => {
                let next_ver = format!("qwen3-lora-v{:03}", self.overview.adapters.len());
                let new_p1 = (self.overview.scorecard.pass_at_1 + 0.035).min(0.97);
                self.overview.scorecard.pass_at_1 = new_p1;
                self.overview.server.active_model = next_ver.clone();
                self.overview.adapters.insert(
                    0,
                    AdapterItem {
                        version: next_ver.clone(),
                        base_model: self.overview.server.base_model.clone(),
                        replay_pass_rate: new_p1,
                        holdout_perplexity: (2.31 - self.overview.adapters.len() as f64 * 0.14)
                            .max(1.20),
                        sanity_score: 1.0,
                        promoted: 1,
                    },
                );
                self.events.push(format!(
                    "[QLORA GATE] Promoted {} to vLLM (Replay@1 {:.1}%)",
                    next_ver,
                    new_p1 * 100.0
                ));
            }
            _ => {}
        }
        true
    }

    pub fn draw(&mut self, f: &mut Frame) {
        let size = f.area();
        f.render_widget(Block::default().style(Style::default().bg(BTOP_MAIN_BG)), size);

        let root_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(9),
                Constraint::Min(7),
                Constraint::Length(1),
            ])
            .split(size);

        let top_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(28),
                Constraint::Percentage(36),
                Constraint::Percentage(36),
            ])
            .split(root_chunks[0]);

        self.draw_gpu_box(f, top_cols[0]);
        self.draw_mem_kpi_box(f, top_cols[1]);
        self.draw_net_stream_box(f, top_cols[2]);

        let mid_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(root_chunks[1]);

        self.draw_nodes_box(f, mid_cols[0]);
        self.draw_bus_box(f, mid_cols[1]);

        let bot_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(root_chunks[2]);

        self.draw_skills_box(f, bot_cols[0]);
        self.draw_adapters_box(f, bot_cols[1]);

        let mut footer_spans = vec![
            Span::styled(
                " MPDC // Man Portable Data Center ",
                Style::default()
                    .bg(Color::Rgb(167, 139, 250))
                    .fg(Color::Rgb(11, 13, 20))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
        ];
        footer_spans.extend(btop_key_label("[", "j/k", "] Select Node  "));
        footer_spans.extend(btop_key_label("[", "p", "] Pause/Resume Stream  "));
        footer_spans.extend(btop_key_label("[", "o", "] Export opencode.json  "));
        footer_spans.extend(btop_key_label("[", "s", "] Toggle Skill  "));
        footer_spans.extend(btop_key_label("[", "t", "] QLoRA Sleep-Train  "));
        footer_spans.extend(btop_key_label("[", "q", "] Quit"));

        f.render_widget(
            Paragraph::new(Line::from(footer_spans)).style(Style::default().bg(BTOP_DIV_LINE)),
            root_chunks[3],
        );
    }
}

pub fn render_once_snapshot(width: u16, height: u16) -> io::Result<()> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend)?;
    let mut app = BtopTuiApp::new();
    app.step_tick();
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
            write!(out, "{}\x1b[0m", cell.symbol())?;
        }
        writeln!(out)?;
    }
    Ok(())
}

pub fn run_tui_entry() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--once") {
        let cols = env::var("COLUMNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(130);
        let lines = env::var("LINES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(28);
        return render_once_snapshot(cols, lines);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = BtopTuiApp::new();
    let res = run_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

fn run_loop<B: Backend>(terminal: &mut Terminal<B>, app: &mut BtopTuiApp) -> io::Result<()> {
    loop {
        if app.last_poll.elapsed() >= Duration::from_millis(1200) {
            app.poll_async();
            app.step_tick();
        }
        terminal.draw(|f| app.draw(f))?;

        if event::poll(Duration::from_millis(150))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && !app.on_key(key.code) {
                    return Ok(());
                }
            }
        }
    }
}
