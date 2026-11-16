//! Bridge between `mpdc-code` and the MPDC-SICA self-improving coding agent.
//!
//! `mpdc-code` does not generate code itself: every prompt is handed to SICA
//! (`python -m base_agent.agent`), which runs a real LLM tool-calling loop
//! (read/write files, run commands, sub-agents, overseer) against the MPDC
//! gateway. This module locates SICA, spawns it, streams its event log back,
//! and computes the real on-disk changes it made.

use super::FileChangeEntry;
use std::{
    collections::BTreeMap,
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Directories never scanned when snapshotting the workspace.
const SKIP_DIRS: &[&str] = &[
    ".git", "target", "node_modules", "__pycache__", ".venv", "venv", ".mpdc", ".mpdc_data",
    "dist", "build", ".direnv", "result", ".pytest_cache", ".mypy_cache",
];
const SKIP_FILES: &[&str] = &[".mpdc_session.json"];
const MAX_SNAPSHOT_FILE_BYTES: u64 = 256 * 1024;
const MAX_SNAPSHOT_FILES: usize = 5000;

pub struct SicaRunRequest {
    pub prompt: String,
    pub cwd: String,
    pub gateway_url: String,
    pub timeout_s: u64,
}

pub enum SicaEvent {
    /// One line of SICA's event stream (`event_type => content`).
    Log(String),
    Finished(SicaOutcome),
}

pub struct SicaOutcome {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub answer: Option<String>,
    pub logdir: PathBuf,
    pub stderr_tail: Vec<String>,
    pub total_tokens: Option<u64>,
    pub duration_s: f64,
    pub launch_error: Option<String>,
}

pub fn default_timeout_s() -> u64 {
    env::var("MPDC_SICA_TIMEOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1800)
}

fn is_sica_root(p: &Path) -> bool {
    p.join("base_agent").join("agent.py").is_file()
}

/// Finds the MPDC-SICA checkout: `$MPDC_SICA_ROOT`, the repo this binary was
/// built from, or a `modules/MPDC-SICA` next to any ancestor of the executable.
pub fn locate_sica_root() -> Option<PathBuf> {
    if let Ok(p) = env::var("MPDC_SICA_ROOT") {
        let p = PathBuf::from(p);
        if is_sica_root(&p) {
            return Some(p);
        }
    }
    let built_from = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../modules/MPDC-SICA");
    if is_sica_root(&built_from) {
        return built_from.canonicalize().ok().or(Some(built_from));
    }
    if let Ok(exe) = env::current_exe() {
        for anc in exe.ancestors() {
            let cand = anc.join("modules").join("MPDC-SICA");
            if is_sica_root(&cand) {
                return Some(cand);
            }
        }
    }
    None
}

/// Python used to run SICA: `$MPDC_SICA_PYTHON`, the repo's
/// `scripts/sica-python.sh` (nix-shell with SICA's deps), else `python3`.
fn sica_python(root: &Path) -> String {
    if let Ok(p) = env::var("MPDC_SICA_PYTHON") {
        if !p.trim().is_empty() {
            return p;
        }
    }
    let wrapper = root.join("../../scripts/sica-python.sh");
    if wrapper.is_file() {
        return wrapper.canonicalize().unwrap_or(wrapper).display().to_string();
    }
    "python3".into()
}

fn run_logdir() -> PathBuf {
    let base = env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|_| env::var("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|_| env::temp_dir());
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    base.join("mpdc").join("sica-runs").join(format!("run-{ts}"))
}

/// Runs SICA to completion on the calling thread, reporting through `on_event`.
pub fn run_sica(req: SicaRunRequest, mut on_event: impl FnMut(SicaEvent)) {
    let started = Instant::now();
    let logdir = run_logdir();
    let fail = |msg: String, logdir: PathBuf| SicaOutcome {
        success: false,
        exit_code: None,
        answer: None,
        logdir,
        stderr_tail: Vec::new(),
        total_tokens: None,
        duration_s: 0.0,
        launch_error: Some(msg),
    };

    let Some(root) = locate_sica_root() else {
        on_event(SicaEvent::Finished(fail(
            "MPDC-SICA not found. Run `git submodule update --init` or set MPDC_SICA_ROOT.".into(),
            logdir,
        )));
        return;
    };
    if let Err(e) = fs::create_dir_all(&logdir) {
        on_event(SicaEvent::Finished(fail(
            format!("cannot create log dir {}: {e}", logdir.display()),
            logdir,
        )));
        return;
    }

    let python = sica_python(&root);
    let mut cmd = Command::new(&python);
    cmd.args(["-m", "base_agent.agent", "--prompt", &req.prompt])
        .arg("--workdir")
        .arg(&req.cwd)
        .arg("--logdir")
        .arg(&logdir)
        .arg("--timeout")
        .arg(req.timeout_s.to_string())
        .current_dir(&root)
        .env("MPDC_GATEWAY_URL", &req.gateway_url)
        .env("SICA_HOST_WORKDIR", &req.cwd)
        .env("MPDC_NODE_ID", "mpdc-code")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            on_event(SicaEvent::Finished(fail(
                format!("failed to launch `{python}` for SICA: {e}. Set MPDC_SICA_PYTHON."),
                logdir,
            )));
            return;
        }
    };

    // stderr carries Python logging; keep a tail for diagnostics and the token summary.
    let stderr = child.stderr.take();
    let (err_tx, err_rx) = mpsc::channel::<(Vec<String>, Option<u64>)>();
    thread::spawn(move || {
        let mut tail: Vec<String> = Vec::new();
        let mut tokens = None;
        if let Some(s) = stderr {
            for line in BufReader::new(s).lines().map_while(Result::ok) {
                // e.g. "Tokens used: 12,345 (cached: 40.00%)"
                if let Some(rest) = line.split("Tokens used: ").nth(1) {
                    let first = rest.split_whitespace().next().unwrap_or("");
                    let n: String = first.chars().filter(|c| c.is_ascii_digit()).collect();
                    tokens = n.parse().ok();
                }
                tail.push(line);
                if tail.len() > 40 {
                    tail.remove(0);
                }
            }
        }
        let _ = err_tx.send((tail, tokens));
    });

    if let Some(out) = child.stdout.take() {
        let mut in_final_tree = false;
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            // After the run SICA prints the full execution tree; it is saved to
            // execution_tree.txt in the logdir, so don't flood the transcript.
            if line.starts_with("EXECUTION TREE") {
                in_final_tree = true;
            }
            if in_final_tree || line.trim().is_empty() {
                continue;
            }
            on_event(SicaEvent::Log(line));
        }
    }

    let status = child.wait();
    let (stderr_tail, total_tokens) = err_rx.recv().unwrap_or_default();
    let exit_code = status.as_ref().ok().and_then(|s| s.code());
    let success = status.map(|s| s.success()).unwrap_or(false);
    let answer = fs::read_to_string(logdir.join("answer.txt"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    on_event(SicaEvent::Finished(SicaOutcome {
        success,
        exit_code,
        answer,
        logdir,
        stderr_tail,
        total_tokens,
        duration_s: started.elapsed().as_secs_f64(),
        launch_error: None,
    }));
}

/// Text contents of the project's source files, keyed by relative path.
pub fn snapshot_workspace(cwd: &str) -> BTreeMap<String, String> {
    let root = Path::new(cwd);
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            if out.len() >= MAX_SNAPSHOT_FILES {
                return out;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let Ok(ft) = entry.file_type() else { continue };
            let path = entry.path();
            if ft.is_dir() {
                if !SKIP_DIRS.contains(&name.as_str()) {
                    stack.push(path);
                }
            } else if ft.is_file() && !SKIP_FILES.contains(&name.as_str()) {
                if entry.metadata().map(|m| m.len() > MAX_SNAPSHOT_FILE_BYTES).unwrap_or(true) {
                    continue;
                }
                if let Ok(text) = fs::read_to_string(&path) {
                    if let Ok(rel) = path.strip_prefix(root) {
                        out.insert(rel.to_string_lossy().to_string(), text);
                    }
                }
            }
        }
    }
    out
}

/// Real per-file changes between two workspace snapshots.
pub fn diff_snapshots(
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
) -> Vec<FileChangeEntry> {
    let mut changes = Vec::new();
    for (path, new_text) in after {
        match before.get(path) {
            Some(old_text) if old_text == new_text => {}
            old => changes.push(file_diff(path, old.map(|s| s.as_str()), Some(new_text))),
        }
    }
    for (path, old_text) in before {
        if !after.contains_key(path) {
            changes.push(file_diff(path, Some(old_text), None));
        }
    }
    changes
}

fn file_diff(path: &str, old: Option<&str>, new: Option<&str>) -> FileChangeEntry {
    let a: Vec<&str> = old.map(|s| s.lines().collect()).unwrap_or_default();
    let b: Vec<&str> = new.map(|s| s.lines().collect()).unwrap_or_default();
    let ops = line_diff(&a, &b);
    let added = ops.iter().filter(|(k, _)| *k == '+').count();
    let removed = ops.iter().filter(|(k, _)| *k == '-').count();

    let mut diff_lines = vec![
        if old.is_some() { format!("--- a/{path}") } else { "--- /dev/null".into() },
        if new.is_some() { format!("+++ b/{path}") } else { "+++ /dev/null".into() },
    ];
    // Emit changed regions with 2 lines of context.
    const CTX: usize = 2;
    let changed: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, (k, _))| *k != ' ')
        .map(|(i, _)| i)
        .collect();
    let mut i = 0;
    while i < changed.len() {
        let start = changed[i].saturating_sub(CTX);
        let mut end = changed[i];
        while i < changed.len() && changed[i] <= end + 2 * CTX + 1 {
            end = changed[i];
            i += 1;
        }
        let end = (end + CTX).min(ops.len() - 1);
        diff_lines.push(format!("@@ lines {}-{} @@", start + 1, end + 1));
        for (k, text) in &ops[start..=end] {
            diff_lines.push(format!("{k}{text}"));
        }
    }
    FileChangeEntry { path: path.to_string(), added, removed, diff_lines }
}

/// LCS line diff; falls back to replace-all for very large inputs.
fn line_diff<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(char, &'a str)> {
    let (n, m) = (a.len(), b.len());
    if n.saturating_mul(m) > 4_000_000 {
        return a.iter().map(|l| ('-', *l)).chain(b.iter().map(|l| ('+', *l))).collect();
    }
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut ops = Vec::with_capacity(n + m);
    while i < n && j < m {
        if a[i] == b[j] {
            ops.push((' ', a[i]));
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            ops.push(('-', a[i]));
            i += 1;
        } else {
            ops.push(('+', b[j]));
            j += 1;
        }
    }
    ops.extend(a[i..].iter().map(|l| ('-', *l)));
    ops.extend(b[j..].iter().map(|l| ('+', *l)));
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_detects_added_modified_and_deleted_files() {
        let mut before = BTreeMap::new();
        before.insert("keep.rs".to_string(), "a\nb\n".to_string());
        before.insert("edit.rs".to_string(), "x\ny\nz\n".to_string());
        before.insert("gone.rs".to_string(), "bye\n".to_string());
        let mut after = before.clone();
        after.insert("edit.rs".into(), "x\nY\nz\n".into());
        after.remove("gone.rs");
        after.insert("new.rs".into(), "hello\n".into());

        let changes = diff_snapshots(&before, &after);
        let by_path: BTreeMap<_, _> = changes.iter().map(|c| (c.path.as_str(), c)).collect();
        assert_eq!(changes.len(), 3);
        assert_eq!((by_path["edit.rs"].added, by_path["edit.rs"].removed), (1, 1));
        assert_eq!((by_path["new.rs"].added, by_path["new.rs"].removed), (1, 0));
        assert_eq!((by_path["gone.rs"].added, by_path["gone.rs"].removed), (0, 1));
        assert!(by_path["edit.rs"].diff_lines.iter().any(|l| l == "-y"));
        assert!(by_path["edit.rs"].diff_lines.iter().any(|l| l == "+Y"));
    }

    #[test]
    fn snapshot_skips_build_dirs() {
        let dir = env::temp_dir().join(format!("mpdc-snap-{}", std::process::id()));
        fs::create_dir_all(dir.join("target")).unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("target/junk.txt"), "x").unwrap();
        fs::write(dir.join("src/main.rs"), "fn main() {}").unwrap();
        let snap = snapshot_workspace(dir.to_str().unwrap());
        let _ = fs::remove_dir_all(&dir);
        assert!(snap.contains_key("src/main.rs"));
        assert!(!snap.keys().any(|k| k.starts_with("target")));
    }

    #[test]
    fn locates_repo_sica_checkout() {
        assert!(locate_sica_root().is_some());
    }
}
