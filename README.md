<div align="center">
  <img src="./assets/mpdc-banner.svg" alt="MPDC - Man Portable Data Center" width="480">
</div>

I MADE THIS TOOL FOR MYSELF BECAUSE I WANTED TO SAVE MONEY ON CLAUDECODE TOKENS. THE FUTURE OF PROGRAMMING DOES NOT RELY ON BIG TECH DATA CENTERS.

Multi-Pool **Programming (`Qwen3` / `Qwen3-Coder` / `DeepSeek-R1`) + Visual & Spatial Intelligence (`Qwen2.5-VL` / `SpatialVLM` / `MASt3R-Depth`)** Local LLM Datacenter stack with a native Rust **Claude-Code-style Coding & Planning TUI (`mpdc-code`)**, **`btop`-style Telemetry TUI (`mpdc-tui`)**, and **Desktop Workbench GUI (`mpdc-gui`)**.

---

## 1. Install (One-Time Setup)

Clone the repository (including the `MPDC-SICA` self-improving agent submodule) and run the global installer:

```bash
git clone --recurse-submodules https://github.com/RA-CAELUM/MPDC.git
cd MPDC
./install.sh
```
*(Or with Nix Flakes directly: `nix profile add .`)*

This compiles and installs four global commands onto your `$PATH` so you can run them inside **any directory**:
- `mpdc-code` (or `mpdc code`) — Interactive Claude-Code-style Coding, Planning & 3D Spatial Swarm TUI
- `mpdc-tui` (or `mpdc top`) — `btop`-style 6-panel Cluster Telemetry & Switchboard TUI
- `mpdc-gui` (or `mpdc gui`) — Native GPU-accelerated Rust `egui` Desktop Telemetry Workbench
- `mpdc` — Unified Datacenter Gateway, Model Pool Manager, and Self-Improvement CLI

---

## 2. How to Run MPDC

### Step A: Start Your Local Model (default: 6 GB GPU via Ollama)
The default config (`config/default.yaml`) targets a single **6 GB GPU such as an RTX 3050 6GB** with **Qwen3-4B-Instruct-2507 (Q4_K_M)** on [Ollama](https://ollama.com), loaded once and served from **two parallel 16k-context slots**: one for the coding agent and one for an independent tester agent that works at the same time.

1. Run the Ollama server with flash attention, an 8-bit KV cache and two slots (that is what makes 2 × 16k fit in 6 GB). On NixOS:
   ```nix
   services.ollama = {
     enable = true;
     # CUDA build of ollama for your nixpkgs (e.g. package = pkgs.ollama-cuda; or acceleration = "cuda"; on older releases)
     environmentVariables = {
       OLLAMA_FLASH_ATTENTION = "1";
       OLLAMA_KV_CACHE_TYPE = "q8_0";
       OLLAMA_NUM_PARALLEL = "2";      # two agents at once; each slot gets num_ctx (16k) of KV cache
       OLLAMA_MAX_LOADED_MODELS = "1";
     };
   };
   ```
   Or ad hoc: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=2 ollama serve`
2. Create the model once (pulls ~2.5 GB, then wraps it as `mpdc-qwen3-4b` with `num_ctx 16384` per slot from [config/ollama/Modelfile](config/ollama/Modelfile)):
   ```bash
   ./scripts/setup-ollama-model.sh
   ```
   Why this model: on a two-task benchmark (new FizzBuzz module; fix bugs in an existing module and add a function, checked by hidden tests the agent never sees, 3 runs each) Qwen3-4B-Instruct-2507 passed **5/6**, typically in 20–120 s. Qwen2.5-Coder-7B (Q4, 16k context, also fits in ~5 GB) passed 2/6: it often narrates instead of calling tools.
3. Check: `ollama ps` should show `mpdc-qwen3-4b ... 100% GPU ... 16384`. Measured footprint with both slots active: **~5.2 GB VRAM**, each slot generating ~100 tok/s on short contexts (RTX 5070 Laptop; expect roughly half on a 3050). If the card also drives your display and it doesn't fit, Ollama offloads a few layers to CPU (slower, still works).

**Bigger GPUs (vLLM):** point `inference.backend_url` / `base_model` in `config/default.yaml` at your server, e.g.
```bash
vllm serve Qwen/Qwen3-30B-A3B --port 8001 --enable-lora --max-model-len 32768
```
and set `thinking_toggle: true` for hybrid Qwen3 models that understand `/think` and `/no_think`.

### Step B: Start the MPDC Datacenter Gateway
In a second terminal, start the MPDC Gateway (streams tokens over SSE to local and remote machines on `:8000`):

```bash
mpdc serve
```

### Step C: Code & Plan in Any Project Folder (`mpdc-code`)
`cd` into **any project folder** on your machine (or another laptop connected over LAN/Tailscale) and launch the interactive Coding & Planning TUI:

```bash
cd ~/your-project
mpdc-code
```

#### How `mpdc-code` works
Every prompt is executed by the **[MPDC-SICA](modules/MPDC-SICA)** agent, which talks to your local model through `mpdc serve`. By default (`MPDC_SICA_MODE=local`) it runs a **single lean agent sized for one small GPU**:

- **Native tool calling** (the model's own function-call format, not SICA's XML protocol) with six tools: `list_files`, `read_file`, `write_file`, `replace_in_file`, `run_command`, `finish`. System prompt + tool schemas are ~1.5k tokens (SICA's full orchestrator uses ~12k).
- **Context fitting:** older tool output is shortened and, if needed, the oldest steps are dropped so requests always fit the served context window.
- **Loop guard:** repeating the exact same tool call is warned on the 3rd time and stops the round on the 5th.
- **Independent tester (second slot):** for Python projects, a second agent writes acceptance tests from the task description *only* (it never sees the new code, and its tests live outside your project) while the coder works. Its tests catch requirements the coder's own tests miss; `MPDC_TESTER=0` turns it off.
- **Verify-and-fix rounds:** after each round `mpdc-code` runs the project's checks (`cargo test`, `pytest`, `ctest` or `nix flake check`) plus the tester's acceptance tests, and fails the round if pre-existing tests were deleted; on failure the output is sent back to the agent for another round (up to `MPDC_VERIFY_ROUNDS`). Verified results are harvested as training data, including the (failed → fixed) pair.

`mpdc-code` streams the agent's events into the chat pane, shows the **actual** file changes (before/after snapshot diff), and saves full traces under `~/.cache/mpdc/sica-runs/`. `MPDC_SICA_MODE=full` restores SICA's original multi-agent orchestrator (meant for larger models).

> The agent edits files and runs shell commands **directly in your project directory** (no sandbox; writes outside the project are refused). Use it inside a git repo so you can review and revert.

| Env var | Default | Purpose |
| :--- | :--- | :--- |
| `MPDC_GATEWAY_URL` | `http://127.0.0.1:8000` | Gateway that the agent's LLM calls go through |
| `MPDC_SICA_MODE` | `local` | `local` = single lean agent; `full` = SICA multi-agent orchestrator |
| `MPDC_VERIFY_ROUNDS` | `3` | Max agent rounds when the independent check fails |
| `MPDC_SICA_TIMEOUT` | `600` | Seconds per round before the agent is cancelled |
| `MPDC_SICA_OVERSEER` | `0` | `1` re-enables SICA's LLM overseer (costs prompt-cache hits on one GPU) |
| `MPDC_SICA_ROOT` / `MPDC_SICA_PYTHON` | repo submodule / `scripts/sica-python.sh` | Where SICA lives and which Python (with pytest) runs it |
| `MPDC_TESTER` | `1` | `0` disables the parallel acceptance-test agent |
| `LOCAL_LLM_CONTEXT_TOKENS` | `16384` | Must match the model's per-slot `num_ctx`; used for context fitting |
| `LOCAL_LLM_MAX_TOKENS` | `4096` | Output-token cap per call |
| `LOCAL_LLM_KEEP_FULL_RESULTS` | `4` | Most recent tool results kept unshortened |
| `SICA_MAX_ITERATIONS` | `40` | Max agent steps per round |

#### `mpdc-code` Keyboard Shortcuts & Slash Commands
| Key / Command | Action |
| :--- | :--- |
| **`Tab` / `Shift+Tab`** | Cycle between **`PLAN MODE (/think)`**, **`CODE & DIFF MODE`**, and **`SPATIAL + CODE SWARM`** |
| **`Ctrl+P`** or `/models` | Open the interactive Model Pool Selector to toggle individual Programming & Spatial/Visual models on/off |
| **`Ctrl+Y`** or `/apply` / `/execute` | Execute the plan produced by the last `/plan` run |
| `/plan <task>` | Run SICA read-only: it inspects the project and returns a numbered plan (no file edits) |
| `/code <task>` | Run SICA autonomously: edit files, build/test, fix, and summarize |
| `/swarm <task>` | Same as `/code` (in `MPDC_SICA_MODE=full`, SICA may also delegate to its `spatial_visual_specialist` sub-agent) |
| `/screen [query]` | Capture a screenshot to `.mpdc_captures/` (not analyzed by a model yet: reference the path in a prompt) |
| `/watch` | Toggle continuous background screen monitoring (`SCREEN WATCH: ON/OFF`) for Spatial AI |
| `/resume` | Restore the previous session (`.mpdc_session.json`) including chat, plan steps, and diffs |
| `/export [file.md]` | Export the full session transcript, plan checklist, and unified file diffs to Markdown |
| `/clear` | Clear the session transcript |
| `/exit` (or `Esc`) | Save session state to `.mpdc_session.json` and exit `mpdc-code` |

#### One-Shot CLI Mode
You can also run multi-agent tasks non-interactively from the command line:

```bash
mpdc code "Fix ROS2 grasp controller Z-axis orientation" \
  --image sim_viewport.png \
  --pointcloud wrist_cam.ply \
  --mode sensor_fusion_loop
```

---

## 3. Monitor & Control Your Datacenter (`mpdc-tui` & `mpdc-gui`)

- **Terminal Monitor (`mpdc-tui`)**:
  ```bash
  mpdc-tui
  ```
  Keys: `j/k` select output computer, `p` pause/resume token stream, `m` cycle swarm coordination mode (`sensor_fusion_loop` / `parallel_ensemble` / `specialist_graph`), `o` export client profile, `s` toggle episodic skill, `t` trigger QLoRA sleep-cycle training, `q` quit.

- **Desktop Workbench GUI (`mpdc-gui`)**:
  ```bash
  mpdc-gui
  ```
  Provides 3 synchronized oscilloscopes, per-node token rate caps, live Programming & Spatial/Visual model pool toggles, Episodic Skill distillation, and QLoRA promotion gate controls.

---

## 4. Manage Model Pools & Self-Improving Pipeline

```bash
# List all active Programming & Visual/Spatial models
mpdc pools

# Register an additional Coding or Spatial/Visual model endpoint
mpdc pools --add-id molmo-spatial-7b --add-name allenai/Molmo-7B-D-0924 --add-category spatial_3d --add-endpoint http://127.0.0.1:8006/v1

# View Online Self-Improvement Scorecard (Pass@1, Iterations-to-Green, Patch Survival Rate, Skill Lift)
mpdc stats

# Export 80% Train (SFT + DPO) and 20% Golden Replay Holdout JSONL datasets
mpdc harvest

# Trigger Tier 3 On-Node QLoRA Training + Gated Evaluation + vLLM LoRA Hot-Swap
mpdc train --now
```

## Submodules
- [`modules/MPDC-SICA`](file:///home/sgk/Projects/MPDC/modules/MPDC-SICA): Self-Improving Coding Agent (`SICA`) pre-configured for local **Qwen3** + **Spatial/Visual Sub-Agents** ([`SpatialVisualAgent`](file:///home/sgk/Projects/MPDC/modules/MPDC-SICA/base_agent/src/agents/implementations/spatial_visual_agent.py)).
