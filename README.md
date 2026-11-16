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

### Step A: Start Your Local Inference Engine(s) (`vLLM` / `SGLang` / `llama-server`)
Run one or more local OpenAI-compatible servers for your **Programming Pool** and **Visual/Spatial Pool**:

```bash
# Port 8001: Programming & Reasoning Pool Backbone (Qwen3 / Qwen3-Coder)
vllm serve Qwen/Qwen3-30B-A3B --port 8001 --enable-lora --max-model-len 32768

# Port 8003 (Optional): 2D / Video Vision-Language Model
vllm serve Qwen/Qwen2.5-VL-32B-Instruct --port 8003

# Port 8004 (Optional): 3D Spatial Grounding Model
vllm serve SpatialVLM/Spatial-Grounded-13B --port 8004
```

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

#### `mpdc-code` Keyboard Shortcuts & Slash Commands
| Key / Command | Action |
| :--- | :--- |
| **`Tab` / `Shift+Tab`** | Cycle between **`PLAN MODE (/think)`**, **`CODE & DIFF MODE`**, and **`SPATIAL + CODE SWARM`** |
| **`Ctrl+P`** or `/models` | Open the interactive Model Pool Selector to toggle individual Programming & Spatial/Visual models on/off |
| **`Ctrl+Y`** or `/apply` | Accept & apply the proposed code diff, run verification, and harvest the trajectory |
| `/plan <task>` | Architect a step-by-step plan using `qwen3-architect` (`<think>`) + Tier 1 Episodic Skills |
| `/code <task>` | Synthesize a code patch across active Programming Pool models |
| `/spatial <path>` | Attach an image, video frame, depth map, or point cloud (`.ply`/`.pcd`) for 3D scene grounding |
| `/clear` | Clear the session transcript |
| **`Esc` / `Ctrl+C`** | Exit `mpdc-code` |

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
