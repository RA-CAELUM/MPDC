# MPDC: Mini-PC Local LLM Datacenter

I MADE THIS TOOL FOR MYSELF BECAUSE I WANTED TO SAVE MONEY ON CLAUDECODE TOKENS. THE FUTURE OF PROGRAMMING DOES NOT RELY ON BIG TECH DATA CENTERS. 
INDEPENDENCE FOR MEAT PROXIES , VIVA LA MEAT PROXY 

OpenCode-centric local **Qwen3** multi-agent & 3-tier self-improving LLM stack designed for NixOS (and portable to Ubuntu via Docker Compose / Nix flakes).

## Quickstart

### 1. Enter the Nix Development Shell (or virtualenv)
```bash
nix develop
# Or with pip/uv:
pip install -e ".[dev]"
```

### 2. Start the MPDC OpenAI Gateway
```bash
mpdc serve
```

### 3. Connect OpenCode
Copy or symlink `opencode.json` into your project root and run `opencode`. OpenCode automatically routes to `mpdc/qwen3-mpdc` (`http://127.0.0.1:8000/v1`) and mounts the `mpdc-agents` MCP server (`mpdc_plan`, `mpdc_learn`, `mpdc_stats`).

### 4. Quantify Self-Improvement & Run On-Node QLoRA Training
```bash
# View Online Production Telemetry (Pass@1, Iterations-to-Green, Patch Survival Rate, Skill Lift)
mpdc stats

# Export 80% Train (SFT + DPO) and 20% Golden Replay Holdout JSONL datasets
mpdc harvest

# Trigger On-Node QLoRA Training + Gated A/B Evaluation + vLLM Dynamic LoRA Hot-Swap
mpdc train --now
```
