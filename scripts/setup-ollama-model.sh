#!/usr/bin/env bash
# One-time setup of MPDC's default local model for 6 GB GPUs.
# Pulls Qwen3-4B-Instruct-2507 (Q4_K_M) and creates `mpdc-qwen3-4b` (16k context per slot).
#
# Run the Ollama server with two parallel slots (coder + independent tester), which
# together fit in ~5.2 GB VRAM:
#   OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=1
# (on NixOS: services.ollama.environmentVariables, see README).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLLAMA_HOST_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if ! curl -fsS -m 3 "$OLLAMA_HOST_URL/api/version" >/dev/null; then
  echo "error: Ollama server not reachable at $OLLAMA_HOST_URL (start it with 'ollama serve' or enable services.ollama)" >&2
  exit 1
fi

ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama create mpdc-qwen3-4b -f "$REPO_ROOT/config/ollama/Modelfile"
echo "Created mpdc-qwen3-4b. Test: ollama run mpdc-qwen3-4b 'say hi'"
