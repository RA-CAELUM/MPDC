#!/usr/bin/env bash
# Python interpreter for the MPDC-SICA agent, used by `mpdc-code`.
# Provides SICA's runtime dependencies via nix-shell. Override with
# MPDC_SICA_PYTHON=/path/to/python if you manage your own environment
# (pip install -r modules/MPDC-SICA/base_agent/requirements.txt).
set -euo pipefail

exec nix-shell -p 'python312.withPackages (p: with p; [
    fastapi uvicorn httpx pydantic pydantic-settings pyyaml rich openai
    jsonlines tiktoken python-dotenv tabulate json-repair jinja2 gitpython
    diff-match-patch anthropic pandas numpy scipy sympy
  ])' --run "python3 ${*@Q}"
