#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

exec nix-shell -p \
  python312 \
  python312Packages.fastapi \
  python312Packages.uvicorn \
  python312Packages.httpx \
  python312Packages.pydantic \
  python312Packages.pydantic-settings \
  python312Packages.pyyaml \
  python312Packages.typer \
  python312Packages.rich \
  python312Packages.numpy \
  --run "PYTHONPATH=\"$REPO_ROOT/src:\${PYTHONPATH:-}\" python3 -m mpdc.cli ${*@Q}"
