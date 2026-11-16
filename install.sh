#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Installing MPDC (mpdc, mpdc-code, mpdc-tui, mpdc-gui) globally..."

if command -v nix >/dev/null 2>&1; then
  # Remove any previous mpdc profile entry and install the self-contained Nix flake package
  nix profile remove mpdc-suite 2>/dev/null || true
  nix profile remove mpdc 2>/dev/null || true
  nix profile add .
  echo "==> Installed via Nix profile! You can now run 'mpdc-code', 'mpdc code', 'mpdc-tui', and 'mpdc-gui' in any folder."
else
  cargo install --path gui/mpdc-gui
  pip install -e .
  echo "==> Installed via Cargo + Pip! You can now run 'mpdc-code', 'mpdc code', 'mpdc-tui', and 'mpdc-gui' in any folder."
fi
