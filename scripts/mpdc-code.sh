#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO_ROOT/gui/mpdc-gui/target/release/mpdc-code"

if [[ ! -x "$BIN" ]]; then
  nix-shell -p cargo rustc pkg-config --run "cargo build --release --manifest-path \"$REPO_ROOT/gui/mpdc-gui/Cargo.toml\" --bin mpdc-code"
fi

# Execute in the caller's current working directory!
exec "$BIN" "$@"
