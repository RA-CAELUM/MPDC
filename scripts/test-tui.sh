#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../gui/mpdc-gui"

if [[ -x "./target/release/mpdc-tui" ]]; then
  exec ./target/release/mpdc-tui "$@"
else
  exec nix-shell -p cargo rustc pkg-config --run "cargo run --release --bin mpdc-tui -- $*"
fi
