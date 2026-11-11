#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../gui/mpdc-gui"

exec nix-shell -p \
  wayland \
  libxkbcommon \
  libGL \
  vulkan-loader \
  libx11 \
  libxcursor \
  libxi \
  libxrandr \
  --run 'export LD_LIBRARY_PATH="$(nix-build "<nixpkgs>" -A wayland --no-out-link)/lib:$(nix-build "<nixpkgs>" -A libxkbcommon --no-out-link)/lib:$(nix-build "<nixpkgs>" -A libGL --no-out-link)/lib:$(nix-build "<nixpkgs>" -A vulkan-loader --no-out-link)/lib:$(nix-build "<nixpkgs>" -A libx11 --no-out-link)/lib:$(nix-build "<nixpkgs>" -A libxcursor --no-out-link)/lib:$(nix-build "<nixpkgs>" -A libxi --no-out-link)/lib:$(nix-build "<nixpkgs>" -A libxrandr --no-out-link)/lib:${LD_LIBRARY_PATH:-}"; exec ./target/release/mpdc-gui "$@"'
