{
  description = "MPDC: Man Portable Data Center (Multi-Pool Qwen3 + Visual/Spatial Swarm + Rust TUI/GUI)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python312;
        pythonEnv = python.withPackages (ps: with ps; [
          fastapi
          uvicorn
          httpx
          pydantic
          pydantic-settings
          pyyaml
          typer
          rich
          numpy
          openai
          pytest
          pytest-asyncio
        ]);

        guiRuntimeLibs = with pkgs; [
          wayland
          libxkbcommon
          libGL
          vulkan-loader
          libx11
          libxcursor
          libxi
          libxrandr
        ];

        guiLibPath = pkgs.lib.makeLibraryPath guiRuntimeLibs;

        mpdcRustUi = pkgs.rustPlatform.buildRustPackage {
          pname = "mpdc-ui";
          version = "0.2.0";
          src = ./gui/mpdc-gui;
          cargoLock = {
            lockFile = ./gui/mpdc-gui/Cargo.lock;
          };
          nativeBuildInputs = [ pkgs.pkg-config pkgs.makeWrapper ];
          buildInputs = guiRuntimeLibs;
          postInstall = ''
            wrapProgram $out/bin/mpdc-gui \
              --prefix LD_LIBRARY_PATH : "${guiLibPath}"
          '';
        };

        mpdcPythonPkg = python.pkgs.buildPythonApplication {
          pname = "mpdc-cli";
          version = "0.2.0";
          pyproject = true;
          src = ./.;
          build-system = [ python.pkgs.hatchling ];
          dependencies = with python.pkgs; [
            fastapi
            uvicorn
            httpx
            pydantic
            pydantic-settings
            pyyaml
            typer
            rich
            numpy
            openai
          ];
          doCheck = false;
          # Remove duplicate wrapper scripts so mpdcRustUi provides native Rust mpdc-code/mpdc-tui/mpdc-gui binaries
          postInstall = ''
            rm -f $out/bin/mpdc-code $out/bin/mpdc-tui $out/bin/mpdc-gui
            wrapProgram $out/bin/mpdc \
              --prefix PATH : "${mpdcRustUi}/bin" \
              --prefix LD_LIBRARY_PATH : "${guiLibPath}"
          '';
        };

        mpdcSuite = pkgs.symlinkJoin {
          name = "mpdc";
          paths = [
            mpdcPythonPkg
            mpdcRustUi
          ];
        };
      in
      {
        packages = {
          default = mpdcSuite;
          mpdc = mpdcSuite;
          ui = mpdcRustUi;
        };

        devShells.default = pkgs.mkShell {
          name = "mpdc-dev";
          packages = [
            pythonEnv
            mpdcSuite
            pkgs.cargo
            pkgs.rustc
            pkgs.pkg-config
            pkgs.uv
            pkgs.git
            pkgs.jq
            pkgs.sqlite
          ] ++ guiRuntimeLibs;
          LD_LIBRARY_PATH = guiLibPath;
          shellHook = ''
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            export MPDC_CONFIG="$PWD/config/default.yaml"
            echo "=================================================================="
            echo " MPDC // Man Portable Data Center"
            echo " • Claude-Style Coding & Planning TUI : mpdc-code  (or: mpdc code)"
            echo " • btop-Style Telemetry TUI           : mpdc-tui   (or: mpdc top)"
            echo " • Desktop Workbench GUI              : mpdc-gui   (or: mpdc gui)"
            echo " • Multi-Agent Gateway Server         : mpdc serve"
            echo " • Programming + Spatial Model Pools  : mpdc pools"
            echo "=================================================================="
          '';
        };
      })) // {
        nixosModules.default = import ./nix/module.nix;
        nixosModules.mpdc = import ./nix/module.nix;
      };
}
