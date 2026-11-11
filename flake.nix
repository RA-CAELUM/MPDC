{
  description = "MPDC: Mini-PC Local LLM Datacenter (Qwen3 + vLLM + Multi-Node Token Switchboard + Self-Improving Pipeline)";

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
          pyyaml
          typer
          rich
          numpy
          pytest
          pytest-asyncio
        ]);

        mpdcPkg = python.pkgs.buildPythonApplication {
          pname = "mpdc";
          version = "0.2.0";
          pyproject = true;
          src = ./.;
          build-system = [ python.pkgs.hatchling ];
          dependencies = with python.pkgs; [
            fastapi
            uvicorn
            httpx
            pydantic
            pyyaml
            typer
            rich
            numpy
          ];
          doCheck = false;
        };

        guiRuntimeLibs = with pkgs; [
          wayland
          libxkbcommon
          libGL
          vulkan-loader
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
        ];
      in
      {
        packages = {
          default = mpdcPkg;
          mpdc = mpdcPkg;
        };

        devShells.default = pkgs.mkShell {
          name = "mpdc-dev";
          packages = [
            pythonEnv
            mpdcPkg
            pkgs.cargo
            pkgs.rustc
            pkgs.pkg-config
            pkgs.uv
            pkgs.git
            pkgs.jq
            pkgs.sqlite
          ] ++ guiRuntimeLibs;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath guiRuntimeLibs;
          shellHook = ''
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            export MPDC_CONFIG="$PWD/config/default.yaml"
            echo "=================================================================="
            echo " MPDC Mission Control DevShell (NixOS -> Ubuntu Portable Stack)"
            echo " Model Family  : Latest Qwen3 (Qwen3-8B / Qwen3-14B / 30B-A3B)"
            echo " CLI btop TUI  : mpdc top"
            echo " Native Rust UI: mpdc gui"
            echo " Gateway Server: mpdc serve --seed-demo"
            echo "=================================================================="
          '';
        };
      })) // {
        nixosModules.default = import ./nix/module.nix;
        nixosModules.mpdc = import ./nix/module.nix;
      };
}
