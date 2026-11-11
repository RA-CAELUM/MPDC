{ config, lib, pkgs, ... }:

with lib;
let
  cfg = config.services.mpdc;
in
{
  options.services.mpdc = {
    enable = mkEnableOption "MPDC Local LLM Datacenter Gateway & Self-Improving Stack";

    port = mkOption {
      type = types.port;
      default = 8000;
      description = "Port for the MPDC OpenAI-compatible Gateway.";
    };

    backendUrl = mkOption {
      type = types.str;
      default = "http://127.0.0.1:8001/v1";
      description = "Upstream vLLM / SGLang inference engine endpoint.";
    };

    baseModel = mkOption {
      type = types.str;
      default = "Qwen/Qwen3-8B-AWQ";
      description = "Primary Qwen3 model backbone served by the inference engine.";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/mpdc";
      description = "State directory for Episodic Skill DB, datasets, telemetry, and LoRA adapters.";
    };

    enableOvernightTraining = mkOption {
      type = types.bool;
      default = false;
      description = "Whether to run automated QLoRA sleep-cycle training & gated evaluation overnight.";
    };

    trainingSchedule = mkOption {
      type = types.str;
      default = "*-*-* 03:00:00";
      description = "Systemd OnCalendar schedule for overnight QLoRA fine-tuning.";
    };
  };

  config = mkIf cfg.enable {
    systemd.tmpfiles.rules = [
      "d ${cfg.dataDir} 0750 root root -"
      "d ${cfg.dataDir}/datasets 0750 root root -"
      "d ${cfg.dataDir}/adapters 0750 root root -"
      "d ${cfg.dataDir}/eval 0750 root root -"
    ];

    systemd.services.mpdc-gateway = {
      description = "MPDC OpenAI-Compatible Multi-Agent Gateway";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      environment = {
        MPDC_DATA_DIR = cfg.dataDir;
        MPDC_BACKEND_URL = cfg.backendUrl;
        MPDC_BASE_MODEL = cfg.baseModel;
      };
      serviceConfig = {
        ExecStart = "${pkgs.python312}/bin/python -m mpdc.cli serve --host 0.0.0.0 --port ${toString cfg.port}";
        Restart = "on-failure";
        WorkingDirectory = cfg.dataDir;
      };
    };

    systemd.services.mpdc-sleep-train = mkIf cfg.enableOvernightTraining {
      description = "MPDC Overnight QLoRA Self-Improvement & Gated Eval";
      serviceConfig = {
        Type = "oneshot";
        Environment = [
          "MPDC_DATA_DIR=${cfg.dataDir}"
          "MPDC_BACKEND_URL=${cfg.backendUrl}"
          "MPDC_BASE_MODEL=${cfg.baseModel}"
        ];
        ExecStart = "${pkgs.python312}/bin/python -m mpdc.cli train --now";
      };
    };

    systemd.timers.mpdc-sleep-train = mkIf cfg.enableOvernightTraining {
      description = "Timer for MPDC Overnight QLoRA Training";
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = cfg.trainingSchedule;
        Persistent = true;
      };
    };
  };
}
