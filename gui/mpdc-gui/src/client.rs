//! Non-blocking HTTP polling & RPC dispatcher shared by `mpdc-tui` and `mpdc-gui`.

use crate::models::OverviewPayload;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;

pub enum BgResult {
    Overview(Result<OverviewPayload, String>),
    ActionLog(String),
    GeneratedConfig(String),
}

pub fn spawn_poll_overview(gateway_url: &str, tx: Sender<BgResult>) {
    let url = format!("{}/v1/mpdc/overview", gateway_url.trim_end_matches('/'));
    thread::spawn(move || {
        let resp = ureq::get(&url).timeout(Duration::from_millis(750)).call();
        match resp {
            Ok(r) => match r.into_json::<OverviewPayload>() {
                Ok(payload) => {
                    let _ = tx.send(BgResult::Overview(Ok(payload)));
                }
                Err(e) => {
                    let _ = tx.send(BgResult::Overview(Err(format!("JSON: {e}"))));
                }
            },
            Err(e) => {
                let _ = tx.send(BgResult::Overview(Err(format!("{e}"))));
            }
        }
    });
}

pub fn spawn_post_action(gateway_url: &str, path: &str, body: serde_json::Value, tx: Sender<BgResult>) {
    let url = format!("{}{}", gateway_url.trim_end_matches('/'), path);
    let path_owned = path.to_string();
    thread::spawn(move || {
        let res = ureq::post(&url)
            .timeout(Duration::from_secs(10))
            .send_json(body);
        match res {
            Ok(r) => {
                if let Ok(val) = r.into_json::<serde_json::Value>() {
                    if let Some(cfg) = val.get("opencode_config") {
                        if let Ok(pretty) = serde_json::to_string_pretty(cfg) {
                            let _ = tx.send(BgResult::GeneratedConfig(pretty));
                        }
                    }
                    let _ = tx.send(BgResult::ActionLog(format!("POST {path_owned} -> 200 OK")));
                }
            }
            Err(e) => {
                let _ = tx.send(BgResult::ActionLog(format!("ERR {path_owned}: {e}")));
            }
        }
    });
}

pub fn format_client_opencode_json(dc_host: &str, port: u16, node_id: &str, mcp: bool) -> String {
    format!(
        r#"{{
  "$schema": "https://opencode.ai/config.json",
  "provider": {{
    "mpdc": {{
      "npm": "@ai-sdk/openai-compatible",
      "name": "MPDC [{dc_host}:{port}] -> {node_id}",
      "options": {{
        "baseURL": "http://{dc_host}:{port}/v1",
        "apiKey": "mpdc-local",
        "headers": {{ "X-MPDC-Client-Node": "{node_id}" }}
      }},
      "models": {{
        "qwen3-mpdc": {{ "name": "Qwen3 (/think + Skills + LoRA)", "reasoning": true, "tool_call": true }},
        "qwen3-mpdc-fast": {{ "name": "Qwen3 (/no_think Fast Coder)", "reasoning": false, "tool_call": true }}
      }}
    }}
  }},
  "model": "mpdc/qwen3-mpdc",
  "small_model": "mpdc/qwen3-mpdc-fast",
  "mcp": {{ "mpdc-agents": {{ "type": "remote", "url": "http://{dc_host}:{port}/mcp", "enabled": {mcp} }} }}
}}"#
    )
}
