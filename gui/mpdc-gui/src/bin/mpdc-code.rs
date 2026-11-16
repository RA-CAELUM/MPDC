//! Binary entrypoint for `mpdc-code` (Claude-Code-style Interactive Coding, Planning & Spatial Swarm TUI).

fn main() -> std::io::Result<()> {
    mpdc_ui::coder::run_coder_tui_entry()
}
