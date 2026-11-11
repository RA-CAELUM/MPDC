//! Entrypoint for the Rust `btop++` Terminal UI (`mpdc-tui`).

fn main() -> std::io::Result<()> {
    mpdc_ui::tui::run_tui_entry()
}
