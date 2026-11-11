//! Entrypoint for the Rust `egui` Industrial Telemetry Workbench (`mpdc-gui`).

fn main() -> eframe::Result<()> {
    mpdc_ui::gui::run_gui_entry()
}
