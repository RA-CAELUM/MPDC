//! Industrial Telemetry & Profiler Workbench styling, custom painter LEDs, and segmented meters.

use eframe::egui::{self, Color32, FontFamily, FontId, Pos2, Rect, RichText, Sense, Stroke, TextStyle, Vec2};

pub const BG_ROOT: Color32 = Color32::from_rgb(13, 15, 18);
pub const BG_PANEL: Color32 = Color32::from_rgb(19, 22, 27);
pub const BG_WELL: Color32 = Color32::from_rgb(10, 12, 15);
pub const BG_HEADER: Color32 = Color32::from_rgb(26, 30, 37);
pub const BORDER_CRISP: Color32 = Color32::from_rgb(44, 50, 62);
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(216, 222, 233);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(130, 142, 160);
pub const ACCENT_STEEL: Color32 = Color32::from_rgb(56, 132, 210);
pub const ACCENT_CYAN: Color32 = Color32::from_rgb(0, 185, 215);
pub const STATUS_OK: Color32 = Color32::from_rgb(46, 168, 86);
pub const STATUS_WARN: Color32 = Color32::from_rgb(218, 150, 35);
pub const STATUS_ERR: Color32 = Color32::from_rgb(215, 60, 60);

pub fn apply_workbench_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(4.0, 3.0);
    style.spacing.window_margin = egui::Margin::same(3.0);
    style.spacing.button_padding = Vec2::new(4.0, 1.5);
    style.spacing.interact_size = Vec2::new(28.0, 16.0);
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = BG_ROOT;
    style.visuals.window_fill = BG_PANEL;
    style.visuals.extreme_bg_color = BG_WELL;
    style.visuals.faint_bg_color = Color32::from_rgb(22, 26, 32);
    style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER_CRISP);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(28, 33, 41);
    style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(52, 60, 74));
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(40, 52, 70);
    style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_STEEL);
    style.visuals.widgets.active.bg_fill = ACCENT_STEEL;
    style.visuals.selection.bg_fill = Color32::from_rgb(38, 78, 128);

    style.text_styles = [
        (TextStyle::Heading, FontId::new(12.5, FontFamily::Monospace)),
        (TextStyle::Body, FontId::new(11.0, FontFamily::Monospace)),
        (TextStyle::Monospace, FontId::new(10.5, FontFamily::Monospace)),
        (TextStyle::Button, FontId::new(10.5, FontFamily::Monospace)),
        (TextStyle::Small, FontId::new(9.5, FontFamily::Monospace)),
    ]
    .into();
    ctx.set_style(style);
}

pub fn draw_led(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 10.0), Sense::hover());
    if ui.is_rect_visible(rect) {
        let sq = Rect::from_center_size(rect.center(), Vec2::new(7.0, 7.0));
        ui.painter().rect_filled(sq, 1.0, color);
        ui.painter()
            .rect_stroke(sq, 1.0, Stroke::new(1.0, Color32::BLACK));
    }
}

fn lerp_c(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
    )
}

pub fn draw_segmented_meter(
    ui: &mut egui::Ui,
    ratio: f32,
    width: f32,
    height: f32,
    c0: Color32,
    c1: Color32,
    c2: Color32,
    text: &str,
) {
    let clamped = ratio.clamp(0.0, 1.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        p.rect_filled(rect, 1.0, BG_WELL);
        let segs = 22usize;
        let seg_w = (rect.width() - 2.0) / segs as f32;
        let filled = (clamped * segs as f32).ceil() as usize;

        for i in 0..filled.min(segs) {
            let pos = i as f32 / (segs - 1).max(1) as f32;
            let col = if pos < 0.5 {
                lerp_c(c0, c1, pos * 2.0)
            } else {
                lerp_c(c1, c2, (pos - 0.5) * 2.0)
            };
            let r = Rect::from_min_size(
                Pos2::new(rect.left() + 1.0 + i as f32 * seg_w, rect.top() + 1.0),
                Vec2::new((seg_w - 1.0).max(1.0), height - 2.0),
            );
            p.rect_filled(r, 0.0, col);
        }
        p.rect_stroke(rect, 1.0, Stroke::new(1.0, BORDER_CRISP));
        p.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            FontId::new(9.5, FontFamily::Monospace),
            Color32::WHITE,
        );
    }
}

pub fn section_header(ui: &mut egui::Ui, code: &str, title: &str) {
    egui::Frame::none()
        .fill(BG_HEADER)
        .stroke(Stroke::new(1.0, BORDER_CRISP))
        .inner_margin(egui::Margin::symmetric(5.0, 2.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("[{code}]"))
                        .color(ACCENT_CYAN)
                        .strong(),
                );
                ui.label(RichText::new(title).color(TEXT_PRIMARY).strong());
            });
        });
}
