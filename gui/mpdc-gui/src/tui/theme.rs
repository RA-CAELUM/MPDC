//! Official `btop++` `default.theme` color constants, 3-stop RGB gradients, and meter/sparkline spans.

use ratatui::{
    style::{Color, Modifier, Style},
    text::Span,
};

pub const BTOP_MAIN_BG: Color = Color::Rgb(10, 12, 16);
pub const BTOP_MAIN_FG: Color = Color::Rgb(204, 220, 255); // #ccdcff
pub const BTOP_TITLE: Color = Color::Rgb(238, 235, 255); // #eeebff
pub const BTOP_HI_FG: Color = Color::Rgb(255, 101, 64); // #ff6540
pub const BTOP_SELECTED_BG: Color = Color::Rgb(106, 47, 47); // #6a2f2f
pub const BTOP_SELECTED_FG: Color = Color::Rgb(255, 255, 255); // #ffffff
pub const BTOP_INACTIVE_FG: Color = Color::Rgb(64, 64, 64); // #404040
pub const BTOP_GRAPH_TEXT: Color = Color::Rgb(144, 144, 144); // #909090
pub const BTOP_PROC_MISC: Color = Color::Rgb(13, 231, 86); // #0de756

pub const BTOP_CPU_BOX: Color = Color::Rgb(85, 123, 91); // #557b5b
pub const BTOP_MEM_BOX: Color = Color::Rgb(138, 136, 88); // #8a8858
pub const BTOP_NET_BOX: Color = Color::Rgb(130, 89, 120); // #825978
pub const BTOP_PROC_BOX: Color = Color::Rgb(73, 109, 138); // #496d8a
pub const BTOP_DIV_LINE: Color = Color::Rgb(48, 48, 48); // #303030

pub type RgbStop = (u8, u8, u8);
pub type Grad3 = (RgbStop, RgbStop, RgbStop);

pub const GRAD_CPU: Grad3 = ((80, 247, 108), (242, 226, 70), (250, 47, 47));
pub const GRAD_TEMP: Grad3 = ((72, 151, 212), (84, 116, 232), (255, 64, 182));
pub const GRAD_MEM: Grad3 = ((91, 114, 250), (124, 159, 254), (217, 248, 255));
pub const GRAD_DL: Grad3 = ((41, 69, 150), (53, 121, 198), (163, 216, 255));
pub const GRAD_UL: Grad3 = ((114, 28, 101), (177, 72, 163), (255, 118, 238));

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t.clamp(0.0, 1.0)).round() as u8
}

pub fn sample_grad(g: Grad3, pos: f32) -> Color {
    let p = pos.clamp(0.0, 1.0);
    let (r, gr, b) = if p < 0.5 {
        let t = p * 2.0;
        (
            lerp_u8(g.0 .0, g.1 .0, t),
            lerp_u8(g.0 .1, g.1 .1, t),
            lerp_u8(g.0 .2, g.1 .2, t),
        )
    } else {
        let t = (p - 0.5) * 2.0;
        (
            lerp_u8(g.1 .0, g.2 .0, t),
            lerp_u8(g.1 .1, g.2 .1, t),
            lerp_u8(g.1 .2, g.2 .2, t),
        )
    };
    Color::Rgb(r, gr, b)
}

pub fn btop_meter_spans(ratio: f64, width: usize, grad: Grad3) -> Vec<Span<'static>> {
    let clamped = ratio.clamp(0.0, 1.0);
    let filled = (clamped * width as f64).round() as usize;
    let mut spans = Vec::with_capacity(width);
    for i in 0..width {
        let pos = i as f32 / (width.saturating_sub(1).max(1)) as f32;
        if i < filled {
            spans.push(Span::styled("■", Style::default().fg(sample_grad(grad, pos))));
        } else {
            spans.push(Span::styled("·", Style::default().fg(BTOP_INACTIVE_FG)));
        }
    }
    spans
}

pub fn btop_spark_spans(values: &[f64], width: usize, grad: Grad3) -> Vec<Span<'static>> {
    const BARS: [char; 8] = ['⣀', '⣄', '⣤', '⣦', '⣶', '⣷', '⣿', '█'];
    if values.is_empty() {
        return vec![];
    }
    let start = values.len().saturating_sub(width);
    let tail = &values[start..];
    let lo = tail.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = tail.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-6);

    let mut out = Vec::with_capacity(width);
    for &v in tail {
        let norm = ((v - lo) / span).clamp(0.0, 1.0);
        let idx = ((norm * (BARS.len() - 1) as f64).round() as usize).min(BARS.len() - 1);
        let col = sample_grad(grad, 0.20 + 0.80 * norm as f32);
        out.push(Span::styled(
            BARS[idx].to_string(),
            Style::default().fg(col).add_modifier(Modifier::BOLD),
        ));
    }
    out
}

pub fn btop_key_label(
    prefix: &'static str,
    key: &'static str,
    suffix: &'static str,
) -> Vec<Span<'static>> {
    vec![
        Span::styled(prefix, Style::default().fg(BTOP_GRAPH_TEXT)),
        Span::styled(
            key,
            Style::default()
                .fg(BTOP_HI_FG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(suffix, Style::default().fg(BTOP_MAIN_FG)),
    ]
}
