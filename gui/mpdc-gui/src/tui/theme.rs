//! `assets/mpdc-banner.svg` (`#38bdf8 -> #a78bfa -> #f472b6`) color constants, 3-stop RGB gradients, and meter/sparkline spans for `mpdc-tui`.

use ratatui::{
    style::{Color, Modifier, Style},
    text::Span,
};

// Base surfaces & typography tuned to `assets/mpdc-banner.svg`
pub const BTOP_MAIN_BG: Color = Color::Rgb(11, 13, 20); // #0b0d14
pub const BTOP_MAIN_FG: Color = Color::Rgb(226, 232, 248); // #e2e8f8
pub const BTOP_TITLE: Color = Color::Rgb(245, 243, 255); // #f5f3ff
pub const BTOP_HI_FG: Color = Color::Rgb(244, 114, 182); // #f472b6 (SVG stop 1.0 Pink)
pub const BTOP_SELECTED_BG: Color = Color::Rgb(58, 44, 92); // #3a2c5c
pub const BTOP_SELECTED_FG: Color = Color::Rgb(255, 255, 255); // #ffffff
pub const BTOP_INACTIVE_FG: Color = Color::Rgb(52, 58, 84); // #343a54
pub const BTOP_GRAPH_TEXT: Color = Color::Rgb(148, 156, 190); // #949cbe
pub const BTOP_PROC_MISC: Color = Color::Rgb(56, 189, 248); // #38bdf8 (SVG stop 0.0 Sky)

// Box border accents derived from the 3 SVG stops (#38bdf8, #a78bfa, #f472b6)
pub const BTOP_CPU_BOX: Color = Color::Rgb(56, 189, 248); // #38bdf8 Sky
pub const BTOP_MEM_BOX: Color = Color::Rgb(167, 139, 250); // #a78bfa Violet
pub const BTOP_NET_BOX: Color = Color::Rgb(244, 114, 182); // #f472b6 Pink
pub const BTOP_PROC_BOX: Color = Color::Rgb(112, 164, 249); // Midpoint Sky-Violet
pub const BTOP_DIV_LINE: Color = Color::Rgb(24, 27, 42); // #181b2a

pub type RgbStop = (u8, u8, u8);
pub type Grad3 = (RgbStop, RgbStop, RgbStop);

// Primary MPDC Banner SVG 3-stop linear gradient: #38bdf8 -> #a78bfa -> #f472b6
pub const GRAD_MPDC: Grad3 = ((56, 189, 248), (167, 139, 250), (244, 114, 182));
pub const GRAD_CPU: Grad3 = GRAD_MPDC;
pub const GRAD_TEMP: Grad3 = ((56, 189, 248), (167, 139, 250), (244, 114, 182));
pub const GRAD_MEM: Grad3 = ((56, 189, 248), (167, 139, 250), (244, 114, 182));
pub const GRAD_DL: Grad3 = ((56, 189, 248), (167, 139, 250), (244, 114, 182));
pub const GRAD_UL: Grad3 = ((167, 139, 250), (205, 126, 216), (244, 114, 182));

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
        let col = sample_grad(grad, norm as f32);
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
