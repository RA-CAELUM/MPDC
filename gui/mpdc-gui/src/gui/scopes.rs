//! `[OSC.00]` 3-Channel Synchronized Oscilloscope Bank and trace visibility strip.

use crate::gui::{
    theme::{ACCENT_CYAN, ACCENT_STEEL, BG_PANEL, STATUS_ERR, STATUS_OK, STATUS_WARN, TEXT_MUTED},
    MpdcWorkbenchApp,
};
use eframe::egui::{self, Color32, RichText, TextStyle};
use egui_plot::{Legend, Line, Plot, PlotPoints};

impl MpdcWorkbenchApp {
    pub fn render_oscilloscope_bank(&mut self, ui: &mut egui::Ui, height: f32) {
        egui::Frame::group(ui.style())
            .fill(BG_PANEL)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("[OSC.00] CHANNELS:").color(ACCENT_CYAN).strong());
                    ui.checkbox(&mut self.show_trace_cluster_tps, "SUM_TPS");
                    ui.checkbox(&mut self.show_trace_node0_tps, "NODE_0");
                    ui.checkbox(&mut self.show_trace_node1_tps, "NODE_1");
                    ui.checkbox(&mut self.show_trace_node2_tps, "NODE_2");
                    ui.separator();
                    ui.checkbox(&mut self.show_trace_pass1, "PASS@1");
                    ui.checkbox(&mut self.show_trace_psr, "PSR%");
                    ui.checkbox(&mut self.show_trace_lift, "LIFT%");
                    ui.separator();
                    ui.checkbox(&mut self.show_trace_ppl, "PPLx10");
                    ui.checkbox(&mut self.show_trace_ttft, "TTFT(ms)");
                    ui.checkbox(&mut self.show_trace_itl, "ITL(ms)");
                    ui.separator();
                    ui.label(RichText::new("WIN").color(TEXT_MUTED));
                    ui.add(egui::DragValue::new(&mut self.scope_window).range(15..=200));
                    if ui.small_button("[CLR]").clicked() {
                        self.tps_cluster_hist.clear();
                        self.tps_node0_hist.clear();
                        self.tps_node1_hist.clear();
                        self.tps_node2_hist.clear();
                        self.pass1_hist.clear();
                        self.psr_hist.clear();
                        self.lift_hist.clear();
                        self.ppl_hist.clear();
                        self.ttft_hist.clear();
                        self.itl_hist.clear();
                    }
                });

                let plot_h = (height - 32.0).max(95.0);
                ui.columns(3, |cols| {
                    Plot::new("osc_throughput")
                        .height(plot_h)
                        .legend(Legend::default().text_style(TextStyle::Small))
                        .allow_zoom(false)
                        .allow_drag(false)
                        .show(&mut cols[0], |plot_ui| {
                            if self.show_trace_cluster_tps {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.tps_cluster_hist.clone()))
                                        .color(STATUS_OK)
                                        .width(2.0)
                                        .name("CLUSTER tok/s"),
                                );
                            }
                            if self.show_trace_node0_tps {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.tps_node0_hist.clone()))
                                        .color(ACCENT_CYAN)
                                        .width(1.3)
                                        .name("rtx5070-laptop"),
                                );
                            }
                            if self.show_trace_node1_tps {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.tps_node1_hist.clone()))
                                        .color(STATUS_WARN)
                                        .width(1.3)
                                        .name("ros2-field-deck"),
                                );
                            }
                            if self.show_trace_node2_tps {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.tps_node2_hist.clone()))
                                        .color(Color32::from_rgb(175, 130, 255))
                                        .width(1.3)
                                        .name("workstation-sim"),
                                );
                            }
                        });

                    Plot::new("osc_self_improve")
                        .height(plot_h)
                        .legend(Legend::default().text_style(TextStyle::Small))
                        .allow_zoom(false)
                        .allow_drag(false)
                        .show(&mut cols[1], |plot_ui| {
                            if self.show_trace_pass1 {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.pass1_hist.clone()))
                                        .color(STATUS_OK)
                                        .width(2.0)
                                        .name("Pass@1 %"),
                                );
                            }
                            if self.show_trace_psr {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.psr_hist.clone()))
                                        .color(ACCENT_STEEL)
                                        .width(2.0)
                                        .name("Patch Survival %"),
                                );
                            }
                            if self.show_trace_lift {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.lift_hist.clone()))
                                        .color(STATUS_WARN)
                                        .width(1.5)
                                        .name("Skill Lift %"),
                                );
                            }
                        });

                    Plot::new("osc_latency_ppl")
                        .height(plot_h)
                        .legend(Legend::default().text_style(TextStyle::Small))
                        .allow_zoom(false)
                        .allow_drag(false)
                        .show(&mut cols[2], |plot_ui| {
                            if self.show_trace_ttft {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.ttft_hist.clone()))
                                        .color(ACCENT_CYAN)
                                        .width(1.8)
                                        .name("TTFT (ms)"),
                                );
                            }
                            if self.show_trace_itl {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.itl_hist.clone()))
                                        .color(STATUS_WARN)
                                        .width(1.8)
                                        .name("ITL (ms/tok)"),
                                );
                            }
                            if self.show_trace_ppl {
                                plot_ui.line(
                                    Line::new(PlotPoints::from(self.ppl_hist.clone()))
                                        .color(STATUS_ERR)
                                        .width(1.8)
                                        .name("Holdout PPL (x10)"),
                                );
                            }
                        });
                });
            });
    }
}
