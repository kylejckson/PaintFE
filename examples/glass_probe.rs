//! Local CPU probe for glass frame painting, excluding native/GPU presentation.
use paintfe::ui::glass::{self, GlassSettings};
use std::time::Instant;
fn main() {
    for enabled in [false, true] {
        let ctx = egui::Context::default();
        let settings = GlassSettings {
            enabled,
            desktop: true,
            desktop_blur: false,
            ..Default::default()
        };
        let mut samples = Vec::new();
        let mut uploads = 0;
        for frame in 0..18 {
            let begin = Instant::now();
            let output = ctx.run_ui(
                egui::RawInput {
                    time: Some(frame as f64 * 0.2),
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 800.),
                    )),
                    ..Default::default()
                },
                |ui| {
                    glass::configure(ui.ctx(), &settings, false);
                    egui::Window::new("Glass probe")
                        .fixed_pos(egui::pos2(180., 140.))
                        .show(ui.ctx(), |ui| {
                            ui.label("Surface with readable controls");
                            let _ = ui.button("Button");
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.label("Nested group");
                            });
                        });
                },
            );
            let _ = ctx.tessellate(output.shapes, output.pixels_per_point);
            if frame > 2 {
                samples.push(begin.elapsed().as_secs_f64() * 1000.);
                uploads += output.textures_delta.set.len();
            }
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "enabled={enabled} median_cpu_ms={:.3} p95_cpu_ms={:.3} texture_updates={uploads}",
            samples[samples.len() / 2],
            samples[(samples.len() * 95 / 100).min(samples.len() - 1)]
        );
    }
}
