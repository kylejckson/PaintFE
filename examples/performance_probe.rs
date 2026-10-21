//! Local diagnostic microbenchmarks, not application/GPU performance claims.
//! Run an optimized build, with no compiler running during measurement.
use std::time::Instant;

fn summary(mut values: Vec<f64>) -> serde_json::Value {
    values.sort_by(f64::total_cmp);
    serde_json::json!({"n": values.len(), "p50_ms": values[values.len()/2],
        "p95_ms": values[(values.len()-1)*95/100], "max_ms": values[values.len()-1]})
}

fn font_probe() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for mode in [
        "egui_default",
        "latin",
        "bundled_scripts",
        "with_system_cjk",
    ] {
        let mut samples = Vec::new();
        for _ in 0..8 {
            let ctx = egui::Context::default();
            let mut fonts = egui::FontDefinitions::default();
            if mode != "egui_default" {
                for (name, data) in [
                    (
                        "dm",
                        include_bytes!("../assets/fonts/DMSans-Regular.ttf").as_slice(),
                    ),
                    (
                        "noto",
                        include_bytes!("../assets/fonts/NotoSans-Regular.ttf").as_slice(),
                    ),
                    (
                        "mono",
                        include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf").as_slice(),
                    ),
                    (
                        "medium",
                        include_bytes!("../assets/fonts/DMSans-Medium.ttf").as_slice(),
                    ),
                ] {
                    fonts
                        .font_data
                        .insert(name.into(), egui::FontData::from_static(data).into());
                }
            }
            if matches!(mode, "bundled_scripts" | "with_system_cjk") {
                for (name, data) in [
                    (
                        "cjk",
                        include_bytes!("../assets/fonts/NotoSansCJK-Regular.otf").as_slice(),
                    ),
                    (
                        "thai",
                        include_bytes!("../assets/fonts/NotoSansThai.ttf").as_slice(),
                    ),
                    (
                        "arabic",
                        include_bytes!("../assets/fonts/NotoSansArabic.ttf").as_slice(),
                    ),
                ] {
                    fonts
                        .font_data
                        .insert(name.into(), egui::FontData::from_static(data).into());
                }
            }
            if mode == "with_system_cjk"
                && let Ok(data) = std::fs::read("C:/Windows/Fonts/msyh.ttc")
            {
                fonts
                    .font_data
                    .insert("system_cjk".into(), egui::FontData::from_owned(data).into());
            }
            let start = Instant::now();
            ctx.set_fonts(fonts);
            let _ = ctx.run_ui(Default::default(), |ui| {
                ui.label("Font registration probe");
            });
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        results
            .push(serde_json::json!({"case": mode, "registration_and_first_ui": summary(samples)}));
    }
    results
}

fn main() {
    let mut results = Vec::new();
    for dark in [false, true] {
        for scale in [1.0, 1.25] {
            for width in [180.0, 600.0] {
                for kind in ["empty", "numeric", "standard", "gradient"] {
                    let ctx = egui::Context::default();
                    let theme = if dark {
                        paintfe::theme::Theme::dark()
                    } else {
                        paintfe::theme::Theme::light()
                    };
                    theme.apply(&ctx);
                    ctx.set_zoom_factor(scale);
                    let mut values = [128.0_f32; 24];
                    let mut ui_times = Vec::new();
                    let mut tess_times = Vec::new();
                    let mut shape_count = 0;
                    for frame in 0..100 {
                        let start = Instant::now();
                        let output = ctx.run_ui(
                            egui::RawInput {
                                screen_rect: Some(egui::Rect::from_min_size(
                                    egui::Pos2::ZERO,
                                    egui::vec2(900.0, 1800.0),
                                )),
                                time: Some(frame as f64 / 60.0),
                                ..Default::default()
                            },
                            |ui| {
                                ui.set_width(width);
                                for (i, value) in values.iter_mut().enumerate() {
                                    ui.push_id(i, |ui| {
                                        ui.set_width(width);
                                        match kind {
                                            "numeric" => {
                                                ui.add(
                                                    paintfe::ui::controls::Numeric::new(value)
                                                        .id_salt(i)
                                                        .range(0.0..=255.0),
                                                );
                                            }
                                            "standard" => {
                                                ui.add(
                                                    paintfe::ui::controls::Slider::new(
                                                        value,
                                                        0.0..=255.0,
                                                    )
                                                    .id_salt(i)
                                                    .steppers(true)
                                                    .reset_to(128.0),
                                                );
                                            }
                                            "gradient" => {
                                                ui.add(
                                                    paintfe::ui::controls::GradientSlider::new(
                                                        value,
                                                        255.0,
                                                        |t| {
                                                            egui::Color32::from_rgba_unmultiplied(
                                                                255,
                                                                100,
                                                                20,
                                                                (t * 255.0) as u8,
                                                            )
                                                        },
                                                    )
                                                    .id_salt(i)
                                                    .alpha(true)
                                                    .theme(theme.clone()),
                                                );
                                            }
                                            _ => {
                                                ui.allocate_space(egui::vec2(width, 24.0));
                                            }
                                        }
                                    });
                                }
                            },
                        );
                        let ui_ms = start.elapsed().as_secs_f64() * 1000.0;
                        shape_count = output.shapes.len();
                        let start = Instant::now();
                        let _ = ctx.tessellate(output.shapes, output.pixels_per_point);
                        if frame >= 20 {
                            ui_times.push(ui_ms);
                            tess_times.push(start.elapsed().as_secs_f64() * 1000.0);
                        }
                    }
                    results.push(
                        serde_json::json!({"kind": kind, "dark": dark, "scale": scale,
                        "width": width, "controls": 24, "shapes": shape_count,
                        "ui": summary(ui_times), "tessellation": summary(tess_times)}),
                    );
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"controls": results, "fonts": font_probe()})
    );
}
