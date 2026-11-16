use crate::{
    config::settings::AppSettings,
    theme::{Theme, ThemePreset},
};
use egui::{Color32, Context};

#[test]
fn theme_finishes_roundtrip_reset_and_do_not_change_permissions() {
    let mut settings = AppSettings::default();
    settings.glass.enabled = true;
    settings.glass.grain = 0.35;
    settings.ui_polish.retro_bevel = true;
    settings.ui_polish.solid_headers = true;
    settings.canvas_grid_opacity = 0.12;
    let mut restored = AppSettings::default();
    let permissions = serde_json::to_value(&restored.automation).unwrap();
    restored.import_theme_from_string(&settings.export_theme_to_string());
    assert_eq!(restored.glass.grain, 0.35);
    assert!(restored.ui_polish.retro_bevel);
    assert!(restored.ui_polish.solid_headers);
    assert_eq!(restored.canvas_grid_opacity, 0.12);
    assert_eq!(
        serde_json::to_value(&restored.automation).unwrap(),
        permissions
    );
    restored.reset_signal_theme();
    assert!(!restored.ui_polish.retro_bevel);
    assert!(!restored.ui_polish.solid_headers);
    assert_eq!(restored.glass.grain, 0.0);
    assert!(!restored.glass.enabled);
}

#[test]
fn retro_headers_keep_text_contrast_and_uniform_fills() {
    crate::i18n::init();
    let ctx = Context::default();
    let mut fonts = egui::FontDefinitions::default();
    fonts.families.insert(
        egui::FontFamily::Name("WidgetTitle".into()),
        fonts.families[&egui::FontFamily::Proportional].clone(),
    );
    ctx.set_fonts(fonts);
    let mut accents = ThemePreset::Signal.accent_colors();
    accents.light_normal = Color32::from_rgb(0, 0, 128);
    let mut theme = Theme::light_with_accent(ThemePreset::Custom, accents);
    theme.polish.solid_headers = true;
    theme.apply(&ctx);
    let output = ctx.run_ui(egui::RawInput::default(), |ui| {
        super::dialogs::core::paint_dialog_header(
            ui,
            &super::dialogs::core::DialogColors::from_ctx(&ctx),
            "",
            "Classic header",
        );
    });
    assert!(output.shapes.iter().any(|shape| matches!(&shape.shape, egui::Shape::Text(text) if text.galley.job.text.contains("Classic header") && text.galley.job.sections.iter().any(|section| section.format.color == Color32::WHITE))));
    assert!(output.shapes.iter().any(|shape| matches!(&shape.shape, egui::Shape::Rect(rect) if rect.fill == accents.light_normal && rect.brush.is_none())));
}

#[test]
fn bevels_are_opt_in_and_do_not_change_control_geometry_or_values() {
    crate::i18n::init();
    for dark in [false, true] {
        let mut rectangles = Vec::new();
        for retro in [false, true] {
            let ctx = Context::default();
            let mut theme = if dark {
                Theme::dark_with_accent(ThemePreset::Signal, ThemePreset::Signal.accent_colors())
            } else {
                Theme::light_with_accent(ThemePreset::Signal, ThemePreset::Signal.accent_colors())
            };
            theme.polish.retro_bevel = retro;
            theme.apply(&ctx);
            assert_eq!(ctx.global_style().visuals.retro_bevel, retro);
            let mut value = 42.;
            let mut text = "Example".to_owned();
            let mut selected = 0;
            let mut rect = egui::Rect::NOTHING;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                rect = ui.button("A button").rect;
                ui.add(crate::ui::controls::Numeric::new(&mut value));
                ui.add(crate::ui::controls::Slider::new(&mut value, 0.0..=100.0));
                ui.text_edit_singleline(&mut text);
                egui::ComboBox::from_id_salt("test_combo").show_index(
                    ui,
                    &mut selected,
                    2,
                    |index| index.to_string(),
                );
            });
            assert_eq!(value, 42.);
            assert_eq!(text, "Example");
            rectangles.push(rect);
        }
        assert_eq!(rectangles[0], rectangles[1]);
    }
}

#[test]
fn raised_and_sunken_bevels_reverse_edges_and_keep_alpha() {
    let ctx = Context::default();
    let mut shapes = Vec::new();
    for sunken in [false, true] {
        let output = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.painter().rect_bevel(
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(60., 24.)),
                sunken,
                Color32::from_rgba_unmultiplied(192, 192, 192, 128),
            );
        });
        let edges: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|shape| match &shape.shape {
                egui::Shape::LineSegment { stroke, .. } => Some(stroke.color),
                _ => None,
            })
            .collect();
        assert_eq!(edges.len(), 8);
        assert!(edges.iter().all(|colour| colour.a() == 128));
        shapes.push(edges);
    }
    assert_ne!(shapes[0][0], shapes[1][0]);
    assert_eq!(shapes[0][0], shapes[1][2]);
}
