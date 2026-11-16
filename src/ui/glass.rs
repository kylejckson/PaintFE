//! Lightweight translucent UI fills. Document pixels are never sampled or blurred.
use egui::{Color32, Context, Rect, Shape};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Surface {
    pub enabled: bool,
    pub opacity: f32,
    pub tint: Option<[u8; 3]>,
    pub border: f32,
}
impl Default for Surface {
    fn default() -> Self {
        Self {
            enabled: true,
            opacity: 0.78,
            tint: None,
            border: 0.7,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GlassSettings {
    pub enabled: bool,
    pub desktop: bool,
    pub desktop_blur: bool,
    pub workspace_opacity: f32,
    pub panels: Surface,
    pub headers: Surface,
    pub menus: Surface,
    pub toolbars: Surface,
    pub buttons: Surface,
}
impl Default for GlassSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            desktop: true,
            desktop_blur: false,
            workspace_opacity: 0.18,
            panels: Surface::default(),
            headers: Surface {
                opacity: 1.,
                ..Default::default()
            },
            menus: Surface::default(),
            toolbars: Surface::default(),
            buttons: Surface {
                opacity: 0.8,
                ..Default::default()
            },
        }
    }
}
impl GlassSettings {
    pub fn sanitize(&mut self) {
        fn limit(v: &mut f32, default: f32, max: f32) {
            *v = if v.is_finite() {
                v.clamp(0., max)
            } else {
                default
            };
        }
        limit(&mut self.workspace_opacity, 0.18, 1.);
        for s in [
            &mut self.panels,
            &mut self.headers,
            &mut self.menus,
            &mut self.toolbars,
            &mut self.buttons,
        ] {
            limit(&mut s.opacity, 0.78, 1.);
            limit(&mut s.border, 0.7, 2.);
        }
    }
}
#[derive(Clone)]
struct Materials {
    settings: GlassSettings,
    dark: bool,
}
pub fn settings(ctx: &Context) -> GlassSettings {
    ctx.data(|d| d.get_temp::<std::sync::Arc<Materials>>(egui::Id::new("glass_materials")))
        .map(|m| m.settings.clone())
        .unwrap_or_default()
}
pub fn configure(ctx: &Context, value: &GlassSettings, dark: bool) {
    let mut value = value.clone();
    value.sanitize();
    let current =
        ctx.data(|d| d.get_temp::<std::sync::Arc<Materials>>(egui::Id::new("glass_materials")));
    let changed = current
        .as_ref()
        .is_none_or(|old| old.settings != value || old.dark != dark);
    if changed {
        ctx.data_mut(|d| {
            d.insert_temp(
                egui::Id::new("glass_materials"),
                std::sync::Arc::new(Materials {
                    settings: value.clone(),
                    dark,
                }),
            );
            let key = egui::Id::new("frame_paint_hook");
            if value.enabled {
                d.insert_temp(
                    key,
                    egui::containers::frame::FramePaintHook(std::sync::Arc::new(material)),
                );
            } else {
                d.remove::<egui::containers::frame::FramePaintHook>(key);
            }
        });
    }
    configure_buttons(ctx, &value, changed);
    #[cfg(not(target_arch = "wasm32"))]
    crate::services::glass::configure(&value);
}
fn material(
    ctx: &Context,
    frame: &egui::Frame,
    content: Rect,
    kind: Option<egui::UiKind>,
) -> Option<Shape> {
    let m =
        ctx.data(|d| d.get_temp::<std::sync::Arc<Materials>>(egui::Id::new("glass_materials")))?;
    if !m.settings.enabled || frame.fill.a() == 0 {
        return None;
    }

    let surface = match kind {
        Some(egui::UiKind::CentralPanel) => return None,
        Some(egui::UiKind::TopPanel | egui::UiKind::BottomPanel) => &m.settings.toolbars,
        Some(
            egui::UiKind::Menu | egui::UiKind::Popup | egui::UiKind::Tooltip | egui::UiKind::Picker,
        ) => &m.settings.menus,
        _ => &m.settings.panels,
    };
    material_surface(frame, content, surface)
}
fn material_surface(frame: &egui::Frame, content: Rect, surface: &Surface) -> Option<Shape> {
    if !surface.enabled {
        return None;
    }
    let mut f = *frame;
    let original = frame.fill.to_srgba_unmultiplied();
    let rgb = surface
        .tint
        .unwrap_or([original[0], original[1], original[2]]);
    // A tint replaces RGB only. Preserve the caller's alpha, so a material
    // cannot turn a deliberately transparent toolbar or panel opaque again.
    let alpha = original[3];
    f.fill = Color32::from_rgba_unmultiplied(
        rgb[0],
        rgb[1],
        rgb[2],
        (f32::from(alpha) * surface.opacity) as u8,
    );
    f.stroke.width *= surface.border;
    Some(f.paint(content))
}
pub fn workspace(ctx: &Context, color: Color32) -> Color32 {
    let s = settings(ctx);
    if s.enabled && s.desktop {
        let rgba = color.to_srgba_unmultiplied();
        Color32::from_rgba_unmultiplied(
            rgba[0],
            rgba[1],
            rgba[2],
            (s.workspace_opacity * 255.) as u8,
        )
    } else {
        color
    }
}
pub fn preferences(ui: &mut egui::Ui, s: &mut GlassSettings) -> bool {
    let before = s.clone();
    ui.checkbox(&mut s.enabled, crate::t!("glass.enable"));
    ui.add_enabled_ui(s.enabled, |ui| {
        ui.checkbox(&mut s.desktop, crate::t!("glass.desktop"));
        ui.add_enabled_ui(s.desktop, |ui| {
            ui.checkbox(&mut s.desktop_blur, crate::t!("glass.desktop_blur"));
            ui.add(
                crate::ui::controls::Slider::new(&mut s.workspace_opacity, 0.0..=1.0)
                    .text(crate::t!("glass.workspace_opacity"))
                    .reset_to(0.18),
            );
        });
        #[cfg(not(target_arch = "wasm32"))]
        if s.desktop && s.desktop_blur {
            ui.weak(crate::services::glass::status());
        }
        for (name, part) in [
            ("glass.panels", &mut s.panels),
            ("glass.headers", &mut s.headers),
            ("glass.menus", &mut s.menus),
            ("glass.toolbars", &mut s.toolbars),
            ("glass.buttons", &mut s.buttons),
        ] {
            ui.push_id(name, |ui| {
                egui::CollapsingHeader::new(crate::i18n::translate(name)).show(ui, |ui| {
                    ui.checkbox(&mut part.enabled, crate::t!("glass.use"));
                    ui.add(
                        crate::ui::controls::Slider::new(&mut part.opacity, 0.0..=1.0)
                            .text(crate::t!("glass.opacity"))
                            .reset_to(match name {
                                "glass.buttons" => 0.8,
                                "glass.headers" => 1.,
                                _ => 0.78,
                            }),
                    );
                    ui.add(
                        crate::ui::controls::Slider::new(&mut part.border, 0.0..=2.0)
                            .text(crate::t!("glass.border"))
                            .reset_to(0.7),
                    );
                    ui.horizontal(|ui| {
                        let mut custom = part.tint.is_some();
                        if ui.checkbox(&mut custom, crate::t!("glass.tint")).changed() {
                            part.tint = custom.then_some([128, 128, 140]);
                        }
                        if let Some(color) = &mut part.tint {
                            crate::ui::controls::color_edit_button_srgb(ui, color);
                        }
                    });
                });
            });
        }
        ui.weak(crate::t!("glass.notice"));
    });
    s.sanitize();
    *s != before
}

#[derive(Clone)]
struct ButtonBase {
    original: egui::style::Widgets,
    painted: egui::style::Widgets,
}
fn configure_buttons(ctx: &Context, s: &GlassSettings, force: bool) {
    let id = egui::Id::new("glass_button_base");
    let old = ctx.data(|d| d.get_temp::<std::sync::Arc<ButtonBase>>(id));
    let style = ctx.global_style();
    let current = &style.visuals.widgets;
    if !s.enabled || !s.buttons.enabled {
        if let Some(old) = old {
            if *current == old.painted {
                ctx.global_style_mut(|style| style.visuals.widgets = old.original.clone());
            }
            ctx.data_mut(|d| d.remove::<std::sync::Arc<ButtonBase>>(id));
        }
        return;
    }
    if !force && old.as_ref().is_some_and(|old| old.painted == *current) {
        return;
    }
    let original = old
        .filter(|old| old.painted == *current)
        .map(|old| old.original.clone())
        .unwrap_or_else(|| current.clone());
    let mut painted = original.clone();
    for w in [
        &mut painted.inactive,
        &mut painted.hovered,
        &mut painted.active,
        &mut painted.open,
    ] {
        for colour in [&mut w.bg_fill, &mut w.weak_bg_fill] {
            let original = colour.to_srgba_unmultiplied();
            let rgb = s
                .buttons
                .tint
                .unwrap_or([original[0], original[1], original[2]]);
            *colour = Color32::from_rgba_unmultiplied(
                rgb[0],
                rgb[1],
                rgb[2],
                (colour.a() as f32 * s.buttons.opacity) as u8,
            );
        }
        w.bg_stroke.width *= s.buttons.border;
    }
    if ctx.global_style().visuals.widgets != painted {
        ctx.global_style_mut(|style| style.visuals.widgets = painted.clone());
    }
    ctx.data_mut(|d| d.insert_temp(id, std::sync::Arc::new(ButtonBase { original, painted })));
}

pub fn paint_header(
    painter: &egui::Painter,
    rect: Rect,
    rounding: egui::CornerRadius,
    fill: Color32,
) {
    let s = settings(painter.ctx());
    let frame = egui::Frame::NONE.fill(fill).corner_radius(rounding);
    if s.enabled
        && let Some(shape) = material_surface(&frame, rect, &s.headers)
    {
        painter.add(shape);
        return;
    }
    painter.rect_filled(rect, rounding, fill);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_glass_has_no_hook_and_preserves_upstream_fill() {
        let ctx = Context::default();
        configure(&ctx, &GlassSettings::default(), false);
        assert!(
            ctx.data(
                |d| d.get_temp::<egui::containers::frame::FramePaintHook>(egui::Id::new(
                    "frame_paint_hook"
                ))
            )
            .is_none()
        );
        assert_eq!(workspace(&ctx, Color32::WHITE), Color32::WHITE);
    }
    #[test]
    fn translucent_material_keeps_premultiplied_colour_at_half_opacity() {
        let frame = egui::Frame::NONE.fill(Color32::from_rgba_unmultiplied(200, 100, 50, 128));
        let surface = Surface {
            opacity: 0.5,
            ..Default::default()
        };
        let Shape::Rect(rect) = material_surface(
            &frame,
            Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(40., 40.)),
            &surface,
        )
        .unwrap() else {
            panic!("Missing material")
        };
        assert_eq!(rect.fill.a(), 64);
        let rgba = rect.fill.to_srgba_unmultiplied();
        assert!((i16::from(rgba[0]) - 200).abs() <= 3);
        assert!((i16::from(rgba[1]) - 100).abs() <= 3);
    }
    #[test]
    fn tinted_materials_preserve_explicit_alpha_without_a_floor() {
        for alpha in [0, 13, 51, 128, 204, 255] {
            for opacity in [0., 0.2, 0.78, 1.] {
                for tint in [None, Some([20, 21, 23])] {
                    let frame = egui::Frame::NONE
                        .fill(Color32::from_rgba_unmultiplied(100, 120, 140, alpha));
                    let surface = Surface {
                        opacity,
                        tint,
                        ..Default::default()
                    };
                    let Shape::Rect(rect) = material_surface(
                        &frame,
                        Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(40., 40.)),
                        &surface,
                    )
                    .unwrap() else {
                        panic!("Missing material")
                    };
                    assert_eq!(rect.fill.a(), (f32::from(alpha) * opacity) as u8);
                }
            }
        }
    }
    #[test]
    fn glass_settings_clamp_nonfinite_and_out_of_range_values() {
        let mut s = GlassSettings {
            workspace_opacity: f32::NAN,
            ..Default::default()
        };
        s.panels.opacity = 5.;
        s.menus.border = -4.;
        s.sanitize();
        assert_eq!(s.workspace_opacity, 0.18);
        assert_eq!(s.panels.opacity, 1.);
        assert_eq!(s.menus.border, 0.);
    }
    #[test]
    fn theme_roundtrip_carries_materials_but_not_permissions() {
        let mut a = crate::config::settings::AppSettings::default();
        a.glass.enabled = true;
        a.glass.menus.opacity = 0.4;
        a.automation.enabled = true;
        let mut b = crate::config::settings::AppSettings::default();
        b.import_theme_from_string(&a.export_theme_to_string());
        assert_eq!(a.glass, b.glass);
        assert!(!b.automation.enabled);
    }
    #[test]
    fn glass_rendering_keeps_frame_geometry_and_button_text_in_both_themes() {
        for dark in [false, true] {
            let ctx = Context::default();
            ctx.set_visuals(if dark {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
            let text = ctx.global_style().visuals.widgets.inactive.fg_stroke;
            let s = GlassSettings {
                enabled: true,
                ..Default::default()
            };
            configure(&ctx, &s, dark);
            let mut rect = None;
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                rect = Some(
                    egui::Frame::group(ui.style())
                        .show(ui, |ui| {
                            ui.label("Readable glass");
                            let _ = ui.button("Button");
                        })
                        .response
                        .rect,
                );
            });
            assert!(rect.unwrap().is_positive());
            assert_eq!(ctx.global_style().visuals.widgets.inactive.fg_stroke, text);
            configure(&ctx, &GlassSettings::default(), dark);
            assert!(
                ctx.data(|d| d
                    .get_temp::<std::sync::Arc<ButtonBase>>(egui::Id::new("glass_button_base")))
                    .is_none()
            );
        }
    }
}

#[cfg(test)]
mod lightweight_tests {
    use super::*;
    fn rects(shape: &Shape, out: &mut Vec<egui::epaint::RectShape>) {
        match shape {
            Shape::Rect(r) => out.push(r.clone()),
            Shape::Vec(items) => {
                for s in items {
                    rects(s, out)
                }
            }
            _ => {}
        }
    }
    #[test]
    fn glass_is_opt_in_and_native_blur_is_not_enabled_with_it() {
        let s = GlassSettings::default();
        assert!(!s.enabled);
        assert!(s.desktop);
        assert!(!s.desktop_blur);
        let old: GlassSettings = serde_json::from_str(
            r#"{"enabled":true,"panels":{"frosting":true,"strength":16.0,"opacity":0.7}}"#,
        )
        .unwrap();
        assert_eq!(old.panels.opacity, 0.7);
    }
    #[test]
    fn moving_glass_windows_uploads_no_backdrop_textures_and_keeps_nested_groups_opaque() {
        for dark in [false, true] {
            let ctx = Context::default();
            let mut fonts = egui::FontDefinitions::default();
            fonts.families.insert(
                egui::FontFamily::Name("WidgetTitle".into()),
                fonts.families[&egui::FontFamily::Proportional].clone(),
            );
            ctx.set_fonts(fonts);
            ctx.set_visuals(if dark {
                egui::Visuals::dark()
            } else {
                egui::Visuals::light()
            });
            let mut s = GlassSettings {
                enabled: true,
                ..Default::default()
            };
            s.panels.opacity = 0.5;
            let nested = Color32::from_rgb(37, 79, 123);
            for frame in 0..8 {
                let output = ctx.run_ui(
                    egui::RawInput {
                        time: Some(frame as f64 * 0.2),
                        screen_rect: Some(Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1000., 700.),
                        )),
                        ..Default::default()
                    },
                    |ui| {
                        configure(ui.ctx(), &s, dark);
                        egui::Window::new("Moving glass")
                            .current_pos(egui::pos2(80. + frame as f32 * 8., 100.))
                            .frame(egui::Frame::window(ui.style()).fill(Color32::WHITE))
                            .show(ui.ctx(), |ui| {
                                crate::ui::dialogs::core::paint_dialog_header(
                                    ui,
                                    &crate::ui::dialogs::core::DialogColors::from_ctx(&ctx),
                                    "",
                                    "Header",
                                );
                                egui::Frame::group(ui.style()).fill(nested).show(ui, |ui| {
                                    ui.label("Nested content");
                                });
                            });
                    },
                );
                if frame > 2 {
                    assert!(
                        output.textures_delta.set.is_empty(),
                        "Glass must not upload sampled image textures"
                    );
                    let mut shapes = Vec::new();
                    for s in &output.shapes {
                        rects(&s.shape, &mut shapes)
                    }
                    assert!(
                        shapes.iter().all(|r| r.brush.is_none()),
                        "Dialog fills and headers must not sample the document"
                    );
                    assert!(
                        shapes.iter().any(|r| r.fill == nested),
                        "Nested group fill must stay opaque"
                    );
                    assert!(
                        shapes
                            .iter()
                            .any(|r| r.fill == Color32::from_rgba_unmultiplied(255, 255, 255, 127)),
                        "Outer panel should have one translucent fill"
                    );
                }
            }
        }
    }
    #[test]
    fn workspace_and_headers_keep_consistent_alpha_without_image_textures() {
        let ctx = Context::default();
        let s = GlassSettings {
            enabled: true,
            ..Default::default()
        };
        configure(&ctx, &s, false);
        assert_eq!(workspace(&ctx, Color32::WHITE).a(), (0.18 * 255.) as u8);
        let fill = Color32::from_rgba_unmultiplied(200, 100, 50, 32);
        let output = ctx.run_ui(egui::RawInput::default(), |ui| {
            paint_header(
                ui.painter(),
                Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200., 30.)),
                egui::CornerRadius::ZERO,
                fill,
            );
        });
        let mut shapes = Vec::new();
        for s in &output.shapes {
            rects(&s.shape, &mut shapes)
        }
        assert!(shapes.iter().any(|r| r.fill == fill && r.brush.is_none()));
        configure(&ctx, &GlassSettings::default(), false);
        assert_eq!(workspace(&ctx, Color32::WHITE), Color32::WHITE);
    }
}
