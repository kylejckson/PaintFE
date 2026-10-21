use egui::{Color32, Response, Sense, Stroke, Ui, Vec2, Widget};

#[derive(Clone)]
struct TrackTexture {
    pixels: std::sync::Arc<[Color32]>,
    texture: egui::TextureHandle,
}

fn track_texture(
    ui: &Ui,
    id: egui::Id,
    pixels: Vec<Color32>,
    size: [usize; 2],
    options: egui::TextureOptions,
) -> egui::TextureHandle {
    let cache = ui.ctx().data(|d| d.get_temp::<TrackTexture>(id));
    if let Some(mut cache) = cache {
        if cache.texture.size() != size || cache.pixels.as_ref() != pixels {
            cache
                .texture
                .set(egui::ColorImage::new(size, pixels.clone()), options);
            cache.pixels = pixels.into();
            ui.ctx().data_mut(|d| d.insert_temp(id, cache.clone()));
        }
        cache.texture
    } else {
        let texture = ui.ctx().load_texture(
            "colour_track",
            egui::ColorImage::new(size, pixels.clone()),
            options,
        );
        ui.ctx().data_mut(|d| {
            d.insert_temp(
                id,
                TrackTexture {
                    pixels: pixels.into(),
                    texture: texture.clone(),
                },
            )
        });
        texture
    }
}

fn paint_track<F: Fn(f32) -> Color32>(
    ui: &Ui,
    id: egui::Id,
    track: egui::Rect,
    radius: f32,
    alpha: bool,
    theme: &crate::theme::Theme,
    sample: F,
) {
    if alpha {
        ui.painter().rect_filled(track, radius as u8, theme.bg3);
        let pixels = (0..64)
            .map(|i| {
                if ((i % 8) / 4 + (i / 8) / 4) % 2 == 0 {
                    theme.panel_bg
                } else {
                    Color32::TRANSPARENT
                }
            })
            .collect();
        let checker = track_texture(
            ui,
            egui::Id::new(("colour_track_checker", theme.panel_bg)),
            pixels,
            [8, 8],
            egui::TextureOptions::NEAREST_REPEAT,
        );
        ui.painter().add(
            egui::epaint::RectShape::filled(track, radius as u8, Color32::WHITE).with_texture(
                checker.id(),
                egui::Rect::from_min_max(
                    egui::Pos2::ZERO,
                    egui::pos2(track.width() / 8.0, track.height() / 8.0),
                ),
            ),
        );
    }
    let count = (track.width() * ui.ctx().pixels_per_point())
        .ceil()
        .clamp(2.0, 1024.0) as usize;
    let pixels = (0..count)
        .map(|x| sample(x as f32 / (count - 1) as f32))
        .collect();
    // Re-sampling makes arbitrary caller closures safe: no stale colour cache.
    // Only changed samples upload; geometry stays constant at every width.
    let gradient = track_texture(
        ui,
        id.with("track_texture"),
        pixels,
        [count, 1],
        egui::TextureOptions::LINEAR,
    );
    ui.painter().add(
        egui::epaint::RectShape::filled(track, radius as u8, Color32::WHITE).with_texture(
            gradient.id(),
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        ),
    );
}

/// Colour-track variation: specialised paint, shared pointer, keyboard and editor.
pub struct GradientSlider<'a, F: Fn(f32) -> Color32> {
    value: &'a mut f32,
    max: f32,
    sample: F,
    id: egui::Id,
    label: String,
    alpha: bool,
    default: Option<f32>,
    theme: crate::theme::Theme,
}

impl<'a, F: Fn(f32) -> Color32> GradientSlider<'a, F> {
    #[track_caller]
    pub fn new(value: &'a mut f32, max: f32, sample: F) -> Self {
        let source = std::panic::Location::caller();
        Self {
            value,
            max,
            sample,
            id: egui::Id::new((source.file(), source.line(), source.column())),
            label: String::new(),
            alpha: false,
            default: None,
            theme: crate::theme::Theme::default(),
        }
    }
    pub fn id_salt(mut self, id: impl egui::AsId) -> Self {
        self.id = egui::Id::new(id);
        self
    }
    pub fn label(mut self, label: &str) -> Self {
        self.label = label.to_owned();
        self
    }
    pub fn alpha(mut self, alpha: bool) -> Self {
        self.alpha = alpha;
        self
    }
    pub fn theme(mut self, theme: crate::theme::Theme) -> Self {
        self.theme = theme;
        self
    }
    pub fn reset_to(mut self, default: f32) -> Self {
        self.default = Some(default);
        self
    }
}

impl<F: Fn(f32) -> Color32> Widget for GradientSlider<'_, F> {
    fn ui(self, ui: &mut Ui) -> Response {
        let _profile = crate::ui::perf::Scope::new(25);
        ui.scope_builder(
            egui::UiBuilder::new().id(ui.make_persistent_id(("control", self.id))),
            |ui| {
                ui.horizontal(|ui| {
                    let before = *self.value;
                    if !self.label.is_empty() {
                        ui.add_sized(
                            [12.0, 22.0],
                            egui::Label::new(egui::RichText::new(&self.label).size(10.0)),
                        );
                    }
                    let width = (ui.available_width() - 47.0).max(40.0);
                    let (rect, mut response) =
                        ui.allocate_exact_size(Vec2::new(width, 22.0), Sense::click_and_drag());
                    #[cfg(test)]
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(egui::Id::new("control_test_track"), response.clone())
                    });
                    response.widget_info(|| {
                        egui::WidgetInfo::slider(ui.is_enabled(), *self.value as f64, &self.label)
                    });
                    let track = egui::Rect::from_center_size(
                        rect.center() + egui::vec2(0.0, 3.0),
                        Vec2::new(width - 4.0, 8.0),
                    );
                    if let (_, Some(pos), _) = super::drag_motion(ui, rect, response.id) {
                        *self.value =
                            ((pos.x - track.left()) / track.width()).clamp(0.0, 1.0) * self.max;
                        response.request_focus();
                    }
                    crate::ui::numeric::slider_input(
                        ui,
                        &response,
                        self.value,
                        0.0..=self.max,
                        1.0,
                    );
                    if let Some(default) = self.default {
                        response = response.on_hover_text(crate::t!("ui.double_click_reset_hint"));
                        if ui.is_enabled()
                            && super::super::polish::settings(ui.ctx()).double_click_slider_reset
                            && response.double_clicked()
                        {
                            *self.value = default.clamp(0.0, self.max);
                        }
                    }
                    if ui.is_rect_visible(rect) {
                        let radius = self.theme.widget_rounding.clamp(0.0, 4.0);
                        paint_track(
                            ui,
                            response.id,
                            track,
                            radius,
                            self.alpha,
                            &self.theme,
                            &self.sample,
                        );
                        ui.painter().rect_stroke(
                            track,
                            radius as u8,
                            Stroke::new(0.5, self.theme.border_color),
                            egui::StrokeKind::Inside,
                        );
                        let x =
                            track.left() + (*self.value / self.max).clamp(0.0, 1.0) * track.width();
                        let tip = egui::pos2(x, track.center().y);
                        ui.painter().add(egui::Shape::convex_polygon(
                            vec![
                                tip,
                                tip + egui::vec2(-4.0, -9.0),
                                tip + egui::vec2(4.0, -9.0),
                            ],
                            self.theme.text_color,
                            Stroke::new(1.0, self.theme.panel_bg),
                        ));
                    }
                    let field = egui::Frame::new()
                        .fill(self.theme.bg2)
                        .corner_radius(self.theme.widget_cr(5))
                        .show(ui, |ui| {
                            ui.add_sized(
                                [37.0, 20.0],
                                super::Numeric::new(self.value)
                                    .id_salt("value")
                                    .range(0.0..=self.max)
                                    .clamp_existing_to_range(false)
                                    .speed(1.0)
                                    .max_decimals(0),
                            )
                        })
                        .inner;
                    response = if field.has_focus() || field.gained_focus() || field.lost_focus() {
                        field.union(response)
                    } else {
                        response.union(field)
                    };
                    if *self.value != before {
                        response.mark_changed();
                    }
                    response
                })
                .inner
            },
        )
        .inner
    }
}

#[cfg(test)]
mod paint_tests {
    use super::*;
    #[test]
    fn track_geometry_is_bounded_and_texture_changes_follow_colour_size_and_scale() {
        for dark in [false, true] {
            let ctx = egui::Context::default();
            let theme = if dark {
                crate::theme::Theme::dark()
            } else {
                crate::theme::Theme::light()
            };
            let draw = |width: f32, colour: Color32| {
                ctx.run_ui(egui::RawInput::default(), |ui| {
                    paint_track(
                        ui,
                        egui::Id::new("test"),
                        egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(width, 12.0)),
                        4.0,
                        true,
                        &theme,
                        |_| colour,
                    );
                })
            };
            let first = draw(600.0, Color32::RED);
            assert!(
                first.shapes.len() <= 4,
                "gradient generated per-pixel geometry"
            );
            let unchanged = draw(600.0, Color32::RED);
            assert!(
                unchanged.textures_delta.set.is_empty(),
                "unchanged track uploaded again"
            );
            let changed = draw(600.0, Color32::BLUE);
            assert_eq!(changed.textures_delta.set.len(), 1);
            assert_eq!(changed.textures_delta.set[0].1.image.size(), [600, 1]);
            assert_eq!(draw(300.0, Color32::BLUE).textures_delta.set.len(), 1);
            ctx.set_pixels_per_point(1.25);
            let scaled = draw(300.0, Color32::BLUE);
            assert_eq!(scaled.textures_delta.set[0].1.image.size(), [375, 1]);
            assert!(scaled.shapes.len() <= 4);
        }
    }
}
