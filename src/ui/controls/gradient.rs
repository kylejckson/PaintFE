use egui::{Color32, Response, Sense, Stroke, Ui, Vec2, Widget};

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
                        if self.alpha {
                            ui.painter()
                                .rect_filled(track, radius as u8, self.theme.bg3);
                            let painter =
                                ui.painter().with_clip_rect(track.intersect(ui.clip_rect()));
                            for y in 0..2 {
                                for x in 0..(track.width() / 4.0).ceil() as usize {
                                    if (x + y) % 2 == 0 {
                                        let cell = egui::Rect::from_min_size(
                                            track.min + Vec2::new(x as f32 * 4.0, y as f32 * 4.0),
                                            Vec2::splat(4.0),
                                        )
                                        .intersect(track.shrink(0.5));
                                        painter.rect_filled(cell, 0, self.theme.panel_bg);
                                    }
                                }
                            }
                        }
                        for x in 0..track.width().ceil() as usize {
                            let px = x as f32 + 0.5;
                            let edge = px.min(track.width() - px).max(0.0);
                            let half = if edge < radius {
                                4.0 - radius + (radius.powi(2) - (radius - edge).powi(2)).sqrt()
                            } else {
                                4.0
                            };
                            let strip = egui::Rect::from_center_size(
                                egui::pos2(track.left() + px, track.center().y),
                                Vec2::new(1.1, half * 2.0),
                            );
                            ui.painter()
                                .rect_filled(strip, 0, (self.sample)(px / track.width()));
                        }
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
