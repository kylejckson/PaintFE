use super::settings_window::{SearchableResponse, SettingsWindow};
use crate::config::settings::AppSettings;
use egui::{Event, Pos2, Rect, TouchPhase, Ui, Vec2};
use std::collections::VecDeque;

#[derive(Default)]
pub(super) struct TabletTest {
    points: VecDeque<Option<(Pos2, f32)>>,
    drawing: bool,
    touch: Option<egui::TouchId>,
    pressure: Option<f32>,
}

impl TabletTest {
    pub(super) fn stop(&mut self) {
        if self.drawing {
            self.finish();
        }
    }

    fn finish(&mut self) {
        self.drawing = false;
        self.touch = None;
        self.points.push_back(None);
        if self.pressure.is_some() {
            self.pressure = Some(0.);
        }
    }

    fn record(&mut self, pos: Pos2, rect: Rect, enabled: bool) {
        let pos = Pos2::new(
            (pos.x - rect.min.x) / rect.width(),
            (pos.y - rect.min.y) / rect.height(),
        );
        let force = if enabled {
            self.pressure.unwrap_or(1.)
        } else {
            1.
        };
        if self.points.back().copied().flatten() != Some((pos, force)) {
            self.points.push_back(Some((pos, force)));
        }
    }

    fn input(
        &mut self,
        events: &[Event],
        rect: Rect,
        enabled: bool,
        can_start: impl Fn(Pos2) -> bool,
    ) {
        for event in events {
            match event {
                Event::Touch {
                    id,
                    phase,
                    pos,
                    force,
                    ..
                } => {
                    if (self.touch.is_none() || self.touch == Some(*id))
                        && let Some(force) = force.filter(|force| force.is_finite())
                    {
                        self.pressure = Some(force.clamp(0., 1.));
                    }
                    match phase {
                        TouchPhase::Start if self.touch.is_none() && can_start(*pos) => {
                            self.points.push_back(None);
                            self.touch = Some(*id);
                            self.drawing = true;
                            self.record(*pos, rect, enabled);
                        }
                        TouchPhase::Move if self.touch == Some(*id) => {
                            self.record(*pos, rect, enabled)
                        }
                        TouchPhase::End | TouchPhase::Cancel if self.touch == Some(*id) => {
                            self.finish()
                        }
                        _ => {}
                    }
                }
                Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    ..
                } if self.touch.is_none() && can_start(*pos) => {
                    self.points.push_back(None);
                    self.drawing = true;
                    // Mouse input has no force. Do not reuse a previous pen sample.
                    self.pressure = None;
                    self.record(*pos, rect, enabled);
                }
                Event::PointerMoved(pos) if self.drawing && self.touch.is_none() => {
                    self.record(*pos, rect, enabled)
                }
                Event::PointerButton {
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    ..
                } if self.drawing && self.touch.is_none() => self.finish(),
                Event::PointerGone if self.drawing => self.finish(),
                _ => {}
            }
            while self.points.len() > 1024 {
                self.points.pop_front();
            }
        }
    }

    fn show(&mut self, ui: &mut Ui, enabled: bool) {
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width().max(1.), 160.),
            egui::Sense::drag(),
        );
        let events = ui.input(|input| input.events.clone());
        self.input(&events, rect, enabled, |pos| {
            rect.intersect(ui.clip_rect()).contains(pos)
                && ui
                    .ctx()
                    .layer_id_at(pos)
                    .is_none_or(|layer| layer == ui.layer_id())
        });
        let painter = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
        let visuals = ui.visuals();
        painter.rect(
            rect,
            visuals.widgets.noninteractive.corner_radius,
            visuals.extreme_bg_color,
            visuals.window_stroke,
            egui::StrokeKind::Inside,
        );
        let mut previous = None;
        for sample in &self.points {
            if let Some((pos, force)) = sample {
                let pos = rect.min + pos.to_vec2() * rect.size();
                let width = 1. + 8. * force;
                if let Some(previous) = previous {
                    painter.line_segment([previous, pos], (width, visuals.hyperlink_color));
                } else {
                    painter.circle_filled(pos, width * 0.5, visuals.hyperlink_color);
                }
                previous = Some(pos);
            } else {
                previous = None;
            }
        }
        if let Some(pressure) = self.pressure {
            ui.label(format!(
                "{}: {:.0}%",
                crate::t!("tools.tablet.pressure"),
                pressure * 100.
            ));
            ui.add(egui::ProgressBar::new(pressure).desired_width(ui.available_width()));
        } else {
            ui.weak(crate::t!("tools.tablet.no_signal"));
        }
        if ui.button(crate::t!("tools.tablet.clear")).clicked() {
            *self = Self::default();
        }
    }
}

impl SettingsWindow {
    pub(super) fn show_tablet_tab(&mut self, ui: &mut Ui, settings: &mut AppSettings) {
        Self::section_header(ui, &crate::t!("tools.tablet.title"));
        ui.checkbox(
            &mut settings.tablet_pressure_enabled,
            crate::t!("tools.tablet.enabled"),
        )
        .search_target(ui, "tools.tablet.enabled");
        ui.weak(crate::t!("tools.tablet.hint"));
        ui.label(crate::t!(if cfg!(target_os = "windows") {
            "tools.tablet.windows"
        } else {
            "tools.tablet.system"
        }));
        if cfg!(target_os = "windows") {
            ui.weak(crate::t!("tools.tablet.windows_hint"));
        }
        Self::section_header(ui, &crate::t!("tools.tablet.test"));
        ui.weak(crate::t!("tools.tablet.test_hint"));
        self.tablet_test.show(ui, settings.tablet_pressure_enabled);
        settings.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(id: u64, phase: TouchPhase, pos: Pos2, force: f32) -> Event {
        Event::Touch {
            device_id: egui::TouchDeviceId(1),
            id: egui::TouchId(id),
            phase,
            pos,
            force: Some(force),
        }
    }

    #[test]
    fn pressure_test_tracks_force_without_duplicate_mouse_samples_and_resets_on_lift() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.));
        let p = Pos2::new(10., 10.);
        let mut test = TabletTest::default();
        test.input(
            &[
                touch(1, TouchPhase::Start, p, 0.2),
                Event::PointerButton {
                    pos: p,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                touch(1, TouchPhase::Move, p + Vec2::splat(20.), 0.8),
                Event::PointerMoved(p + Vec2::splat(20.)),
            ],
            rect,
            true,
            |p| rect.contains(p),
        );
        let samples: Vec<_> = test.points.iter().flatten().copied().collect();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], (Pos2::new(0.1, 0.1), 0.2));
        assert_eq!(samples[1].1, 0.8);
        test.input(&[touch(2, TouchPhase::Move, p, 0.3)], rect, true, |_| true);
        assert_eq!(test.pressure, Some(0.8));
        test.input(&[touch(1, TouchPhase::End, p, 0.)], rect, true, |_| true);
        assert!(!test.drawing);
        assert_eq!(test.pressure, Some(0.));
        assert_eq!(test.points.back(), Some(&None));
    }

    #[test]
    fn pressure_test_is_primary_only_bounded_and_respects_the_enable_setting() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.));
        let p = Pos2::new(20., 20.);
        let mut test = TabletTest::default();
        test.input(
            &[Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Secondary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }],
            rect,
            true,
            |_| true,
        );
        assert!(test.points.is_empty());
        test.input(&[touch(1, TouchPhase::Start, p, 0.2)], rect, false, |_| {
            true
        });
        assert_eq!(test.points.back().copied().flatten().unwrap().1, 1.);
        for n in 0..1500 {
            test.input(
                &[touch(1, TouchPhase::Move, Pos2::new(n as f32, 20.), 0.5)],
                rect,
                false,
                |_| true,
            );
        }
        assert!(test.points.len() <= 1024);
        test.stop();
        assert!(!test.drawing);
        assert_eq!(test.touch, None);
        test.input(
            &[Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }],
            rect,
            true,
            |_| false,
        );
        assert!(
            !test.drawing,
            "A popup or clipped area must prevent starting a test stroke"
        );
    }

    #[test]
    fn pressure_test_mouse_falls_back_to_full_pressure_after_using_a_pen() {
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.));
        let p = Pos2::new(20., 20.);
        let mut test = TabletTest::default();
        test.input(
            &[
                touch(1, TouchPhase::Start, p, 0.1),
                touch(1, TouchPhase::End, p, 0.),
            ],
            rect,
            true,
            |_| true,
        );
        test.input(
            &[Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }],
            rect,
            true,
            |_| true,
        );
        assert_eq!(test.pressure, None);
        assert_eq!(test.points.back().copied().flatten().unwrap().1, 1.);
        test.input(&[Event::PointerGone], rect, true, |_| true);
        assert!(!test.drawing);
    }
}
