//! Variable-height Preferences sections retain their measured space offscreen.
//! Each section has an explicit scope, so skipping its widgets cannot renumber
//! controls in later sections or change their focus/drag identity.
use egui::{Id, Ui, Vec2};

#[derive(Clone)]
struct Measurement {
    size: Vec2,
    width: f32,
    revision: u64,
    focused: Option<Id>,
}

pub(super) fn show(
    ui: &mut Ui,
    key: &'static str,
    revision: u64,
    force: bool,
    contents: impl FnOnce(&mut Ui),
) {
    #[cfg(test)]
    let force = force
        || ui.ctx().data(|d| {
            d.get_temp::<bool>(Id::new("profile_disable_section_cache"))
                .unwrap_or(false)
        });
    let id = ui.make_persistent_id(("settings_section", key));
    let cache_id = id.with("measurement");
    let width = ui.available_width();
    let cache = ui.ctx().data(|d| d.get_temp::<Measurement>(cache_id));
    let focused = ui.memory(|m| m.focused());
    let skip = cache.as_ref().is_some_and(|cache| {
        cache.revision == revision
            && (cache.width - width).abs() < 0.25
            && !force
            && !(focused.is_some() && focused == cache.focused)
            && !ui.is_rect_visible(
                egui::Rect::from_min_size(ui.next_widget_position(), cache.size).expand(80.0),
            )
    });
    let result = ui.scope_builder(egui::UiBuilder::new().id(id), |ui| {
        let _profile = crate::ui::perf::Scope::new(if skip { 27 } else { 26 });
        if skip {
            ui.allocate_space(cache.as_ref().unwrap().size);
        } else {
            contents(ui);
        }
    });
    if !skip {
        let focused = ui.memory(|m| m.focused()).filter(|&id| {
            ui.ctx()
                .read_response(id)
                .is_some_and(|r| result.response.rect.intersects(r.rect))
        });
        ui.ctx().data_mut(|d| {
            d.insert_temp(
                cache_id,
                Measurement {
                    size: result.response.rect.size(),
                    width,
                    revision,
                    focused,
                },
            )
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offscreen_sections_preserve_height_and_invalidate_on_revision() {
        let ctx = egui::Context::default();
        let mut calls = [0; 3];
        let mut heights = Vec::new();
        for revision in [0, 0, 1] {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                ui.set_clip_rect(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(300.0, 100.0),
                ));
                for (i, calls) in calls.iter_mut().enumerate() {
                    show(ui, ["a", "b", "c"][i], revision, false, |ui| {
                        *calls += 1;
                        ui.allocate_space(egui::vec2(200.0, 180.0));
                    });
                }
                heights.push(ui.min_rect().height());
            });
        }
        assert_eq!(calls, [3, 2, 2]);
        assert!((heights[0] - heights[1]).abs() < 0.1);
        assert!((heights[1] - heights[2]).abs() < 0.1);
    }
}
