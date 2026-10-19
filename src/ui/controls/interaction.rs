use egui::Ui;

pub(crate) fn drag_motion(
    ui: &Ui,
    rect: egui::Rect,
    id: egui::Id,
) -> (Option<egui::Pos2>, Option<egui::Pos2>, bool) {
    let id = id.with("pointer_capture");
    let mut active = ui
        .ctx()
        .data_mut(|d| d.get_temp::<bool>(id))
        .unwrap_or(false)
        && ui.is_enabled();
    let mut start = None;
    let mut position = None;
    let events = ui.input(|input| input.events.clone());
    for event in &events {
        match event {
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                ..
            } if ui.is_enabled()
                && rect.intersect(ui.clip_rect()).contains(*pos)
                && ui
                    .ctx()
                    .layer_id_at(*pos)
                    .is_none_or(|layer| layer == ui.layer_id()) =>
            {
                active = true;
                start = Some(*pos);
                position = Some(*pos);
            }
            egui::Event::PointerMoved(pos) if active => position = Some(*pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                ..
            } if active => {
                position = Some(*pos);
                active = false;
            }
            egui::Event::PointerGone => active = false,
            _ => {}
        }
    }
    ui.ctx().data_mut(|d| d.insert_temp(id, active));
    (start, position, active)
}
