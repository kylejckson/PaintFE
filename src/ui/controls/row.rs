use egui::{Response, Ui, WidgetText};

/// Labelled row composition, with identity independent of its translated label.
pub struct ValueRow {
    label: WidgetText,
    width: Option<f32>,
    id: egui::Id,
}

pub struct ValueRowResponse<R> {
    pub label: Response,
    pub inner: R,
}

impl ValueRow {
    #[track_caller]
    pub fn new(label: impl Into<WidgetText>) -> Self {
        let source = std::panic::Location::caller();
        Self {
            label: label.into(),
            width: None,
            id: egui::Id::new((source.file(), source.line(), source.column())),
        }
    }
    pub fn label_width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }
    pub fn id_salt(mut self, id: impl egui::AsId) -> Self {
        self.id = egui::Id::new(id);
        self
    }
    pub fn show<R>(self, ui: &mut Ui, contents: impl FnOnce(&mut Ui) -> R) -> ValueRowResponse<R> {
        ui.scope_builder(
            egui::UiBuilder::new().id(ui.make_persistent_id(("value_row", self.id))),
            |ui| {
                ui.horizontal(|ui| {
                    let label = if let Some(width) = self.width {
                        ui.add_sized(
                            [width, ui.spacing().interact_size.y],
                            egui::Label::new(self.label).wrap(),
                        )
                    } else {
                        ui.label(self.label)
                    };
                    ValueRowResponse {
                        label,
                        inner: contents(ui),
                    }
                })
                .inner
            },
        )
        .inner
    }
}
