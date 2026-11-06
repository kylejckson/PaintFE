use crate::{config::settings::ToolShelfSettings, theme::Theme};
use egui::{Color32, Frame, Ui};

/// Geometry is resolved before tool content, so overflow cannot move the canvas.
pub(crate) struct ShelfStyle {
    pub frame: Frame,
    pub row_height: f32,
    pub scroll_space: f32,
    pub separator: Color32,
}

impl ShelfStyle {
    pub fn resolve(
        ui: &Ui,
        theme: &Theme,
        settings: &ToolShelfSettings,
        rounding: Option<f32>,
    ) -> Self {
        let mut frame = theme.tool_shelf_frame();
        frame.fill = theme
            .tool_shelf_bg
            .gamma_multiply(settings.effective_opacity());
        frame.inner_margin =
            egui::Margin::symmetric(settings.padding_x as i8, settings.padding_y as i8);
        frame.corner_radius = egui::CornerRadius::same(rounding.unwrap_or(if settings.docked {
            0.
        } else {
            theme.tool_shelf_rounding
        }) as u8);
        if settings.docked {
            frame.shadow = egui::epaint::Shadow::NONE;
            frame.stroke = egui::Stroke::NONE;
        }
        Self {
            frame,
            row_height: ui
                .spacing()
                .interact_size
                .y
                .max(ui.text_style_height(&egui::TextStyle::Body))
                .max(20.)
                + ui.spacing().button_padding.y * 2.
                + 4.,
            scroll_space: ui.spacing().scroll.bar_width + ui.spacing().scroll.bar_inner_margin,
            separator: settings
                .top_separator
                .map(|[r, g, b]| Color32::from_rgb(r, g, b))
                .unwrap_or(theme.tool_shelf_border)
                .gamma_multiply(settings.separator_opacity),
        }
    }
    pub fn height(&self) -> f32 {
        self.row_height + self.scroll_space + self.frame.total_margin().sum().y
    }
}

/// Paint the workspace behind translucent shelves without changing layout.
pub(crate) fn paint_workspace_background(
    painter: &egui::Painter,
    rect: egui::Rect,
    top: Color32,
    bottom: Color32,
) {
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 2, 3);
    painter.add(egui::Shape::mesh(mesh));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn appearance_keeps_theme_alpha_opacity_rounding_and_separator_overrides() {
        let ctx = egui::Context::default();
        let mut theme = Theme::light();
        theme.tool_shelf_bg = Color32::from_rgba_unmultiplied(180, 90, 30, 128);
        let mut settings = ToolShelfSettings {
            docked: true,
            opacity: Some(0.5),
            top_separator: Some([10, 80, 240]),
            separator_opacity: 0.5,
            ..Default::default()
        };
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let style = ShelfStyle::resolve(ui, &theme, &settings, Some(12.));
            assert_eq!(style.frame.fill, theme.tool_shelf_bg.gamma_multiply(0.5));
            assert_eq!(style.frame.fill.a(), 64);
            assert_eq!(style.frame.corner_radius.nw, 12);
            assert_eq!(
                style.separator,
                Color32::from_rgb(10, 80, 240).gamma_multiply(0.5)
            );
            settings.top_separator = None;
            assert_eq!(
                ShelfStyle::resolve(ui, &theme, &settings, None).separator,
                theme.tool_shelf_border.gamma_multiply(0.5)
            );
            settings.opacity = Some(0.);
            assert_eq!(
                ShelfStyle::resolve(ui, &theme, &settings, None).frame.fill,
                Color32::TRANSPARENT
            );
        });
    }
}
