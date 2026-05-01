impl ToolsPanel {
    pub fn show_compact(
        &mut self,
        ui: &mut egui::Ui,
        assets: &Assets,
        primary_color: egui::Color32,
        secondary_color: egui::Color32,
        keybindings: &crate::assets::KeyBindings,
        is_text_layer: bool,
    ) -> ToolsPanelAction {
        let mut action = ToolsPanelAction::None;

        // Clear tool hint each frame ÔÇö only set when hovering a tool button
        self.tool_hint.clear();

        let btn_size = 26.0; // visual button size
        let cols = 3;
        let gap = 6.0; // 1px gap between buttons

        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
        ui.spacing_mut().button_padding = egui::vec2(2.0, 2.0);

        ui.add_space(4.0);

        // Separator color for between-group dividers (accent-tinted for Signal Grid)
        let accent = ui.visuals().selection.bg_fill;
        let sep_color = if ui.visuals().dark_mode {
            egui::Color32::from_rgb(
                60u8.saturating_add(accent.r() / 8),
                60u8.saturating_add(accent.g() / 8),
                60u8.saturating_add(accent.b() / 8),
            )
        } else {
            egui::Color32::from_gray(200)
        };

        // Tool groups ÔÇö 4 groups of 6, each 2 rows ├ù 3 cols, with separators
        // PAINT: Core painting & fill tools
        let paint_tools: Vec<(Icon, Tool)> = vec![
            (Icon::Brush, Tool::Brush),
            (Icon::Pencil, Tool::Pencil),
            (Icon::Eraser, Tool::Eraser),
            (Icon::Line, Tool::Line),
            (Icon::Fill, Tool::Fill),
            (Icon::Gradient, Tool::Gradient),
        ];
        // SELECT: Region selection & movement
        let select_tools: Vec<(Icon, Tool)> = vec![
            (Icon::RectSelect, Tool::RectangleSelect),
            (Icon::EllipseSelect, Tool::EllipseSelect),
            (Icon::Lasso, Tool::Lasso),
            (Icon::MagicWand, Tool::MagicWand),
            (Icon::MovePixels, Tool::MovePixels),
            (Icon::MoveSelection, Tool::MoveSelection),
        ];
        // RETOUCH & WARP: Repair/clone + distort/transform
        let retouch_tools: Vec<(Icon, Tool)> = vec![
            (Icon::CloneStamp, Tool::CloneStamp),
            (Icon::ContentAwareBrush, Tool::ContentAwareBrush),
            (Icon::ColorRemover, Tool::ColorRemover),
            (Icon::Liquify, Tool::Liquify),
            (Icon::MeshWarp, Tool::MeshWarp),
            (Icon::PerspectiveCrop, Tool::PerspectiveCrop),
        ];
        // UTILITY: Sample, create, navigate
        let utility_tools: Vec<(Icon, Tool)> = vec![
            (Icon::ColorPicker, Tool::ColorPicker),
            (Icon::Text, Tool::Text),
            (Icon::Zoom, Tool::Zoom),
            (Icon::Pan, Tool::Pan),
        ];

        let groups: Vec<&Vec<(Icon, Tool)>> =
            vec![&paint_tools, &select_tools, &retouch_tools, &utility_tools];
        let sep_gap = 11.0; // vertical space for separator lines between groups (5px padding each side + 1px line)
        let grid_w = cols as f32 * btn_size + (cols - 1) as f32 * gap;

        // Calculate total height: all tool rows + separators between groups
        let total_tool_rows: usize = groups.iter().map(|g| g.len().div_ceil(cols)).sum();
        let num_separators = groups.len() - 1;
        let grid_h = total_tool_rows as f32 * btn_size
            + (total_tool_rows - 1) as f32 * gap
            + num_separators as f32 * sep_gap;

        // Allocate exact space for entire tool grid (all groups + separators)
        let (grid_rect, _) =
            ui.allocate_exact_size(egui::vec2(grid_w, grid_h), egui::Sense::hover());

        let mut current_y = grid_rect.min.y;
        let dark_mode = ui.visuals().dark_mode;
        let tool_btn_fill = crate::theme::Theme::icon_button_bg_for(ui);
        let tool_btn_active = crate::theme::Theme::icon_button_active_for(ui);
        let tool_btn_disabled = crate::theme::Theme::icon_button_disabled_for(ui);
        let tool_btn_rounding = crate::theme::Theme::tool_button_rounding_for(ui);

        for (gi, group) in groups.iter().enumerate() {
            let group_rows = group.len().div_ceil(cols);

            for (i, (icon, tool)) in group.iter().enumerate() {
                let col = i % cols;
                let row = i / cols;
                let x = grid_rect.min.x + col as f32 * (btn_size + gap);
                let y = current_y + row as f32 * (btn_size + gap);
                let btn_rect =
                    egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(btn_size, btn_size));

                let selected = self.active_tool == *tool;

                // On text layers, only Text/Zoom/Pan are enabled
                let tool_disabled =
                    is_text_layer && !matches!(tool, Tool::Text | Tool::Zoom | Tool::Pan);

                // Manual painting (like Shapes button) for full control over fill/tint
                let resp = ui.allocate_rect(btn_rect, egui::Sense::click());
                let hovered = resp.hovered() && !tool_disabled;

                // Background fill ÔÇö selected > hovered > recessed default
                let fill = if tool_disabled {
                    tool_btn_disabled
                } else if selected {
                    tool_btn_active
                } else if hovered {
                    ui.visuals().widgets.hovered.bg_fill
                } else {
                    tool_btn_fill
                };

                // Accent glow behind active tool
                if selected {
                    let glow_expand = 3.0;
                    let glow_rect = btn_rect.expand(glow_expand);
                    let sel = ui.visuals().selection.bg_fill;
                    let glow_color =
                        egui::Color32::from_rgba_unmultiplied(sel.r(), sel.g(), sel.b(), 40);
                    ui.painter().rect_filled(glow_rect, tool_btn_rounding + 2.0, glow_color);
                }

                ui.painter().rect_filled(btn_rect, tool_btn_rounding, fill);

                // Border
                if selected {
                    ui.painter().rect_stroke(
                        btn_rect,
                        tool_btn_rounding,
                        egui::Stroke::new(2.0, ui.visuals().selection.bg_fill),
                        egui::StrokeKind::Middle,
                    );
                } else if hovered {
                    ui.painter().rect_stroke(
                        btn_rect,
                        tool_btn_rounding,
                        ui.visuals().widgets.hovered.bg_stroke,
                        egui::StrokeKind::Middle,
                    );
                }

                // Draw icon image or emoji fallback
                if let Some(texture) = assets.get_texture(*icon) {
                    let sized_texture = egui::load::SizedTexture::from_handle(texture);
                    let img_size = egui::vec2(btn_size * 0.75, btn_size * 0.75);
                    // Dim disabled tools, tint white in dark mode for contrast
                    let tint = if tool_disabled {
                        egui::Color32::from_rgba_unmultiplied(128, 128, 128, 80)
                    } else {
                        egui::Color32::WHITE
                    };
                    let img = egui::Image::from_texture(sized_texture)
                        .fit_to_exact_size(img_size)
                        .tint(tint);
                    let img_rect = egui::Rect::from_center_size(btn_rect.center(), img_size);
                    img.paint_at(ui, img_rect);
                } else {
                    let text_color = if tool_disabled {
                        egui::Color32::from_rgba_unmultiplied(128, 128, 128, 80)
                    } else if selected {
                        egui::Color32::WHITE
                    } else {
                        ui.visuals().text_color()
                    };
                    ui.painter().text(
                        btn_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        icon.emoji(),
                        egui::FontId::proportional(btn_size * 0.42),
                        text_color,
                    );
                }

                if !tool_disabled
                    && resp
                        .on_hover_text(icon.tooltip_with_keybind(keybindings))
                        .clicked()
                {
                    self.change_tool(*tool);
                }
                if hovered {
                    self.tool_hint = Self::tool_hint_for(*tool);
                }
            }

            // Advance Y past this group's rows
            current_y += group_rows as f32 * btn_size + (group_rows - 1) as f32 * gap;

            // Draw separator line between groups (not after the last group)
            if gi < groups.len() - 1 {
                let sep_y = current_y + sep_gap * 0.5;
                ui.painter().line_segment(
                    [
                        egui::pos2(grid_rect.min.x, sep_y),
                        egui::pos2(grid_rect.min.x + grid_w, sep_y),
                    ],
                    egui::Stroke::new(1.0, sep_color),
                );
                current_y += sep_gap;
            }
        }

        // Shapes ÔÇö double-wide, sits in the last row of the utility group
        {
            // The utility group has 4 tools ÔåÆ row0: 3 tools, row1: Pan only (col 0)
            // Shapes goes in col 1-2 of that last row
            let shape_x = grid_rect.min.x + 1.0 * (btn_size + gap);
            let shape_y = current_y - btn_size; // last row Y (Pan's row)
            let remaining_cols = cols - 1; // 2 columns
            let shape_w = remaining_cols as f32 * btn_size + (remaining_cols - 1) as f32 * gap;
            let shape_rect = egui::Rect::from_min_size(
                egui::pos2(shape_x, shape_y),
                egui::vec2(shape_w, btn_size),
            );

            let icon = Icon::Shapes;
            let is_shapes = self.active_tool == Tool::Shapes;
            let shapes_disabled = is_text_layer;

            let resp = ui.allocate_rect(shape_rect, egui::Sense::click());
            let hovered = resp.hovered() && !shapes_disabled;

            let fill = if shapes_disabled {
                tool_btn_disabled
            } else if is_shapes {
                tool_btn_active
            } else if hovered {
                ui.visuals().widgets.hovered.bg_fill
            } else {
                tool_btn_fill
            };

            // Accent glow behind active Shapes button
            if is_shapes {
                let glow_expand = 3.0;
                let glow_rect = shape_rect.expand(glow_expand);
                let sel = ui.visuals().selection.bg_fill;
                let glow_color =
                    egui::Color32::from_rgba_unmultiplied(sel.r(), sel.g(), sel.b(), 40);
                ui.painter().rect_filled(glow_rect, 6.0, glow_color);
            }

            ui.painter().rect_filled(shape_rect, 4.0, fill);

            if is_shapes {
                ui.painter().rect_stroke(
                    shape_rect,
                    2.0,
                    egui::Stroke::new(2.0, ui.visuals().selection.bg_fill),
                    egui::StrokeKind::Middle,
                );
            } else if hovered {
                ui.painter().rect_stroke(
                    shape_rect,
                    4.0,
                    ui.visuals().widgets.hovered.bg_stroke,
                    egui::StrokeKind::Middle,
                );
            }

            if let Some(texture) = assets.get_texture(icon) {
                let sized_texture = egui::load::SizedTexture::from_handle(texture);
                let img = egui::Image::from_texture(sized_texture)
                    .fit_to_exact_size(egui::vec2(shape_w * 0.75, btn_size * 0.75));
                let img = if shapes_disabled {
                    img.tint(egui::Color32::from_rgba_unmultiplied(128, 128, 128, 80))
                } else if dark_mode && !is_shapes {
                    img.tint(egui::Color32::WHITE)
                } else {
                    img
                };
                let img_rect = egui::Rect::from_center_size(
                    shape_rect.center(),
                    egui::vec2(shape_w * 0.75, btn_size * 0.75),
                );
                img.paint_at(ui, img_rect);
            } else {
                let text_color = if shapes_disabled {
                    egui::Color32::from_rgba_unmultiplied(128, 128, 128, 80)
                } else if is_shapes {
                    egui::Color32::WHITE
                } else {
                    ui.visuals().text_color()
                };
                ui.painter().text(
                    shape_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    icon.emoji(),
                    egui::FontId::proportional(btn_size * 0.42),
                    text_color,
                );
            }

            if !shapes_disabled
                && resp
                    .on_hover_text(icon.tooltip_with_keybind(keybindings))
                    .clicked()
            {
                self.change_tool(Tool::Shapes);
            }
            if hovered {
                self.tool_hint = Self::tool_hint_for(Tool::Shapes);
            }
        }

        ui.add_space(8.0);

        // Separator line before color swatches
        let sep_width = grid_w;
        let (sep_rect, _) =
            ui.allocate_exact_size(egui::vec2(sep_width, 1.0), egui::Sense::hover());
        ui.painter().line_segment(
            [sep_rect.left_center(), sep_rect.right_center()],
            egui::Stroke::new(1.0, sep_color),
        );
        ui.add_space(10.0);

        // Color swatches ÔÇö centered in panel
        ui.horizontal(|ui| {
            ui.add_space(28.0);
            if Self::draw_color_swatch_compact(ui, primary_color, secondary_color) {
                action = ToolsPanelAction::OpenColors;
            }
        });

        ui.add_space(-8.0);

        // Swap button ÔÇö centered in panel (frameless)
        ui.horizontal(|ui| {
            ui.add_space(28.0);
            let clicked = if let Some(texture) = assets.get_texture(Icon::SwapColors) {
                let sized_texture = egui::load::SizedTexture::from_handle(texture);
                let img = egui::Image::from_texture(sized_texture)
                    .fit_to_exact_size(egui::vec2(16.0, 16.0));
                ui.add(egui::Button::image(img).frame(false))
                    .on_hover_text(Icon::SwapColors.tooltip())
                    .clicked()
            } else {
                assets.small_icon_button(ui, Icon::SwapColors).clicked()
            };
            if clicked {
                action = ToolsPanelAction::SwapColors;
            }
        });

        ui.add_space(4.0);

        action
    }

    /// Draw compact overlapping primary/secondary color swatch
    fn draw_color_swatch_compact(
        ui: &mut egui::Ui,
        primary: egui::Color32,
        secondary: egui::Color32,
    ) -> bool {
        let swatch_size = 24.0;
        let offset = 8.0;
        let total_size = egui::vec2(swatch_size + offset, swatch_size + offset);

        let (rect, response) = ui.allocate_exact_size(total_size, egui::Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Secondary color (back, offset down-right)
            let secondary_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(offset, offset),
                egui::vec2(swatch_size, swatch_size),
            );
            painter.rect_filled(secondary_rect, 2.0, secondary);
            painter.rect_stroke(
                secondary_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(100)),
                egui::StrokeKind::Middle,
            );

            // Primary color (front, top-left)
            let primary_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(swatch_size, swatch_size));
            painter.rect_filled(primary_rect, 2.0, primary);
            painter.rect_stroke(
                primary_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(100)),
                egui::StrokeKind::Middle,
            );
        }

        response.on_hover_text("Click to open Colors").clicked()
    }

    /// Draw larger overlapping primary/secondary color swatch (centered)
    /// Returns true if clicked (to open colors panel)
    fn draw_color_swatch_large(
        ui: &mut egui::Ui,
        primary: egui::Color32,
        secondary: egui::Color32,
    ) -> bool {
        let swatch_size = 32.0;
        let offset = 10.0;
        let total_size = egui::vec2(swatch_size + offset, swatch_size + offset);

        let (rect, response) = ui.allocate_exact_size(total_size, egui::Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Secondary color (back, offset down-right)
            let secondary_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(offset, offset),
                egui::vec2(swatch_size, swatch_size),
            );
            painter.rect_filled(secondary_rect, 3.0, secondary);
            painter.rect_stroke(
                secondary_rect,
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
                egui::StrokeKind::Middle,
            );

            // Primary color (front, top-left)
            let primary_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(swatch_size, swatch_size));
            painter.rect_filled(primary_rect, 3.0, primary);
            painter.rect_stroke(
                primary_rect,
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
                egui::StrokeKind::Middle,
            );
        }

        response.on_hover_text("Click to open Colors").clicked()
    }

    /// Draw overlapping primary/secondary color swatch
    /// Returns true if clicked (to open colors panel)
    pub fn draw_color_swatch(
        ui: &mut egui::Ui,
        primary: egui::Color32,
        secondary: egui::Color32,
    ) -> bool {
        let swatch_size = 24.0;
        let offset = 8.0;
        let total_size = egui::vec2(swatch_size + offset, swatch_size + offset);

        let (rect, response) = ui.allocate_exact_size(total_size, egui::Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Secondary color (back, offset down-right)
            let secondary_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(offset, offset),
                egui::vec2(swatch_size, swatch_size),
            );
            painter.rect_filled(secondary_rect, 2.0, secondary);
            painter.rect_stroke(
                secondary_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
                egui::StrokeKind::Middle,
            );

            // Primary color (front, top-left)
            let primary_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(swatch_size, swatch_size));
            painter.rect_filled(primary_rect, 2.0, primary);
            painter.rect_stroke(
                primary_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
                egui::StrokeKind::Middle,
            );
        }

        response.on_hover_text("Click to open Colors").clicked()
    }

    /// Original full show method for sidebar (kept for compatibility)
    pub fn show(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        ui.vertical(|ui| {
            ui.heading("Tools");
            ui.separator();

            // Large icon-style tool buttons
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;

                let is_brush = self.active_tool == Tool::Brush;
                if assets.icon_selectable(ui, Icon::Brush, is_brush) {
                    self.change_tool(Tool::Brush);
                }

                let is_eraser = self.active_tool == Tool::Eraser;
                if assets.icon_selectable(ui, Icon::Eraser, is_eraser) {
                    self.change_tool(Tool::Eraser);
                }

                let is_line = self.active_tool == Tool::Line;
                if assets.icon_selectable(ui, Icon::Line, is_line) {
                    self.change_tool(Tool::Line);
                }

                let is_rect_sel = self.active_tool == Tool::RectangleSelect;
                if assets.icon_selectable(ui, Icon::RectSelect, is_rect_sel) {
                    self.change_tool(Tool::RectangleSelect);
                }

                let is_ellipse_sel = self.active_tool == Tool::EllipseSelect;
                if assets.icon_selectable(ui, Icon::EllipseSelect, is_ellipse_sel) {
                    self.change_tool(Tool::EllipseSelect);
                }

                let is_move_px = self.active_tool == Tool::MovePixels;
                if assets.icon_selectable(ui, Icon::MovePixels, is_move_px) {
                    self.change_tool(Tool::MovePixels);
                }

                let is_move_sel = self.active_tool == Tool::MoveSelection;
                if assets.icon_selectable(ui, Icon::MoveSelection, is_move_sel) {
                    self.change_tool(Tool::MoveSelection);
                }
            });

            ui.separator();

            // Tool name label
            let tool_name = match self.active_tool {
                Tool::Brush => t!("tool.brush"),
                Tool::Eraser => t!("tool.eraser"),
                Tool::Pencil => t!("tool.pencil"),
                Tool::Line => t!("tool.line"),
                Tool::RectangleSelect => t!("tool.rectangle_select"),
                Tool::EllipseSelect => t!("tool.ellipse_select"),
                Tool::MovePixels => t!("tool.move_pixels"),
                Tool::MoveSelection => t!("tool.move_selection"),
                Tool::MagicWand => t!("tool.magic_wand"),
                Tool::Fill => t!("tool.fill"),
                Tool::ColorPicker => t!("tool.color_picker"),
                Tool::Gradient => t!("tool.gradient"),
                Tool::ContentAwareBrush => t!("tool.content_aware_fill"),
                Tool::Liquify => t!("tool.liquify"),
                Tool::MeshWarp => t!("tool.mesh_warp"),
                Tool::ColorRemover => t!("tool.color_remover"),
                Tool::Smudge => "Smudge".to_string(),
                Tool::CloneStamp => t!("tool.clone_stamp"),
                Tool::Text => t!("tool.text"),
                Tool::PerspectiveCrop => t!("tool.perspective_crop"),
                Tool::Lasso => t!("tool.lasso"),
                Tool::Zoom => t!("tool.zoom"),
                Tool::Pan => t!("tool.pan"),
                Tool::Shapes => t!("tool.shapes"),
            };
            ui.label(egui::RichText::new(tool_name).strong());
        });
    }
}
