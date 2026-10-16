impl ToolsPanel {
    pub(crate) fn has_escape_operation(&self) -> bool {
        self.stroke_tracker.is_active
            || self.tool_state.brush_resize_drag_active
            || self.selection_state.dragging
            || self.lasso_state.dragging
            || self.text_state.is_editing
            || self.line_state.line_tool.stage != LineStage::Idle
            || self.shapes_state.placed.is_some()
            || self.shapes_state.is_drawing
            || self.gradient_state.drag_start.is_some()
            || self.mesh_warp_state.is_active
            || self.liquify_state.is_active
            || self.perspective_crop_state.active
            || self.zoom_tool_state.dragging
            || self.fill_state.active_fill.is_some()
            || (self.active_tool == Tool::CloneStamp && self.clone_stamp_state.source.is_some())
    }

    pub(crate) fn cancel_line_preview(&mut self, canvas: &mut CanvasState, pointer_down: bool) {
        if self.active_tool == Tool::Line && self.line_state.line_tool.stage != LineStage::Idle {
            let options = self.line_state.line_tool.options.clone();
            self.line_state.line_tool = LineToolState {
                options,
                require_mouse_release: pointer_down,
                ..Default::default()
            };
            self.stroke_tracker = StrokeTracker::default();
            canvas.clear_preview_state();
            canvas.mark_dirty(None);
        }
    }

    pub(crate) fn clear_selection_tool_session(&mut self) {
        self.clear_magic_wand_async_state();
    }

    pub fn show_compact(
        &mut self,
        ui: &mut egui::Ui,
        assets: &Assets,
        colors: (egui::Color32, egui::Color32),
        keybindings: &crate::assets::KeyBindings,
        is_text_layer: bool,
        show_colors: bool,
        palette: &crate::config::settings::ToolsPaletteSettings,
        layout: &crate::ui::panels::tools::palette::PaletteLayout,
    ) -> ToolsPanelAction {
        let (primary_color, secondary_color) = colors;
        let mut action = ToolsPanelAction::None;

        // Clear tool hint each frame; only set when hovering a tool button
        self.tool_hint.clear();

        let btn_size = 26.0; // visual button size

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

        let owned: [Vec<(Icon, Tool)>; 4] =
            crate::ui::panels::tools::palette::GROUPS.map(|group| {
                group
                    .iter()
                    .filter(|(_, id)| crate::ui::panels::tools::palette::visible(palette, id))
                    .map(|(tool, _)| (tool.icon(), *tool))
                    .collect()
            });
        let groups: Vec<_> = owned.iter().collect();
        let (grid_rect, _) = ui.allocate_exact_size(layout.size, egui::Sense::hover());
        let offset = grid_rect.min.to_vec2();
        if crate::ui::panels::tools::palette::count(palette, show_colors) == 0 {
            let response = ui.put(grid_rect, egui::Button::new(t!("settings.tools.configure")));
            if response.clicked() {
                action = ToolsPanelAction::Configure;
            }
        }
        let dark_mode = ui.visuals().dark_mode;
        let tool_btn_fill = crate::theme::Theme::icon_button_bg_for(ui);
        let tool_btn_active = crate::theme::Theme::icon_button_active_for(ui);
        let tool_btn_disabled = crate::theme::Theme::icon_button_disabled_for(ui);
        let tool_btn_rounding = crate::theme::Theme::tool_button_rounding_for(ui);

        // Paint every base first, then the moving selection, then icons/focus.
        // This prevents later buttons from covering the highlight during a transition.
        let mut target = None;
        for (gi, group) in groups.iter().enumerate() {
            for (i, (_, tool)) in group.iter().enumerate() {
                let rect = layout.groups[gi][i].translate(offset);
                let disabled =
                    is_text_layer && !matches!(tool, Tool::Text | Tool::Zoom | Tool::Pan);
                ui.painter().rect_filled(
                    rect,
                    tool_btn_rounding,
                    if disabled {
                        tool_btn_disabled
                    } else {
                        tool_btn_fill
                    },
                );
                if self.active_tool == *tool && !disabled {
                    target = Some(rect);
                }
            }
        }
        let shapes_rect = layout.shapes.translate(offset);
        if shapes_rect.is_positive() {
            ui.painter().rect_filled(
                shapes_rect,
                tool_btn_rounding,
                if is_text_layer {
                    tool_btn_disabled
                } else {
                    tool_btn_fill
                },
            );
            if self.active_tool == Tool::Shapes && !is_text_layer {
                target = Some(shapes_rect);
            }
        }
        if let Some(target) = target {
            let local = target.min - grid_rect.min;
            let animate = |key, value| {
                crate::ui::polish::animate(
                    ui.ctx(),
                    ui.id().with(key),
                    value,
                    crate::ui::polish::MotionKind::Selection,
                )
            };
            let highlight = egui::Rect::from_min_size(
                grid_rect.min
                    + egui::vec2(
                        animate("tool_selection_x", local.x),
                        animate("tool_selection_y", local.y),
                    ),
                egui::vec2(animate("tool_selection_width", target.width()), btn_size),
            );
            ui.painter()
                .rect_filled(highlight, tool_btn_rounding, tool_btn_active);
            ui.painter().rect_stroke(
                highlight,
                tool_btn_rounding,
                egui::Stroke::new(
                    crate::ui::polish::settings(ui.ctx()).border_width,
                    ui.visuals().selection.stroke.color,
                ),
                egui::StrokeKind::Inside,
            );
        }

        for (gi, group) in groups.iter().enumerate() {
            for (i, (icon, tool)) in group.iter().enumerate() {
                let btn_rect = layout.groups[gi][i].translate(offset);

                let selected = self.active_tool == *tool;

                // On text layers, only Text/Zoom/Pan are enabled
                let tool_disabled = !ui.is_enabled()
                    || (is_text_layer && !matches!(tool, Tool::Text | Tool::Zoom | Tool::Pan));

                // Manual painting (like Shapes button) for full control over fill/tint
                let resp = ui.interact(
                    btn_rect,
                    ui.id().with(("tool", icon)),
                    if tool_disabled {
                        egui::Sense::hover()
                    } else {
                        egui::Sense::click()
                    },
                );
                resp.widget_info(|| {
                    egui::WidgetInfo::selected(
                        egui::WidgetType::Button,
                        ui.is_enabled() && !tool_disabled,
                        selected,
                        icon.tooltip(),
                    )
                });
                let hovered = resp.hovered() && !tool_disabled;

                let hover_t = crate::ui::polish::hover(ui.ctx(), resp.id.with("hover"), hovered);
                if !tool_disabled && !selected {
                    let fill = if resp.is_pointer_button_down_on() {
                        ui.visuals().widgets.active.bg_fill
                    } else {
                        ui.visuals().widgets.hovered.bg_fill.gamma_multiply(hover_t)
                    };
                    ui.painter().rect_filled(btn_rect, tool_btn_rounding, fill);
                }
                crate::ui::polish::focus(ui, &resp, tool_btn_rounding);

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
        }
        for line in &layout.separators {
            ui.painter().line_segment(
                [line[0] + offset, line[1] + offset],
                egui::Stroke::new(1.0, sep_color),
            );
        }

        if shapes_rect.is_positive() {
            let shape_rect = shapes_rect;
            let shape_w = shape_rect.width();
            let icon = if shape_w > 26. {
                Icon::Shapes
            } else {
                Icon::ShapesSmall
            };
            let icon_size = if shape_w > 26. {
                egui::vec2(44., 22.)
            } else {
                egui::Vec2::splat(btn_size * 0.75)
            };
            let is_shapes = self.active_tool == Tool::Shapes;
            let shapes_disabled = is_text_layer || !ui.is_enabled();

            let resp = ui.interact(
                shape_rect,
                ui.id().with("shapes"),
                if shapes_disabled {
                    egui::Sense::hover()
                } else {
                    egui::Sense::click()
                },
            );
            resp.widget_info(|| {
                egui::WidgetInfo::selected(
                    egui::WidgetType::Button,
                    ui.is_enabled() && !shapes_disabled,
                    is_shapes,
                    t!("tool.shapes"),
                )
            });
            let hovered = resp.hovered() && !shapes_disabled;
            let hover = crate::ui::polish::hover(ui.ctx(), resp.id.with("hover"), hovered);
            if !shapes_disabled && !is_shapes {
                ui.painter().rect_filled(
                    shape_rect,
                    tool_btn_rounding,
                    ui.visuals().widgets.hovered.bg_fill.gamma_multiply(hover),
                );
            }
            crate::ui::polish::focus(ui, &resp, tool_btn_rounding);

            if let Some(texture) = assets.get_texture(icon) {
                let sized_texture = egui::load::SizedTexture::from_handle(texture);
                let img = egui::Image::from_texture(sized_texture).fit_to_exact_size(icon_size);
                let img = if shapes_disabled {
                    img.tint(egui::Color32::from_rgba_unmultiplied(128, 128, 128, 80))
                } else if dark_mode && !is_shapes {
                    img.tint(egui::Color32::WHITE)
                } else {
                    img
                };
                let img_rect = egui::Rect::from_center_size(shape_rect.center(), icon_size);
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

        if let Some([swatch, swap]) = layout.colors {
            let swatch = swatch.translate(offset);
            let swap = swap.translate(offset);
            let response = ui.interact(swatch, ui.id().with("tool_colors"), egui::Sense::click());
            response.widget_info(|| {
                egui::WidgetInfo::labeled(
                    egui::WidgetType::Button,
                    ui.is_enabled(),
                    crate::t!("ui.click_to_open_colors"),
                )
            });
            ui.painter().rect_filled(
                egui::Rect::from_min_size(
                    swatch.min + egui::vec2(8.0, 8.0),
                    egui::Vec2::splat(24.0),
                ),
                3.0,
                secondary_color,
            );
            ui.painter().rect_stroke(
                egui::Rect::from_min_size(swatch.min + egui::vec2(8., 8.), egui::Vec2::splat(24.)),
                3.,
                egui::Stroke::new(1., sep_color),
                egui::StrokeKind::Inside,
            );
            ui.painter().rect_filled(
                egui::Rect::from_min_size(swatch.min, egui::Vec2::splat(24.0)),
                3.0,
                primary_color,
            );
            ui.painter().rect_stroke(
                egui::Rect::from_min_size(swatch.min, egui::Vec2::splat(24.0)),
                3.0,
                egui::Stroke::new(1.0, sep_color),
                egui::StrokeKind::Inside,
            );
            if response
                .on_hover_text(crate::t!("ui.click_to_open_colors"))
                .clicked()
            {
                action = ToolsPanelAction::OpenColors;
            }
            let response = ui.interact(swap, ui.id().with("tool_swap"), egui::Sense::click());
            response.widget_info(|| {
                egui::WidgetInfo::labeled(
                    egui::WidgetType::Button,
                    ui.is_enabled(),
                    Icon::SwapColors.tooltip(),
                )
            });
            let theme = crate::components::color_widgets::theme(ui);
            if response.hovered() {
                ui.painter()
                    .rect_filled(swap, theme.widget_cr(5), theme.bg2);
            }
            let c = swap.center();
            let stroke = egui::Stroke::new(
                1.3,
                if response.hovered() {
                    theme.accent
                } else {
                    theme.text_muted
                },
            );
            for (y, direction) in [(-2.5, 1.0), (2.5, -1.0)] {
                ui.painter()
                    .line_segment([c + egui::vec2(-5., y), c + egui::vec2(5., y)], stroke);
                let end = c + egui::vec2(5. * direction, y);
                for dy in [-2., 2.] {
                    ui.painter()
                        .line_segment([end, end + egui::vec2(-2. * direction, dy)], stroke);
                }
            }
            if response.on_hover_text(Icon::SwapColors.tooltip()).clicked() {
                action = ToolsPanelAction::SwapColors;
            }
        }

        action
    }

    /// Draw compact overlapping primary/secondary color swatch
    fn draw_color_swatch_compact(
        ui: &mut egui::Ui,
        primary: egui::Color32,
        secondary: egui::Color32,
    ) -> bool {
        let theme = crate::components::color_widgets::theme(ui);
        let rounding = theme.widget_cr(3);
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
            painter.rect_filled(secondary_rect, rounding, secondary);
            painter.rect_stroke(
                secondary_rect,
                rounding,
                egui::Stroke::new(1.0, theme.border_color),
                egui::StrokeKind::Middle,
            );

            // Primary color (front, top-left)
            let primary_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(swatch_size, swatch_size));
            painter.rect_filled(primary_rect, rounding, primary);
            painter.rect_stroke(
                primary_rect,
                rounding,
                egui::Stroke::new(1.0, theme.border_color),
                egui::StrokeKind::Middle,
            );
        }

        response
            .on_hover_text(crate::t!("ui.click_to_open_colors"))
            .clicked()
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

        response
            .on_hover_text(crate::t!("ui.click_to_open_colors"))
            .clicked()
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

        response
            .on_hover_text(crate::t!("ui.click_to_open_colors"))
            .clicked()
    }

    /// Original full show method for sidebar (kept for compatibility)
    pub fn show(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        ui.vertical(|ui| {
            ui.heading(crate::t!("keybind_category.tools"));
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
                Tool::Smudge => crate::t!("tool.smudge"),
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
