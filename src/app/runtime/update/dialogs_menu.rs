include!("dialogs_menu/modal_flow.rs");

impl PaintFEApp {
    fn show_runtime_dialogs_menu(&mut self, ctx: &egui::Context, root_ui: &mut egui::Ui) {
        let popup_was_open = egui::Popup::is_any_open(ctx);
        self.reset_ui_cursor_blocking_rects();
        let modal_open = self.handle_runtime_modal_flow(ctx) || self.processing_busy();

        // --- Top Menu Bar ---
        let menu_kb = self.settings.keybindings.clone();
        ctx.data_mut(|data| {
            data.insert_temp(
                egui::Id::new("paintfe_command_keybindings"),
                std::sync::Arc::new(menu_kb.clone()),
            )
        });
        let has_project = !self.projects.is_empty();
        #[allow(deprecated)]
        let menu_resp = egui::Panel::top("menu_bar")
            .frame(self.theme.menu_frame())
            .exact_size(28.0)
            .show(root_ui, |ui| {
                if modal_open { ui.disable(); }
                egui::MenuBar::new().config(crate::ui::menus::config()).ui(ui, |ui| {
                    ui.menu_button(t!("menu.file"), |ui| {
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuFileNew, &t!("menu.file.new"), true, &menu_kb, crate::config::keybindings::BindableAction::NewFile).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::NewFile); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuFileOpen, &t!("menu.file.open"), true, &menu_kb, crate::config::keybindings::BindableAction::OpenFile).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::OpenFile); ui.close(); }
                        ui.menu_button(crate::t!("ui.recent_files"), |ui| {
                            let recent = self.settings.recent_files.clone();
                            let cap = (ctx.content_rect().width() * 0.5).clamp(1.0, 700.0);
                            let font = egui::TextStyle::Button.resolve(ui.style());
                            let text_width = recent.iter().map(|path| {
                                let label = crate::services::io::display_path(std::path::Path::new(path));
                                ui.painter().layout_no_wrap(label, font.clone(), ui.visuals().text_color()).size().x
                            }).fold(0.0_f32, f32::max);
                            let width = (text_width + 24.0).max(160.0).min(cap);
                            ui.set_width(width);
                            if recent.is_empty() { ui.weak(crate::t!("ui.no_recent_files")); }
                            for path in recent {
                                let label = crate::services::io::display_path(std::path::Path::new(&path));
                                let clipped = ui.painter().layout_no_wrap(label.clone(), font.clone(), ui.visuals().text_color()).size().x > width - 2.0 * ui.spacing().button_padding.x;
                                let response = ui.add_sized([width, 22.0], egui::Button::new("").left_text(egui::RichText::new(&label)).truncate());
                                let response = if clipped { response.on_hover_text(&label) } else { response };
                                if response.clicked() {
                                    self.open_file_by_path(PathBuf::from(path), ctx.input(|i| i.time));
                                    ui.close();
                                }
                            }
                            if self.show_command_button(ui,ctx,crate::config::commands::MenuCommand::FileClearRecent) { ui.close(); }
                        });
                        ui.separator();
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuFileSave, &t!("menu.file.save"), has_project, &menu_kb, crate::config::keybindings::BindableAction::Save).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Save); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuFileSaveAs, &t!("menu.file.save_as"), has_project, &menu_kb, crate::config::keybindings::BindableAction::SaveAs).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::SaveAs); ui.close(); }
                        {
                            let any_dirty =
                                self.projects.iter().any(Project::needs_save);
                            if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuFileSaveAll, &t!("menu.file.save_all"), any_dirty, &menu_kb, crate::config::keybindings::BindableAction::SaveAll).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::SaveAll); ui.close(); }
                        }
                        ui.separator();
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuFileInfo, &t!("menu.file.info"), has_project, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FileInfo)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FileInfo)); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::Duplicate, &t!("menu.file.duplicate"), has_project, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FileDuplicate)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FileDuplicate)); ui.close(); }
                        ui.separator();
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuFilePrint, &t!("menu.file.print"), has_project, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FilePrint)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FilePrint)); ui.close(); }
                        ui.separator();
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::Close, &t!("ui.close"), has_project, &menu_kb, crate::config::keybindings::BindableAction::CloseProject).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::CloseProject); ui.close(); }
                        ui.separator();
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuFileQuit, &t!("menu.file.quit"), true, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FileQuit)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FileQuit)); ui.close(); }
                    });

                    ui.menu_button(t!("menu.edit"), |ui| {
                        let can_undo = self.can_undo_action();
                        let can_redo = self.can_redo_action();

                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuEditUndo, &t!("menu.edit.undo"), can_undo, &menu_kb, crate::config::keybindings::BindableAction::Undo).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Undo); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuEditRedo, &t!("menu.edit.redo"), can_redo, &menu_kb, crate::config::keybindings::BindableAction::Redo).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Redo); ui.close(); }

                        ui.separator();

                        let has_sel = self
                            .active_project()
                            .is_some_and(|p| p.canvas_state.has_selection());
                        let can_copy = has_sel || self.paste_overlay.is_some();
                        let has_clip = crate::ops::clipboard::has_clipboard_image(ctx);

                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Cut.icon(), &t!("menu.edit.cut"), has_sel, &menu_kb, crate::config::keybindings::BindableAction::Cut).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Cut); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuEditCopy, &t!("menu.edit.copy"), can_copy, &menu_kb, crate::config::keybindings::BindableAction::Copy).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Copy); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Paste.icon(), &t!("menu.edit.paste"), has_project && has_clip, &menu_kb, crate::config::keybindings::BindableAction::Paste).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Paste); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::PasteLayer.icon(), &t!("menu.edit.paste_as_layer"), has_project && has_clip, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::EditPasteAsLayer)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::EditPasteAsLayer)); ui.close(); }

                        ui.separator();

                        // Selection operations
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::SelectAll.icon(), &t!("menu.edit.select_all"), has_project, &menu_kb, crate::config::keybindings::BindableAction::SelectAll).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::SelectAll); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Deselect.icon(), &t!("menu.edit.deselect"), has_sel, &menu_kb, crate::config::keybindings::BindableAction::Deselect).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Deselect); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::InvertSelection.icon(), &t!("menu.edit.invert_selection"), has_sel, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::EditInvertSelection)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::EditInvertSelection)); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::ColorRange.icon(), &format!("{}...", crate::t!("ui.select_color_range")), has_project, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::SelectColorRange)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::SelectColorRange)); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::SelectAll.icon(), &t!("menu.edit.select_content_bounds"), has_project, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::EditSelectContentBounds)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::EditSelectContentBounds)); ui.close(); }

                        ui.separator();

                        if self.assets.menu_item_shortcut_enabled(ui, Icon::MenuEditPreferences, &t!("menu.edit.preferences"), true, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::EditPreferences)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::EditPreferences)); ui.close(); }
                    });

                    // ==================== CANVAS MENU (was: Image) ====================
                    let no_dialog = !modal_open;
                    ui.add_enabled_ui(has_project, |ui| ui.menu_button(t!("menu.canvas"), |ui| {
                        if !has_project {
                            ui.disable();
                        }
                        if self.assets.menu_item_shortcut_below_enabled(ui, HistoryAction::ResizeImage.icon(), &t!("menu.canvas.resize_image"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ResizeImage).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ResizeImage); ui.close(); }
                        if self.assets.menu_item_shortcut_below_enabled(ui, HistoryAction::ResizeCanvas.icon(), &t!("menu.canvas.resize_canvas"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ResizeCanvas).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ResizeCanvas); ui.close(); }
                        let has_sel = self
                            .active_project()
                            .is_some_and(|p| p.canvas_state.has_selection());
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Crop.icon(), &t!("menu.canvas.crop_to_selection"), has_sel, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasCropToSelection)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasCropToSelection)); ui.close(); }
                        ui.separator();
                        if self.assets.menu_item_shortcut_enabled(ui, Icon::Rename, &t!("menu.canvas.new_text_layer"), true, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasNewTextLayer)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasNewTextLayer)); ui.close(); }
                        ui.separator();
                        let align_enabled = no_dialog
                            && self.active_project().is_some_and(|p| {
                                p.canvas_state
                                    .layers
                                    .get(p.canvas_state.active_layer_index)
                                    .is_some_and(|l| !l.is_text_layer())
                            });
                        let align_resp = self.assets.menu_item_enabled(
                            ui,
                            HistoryAction::AlignLayer.icon(),
                            &t!("menu.canvas.align"),
                            align_enabled,
                        );
                        if !align_enabled {
                            align_resp
                                .clone()
                                .on_disabled_hover_text(crate::t!("ui.align_works_only_on_raster_layers"));
                        }
                        if align_resp.clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasAlign)); ui.close(); }

                        ui.separator();
                        ui.menu_button(t!("menu.canvas.flip_canvas"), |ui| {
                            ui.set_min_width(ui.min_rect().width().max(160.0));
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::FlipHorizontal.icon(), &t!("menu.canvas.flip_horizontal"), true, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasFlipHorizontal)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasFlipHorizontal)); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::FlipVertical.icon(), &t!("menu.canvas.flip_vertical"), true, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasFlipVertical)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasFlipVertical)); ui.close(); }
                        });
                        ui.menu_button(t!("menu.canvas.rotate_canvas"), |ui| {
                            ui.set_min_width(ui.min_rect().width().max(160.0));
                            if self.show_command_button(ui,ctx,crate::config::commands::MenuCommand::CanvasStraighten) { ui.close(); }
                            ui.separator();
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::RotateCw.icon(), &t!("menu.canvas.rotate_90cw"), true, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate90Cw)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate90Cw)); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::RotateCcw.icon(), &t!("menu.canvas.rotate_90ccw"), true, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate90Ccw)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate90Ccw)); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Rotate180.icon(), &t!("menu.canvas.rotate_180"), true, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate180)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate180)); ui.close(); }
                        });
                        ui.separator();
                        if self.assets.menu_item_shortcut_below_enabled(ui, HistoryAction::Flatten.icon(), &t!("menu.canvas.flatten_all_layers"), true, &menu_kb, crate::config::keybindings::BindableAction::FlattenLayers).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FlattenLayers); ui.close(); }
                    }));

                    // (Layers menu removed -- layer operations are now in the Layers Panel context menu)

                    // ==================== COLOR MENU (was: Adjustments) ====================
                    ui.add_enabled_ui(has_project, |ui| ui.menu_button(t!("menu.color"), |ui| {
                        if !has_project {
                            ui.disable();
                        }
                        // --- Instant adjustments (no dialog) ---
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::AutoLevels.icon(), &t!("menu.color.auto_levels"), true, &menu_kb, crate::config::keybindings::BindableAction::ColorAutoLevels).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorAutoLevels); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Desaturate.icon(), &t!("menu.color.desaturate"), true, &menu_kb, crate::config::keybindings::BindableAction::ColorDesaturate).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorDesaturate); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::InvertColors.icon(), &t!("menu.color.invert_colors"), true, &menu_kb, crate::config::keybindings::BindableAction::ColorInvertColors).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorInvertColors); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::InvertAlpha.icon(), &t!("menu.color.invert_alpha"), true, &menu_kb, crate::config::keybindings::BindableAction::ColorInvertAlpha).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorInvertAlpha); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::ColorToAlpha.icon(), &t!("menu.color.color_to_alpha"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::ColorColorToAlpha)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::ColorColorToAlpha)); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::RecoverTransparency.icon(), &t!("menu.color.recover_transparency"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::ColorRecoverTransparency)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::ColorRecoverTransparency)); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Sepia.icon(), &t!("menu.color.sepia_tone"), true, &menu_kb, crate::config::keybindings::BindableAction::ColorSepiaTone).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorSepiaTone); ui.close(); }
                        ui.separator();
                        // --- Parameterized adjustments (with dialog) ---
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::BrightnessContrast.icon(), &t!("menu.color.brightness_contrast"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorBrightnessContrast).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorBrightnessContrast); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Curves.icon(), &t!("menu.color.curves"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorCurves).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorCurves); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Exposure.icon(), &t!("menu.color.exposure"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorExposure).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorExposure); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::HighlightsShadows.icon(), &t!("menu.color.highlights_shadows"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorHighlightsShadows).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorHighlightsShadows); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::HueSaturation.icon(), &t!("menu.color.hue_saturation"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorHueSaturation).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorHueSaturation); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Levels.icon(), &t!("menu.color.levels"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorLevels).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorLevels); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::TemperatureTint.icon(), &t!("menu.color.temperature_tint"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorTemperatureTint).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorTemperatureTint); ui.close(); }
                        ui.separator();
                        // --- Additional color adjustments ---
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Vibrance.icon(), &t!("menu.color.vibrance"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorVibrance).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorVibrance); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Threshold.icon(), &t!("menu.color.threshold"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorThreshold).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorThreshold); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Posterize.icon(), &t!("menu.color.posterize"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorPosterize).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorPosterize); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::ColorBalance.icon(), &t!("menu.color.color_balance"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorBalance).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorBalance); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::GradientMap.icon(), &t!("menu.color.gradient_map"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorGradientMap).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorGradientMap); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::BlackAndWhite.icon(), &t!("menu.color.black_and_white"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::ColorBlackAndWhite).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ColorBlackAndWhite); ui.close(); }
                    }));

                    // ==================== FILTER MENU (was: Effects) ====================
                    ui.add_enabled_ui(has_project, |ui| ui.menu_button(t!("menu.filter"), |ui| {
                        if !has_project {
                            ui.disable();
                        }
                        // -- Blur submenu --
                        ui.menu_button(t!("menu.filter.blur"), |ui| {
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::GaussianBlur.icon(), &t!("menu.filter.blur.gaussian"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterGaussianBlur).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterGaussianBlur); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::BokehBlur.icon(), &t!("menu.filter.blur.bokeh"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterBokehBlur).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterBokehBlur); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::MotionBlur.icon(), &t!("menu.filter.blur.motion"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterMotionBlur).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterMotionBlur); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::BoxBlur.icon(), &t!("menu.filter.blur.box"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterBoxBlur).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterBoxBlur); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::ZoomBlur.icon(), &t!("menu.filter.blur.zoom"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterZoomBlur).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterZoomBlur); ui.close(); }
                        });

                        // -- Sharpen submenu (was in Stylize + Noise) --
                        ui.menu_button(t!("menu.filter.sharpen"), |ui| {
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Sharpen.icon(), &t!("menu.filter.sharpen.sharpen"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterSharpen).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterSharpen); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::ReduceNoise.icon(), &t!("menu.filter.sharpen.reduce_noise"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterReduceNoise).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterReduceNoise); ui.close(); }
                        });

                        // -- Distort submenu --
                        ui.menu_button(t!("menu.filter.distort"), |ui| {
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Crystallize.icon(), &t!("menu.filter.distort.crystallize"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterCrystallize).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterCrystallize); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Dents.icon(), &t!("menu.filter.distort.dents"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterDents).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterDents); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Pixelate.icon(), &t!("menu.filter.distort.pixelate"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterPixelate).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterPixelate); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Bulge.icon(), &t!("menu.filter.distort.bulge_pinch"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterBulge).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterBulge); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Twist.icon(), &t!("menu.filter.distort.twist"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterTwist).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterTwist); ui.close(); }
                        });

                        // -- Noise submenu --
                        ui.menu_button(t!("menu.filter.noise"), |ui| {
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::AddNoise.icon(), &t!("menu.filter.noise.add_noise"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterAddNoise).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterAddNoise); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Median.icon(), &t!("menu.filter.noise.median"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterMedian).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterMedian); ui.close(); }
                        });

                        // -- Stylize submenu (absorbs old Artistic) --
                        ui.menu_button(t!("menu.filter.stylize"), |ui| {
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Glow.icon(), &t!("menu.filter.stylize.glow"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterGlow).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterGlow); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Vignette.icon(), &t!("menu.filter.stylize.vignette"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterVignette).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterVignette); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Halftone.icon(), &t!("menu.filter.stylize.halftone"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterHalftone).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterHalftone); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Ink.icon(), &t!("menu.filter.stylize.ink"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterInk).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterInk); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::OilPainting.icon(), &t!("menu.filter.stylize.oil_painting"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterOilPainting).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterOilPainting); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::ColorFilter.icon(), &t!("menu.filter.stylize.color_filter"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterColorFilter).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterColorFilter); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::CanvasBorder.icon(), &t!("menu.filter.stylize.canvas_border"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FilterStylizeCanvasBorder)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FilterStylizeCanvasBorder)); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::SeamlessTexture.icon(), &t!("menu.filter.stylize.seamless"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FilterStylizeSeamless)).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::FilterStylizeSeamless)); ui.close(); }
                            #[cfg(not(target_arch = "wasm32"))]
                            if self.settings.paintdotnet_plugins_enabled {
                                let plugins: Vec<_> = crate::paintdotnet_plugins::PluginManager::load()
                                    .enabled_plugins()
                                    .filter(|plugin| plugin.category.eq_ignore_ascii_case("stylize"))
                                    .cloned()
                                    .collect();
                                if !plugins.is_empty() {
                                    ui.separator();
                                    for plugin in plugins {
                                        let label = format!("🧩 {}", plugin.name);
                                        if ui
                                            .add_enabled(no_dialog, egui::Button::new(label).frame(false))
                                            .clicked()
                                        {
                                            if let Some(project) = self.active_project()
                                                && let Some(dialog) = crate::paintdotnet_plugins::PaintDotNetPluginDialog::new(
                                                    &project.canvas_state,
                                                    plugin,
                                                )
                                            {
                                                self.active_dialog = ActiveDialog::PaintDotNetPlugin(Box::new(dialog));
                                            }
                                            ui.close();
                                        }
                                    }
                                }
                            }
                        });

                        #[cfg(not(target_arch = "wasm32"))]
                        if self.settings.paintdotnet_plugins_enabled {
                            let plugins: Vec<_> = crate::paintdotnet_plugins::PluginManager::load()
                                .enabled_plugins()
                                .filter(|plugin| !plugin.category.eq_ignore_ascii_case("stylize"))
                                .cloned()
                                .collect();
                            if !plugins.is_empty() {
                                ui.menu_button(t!("menu.filter.paintdotnet_plugins"), |ui| {
                                    for plugin in plugins {
                                        let label = format!("{} · {}", plugin.category, plugin.name);
                                        if ui
                                            .add_enabled(
                                                no_dialog,
                                                egui::Button::new(label).frame(false),
                                            )
                                            .clicked()
                                        {
                                            if let Some(project) = self.active_project()
                                                && let Some(dialog) = crate::paintdotnet_plugins::PaintDotNetPluginDialog::new(
                                                    &project.canvas_state,
                                                    plugin,
                                                )
                                            {
                                                self.active_dialog = ActiveDialog::PaintDotNetPlugin(Box::new(dialog));
                                            }
                                            ui.close();
                                        }
                                    }
                                });
                            }
                        }

                        // -- Glitch submenu --
                        ui.menu_button(t!("menu.filter.glitch"), |ui| {
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::PixelDrag.icon(), &t!("menu.filter.glitch.pixel_drag"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterPixelDrag).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterPixelDrag); ui.close(); }
                            if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::RgbDisplace.icon(), &t!("menu.filter.glitch.rgb_displace"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::FilterRgbDisplace).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterRgbDisplace); ui.close(); }
                        });

                        // -- AI submenu --
                        ui.separator();
                        let remove_bg_resp = self.assets.menu_item_enabled(
                            ui,
                            HistoryAction::RemoveBackground.icon(),
                            &t!("menu.filter.remove_background"),
                            no_dialog && !self.background_removal_pending,
                        );
                        if !self.onnx_available {
                            remove_bg_resp.clone().on_hover_text(
                                crate::t!("menu.filter.remove_background.disabled_hint"),
                            );
                        }
                        if remove_bg_resp.clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::FilterRemoveBackground); ui.close(); }

                        // -- Custom Scripts submenu (only shown if scripts exist) --
                        if !self.custom_scripts.is_empty() {
                            ui.separator();
                            ui.menu_button(crate::t!("theme_preset.custom"), |ui| {
                                let mut action: Option<CustomScriptAction> = None;
                                for (idx, effect) in self.custom_scripts.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        // Run button -- effect name, takes remaining space
                                        if ui.button(&effect.name).clicked() {
                                            action = Some(CustomScriptAction::Run(idx));
                                            ui.close();
                                        }
                                        // Push Edit and Delete to the right
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                let del_btn = ui
                                                    .small_button(
                                                        egui::RichText::new(crate::t!("ui.del"))
                                                            .color(egui::Color32::from_rgb(
                                                                180, 80, 80,
                                                            ))
                                                            .size(10.0),
                                                    )
                                                    .on_hover_text(crate::t!("ui.delete_this_custom_effect"));
                                                if del_btn.clicked() {
                                                    action = Some(CustomScriptAction::Delete(idx));
                                                    ui.close();
                                                }
                                                let edit_btn = ui
                                                    .small_button(
                                                        egui::RichText::new(crate::t!("keybind_category.edit")).size(10.0),
                                                    )
                                                    .on_hover_text(crate::t!("ui.edit_in_script_editor"));
                                                if edit_btn.clicked() {
                                                    action = Some(CustomScriptAction::Edit(idx));
                                                    ui.close();
                                                }
                                            },
                                        );
                                    });
                                }
                                if let Some(act) = action {
                                    match act {
                                        CustomScriptAction::Run(idx) => {
                                            if let Some(effect) = self.custom_scripts.get(idx) {
                                                let code = effect.code.clone();
                                                let name = effect.name.clone();
                                                self.run_custom_script(code, name);
                                            }
                                        }
                                        CustomScriptAction::Edit(idx) => {
                                            if let Some(effect) = self.custom_scripts.get(idx) {
                                                self.script_editor.code = effect.code.clone();
                                                self.window_visibility.script_editor = true;
                                            }
                                        }
                                        CustomScriptAction::Delete(idx) => {
                                            if let Some(effect) = self.custom_scripts.get(idx) {
                                                script_editor::delete_custom_effect(&effect.name);
                                            }
                                            self.custom_scripts.remove(idx);
                                        }
                                    }
                                }
                            });
                        }
                    }));

                    // ==================== GENERATE MENU (was: Effects > Render) ====================
                    ui.add_enabled_ui(has_project, |ui| ui.menu_button(t!("menu.generate"), |ui| {
                        if !has_project {
                            ui.disable();
                        }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Grid.icon(), &t!("menu.generate.grid"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::GenerateGrid).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::GenerateGrid); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::DropShadow.icon(), &t!("menu.generate.drop_shadow"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::GenerateDropShadow).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::GenerateDropShadow); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Outline.icon(), &t!("menu.generate.outline"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::GenerateOutline).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::GenerateOutline); ui.close(); }
                        if self.assets.menu_item_shortcut_enabled(ui, HistoryAction::Contours.icon(), &t!("menu.generate.contours"), no_dialog, &menu_kb, crate::config::keybindings::BindableAction::GenerateContours).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::GenerateContours); ui.close(); }
                    }));

                    let view_menu = ui.menu_button(t!("menu.view"), |ui| {
                        // Panel toggles
                        ui.label(egui::RichText::new(crate::t!("menu.view.panels_heading")).strong().size(11.0));
                        self.show_command_toggle(ui,ctx,crate::config::commands::MenuCommand::ViewTools,self.window_visibility.tools);
                        self.show_command_toggle(ui,ctx,crate::config::commands::MenuCommand::ViewLayers,self.window_visibility.layers);
                        self.show_command_toggle(ui,ctx,crate::config::commands::MenuCommand::ViewHistory,self.window_visibility.history);
                        self.show_command_toggle(ui,ctx,crate::config::commands::MenuCommand::ViewColors,self.window_visibility.colors);
                        self.show_command_toggle(ui,ctx,crate::config::commands::MenuCommand::ViewPalette,self.window_visibility.palette);
                        // Script Editor needs a native file picker to load .rhai
                        // scripts from disk — not available on web, so don't
                        // offer the menu entry at all there.
                        #[cfg(not(target_arch = "wasm32"))]
                        self.show_command_toggle(ui,ctx,crate::config::commands::MenuCommand::ViewScriptEditor,self.window_visibility.script_editor);

                        ui.separator();

                        ui.add_enabled_ui(has_project, |ui| {
                        self.show_command_toggle(ui,ctx,crate::config::commands::MenuCommand::ViewPixelGrid,self.active_project().is_some_and(|p|p.canvas_state.show_pixel_grid));
                        if let Some(project) = self.active_project_mut() {
                            ui.menu_button(t!("guides.menu"), |ui| crate::ui::guides::menu(ui, project));
                        }

                        self.show_command_toggle(ui,ctx,crate::config::commands::MenuCommand::ViewCmyk,self.active_project().is_some_and(|p|p.canvas_state.cmyk_preview));

                        ui.separator();

                        // Zoom controls
                        if self.assets.menu_item_shortcut_below_enabled(ui, Icon::MenuViewZoomIn, &t!("menu.view.zoom_in"), true, &menu_kb, crate::config::keybindings::BindableAction::ViewZoomIn).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ViewZoomIn); ui.close(); }
                        if self.assets.menu_item_shortcut_below_enabled(ui, Icon::MenuViewZoomOut, &t!("menu.view.zoom_out"), true, &menu_kb, crate::config::keybindings::BindableAction::ViewZoomOut).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ViewZoomOut); ui.close(); }
                        let fit_menu = self
                            .assets
                            .menu_item_shortcut_below(
                                ui,
                                Icon::MenuViewFitWindow,
                                &t!("menu.view.fit_to_window"),
                                &menu_kb,
                                BindableAction::ViewFitToWindow,
                            );
                        #[cfg(test)]
                        ctx.data_mut(|d|d.insert_temp(egui::Id::new("menu_fit_test_rect"),fit_menu.rect));
                        if fit_menu.clicked() {
                            self.execute_menu_command(ctx, BindableAction::ViewFitToWindow);
                            ui.close();
                        }
                        if self.assets.menu_item_shortcut_below_enabled(ui, Icon::MenuViewZoomIn, &t!("menu.view.zoom_100"), true, &menu_kb, crate::config::keybindings::BindableAction::ViewZoom100).clicked() { self.execute_menu_command(ctx, crate::config::keybindings::BindableAction::ViewZoom100); ui.close(); }

                        });
                        ui.separator();

                        // Theme submenu
                        ui.menu_button(t!("menu.view.theme"), |ui| {
                            ui.set_min_width(ui.min_rect().width().max(160.0));
                            let is_light =
                                matches!(self.theme.mode, crate::theme::ThemeMode::Light);
                            let is_dark = matches!(self.theme.mode, crate::theme::ThemeMode::Dark);
                            if ui.radio(is_light,t!("menu.view.theme.light")).clicked() {self.execute_menu_command(ctx,crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::ViewLight));ui.close();}
                            if ui.radio(is_dark,t!("menu.view.theme.dark")).clicked() {self.execute_menu_command(ctx,crate::config::keybindings::BindableAction::Menu(crate::config::commands::MenuCommand::ViewDark));ui.close();}
                        });
                    });
                    #[cfg(test)]
                    ctx.data_mut(|d|d.insert_temp(egui::Id::new("view_menu_test_rect"),view_menu.response.rect));
                    let _=view_menu;
                    ui.menu_button(t!("help.title"), |ui| {
                        if self.show_command_button(ui,ctx,crate::config::commands::MenuCommand::HelpQuickStart) { ui.close(); }
                        if self.show_command_button(ui,ctx,crate::config::commands::MenuCommand::HelpAiSetup) { ui.close(); }
                        self.show_command_button(ui,ctx,crate::config::commands::MenuCommand::HelpDocumentation);
                    });
                    if let Some(response) = crate::ui::help::show(ctx) {
                        self.remember_ui_cursor_rect(response.rect);
                        match response.action {
                            Some(crate::ui::help::HelpAction::AiSetup) => self.settings_window.open_ai_settings(&self.settings),
                            Some(crate::ui::help::HelpAction::Preferences) => self.settings_window.open = true,
                            Some(crate::ui::help::HelpAction::Keybindings) => self.settings_window.open_keybindings_settings(&self.settings),
                            None => {},
                        }
                    }

                });
            });

        // Bottom line below menu bar
        {
            let menu_rect = menu_resp.response.rect;
            self.remember_ui_cursor_rect(menu_rect);
            let line_color = self.theme.border_color;
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Background,
                egui::Id::new("menu_bottom_line"),
            ));
            let line_rect = egui::Rect::from_min_max(
                egui::pos2(menu_rect.left(), menu_rect.bottom() - 1.0),
                egui::pos2(menu_rect.right(), menu_rect.bottom()),
            );
            painter.rect_filled(line_rect, 0.0, line_color);
        }

        // --- Row 2: Actions + Project Tabs ---
        #[allow(deprecated)]
        let toolbar_resp = egui::Panel::top("toolbar_tabs")
            .frame(self.theme.toolbar_frame())
            .show(root_ui, |ui| {
                if modal_open {
                    ui.disable();
                }
                ui.horizontal(|ui| {
                    // === File Actions ===
                    if self.assets.small_icon_button(ui, Icon::New).clicked() {
                        self.new_file_dialog.load_clipboard_dimensions();
                        self.new_file_dialog
                            .set_lock_aspect_ratio(self.settings.persist_new_file_lock_aspect);
                        self.new_file_dialog.open_dialog();
                    }
                    if self.assets.small_icon_button(ui, Icon::Open).clicked() {
                        self.handle_open_file(ctx.input(|i| i.time));
                    }
                    let is_dirty = self.active_project().is_some_and(|p| p.is_dirty);
                    let save_button = self.assets.icon_button_enabled(ui, Icon::Save, is_dirty);
                    #[cfg(test)]
                    ctx.data_mut(|d| {
                        d.insert_temp(egui::Id::new("toolbar_save_test_rect"), save_button.rect)
                    });
                    if save_button.clicked() {
                        self.handle_save(ctx.input(|i| i.time));
                    }

                    ui.separator();

                    // === Undo/Redo ===
                    let can_undo = self.can_undo_action();
                    let can_redo = self.can_redo_action();

                    if self
                        .assets
                        .icon_button_enabled(ui, Icon::Undo, can_undo)
                        .clicked()
                    {
                        self.perform_undo();
                    }
                    if self
                        .assets
                        .icon_button_enabled(ui, Icon::Redo, can_redo)
                        .clicked()
                    {
                        self.perform_redo();
                    }

                    ui.separator();

                    // === Zoom Controls ===
                    if self
                        .assets
                        .icon_button_enabled(ui, Icon::ZoomIn, has_project)
                        .clicked()
                    {
                        self.canvas.zoom_in();
                    }
                    if self
                        .assets
                        .icon_button_enabled(ui, Icon::ZoomOut, has_project)
                        .clicked()
                    {
                        self.canvas.zoom_out();
                    }
                    let fit_button =
                        self.assets
                            .icon_button_enabled(ui, Icon::ResetZoom, has_project);
                    #[cfg(test)]
                    ctx.data_mut(|d| {
                        d.insert_temp(egui::Id::new("toolbar_fit_test_rect"), fit_button.rect)
                    });
                    if fit_button.clicked() {
                        self.execute_menu_command(ctx, BindableAction::ViewFitToWindow);
                    }

                    ui.separator();

                    // Pixel grid toggle (respects settings mode)
                    let pixel_grid_mode = self.settings.pixel_grid_mode;
                    let show_grid = self
                        .active_project()
                        .map(|p| p.canvas_state.show_pixel_grid)
                        .unwrap_or(false);

                    match pixel_grid_mode {
                        PixelGridMode::Auto => {
                            // Auto mode: Show toggle for manual override
                            let grid_icon = if show_grid {
                                Icon::GridOn
                            } else {
                                Icon::GridOff
                            };
                            if self
                                .assets
                                .icon_button_enabled(ui, grid_icon, has_project)
                                .clicked()
                                && let Some(project) = self.active_project_mut()
                            {
                                project.canvas_state.show_pixel_grid =
                                    !project.canvas_state.show_pixel_grid;
                            }
                        }
                        PixelGridMode::AlwaysOn => {
                            if let Some(project) = self.active_project_mut() {
                                project.canvas_state.show_pixel_grid = true;
                            }
                            self.assets.icon_button_enabled(ui, Icon::GridOn, false);
                        }
                        PixelGridMode::AlwaysOff => {
                            if let Some(project) = self.active_project_mut() {
                                project.canvas_state.show_pixel_grid = false;
                            }
                            self.assets.icon_button_enabled(ui, Icon::GridOff, false);
                        }
                    }

                    // Guidelines toggle
                    {
                        let show_guides = self
                            .active_project()
                            .map(|p| p.canvas_state.show_guidelines)
                            .unwrap_or(false);
                        let guide_icon = if show_guides {
                            Icon::GuidesOn
                        } else {
                            Icon::GuidesOff
                        };
                        if self
                            .assets
                            .icon_button_enabled(ui, guide_icon, has_project)
                            .clicked()
                            && let Some(project) = self.active_project_mut()
                        {
                            project.canvas_state.show_guidelines =
                                !project.canvas_state.show_guidelines;
                        }
                    }

                    // Mirror mode toggle (cycles: None ÔåÆ H ÔåÆ V ÔåÆ Quarters ÔåÆ None)
                    {
                        use crate::canvas::MirrorMode;
                        let mode = self
                            .active_project()
                            .map(|p| p.canvas_state.mirror_mode)
                            .unwrap_or(MirrorMode::None);
                        let mirror_icon = match mode {
                            MirrorMode::None => Icon::MirrorOff,
                            MirrorMode::Horizontal => Icon::MirrorH,
                            MirrorMode::Vertical => Icon::MirrorV,
                            MirrorMode::Quarters => Icon::MirrorQ,
                        };
                        if self
                            .assets
                            .icon_button_enabled(ui, mirror_icon, has_project)
                            .clicked()
                            && let Some(project) = self.active_project_mut()
                        {
                            project.canvas_state.mirror_mode =
                                project.canvas_state.mirror_mode.next();
                            project.canvas_state.clear_preview_state();
                            project.canvas_state.mark_dirty(None);
                        }
                    }

                    // Four-side seamless edge preview toggle
                    {
                        let show_wrap_preview = self
                            .active_project()
                            .map(|p| p.canvas_state.show_wrap_preview)
                            .unwrap_or(false);
                        let wrap_icon = if show_wrap_preview {
                            Icon::WrapPreviewOn
                        } else {
                            Icon::WrapPreviewOff
                        };
                        if self
                            .assets
                            .icon_button_enabled(ui, wrap_icon, has_project)
                            .clicked()
                            && let Some(project) = self.active_project_mut()
                        {
                            project.canvas_state.show_wrap_preview =
                                !project.canvas_state.show_wrap_preview;
                        }
                    }

                    ui.separator();

                    // === Project Tabs Section (clean, no tray) ===
                    ui.add_space(4.0);

                    // Collect project info before mutable operations
                    let project_infos: Vec<(String, bool, u32, u32)> = self
                        .projects
                        .iter()
                        .map(|p| {
                            (
                                p.name.clone(),
                                p.is_dirty,
                                p.canvas_state.width,
                                p.canvas_state.height,
                            )
                        })
                        .collect();

                    let mut tab_to_switch: Option<usize> = None;
                    let mut tab_to_close: Option<usize> = None;
                    let mut document_action: Option<(uuid::Uuid, &str)> = None;
                    let mut tab_reorder: Option<(usize, usize)> = None; // (from, to)
                    let _tab_count = project_infos.len();

                    // Drag state IDs
                    let drag_src_id = egui::Id::new("tab_drag_source");
                    let dragging_tab: Option<usize> =
                        ui.ctx().memory(|m| m.data.get_temp::<usize>(drag_src_id));

                    // Collect tab rects for drop target computation
                    let mut tab_rects: Vec<egui::Rect> = Vec::new();

                    let offset_id = egui::Id::new("project_tabs_offset");
                    let offset = ctx.data(|d| d.get_temp::<f32>(offset_id)).unwrap_or(0.0);
                    let width_id = egui::Id::new("project_tabs_content_width");
                    let previous_width = ctx.data(|d| d.get_temp::<f32>(width_id)).unwrap_or(0.0);
                    let strip = egui::Rect::from_min_size(
                        ui.cursor().min,
                        egui::vec2(ui.available_width(), ui.available_height()),
                    );
                    let overflowing = previous_width > strip.width() + 1.0;
                    // Keep both gutters stable while scrolling; only paint arrows
                    // for the directions that still have hidden tabs.
                    let gutter = if overflowing { 22.0 } else { 0.0 };
                    if overflowing {
                        ui.add_space(gutter);
                    }
                    let scroll_out = egui::ScrollArea::horizontal()
                        .id_salt("project_tabs_scroll")
                        .horizontal_scroll_offset(offset)
                        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                        .max_width((strip.width() - 2.0 * gutter).max(1.0))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 2.0;

                                for (idx, (name, is_dirty, cw, ch)) in
                                    project_infos.iter().enumerate()
                                {
                                    let is_active = idx == self.active_project_index;

                                    // Animated crossfade
                                    let tab_anim_id = egui::Id::new("tab_active_anim").with(idx);
                                    let active_t = ui.ctx().animate_bool(tab_anim_id, is_active);

                                    // 1-frame-delayed hover for fill computation (avoids Background layer z-order issues)
                                    let hover_mem_id = egui::Id::new("tab_hover_mem").with(idx);
                                    let was_hovered: bool = ui.ctx().memory(|m| {
                                        m.data.get_temp::<bool>(hover_mem_id).unwrap_or(false)
                                    });

                                    // --- Colors ---
                                    let (active_fill, inactive_fill, hover_fill) =
                                        match self.theme.mode {
                                            crate::theme::ThemeMode::Dark => (
                                                egui::Color32::from_gray(48), // distinct lift from toolbar
                                                egui::Color32::from_gray(36), // visible against toolbar bg
                                                egui::Color32::from_gray(42), // visible hover
                                            ),
                                            crate::theme::ThemeMode::Light => (
                                                egui::Color32::WHITE,          // crisp white
                                                egui::Color32::from_gray(232), // visible but quiet
                                                egui::Color32::from_gray(242), // lighter on hover
                                            ),
                                        };
                                    let (active_text_color, inactive_text_color, hover_text_color) =
                                        match self.theme.mode {
                                            crate::theme::ThemeMode::Dark => (
                                                egui::Color32::from_gray(245),
                                                egui::Color32::from_gray(140),
                                                egui::Color32::from_gray(200), // brighter on hover
                                            ),
                                            crate::theme::ThemeMode::Light => (
                                                egui::Color32::from_gray(10),
                                                egui::Color32::from_gray(100),
                                                egui::Color32::from_gray(40),
                                            ),
                                        };

                                    // Compute fill: active crossfade takes priority, then hover, then inactive
                                    let fill = if active_t > 0.01 {
                                        crate::theme::Theme::lerp_color(
                                            inactive_fill,
                                            active_fill,
                                            active_t,
                                        )
                                    } else if was_hovered {
                                        hover_fill
                                    } else {
                                        inactive_fill
                                    };
                                    let text_color = if active_t > 0.01 {
                                        crate::theme::Theme::lerp_color(
                                            inactive_text_color,
                                            active_text_color,
                                            active_t,
                                        )
                                    } else if was_hovered {
                                        hover_text_color
                                    } else {
                                        inactive_text_color
                                    };

                                    // Active tab stroke for contrast
                                    let stroke = if active_t > 0.5 {
                                        egui::Stroke::new(
                                            1.0,
                                            match self.theme.mode {
                                                crate::theme::ThemeMode::Dark => {
                                                    egui::Color32::from_gray(62)
                                                }
                                                crate::theme::ThemeMode::Light => {
                                                    egui::Color32::from_gray(200)
                                                }
                                            },
                                        )
                                    } else {
                                        egui::Stroke::NONE
                                    };

                                    // --- Build tab text ---
                                    let tab_label = if name.chars().count() > 24 {
                                        format!("{}…", name.chars().take(23).collect::<String>())
                                    } else {
                                        name.clone()
                                    };
                                    let text = egui::RichText::new(&tab_label).color(text_color);
                                    let text = if is_active { text.strong() } else { text };

                                    // --- Flat-bottom rounding (top-left, top-right, bottom-right, bottom-left) ---
                                    let tr = self.theme.tab_rounding as u8;
                                    let rounding = egui::CornerRadius {
                                        nw: tr,
                                        ne: tr,
                                        sw: 0,
                                        se: 0,
                                    };

                                    let mut close_rect = None;

                                    // --- Tab frame ---
                                    let tab_resp = egui::Frame::NONE
                                        .fill(fill)
                                        .stroke(stroke)
                                        .inner_margin(egui::Margin::symmetric(10, 4))
                                        .corner_radius(rounding)
                                        .show(ui, |ui| {
                                            ui.set_min_height(18.0);
                                            ui.horizontal(|ui| {
                                                // Dirty dot indicator
                                                if *is_dirty {
                                                    let dot_size = 5.0;
                                                    let (dot_rect, _) = ui.allocate_exact_size(
                                                        egui::vec2(dot_size, dot_size),
                                                        egui::Sense::hover(),
                                                    );
                                                    let center = dot_rect.center();
                                                    ui.painter().circle_filled(
                                                        center,
                                                        dot_size / 2.0,
                                                        self.theme.accent,
                                                    );
                                                }

                                                // Tab label (clickable + draggable for reorder)
                                                // Use Label with click sense so only text area is clickable,
                                                // not the full tab width (avoids tabs filling the entire strip).
                                                let label_resp = ui.add(
                                                    egui::Label::new(text)
                                                        .sense(egui::Sense::click_and_drag()),
                                                );
                                                if label_resp.clicked() {
                                                    tab_to_switch = Some(idx);
                                                }
                                                if label_resp.drag_started() {
                                                    ui.ctx().memory_mut(|m| {
                                                        m.data.insert_temp(drag_src_id, idx);
                                                    });
                                                }

                                                // Always reserve the size-label slot so the
                                                // tab row doesn't jitter when switching projects.
                                                let dim_text =
                                                    egui::RichText::new(format!("{}x{}", cw, ch))
                                                        .size(10.0)
                                                        .color(match self.theme.mode {
                                                            crate::theme::ThemeMode::Dark => {
                                                                egui::Color32::from_gray(80)
                                                            }
                                                            crate::theme::ThemeMode::Light => {
                                                                egui::Color32::from_gray(160)
                                                            }
                                                        });
                                                ui.add_visible(
                                                    is_active,
                                                    egui::Label::new(dim_text),
                                                );

                                                // Close button -- always visible (subtle when not hovered)
                                                // Previously gated on `is_active || was_hovered` which caused the
                                                // button to flicker on non-active tabs because the outer frame's
                                                // `hovered()` goes false when the child close-button steals hover.
                                                let close_resp =
                                                    crate::signal_widgets::close_button(
                                                        ui,
                                                        &self.theme,
                                                    );
                                                close_rect = Some(close_resp.rect);
                                                if close_resp.clicked() {
                                                    tab_to_close = Some(idx);
                                                }
                                            });
                                        });

                                    // Track tab rect for drag-drop (use the frame response rect)
                                    tab_rects.push(tab_resp.response.rect);
                                    let document_id = self.projects[idx].id;
                                    let last_active_id = egui::Id::new("project_tabs_last_active");
                                    if is_active
                                        && ctx.data(|d| d.get_temp::<uuid::Uuid>(last_active_id))
                                            != Some(document_id)
                                    {
                                        ui.scroll_to_rect(tab_resp.response.rect, None);
                                        ctx.data_mut(|d| {
                                            d.insert_temp(last_active_id, document_id)
                                        });
                                    }
                                    let full_tab_resp = ui.interact(
                                        tab_resp.response.rect,
                                        ui.id().with(("project_tab_full", idx)),
                                        egui::Sense::click_and_drag(),
                                    );
                                    crate::ui::menus::context_menu(&full_tab_resp, |ui| {
                                        ui.label(name);
                                        for action in [
                                            "Save",
                                            "Save As",
                                            "Duplicate",
                                            "Info",
                                            "Reveal in Folder",
                                            "Close",
                                        ] {
                                            let enabled = action != "Reveal in Folder"
                                                || self.projects[idx].path.is_some();
                                            if ui
                                                .add_enabled(enabled, egui::Button::new(action))
                                                .clicked()
                                            {
                                                document_action = Some((document_id, action));
                                                ui.close();
                                            }
                                        }
                                    });
                                    full_tab_resp.clone().on_hover_ui(|ui| {
                                        ui.label(name);
                                        ui.label(crate::i18n::format_translation(
                                            "ui.0_1_2_layers",
                                            &[
                                                ("0", format!("{}", cw)),
                                                ("1", format!("{}", ch)),
                                                (
                                                    "2",
                                                    format!(
                                                        "{}",
                                                        self.projects[idx]
                                                            .canvas_state
                                                            .layers
                                                            .len()
                                                    ),
                                                ),
                                            ],
                                        ));
                                        let state = &self.projects[idx].canvas_state;
                                        // Compositing changes can leave pixel generations unchanged.
                                        // Include layer order, visibility and blend settings as well.
                                        let generation = egui::Id::new((
                                            cw,
                                            ch,
                                            state.dirty_generation,
                                            state
                                                .layers
                                                .iter()
                                                .enumerate()
                                                .map(|(i, l)| {
                                                    (
                                                        l.gpu_generation,
                                                        state.layer_effectively_visible(i),
                                                        l.opacity.to_bits(),
                                                        l.blend_mode as u8,
                                                        l.mask_enabled,
                                                    )
                                                })
                                                .collect::<Vec<_>>(),
                                        ))
                                        .value();
                                        let cache_id =
                                            egui::Id::new(("tab_thumbnail", document_id));
                                        let cached = ctx.data(|d| {
                                            d.get_temp::<(u64, egui::TextureHandle)>(cache_id)
                                        });
                                        let texture = if let Some((old, texture)) =
                                            cached.filter(|(old, _)| *old == generation)
                                        {
                                            let _ = old;
                                            texture
                                        } else {
                                            let project = &mut self.projects[idx];
                                            project
                                                .canvas_state
                                                .ensure_all_text_layers_rasterized();
                                            let state = &project.canvas_state;
                                            let scale = state
                                                .width
                                                .div_ceil(160)
                                                .max(state.height.div_ceil(120))
                                                .max(1);
                                            let (thumbnail, _) = state
                                                .composite_partial_downscaled(
                                                    egui::Rect::from_min_size(
                                                        egui::Pos2::ZERO,
                                                        egui::vec2(
                                                            state.width as f32,
                                                            state.height as f32,
                                                        ),
                                                    ),
                                                    scale,
                                                );
                                            let texture = ctx.load_texture(
                                                format!("tab_{document_id}"),
                                                thumbnail,
                                                egui::TextureOptions::LINEAR,
                                            );
                                            ctx.data_mut(|d| {
                                                d.insert_temp(
                                                    cache_id,
                                                    (generation, texture.clone()),
                                                )
                                            });
                                            texture
                                        };
                                        ui.image((texture.id(), texture.size_vec2()));
                                    });
                                    if self.settings.middle_click_close_tabs
                                        && full_tab_resp.clicked_by(egui::PointerButton::Middle)
                                    {
                                        tab_to_close = Some(idx);
                                    }
                                    if full_tab_resp.clicked() {
                                        let click_pos = ui.input(|i| i.pointer.interact_pos());
                                        if close_rect
                                            .zip(click_pos)
                                            .is_some_and(|(r, p)| r.contains(p))
                                        {
                                            tab_to_close = Some(idx);
                                        } else {
                                            tab_to_switch = Some(idx);
                                        }
                                    }
                                    if full_tab_resp.drag_started() {
                                        ui.ctx().memory_mut(|m| {
                                            m.data.insert_temp(drag_src_id, idx);
                                        });
                                    }

                                    // Drag cursor: show grab while dragging this tab
                                    if dragging_tab == Some(idx)
                                        && ui.input(|i| i.pointer.any_down())
                                    {
                                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                                    }

                                    // Accent bottom stripe for active tab (2px)
                                    if active_t > 0.01 {
                                        let tab_rect = tab_resp.response.rect;
                                        let stripe_rect = egui::Rect::from_min_max(
                                            egui::pos2(
                                                tab_rect.left() + 2.0,
                                                tab_rect.bottom() - 3.0,
                                            ),
                                            egui::pos2(
                                                tab_rect.right() - 2.0,
                                                tab_rect.bottom() - 1.0,
                                            ),
                                        );
                                        let stripe_alpha = (active_t * 255.0) as u8;
                                        let stripe_color = egui::Color32::from_rgba_unmultiplied(
                                            self.theme.accent.r(),
                                            self.theme.accent.g(),
                                            self.theme.accent.b(),
                                            stripe_alpha,
                                        );
                                        ui.painter().rect_filled(
                                            stripe_rect,
                                            egui::CornerRadius::same(1),
                                            stripe_color,
                                        );
                                    }
                                }

                                // --- Drag-drop indicator + release logic ---
                                if let Some(src_idx) = dragging_tab {
                                    let pointer_released = ui.input(|i| i.pointer.any_released());
                                    let pointer_pos = ui.input(|i| i.pointer.hover_pos());

                                    if let Some(pos) = pointer_pos {
                                        // Compute drop index from pointer x vs tab center positions
                                        let mut drop_idx = tab_rects.len(); // default: after last
                                        for (i, rect) in tab_rects.iter().enumerate() {
                                            if pos.x < rect.center().x {
                                                drop_idx = i;
                                                break;
                                            }
                                        }

                                        // Draw drop indicator line (only if drop position differs from source)
                                        if drop_idx != src_idx && drop_idx != src_idx + 1 {
                                            let indicator_x = if drop_idx < tab_rects.len() {
                                                tab_rects[drop_idx].left() - 1.0
                                            } else if !tab_rects.is_empty() {
                                                tab_rects.last().unwrap().right() + 1.0
                                            } else {
                                                0.0
                                            };
                                            if !tab_rects.is_empty() {
                                                let top = tab_rects[0].top();
                                                let bottom = tab_rects[0].bottom();
                                                ui.painter().line_segment(
                                                    [
                                                        egui::pos2(indicator_x, top),
                                                        egui::pos2(indicator_x, bottom),
                                                    ],
                                                    egui::Stroke::new(2.0, self.theme.accent),
                                                );
                                            }
                                        }

                                        // On release: perform the reorder
                                        if pointer_released {
                                            // Adjust drop index for removal shift
                                            let effective_drop = if drop_idx > src_idx {
                                                drop_idx - 1
                                            } else {
                                                drop_idx
                                            };
                                            if effective_drop != src_idx {
                                                tab_reorder = Some((src_idx, effective_drop));
                                            }
                                            ui.ctx().memory_mut(|m| {
                                                m.data.remove::<usize>(drag_src_id);
                                            });
                                        }
                                    } else if pointer_released {
                                        // Released outside tab area -- cancel
                                        ui.ctx().memory_mut(|m| {
                                            m.data.remove::<usize>(drag_src_id);
                                        });
                                    }
                                }

                                // "+" button -- use a Frame so hover bg is behind the text
                                ui.add_space(4.0);
                                let plus_hover_id = egui::Id::new("plus_tab_hover");
                                let plus_was_hovered: bool = ui.ctx().memory(|m| {
                                    m.data.get_temp::<bool>(plus_hover_id).unwrap_or(false)
                                });
                                let plus_fill = if plus_was_hovered {
                                    match self.theme.mode {
                                        crate::theme::ThemeMode::Dark => {
                                            egui::Color32::from_white_alpha(15)
                                        }
                                        crate::theme::ThemeMode::Light => {
                                            egui::Color32::from_gray(228)
                                        }
                                    }
                                } else {
                                    egui::Color32::TRANSPARENT
                                };
                                let plus_color = match self.theme.mode {
                                    crate::theme::ThemeMode::Dark => {
                                        if plus_was_hovered {
                                            egui::Color32::from_gray(180)
                                        } else {
                                            egui::Color32::from_gray(100)
                                        }
                                    }
                                    crate::theme::ThemeMode::Light => {
                                        if plus_was_hovered {
                                            egui::Color32::from_gray(60)
                                        } else {
                                            egui::Color32::from_gray(140)
                                        }
                                    }
                                };
                                let plus_resp = egui::Frame::NONE
                                    .fill(plus_fill)
                                    .corner_radius(egui::CornerRadius::same(4))
                                    .inner_margin(egui::Margin::symmetric(4, 2))
                                    .show(ui, |ui| {
                                        let plus_text =
                                            egui::RichText::new("+").size(14.0).color(plus_color);
                                        ui.add(
                                            egui::Label::new(plus_text).sense(egui::Sense::click()),
                                        )
                                    });
                                ui.ctx().memory_mut(|m| {
                                    m.data
                                        .insert_temp(plus_hover_id, plus_resp.response.hovered());
                                });
                                if plus_resp.response.hovered() {
                                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                }
                                if plus_resp.inner.clicked() {
                                    self.new_file_dialog.load_clipboard_dimensions();
                                    self.new_file_dialog.set_lock_aspect_ratio(
                                        self.settings.persist_new_file_lock_aspect,
                                    );
                                    self.new_file_dialog.open_dialog();
                                }
                            });
                        });
                    let mut next_offset = scroll_out.state.offset.x;
                    let max_offset =
                        (scroll_out.content_size.x - scroll_out.inner_rect.width()).max(0.0);
                    if overflowing {
                        for (right, visible) in [
                            (false, next_offset > 1.0),
                            (true, next_offset < max_offset - 1.0),
                        ] {
                            if !visible {
                                continue;
                            }
                            let center = egui::pos2(
                                if right {
                                    strip.right() - 11.0
                                } else {
                                    strip.left() + 11.0
                                },
                                scroll_out.inner_rect.center().y,
                            );
                            let response = ui
                                .interact(
                                    egui::Rect::from_center_size(center, egui::vec2(20.0, 24.0)),
                                    egui::Id::new(("project_tabs_arrow", right)),
                                    egui::Sense::click(),
                                )
                                .on_hover_text(if right {
                                    crate::t!("ui.scroll_tabs_right")
                                } else {
                                    crate::t!("ui.scroll_tabs_left")
                                });
                            let color = if response.hovered() {
                                self.theme.accent
                            } else {
                                ui.visuals().weak_text_color().gamma_multiply(0.6)
                            };
                            let direction = if right { 1.0 } else { -1.0 };
                            ui.painter().add(egui::Shape::line(
                                vec![
                                    center + egui::vec2(-direction * 2.0, -4.0),
                                    center + egui::vec2(direction * 2.0, 0.0),
                                    center + egui::vec2(-direction * 2.0, 4.0),
                                ],
                                egui::Stroke::new(1.5, color),
                            ));
                            if response.clicked() {
                                next_offset =
                                    (next_offset + direction * 240.0).clamp(0.0, max_offset);
                                ctx.request_repaint();
                            }
                        }
                    }
                    if (previous_width - scroll_out.content_size.x).abs() > 1.0 {
                        ctx.request_repaint();
                    }
                    ctx.data_mut(|d| d.insert_temp(width_id, scroll_out.content_size.x));
                    if scroll_out.inner_rect.contains(
                        ctx.input(|i| i.pointer.hover_pos())
                            .unwrap_or(egui::Pos2::ZERO),
                    ) {
                        let wheel = ctx.input_mut(|i| {
                            let y = i.smooth_scroll_delta.y;
                            i.smooth_scroll_delta.y = 0.0;
                            y
                        });
                        next_offset = (next_offset - wheel).max(0.0);
                    }
                    ctx.data_mut(|d| d.insert_temp(offset_id, next_offset.clamp(0.0, max_offset)));
                    if let Some((id, action)) = document_action
                        && let Some(idx) = self.projects.iter().position(|p| p.id == id)
                    {
                        match action {
                            "Close" => self.close_project(idx),
                            "Save As" => self.open_save_as_for_project(idx),
                            "Save" => {
                                self.switch_to_project(idx);
                                self.handle_save(ctx.input(|i| i.time));
                            }
                            "Duplicate" => self.duplicate_project(id),
                            "Info" => {
                                ctx.data_mut(|d| d.insert_temp(egui::Id::new("document_info"), id));
                            }
                            "Reveal in Folder" => {
                                #[cfg(not(target_arch = "wasm32"))]
                                if let Some(path) = self.projects[idx].path.as_ref()
                                    && let Err(error) = Self::reveal_document(path)
                                {
                                    self.filter_error = Some(error.to_string());
                                }
                            }
                            _ => {}
                        }
                    }

                    // Process tab actions after iteration
                    if let Some(idx) = tab_to_switch {
                        self.switch_to_project(idx);
                    }
                    if let Some(idx) = tab_to_close {
                        self.close_project(idx);
                    }
                    // Reorder tabs via drag-drop
                    if let Some((from, to)) = tab_reorder
                        && from < self.projects.len()
                        && to <= self.projects.len()
                    {
                        self.persist_active_project_view();
                        let project = self.projects.remove(from);
                        let insert_at = to.min(self.projects.len());
                        self.projects.insert(insert_at, project);
                        // Fix active_project_index to follow the active tab
                        if self.active_project_index == from {
                            self.active_project_index = insert_at;
                        } else if from < self.active_project_index
                            && insert_at >= self.active_project_index
                        {
                            self.active_project_index -= 1;
                        } else if from > self.active_project_index
                            && insert_at <= self.active_project_index
                        {
                            self.active_project_index += 1;
                        }
                        self.restore_active_project_view();
                    }
                });
            });

        self.persist_active_project_view();

        let info_id = egui::Id::new("document_info");
        if let Some(id) = ctx.data(|d| d.get_temp::<uuid::Uuid>(info_id)) {
            let mut open = true;
            if let Some(project) = self.projects.iter().find(|p| p.id == id) {
                crate::ui::polish::window(ctx, "document_info_inspector")
                    .title_bar(false)
                    .collapsible(false)
                    .resizable(true)
                    .default_width(440.0)
                    .min_width(280.0)
                    .max_width((ctx.content_rect().width() - 24.0).max(280.0))
                    .frame(self.theme.dialog_frame())
                    .show(ctx, |ui| {
                        let path = project
                            .path
                            .as_ref()
                            .map(|p| crate::services::io::display_path(p));
                        open &= !crate::ui::panels::document_info::contents(
                            ui,
                            &self.assets,
                            crate::ui::panels::document_info::Info {
                                name: &project.name,
                                width: project.canvas_state.width,
                                height: project.canvas_state.height,
                                layers: project.canvas_state.layers.len(),
                                path: path.as_deref(),
                            },
                        );
                    });
            } else {
                open = false;
            }
            if !open {
                ctx.data_mut(|d| d.remove::<uuid::Uuid>(info_id));
            }
        }

        // Sync primary color from colors panel to tools
        self.tools_panel.properties.color = self.colors_panel.get_primary_color();

        // Sync color picker result back to colors panel
        // If the color picker tool picked a color this frame, update the colors panel
        if let Some((picked_color, use_secondary)) = self.tools_panel.last_picked_color.take() {
            if use_secondary {
                self.colors_panel.set_secondary_color(picked_color);
            } else {
                self.colors_panel.set_primary_color(picked_color);
            }
        }

        for color in self.tools_panel.completed_paint_colors.drain(..) {
            self.palette_panel.observe_color(color);
        }

        self.persist_tool_settings_if_changed();
        crate::ui::polish::remember_popup_input(ctx, popup_was_open);

        // Thin bottom border on toolbar -- subtle divider (lighter than border_color)
        {
            let toolbar_rect = toolbar_resp.response.rect;
            self.remember_ui_cursor_rect(toolbar_rect);
            let screen_rect = ctx.content_rect();
            let line_color = match self.theme.mode {
                crate::theme::ThemeMode::Dark => egui::Color32::from_white_alpha(12),
                crate::theme::ThemeMode::Light => egui::Color32::from_black_alpha(18),
            };
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Background,
                egui::Id::new("toolbar_bottom_line"),
            ));
            let line_rect = egui::Rect::from_min_max(
                egui::pos2(screen_rect.left(), toolbar_rect.bottom()),
                egui::pos2(screen_rect.right(), toolbar_rect.bottom() + 1.0),
            );
            painter.rect_filled(line_rect, 0.0, line_color);
        }
    }
}
