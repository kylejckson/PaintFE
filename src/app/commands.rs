impl PaintFEApp {
    pub(crate) fn menu_command_enabled(
        &self,
        ctx: &egui::Context,
        action: crate::config::keybindings::BindableAction,
    ) -> bool {
        use crate::config::keybindings::BindableAction;
        let Some(spec) = crate::config::commands::command_spec(action) else {
            return false;
        };
        let has_project = !self.projects.is_empty();
        if spec.needs_document && !has_project {
            return false;
        }
        if self.document_modal_open() || self.processing_busy() {
            return false;
        }
        let no_dialog = matches!(self.active_dialog, ActiveDialog::None);
        let has_sel = self
            .active_project()
            .is_some_and(|p| p.canvas_state.has_selection());
        let can_copy = has_sel || self.paste_overlay.is_some();
        let can_undo = self.can_undo_action();
        let can_redo = self.can_redo_action();
        let any_dirty = self.projects.iter().any(Project::needs_save);
        let align_enabled = self.active_project().is_some_and(|p| {
            !p.canvas_state.layers[p.canvas_state.active_layer_index].is_text_layer()
        }) && no_dialog;
        let has_clip = if matches!(
            action,
            BindableAction::Paste
                | BindableAction::Menu(crate::config::commands::MenuCommand::EditPasteAsLayer)
        ) {
            crate::ops::clipboard::has_clipboard_image(ctx)
        } else {
            false
        };
        match action {
            BindableAction::NewFile => true,
            BindableAction::OpenFile => true,
            BindableAction::Save => has_project,
            BindableAction::SaveAs => has_project,
            BindableAction::SaveAll => any_dirty,
            BindableAction::Menu(crate::config::commands::MenuCommand::FileInfo) => has_project,
            BindableAction::Menu(crate::config::commands::MenuCommand::FileDuplicate) => {
                has_project
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::FilePrint) => has_project && !cfg!(target_arch = "wasm32"),
            BindableAction::CloseProject => has_project,
            BindableAction::Menu(crate::config::commands::MenuCommand::FileQuit) => true,
            BindableAction::Undo => can_undo,
            BindableAction::Redo => can_redo,
            BindableAction::Cut => has_sel,
            BindableAction::Copy => can_copy,
            BindableAction::Paste => has_project && has_clip,
            BindableAction::Menu(crate::config::commands::MenuCommand::EditPasteAsLayer) => {
                has_project && has_clip
            }
            BindableAction::SelectAll => has_project,
            BindableAction::Deselect => has_sel,
            BindableAction::Menu(crate::config::commands::MenuCommand::EditInvertSelection) => {
                has_sel
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::SelectColorRange) => {
                has_project
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::EditSelectContentBounds) => {
                has_project
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::EditPreferences) => true,
            BindableAction::ResizeImage => no_dialog,
            BindableAction::ResizeCanvas => no_dialog,
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasCropToSelection) => {
                has_sel
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasNewTextLayer) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasFlipHorizontal) => {
                true
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasFlipVertical) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate90Cw) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate90Ccw) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate180) => true,
            BindableAction::FlattenLayers => true,
            BindableAction::ColorAutoLevels => true,
            BindableAction::ColorDesaturate => true,
            BindableAction::ColorInvertColors => true,
            BindableAction::ColorInvertAlpha => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ColorColorToAlpha) => {
                no_dialog
            }
            BindableAction::Menu(
                crate::config::commands::MenuCommand::ColorRecoverTransparency,
            ) => no_dialog,
            BindableAction::ColorSepiaTone => true,
            BindableAction::ColorBrightnessContrast => no_dialog,
            BindableAction::ColorCurves => no_dialog,
            BindableAction::ColorExposure => no_dialog,
            BindableAction::ColorHighlightsShadows => no_dialog,
            BindableAction::ColorHueSaturation => no_dialog,
            BindableAction::ColorLevels => no_dialog,
            BindableAction::ColorTemperatureTint => no_dialog,
            BindableAction::ColorVibrance => no_dialog,
            BindableAction::ColorThreshold => no_dialog,
            BindableAction::ColorPosterize => no_dialog,
            BindableAction::ColorBalance => no_dialog,
            BindableAction::ColorGradientMap => no_dialog,
            BindableAction::ColorBlackAndWhite => no_dialog,
            BindableAction::FilterGaussianBlur => no_dialog,
            BindableAction::FilterBokehBlur => no_dialog,
            BindableAction::FilterMotionBlur => no_dialog,
            BindableAction::FilterBoxBlur => no_dialog,
            BindableAction::FilterZoomBlur => no_dialog,
            BindableAction::FilterSharpen => no_dialog,
            BindableAction::FilterReduceNoise => no_dialog,
            BindableAction::FilterCrystallize => no_dialog,
            BindableAction::FilterDents => no_dialog,
            BindableAction::FilterPixelate => no_dialog,
            BindableAction::FilterBulge => no_dialog,
            BindableAction::FilterTwist => no_dialog,
            BindableAction::FilterAddNoise => no_dialog,
            BindableAction::FilterMedian => no_dialog,
            BindableAction::FilterGlow => no_dialog,
            BindableAction::FilterVignette => no_dialog,
            BindableAction::FilterHalftone => no_dialog,
            BindableAction::FilterInk => no_dialog,
            BindableAction::FilterOilPainting => no_dialog,
            BindableAction::FilterColorFilter => no_dialog,
            BindableAction::Menu(
                crate::config::commands::MenuCommand::FilterStylizeCanvasBorder,
            ) => no_dialog,
            BindableAction::Menu(crate::config::commands::MenuCommand::FilterStylizeSeamless) => {
                no_dialog
            }
            BindableAction::FilterPixelDrag => no_dialog,
            BindableAction::FilterRgbDisplace => no_dialog,
            BindableAction::GenerateGrid => no_dialog,
            BindableAction::GenerateDropShadow => no_dialog,
            BindableAction::GenerateOutline => no_dialog,
            BindableAction::GenerateContours => no_dialog,
            BindableAction::ViewZoomIn => true,
            BindableAction::ViewZoomOut => true,
            BindableAction::ViewZoom100 => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::FileClearRecent) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasStraighten) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewTools) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewLayers) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewHistory) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewColors) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewPalette) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewScriptEditor) => !cfg!(target_arch = "wasm32"),
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewPixelGrid) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewCmyk) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewLight) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewDark) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::HelpQuickStart) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::HelpAiSetup) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::HelpDocumentation) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasAlign) => {
                align_enabled
            }
            BindableAction::FilterRemoveBackground => no_dialog && !self.background_removal_pending,
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolEllipseSelect) => self.paste_overlay.is_none() && !(self.tools_panel.active_tool==crate::components::tools::Tool::Text && self.tools_panel.text_state.is_editing),
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolMoveSelection) => self.paste_overlay.is_none() && !(self.tools_panel.active_tool==crate::components::tools::Tool::Text && self.tools_panel.text_state.is_editing),
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolContentAwareBrush) => self.paste_overlay.is_none() && !(self.tools_panel.active_tool==crate::components::tools::Tool::Text && self.tools_panel.text_state.is_editing),
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolLiquify) => self.paste_overlay.is_none() && !(self.tools_panel.active_tool==crate::components::tools::Tool::Text && self.tools_panel.text_state.is_editing),
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolSmudge) => self.paste_overlay.is_none() && !(self.tools_panel.active_tool==crate::components::tools::Tool::Text && self.tools_panel.text_state.is_editing),
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolPerspectiveCrop) => self.paste_overlay.is_none() && !(self.tools_panel.active_tool==crate::components::tools::Tool::Text && self.tools_panel.text_state.is_editing),
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesRulers) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesVisible) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesLocked) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesSnap) => true,
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesNew) => self.active_project().is_some_and(|p| !p.canvas_state.guides.locked),
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesClear) => self.active_project().is_some_and(|p| !p.canvas_state.guides.locked && !p.canvas_state.guides.items.is_empty()),
            BindableAction::ViewFitToWindow => true,
            _ => false,
        }
    }
    pub(crate) fn execute_menu_command(
        &mut self,
        ctx: &egui::Context,
        action: crate::config::keybindings::BindableAction,
    ) -> bool {
        use crate::config::keybindings::BindableAction;
        if !self.menu_command_enabled(ctx, action) {
            return false;
        }
        self.finish_pending_paint_stroke();
        match action {
            BindableAction::NewFile => {
                self.new_file_dialog.load_clipboard_dimensions();
                self.new_file_dialog
                    .set_lock_aspect_ratio(self.settings.persist_new_file_lock_aspect);
                self.new_file_dialog.open_dialog();
            }
            BindableAction::OpenFile => {
                self.handle_open_file(ctx.input(|i| i.time));
            }
            BindableAction::Save => {
                self.handle_save(ctx.input(|i| i.time));
            }
            BindableAction::SaveAs => {
                self.open_save_as_for_project(self.active_project_index);
            }
            BindableAction::SaveAll => {
                self.handle_save_all(ctx.input(|i| i.time));
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::FileInfo) => {
                if let Some(project) = self.active_project() {
                    ctx.data_mut(|d| d.insert_temp(egui::Id::new("document_info"), project.id));
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::FileDuplicate) => {
                if let Some(project) = self.active_project() {
                    self.duplicate_project(project.id);
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::FilePrint) => {
                if let Some(project) = self.active_project_mut() {
                    project.canvas_state.ensure_all_text_layers_rasterized();
                    let composite = project.canvas_state.composite();
                    if let Err(e) = crate::ops::print::print_image(&composite) {
                        eprintln!("Print error: {}", e);
                    }
                }
            }
            BindableAction::CloseProject => {
                self.close_project(self.active_project_index);
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::FileQuit) => {
                self.finish_pending_line_edit();
                let dirty_count = self.projects.iter().filter(|p| p.is_dirty).count();
                if self.settings.confirm_on_exit && dirty_count > 0 {
                    self.pending_exit = true;
                } else {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            BindableAction::Undo => {
                self.perform_undo();
            }
            BindableAction::Redo => {
                self.perform_redo();
            }
            BindableAction::Cut => {
                let transparent_cutout = self.settings.clipboard_copy_transparent_cutout;
                self.cut_selection_with_history(transparent_cutout);
            }
            BindableAction::Copy => {
                self.copy_active_selection_or_overlay();
            }
            BindableAction::Paste => {
                let cursor_canvas = if self.active_project_index < self.projects.len() {
                    ctx.input(|i| i.pointer.latest_pos()).and_then(|pp| {
                        self.canvas.last_canvas_rect.and_then(|rect| {
                            let state = &self.projects[self.active_project_index].canvas_state;
                            self.canvas.screen_to_canvas_f32_pub(pp, rect, state)
                        })
                    })
                } else {
                    None
                };
                self.queue_paste_from_clipboard(ctx, cursor_canvas);
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::EditPasteAsLayer) => {
                // Paste clipboard contents as a new layer
                let img = crate::ops::clipboard::get_from_system_clipboard()
                    .or_else(crate::ops::clipboard::get_clipboard_image_pub);
                if let Some(src) = img {
                    self.do_snapshot_op(
                        HistoryAction::PasteLayer.label("Paste as New Layer"),
                        |s| {
                            crate::ops::adjustments::import_layer_from_image(
                                s,
                                &src,
                                "Pasted Layer",
                            );
                        },
                    );
                }
            }
            BindableAction::SelectAll => {
                self.select_all_canvas();
            }
            BindableAction::Deselect => {
                self.clear_selection_with_history();
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::EditInvertSelection) => {
                self.invert_selection_with_history();
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::SelectColorRange) => {
                if let Some(project) = self.active_project_mut() {
                    let dlg = crate::ops::dialogs::ColorRangeDialog::new(&project.canvas_state);
                    self.active_dialog = crate::ops::dialogs::ActiveDialog::ColorRange(dlg);
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::EditSelectContentBounds) => {
                if let Some(project) = self.active_project_mut() {
                    crate::ops::adjustments::select_layer_content_bounds(&mut project.canvas_state);
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::EditPreferences) => {
                self.settings_window.open = true;
            }
            BindableAction::ResizeImage => {
                if let Some(project) = self.active_project() {
                    let mut dlg =
                        crate::ops::dialogs::ResizeImageDialog::new(&project.canvas_state);
                    dlg.lock_aspect = self.settings.persist_resize_lock_aspect;
                    self.active_dialog = ActiveDialog::ResizeImage(dlg);
                }
            }
            BindableAction::ResizeCanvas => {
                if let Some(project) = self.active_project() {
                    let mut dlg =
                        crate::ops::dialogs::ResizeCanvasDialog::new(&project.canvas_state);
                    dlg.lock_aspect = self.settings.persist_resize_lock_aspect;
                    self.active_dialog = ActiveDialog::ResizeCanvas(dlg);
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasCropToSelection) => {
                self.do_snapshot_op(HistoryAction::Crop.label("Crop to Selection"), |s| {
                    crate::ops::adjustments::crop_to_selection(s);
                });
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasNewTextLayer) => {
                if let Some(project) = self.active_project_mut() {
                    crate::ops::canvas_ops::add_text_layer(
                        &mut project.canvas_state,
                        &mut project.history,
                    );
                    self.canvas.gpu_clear_layers();
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasFlipHorizontal) => {
                self.do_snapshot_op(
                    HistoryAction::FlipHorizontal.label("Flip Horizontal"),
                    |s| {
                        crate::ops::transform::flip_canvas_horizontal(s);
                    },
                );
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasFlipVertical) => {
                self.do_snapshot_op(HistoryAction::FlipVertical.label("Flip Vertical"), |s| {
                    crate::ops::transform::flip_canvas_vertical(s);
                });
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate90Cw) => {
                self.do_snapshot_op(HistoryAction::RotateCw.label("Rotate 90 deg CW"), |s| {
                    crate::ops::transform::rotate_canvas_90cw(s);
                });
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate90Ccw) => {
                self.do_snapshot_op(HistoryAction::RotateCcw.label("Rotate 90 deg CCW"), |s| {
                    crate::ops::transform::rotate_canvas_90ccw(s);
                });
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasRotate180) => {
                self.do_snapshot_op(HistoryAction::Rotate180.label("Rotate 180 deg"), |s| {
                    crate::ops::transform::rotate_canvas_180(s);
                });
            }
            BindableAction::FlattenLayers => {
                self.do_snapshot_op(HistoryAction::Flatten.label("Flatten All Layers"), |s| {
                    crate::ops::transform::flatten_image(s);
                });
            }
            BindableAction::ColorAutoLevels => {
                self.do_layer_snapshot_op(HistoryAction::AutoLevels.label("Auto Levels"), |s| {
                    let idx = s.active_layer_index;
                    crate::ops::adjustments::auto_levels(s, idx);
                });
            }
            BindableAction::ColorDesaturate => {
                self.do_layer_snapshot_op(HistoryAction::Desaturate.label("Desaturate"), |s| {
                    let idx = s.active_layer_index;
                    crate::ops::filters::desaturate_layer(s, idx);
                });
            }
            BindableAction::ColorInvertColors => {
                self.do_gpu_snapshot_op(
                    HistoryAction::InvertColors.label("Invert Colors"),
                    |s, gpu| {
                        let idx = s.active_layer_index;
                        crate::ops::adjustments::invert_colors_gpu(s, idx, gpu);
                    },
                );
            }
            BindableAction::ColorInvertAlpha => {
                self.do_layer_snapshot_op(HistoryAction::InvertAlpha.label("Invert Alpha"), |s| {
                    let idx = s.active_layer_index;
                    crate::ops::adjustments::invert_alpha(s, idx);
                });
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ColorColorToAlpha) => {
                let target = self.colors_panel.get_primary_color();
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::ColorToAlpha(
                        crate::ops::effect_dialogs::ColorToAlphaDialog::new_with_target(
                            &project.canvas_state,
                            target,
                        ),
                    );
                }
            }
            BindableAction::Menu(
                crate::config::commands::MenuCommand::ColorRecoverTransparency,
            ) => {
                let target = self.colors_panel.get_primary_color();
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::RecoverTransparency(
                        crate::ops::effect_dialogs::RecoverTransparencyDialog::new_with_target(
                            &project.canvas_state,
                            target,
                        ),
                    );
                }
            }
            BindableAction::ColorSepiaTone => {
                self.do_layer_snapshot_op(HistoryAction::Sepia.label("Sepia Tone"), |s| {
                    let idx = s.active_layer_index;
                    crate::ops::adjustments::sepia(s, idx);
                });
            }
            BindableAction::ColorBrightnessContrast => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::BrightnessContrast(
                        crate::ops::dialogs::BrightnessContrastDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorCurves => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Curves(
                        crate::ops::dialogs::CurvesDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorExposure => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Exposure(
                        crate::ops::dialogs::ExposureDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorHighlightsShadows => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::HighlightsShadows(
                        crate::ops::dialogs::HighlightsShadowsDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorHueSaturation => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::HueSaturation(
                        crate::ops::dialogs::HueSaturationDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorLevels => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Levels(
                        crate::ops::dialogs::LevelsDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorTemperatureTint => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::TemperatureTint(
                        crate::ops::dialogs::TemperatureTintDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorVibrance => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Vibrance(
                        crate::ops::dialogs::VibranceDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorThreshold => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Threshold(
                        crate::ops::dialogs::ThresholdDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorPosterize => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Posterize(
                        crate::ops::dialogs::PosterizeDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorBalance => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::ColorBalance(
                        crate::ops::dialogs::ColorBalanceDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorGradientMap => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::GradientMap(
                        crate::ops::dialogs::GradientMapDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ColorBlackAndWhite => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::BlackAndWhite(
                        crate::ops::dialogs::BlackAndWhiteDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterGaussianBlur => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::GaussianBlur(
                        crate::ops::dialogs::GaussianBlurDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterBokehBlur => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::BokehBlur(
                        crate::ops::effect_dialogs::BokehBlurDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterMotionBlur => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::MotionBlur(
                        crate::ops::effect_dialogs::MotionBlurDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterBoxBlur => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::BoxBlur(
                        crate::ops::effect_dialogs::BoxBlurDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterZoomBlur => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::ZoomBlur(
                        crate::ops::effect_dialogs::ZoomBlurDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterSharpen => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Sharpen(
                        crate::ops::effect_dialogs::SharpenDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterReduceNoise => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::ReduceNoise(
                        crate::ops::effect_dialogs::ReduceNoiseDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterCrystallize => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Crystallize(
                        crate::ops::effect_dialogs::CrystallizeDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterDents => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Dents(
                        crate::ops::effect_dialogs::DentsDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterPixelate => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Pixelate(
                        crate::ops::effect_dialogs::PixelateDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterBulge => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Bulge(
                        crate::ops::effect_dialogs::BulgeDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterTwist => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Twist(
                        crate::ops::effect_dialogs::TwistDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterAddNoise => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::AddNoise(
                        crate::ops::effect_dialogs::AddNoiseDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterMedian => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Median(
                        crate::ops::effect_dialogs::MedianDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterGlow => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Glow(
                        crate::ops::effect_dialogs::GlowDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterVignette => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Vignette(
                        crate::ops::effect_dialogs::VignetteDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterHalftone => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Halftone(
                        crate::ops::effect_dialogs::HalftoneDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterInk => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Ink(
                        crate::ops::effect_dialogs::InkDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterOilPainting => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::OilPainting(
                        crate::ops::effect_dialogs::OilPaintingDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterColorFilter => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::ColorFilter(
                        crate::ops::effect_dialogs::ColorFilterDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::Menu(
                crate::config::commands::MenuCommand::FilterStylizeCanvasBorder,
            ) => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::CanvasBorder(
                        crate::ops::effect_dialogs::CanvasBorderDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::FilterStylizeSeamless) => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::SeamlessTexture(
                        crate::ops::effect_dialogs::SeamlessTextureDialog::new(
                            &project.canvas_state,
                        ),
                    );
                }
            }
            BindableAction::FilterPixelDrag => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::PixelDrag(
                        crate::ops::effect_dialogs::PixelDragDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterRgbDisplace => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::RgbDisplace(
                        crate::ops::effect_dialogs::RgbDisplaceDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::GenerateGrid => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Grid(
                        crate::ops::effect_dialogs::GridDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::GenerateDropShadow => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::DropShadow(
                        crate::ops::effect_dialogs::DropShadowDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::GenerateOutline => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Outline(
                        crate::ops::effect_dialogs::OutlineDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::GenerateContours => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::Contours(
                        crate::ops::effect_dialogs::ContoursDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::ViewZoomIn => {
                self.canvas.zoom_in();
            }
            BindableAction::ViewZoomOut => {
                self.canvas.zoom_out();
            }
            BindableAction::ViewZoom100 => {
                self.canvas.reset_zoom();
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::FileClearRecent) => {
                self.settings.recent_files.clear();
                self.settings.save();
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasStraighten) => {
                self.start_straighten();
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewTools) => {
                self.window_visibility.tools = !self.window_visibility.tools;
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewLayers) => {
                self.window_visibility.layers = !self.window_visibility.layers;
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewHistory) => {
                self.window_visibility.history = !self.window_visibility.history;
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewColors) => {
                self.window_visibility.colors = !self.window_visibility.colors;
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewPalette) => {
                self.window_visibility.palette = !self.window_visibility.palette;
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewScriptEditor) => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.window_visibility.script_editor = !self.window_visibility.script_editor;
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewPixelGrid) => {
                if let Some(project) = self.active_project_mut() {
                    project.canvas_state.show_pixel_grid = !project.canvas_state.show_pixel_grid;
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewCmyk) => {
                if let Some(project) = self.active_project_mut() {
                    project.canvas_state.cmyk_preview = !project.canvas_state.cmyk_preview;
                    project.canvas_state.composite_cache = None;
                    project.canvas_state.mark_render_dirty(None);
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewLight) => {
                if self.theme.mode != crate::theme::ThemeMode::Light {
                    self.theme.toggle();
                    self.theme.apply(ctx);
                    self.settings.theme_mode = self.theme.mode;
                    self.settings.save();
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ViewDark) => {
                if self.theme.mode != crate::theme::ThemeMode::Dark {
                    self.theme.toggle();
                    self.theme.apply(ctx);
                    self.settings.theme_mode = self.theme.mode;
                    self.settings.save();
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::HelpQuickStart) => {
                crate::ui::help::open(ctx);
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::HelpAiSetup) => {
                self.settings_window.open_ai_settings(&self.settings);
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::HelpDocumentation) => {
                ctx.open_url(egui::OpenUrl::new_tab(crate::ui::help::DOCUMENTATION_URL));
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::CanvasAlign) => {
                if let Some(project) = self.active_project() {
                    self.active_dialog = ActiveDialog::AlignLayer(
                        crate::ops::dialogs::AlignLayerDialog::new(&project.canvas_state),
                    );
                }
            }
            BindableAction::FilterRemoveBackground => {
                if !self.onnx_available {
                    self.settings_window.open_ai_settings(&self.settings);
                } else {
                    self.active_dialog = ActiveDialog::RemoveBackground(
                        crate::ops::effect_dialogs::RemoveBackgroundDialog::new(),
                    );
                }
            }
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolEllipseSelect) => {self.tools_panel.change_tool(crate::components::tools::Tool::EllipseSelect);},
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolMoveSelection) => {self.tools_panel.change_tool(crate::components::tools::Tool::MoveSelection);},
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolContentAwareBrush) => {self.tools_panel.change_tool(crate::components::tools::Tool::ContentAwareBrush);},
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolLiquify) => {self.tools_panel.change_tool(crate::components::tools::Tool::Liquify);},
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolSmudge) => {self.tools_panel.change_tool(crate::components::tools::Tool::Smudge);},
            BindableAction::Menu(crate::config::commands::MenuCommand::ToolPerspectiveCrop) => {self.tools_panel.change_tool(crate::components::tools::Tool::PerspectiveCrop);},
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesRulers) => {if let Some(project)=self.active_project_mut() {crate::ui::guides::run_menu_command(ctx,project,crate::config::commands::MenuCommand::GuidesRulers);}},
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesVisible) => {if let Some(project)=self.active_project_mut() {crate::ui::guides::run_menu_command(ctx,project,crate::config::commands::MenuCommand::GuidesVisible);}},
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesLocked) => {if let Some(project)=self.active_project_mut() {crate::ui::guides::run_menu_command(ctx,project,crate::config::commands::MenuCommand::GuidesLocked);}},
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesSnap) => {if let Some(project)=self.active_project_mut() {crate::ui::guides::run_menu_command(ctx,project,crate::config::commands::MenuCommand::GuidesSnap);}},
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesNew) => {if let Some(project)=self.active_project_mut() {crate::ui::guides::run_menu_command(ctx,project,crate::config::commands::MenuCommand::GuidesNew);}},
            BindableAction::Menu(crate::config::commands::MenuCommand::GuidesClear) => {if let Some(project)=self.active_project_mut() {crate::ui::guides::run_menu_command(ctx,project,crate::config::commands::MenuCommand::GuidesClear);}},
            BindableAction::ViewFitToWindow => {self.fit_active_document();},
            _ => return false,
        }
        true
    }
}

impl PaintFEApp {
    fn show_command_toggle(&mut self,ui:&mut egui::Ui,ctx:&egui::Context,command:crate::config::commands::MenuCommand,value:bool) {
        let action=crate::config::keybindings::BindableAction::Menu(command);
        let mut checked=value;
        let response=ui.push_id(action,|ui|ui.add_enabled(self.menu_command_enabled(ctx,action),egui::Checkbox::new(&mut checked,command.label()))).inner;
        let response=if let Some(combo)=self.settings.keybindings.get(action) {response.on_hover_text(combo.display())} else {response};
        if response.changed() {self.execute_menu_command(ctx,action);}
    }
    fn show_command_button(&mut self,ui:&mut egui::Ui,ctx:&egui::Context,command:crate::config::commands::MenuCommand)->bool {
        let action=crate::config::keybindings::BindableAction::Menu(command);
        let shortcut=self.settings.keybindings.get(action).map(|combo|combo.display()).unwrap_or_default();
        let response=ui.push_id(action,|ui|ui.add_enabled(self.menu_command_enabled(ctx,action),egui::Button::new(command.label()).right_text(shortcut))).inner;
        if response.clicked(){self.execute_menu_command(ctx,action);true}else{false}
    }
}
