impl PaintFEApp {
    fn fit_active_document(&mut self) {
        if let Some(project) = self.active_project() {
            let size = (project.canvas_state.width, project.canvas_state.height);
            self.canvas.fit_to_window(size);
        }
    }
    fn restore_script_original(&mut self) {
        if let Some((id, layer, original, revision)) = self.script_original_pixels.take()
            && let Some(project) = self
                .projects
                .iter_mut()
                .find(|p| p.id == id && p.edit_revision == revision)
            && let Some(target) = project.canvas_state.layers.get_mut(layer)
        {
            target.pixels = original;
            project.canvas_state.mark_dirty(None);
        }
    }

    fn processing_busy(&self) -> bool {
        self.pending_filter_jobs > 0 && matches!(self.active_dialog, ActiveDialog::None)
    }

    fn processing_dialog(&mut self, ctx: &egui::Context) {
        if !self.processing_busy() {
            return;
        }
        egui::Modal::new(egui::Id::new("processing_document")).show(ctx, |ui| {
            ui.set_max_width(380.0);
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(&self.filter_status_description);
            });
            if self
                .operation_cancel
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                ui.label(crate::t!("processing.cancelling"));
            } else if ui.button(crate::t!("common.cancel")).clicked() {
                self.operation_cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.filter_cancel
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.script_editor
                    .cancel_flag
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.preview_job_token = self.preview_job_token.wrapping_add(1);
                self.pending_preview = None;
                crate::services::optional_packs::avif::cancel();
            }
        });
    }

    fn document_modal_open(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        let welcome_open = self.show_welcome_popup;
        #[cfg(not(target_arch = "wasm32"))]
        let welcome_open = false;
        self.processing_busy()
            || self.filter_error.is_some()
            || self.guide_dialog_open
            || welcome_open
            || self.settings_window.open
            || self.save_file_dialog.open
            || self.new_file_dialog.open
            || !matches!(self.active_dialog, ActiveDialog::None)
            || self.pending_paste_request.is_some()
            || !self.pending_import_queue.is_empty()
            || self.pending_oversized_import.is_some()
            || self.pending_exit
            || self.pending_close_index.is_some()
            || self.save_warning.is_some()
            || self.save_sequence.is_some()
    }
}

impl eframe::App for PaintFEApp {
    fn raw_input_hook(&mut self, ctx: &egui::Context, input: &mut egui::RawInput) {
        #[cfg(target_os = "windows")]
        {
            crate::windows_key_probe::bridge_raw_input(input);
            ctx.data_mut(|d| {
                d.insert_temp(egui::Id::new("native_keyboard_bridge"), true);
            });
            if !input.focused {
                self.settings.keybindings.discard_pending_presses(ctx);
            }
        }
        #[cfg(not(target_os = "windows"))]
        let _ = (ctx, input);
    }
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        let c = self.theme.canvas_bg_bottom;
        [
            c.r() as f32 / 255.0,
            c.g() as f32 / 255.0,
            c.b() as f32 / 255.0,
            1.0,
        ]
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.guide_dialog_open = crate::ui::guides::dialog_open(&ctx);
        crate::ui::guides::set_appearance(&ctx, &self.settings.ruler);
        crate::io::heic::set_library_path(&self.settings.heic_library_path);
        let _profile = crate::ui::perf::Frame::begin(&ctx, frame.info().cpu_usage);
        ctx.data_mut(|d| {
            d.insert_temp(
                egui::Id::new("resize_grip_appearance"),
                (
                    self.settings.resize_grip_color,
                    self.settings.resize_grip_opacity,
                ),
            )
        });
        for theme in [egui::Theme::Dark, egui::Theme::Light] {
            if ctx.style_of(theme).interaction.tooltip_delay != self.settings.tooltip_delay {
                ctx.style_mut_of(theme, |style| {
                    style.interaction.tooltip_delay = self.settings.tooltip_delay
                });
            }
        }
        {
            let _scope = crate::ui::perf::Scope::new(0);
            self.update_runtime_lifecycle_async(&ctx);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let pack_modal = crate::ui::panels::optional_features::frame(&ctx, &mut self.settings);
        #[cfg(target_arch = "wasm32")]
        let pack_modal = false;
        self.processing_dialog(&ctx);
        let modal_open = {
            let _scope = crate::ui::perf::Scope::new(4);
            self.update_runtime_input(&ctx) || pack_modal
        };
        {
            let _scope = crate::ui::perf::Scope::new(5);
            self.show_runtime_dialogs_menu(&ctx, ui);
        }
        {
            let _scope = crate::ui::perf::Scope::new(6);
            self.show_runtime_canvas_tail(&ctx, ui, modal_open);
        }
        if let Some(project) = self.active_project_mut() {
            crate::ui::guides::begin(&ctx, project.id);
            crate::ui::guides::dialog(&ctx, project);
            crate::ui::guides::drain(&ctx, project);
        }
    }
}

include!("runtime/update/lifecycle_async.rs");
include!("runtime/update/input_shortcuts.rs");
include!("runtime/update/dialogs_menu.rs");
include!("runtime/update/canvas_tail.rs");

impl Drop for PaintFEApp {
    fn drop(&mut self) {
        self.operation_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(target_arch = "wasm32"))]
        crate::services::resize_worker::drain_cancelled();
        if let Err(error) = crate::config::persistence::drain() {
            crate::logger::write("WARN", &format!("Settings shutdown save failed: {error}"));
        }
    }
}
