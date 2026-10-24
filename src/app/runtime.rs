impl PaintFEApp {
    fn document_modal_open(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        let welcome_open = self.show_welcome_popup;
        #[cfg(not(target_arch = "wasm32"))]
        let welcome_open = false;
        self.filter_error.is_some()
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
        if let Err(error) = crate::config::persistence::drain() {
            crate::logger::write("WARN", &format!("Settings shutdown save failed: {error}"));
        }
    }
}
