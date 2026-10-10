impl eframe::App for PaintFEApp {
    fn raw_input_hook(&mut self, ctx: &egui::Context, input: &mut egui::RawInput) {
        #[cfg(target_os = "windows")]
        {
            crate::windows_key_probe::bridge_raw_input(input);
            ctx.data_mut(|d| { d.insert_temp(egui::Id::new("native_keyboard_bridge"), true); });
            if !input.focused { self.settings.keybindings.discard_pending_presses(ctx); }
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
        for theme in [egui::Theme::Dark, egui::Theme::Light] {
            if ctx.style_of(theme).interaction.tooltip_delay != self.settings.tooltip_delay {
                ctx.style_mut_of(theme, |style| style.interaction.tooltip_delay = self.settings.tooltip_delay);
            }
        }
        let _profile = crate::ui::perf::Frame::begin(&ctx, frame.info().cpu_usage);
        {
            let _scope = crate::ui::perf::Scope::new(0);
            self.update_runtime_lifecycle_async(&ctx);
        }
        let modal_open = {
            let _scope = crate::ui::perf::Scope::new(4);
            self.update_runtime_input(&ctx)
        };
        {
            let _scope = crate::ui::perf::Scope::new(5);
            self.show_runtime_dialogs_menu(&ctx, ui);
        }
        {
            let _scope = crate::ui::perf::Scope::new(6);
            self.show_runtime_canvas_tail(&ctx, ui, modal_open);
        }
    }
}

include!("runtime/update/lifecycle_async.rs");
include!("runtime/update/input_shortcuts.rs");
include!("runtime/update/dialogs_menu.rs");
include!("runtime/update/canvas_tail.rs");
