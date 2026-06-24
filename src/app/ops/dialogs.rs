impl PaintFEApp {
    fn process_active_dialog(&mut self, ctx: &egui::Context) {
        let mut dialog = std::mem::take(&mut self.active_dialog);

        // Log dialog open
        if !dialog.is_none() {
            log_info!("Dialog: open ({})", dialog.name());
        }

        if self.process_brush_tip_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_canvas_and_transform_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_adjustments_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_blur_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_distort_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_noise_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_stylize_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_render_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_glitch_and_artistic_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_ai_and_color_selection_dialog(ctx, &mut dialog) {
            return;
        }

        if self.process_paintdotnet_plugin_dialog(ctx, &mut dialog) {
            return;
        }

        self.active_dialog = dialog;
    }
}

include!("dialogs/brush_tip.rs");
include!("dialogs/canvas_and_transform.rs");
include!("dialogs/adjustments.rs");
include!("dialogs/blur.rs");
include!("dialogs/distort.rs");
include!("dialogs/noise.rs");
include!("dialogs/stylize.rs");
include!("dialogs/render.rs");
include!("dialogs/glitch_and_artistic.rs");
include!("dialogs/ai_and_color_selection.rs");
include!("dialogs/paintdotnet_plugin.rs");
