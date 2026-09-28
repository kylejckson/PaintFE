impl PaintFEApp {
    fn process_active_dialog(&mut self, ctx: &egui::Context) {
        let mut dialog = std::mem::take(&mut self.active_dialog);

        // Log dialog open
        if !dialog.is_none() {
            log_info!("Dialog: open ({})", dialog.name());
        }

        #[cfg(not(target_arch = "wasm32"))]
        let matched = self.process_dialog_dispatch(ctx, &mut dialog)
            || self.process_paintdotnet_plugin_dialog(ctx, &mut dialog);
        #[cfg(target_arch = "wasm32")]
        let matched = self.process_dialog_dispatch(ctx, &mut dialog);

        // Matched handlers store the dialog back into `self.active_dialog`
        // (or clear it on OK/Cancel); unmatched dialogs are put back here.
        if !matched {
            self.active_dialog = dialog;
        }

        // "Live preview" was unchecked → put the original layer pixels back.
        self.sync_preview_restore_request(ctx);
    }

    /// Route the active dialog to its category handler. Returns true when the
    /// dialog was handled.
    fn process_dialog_dispatch(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        self.process_brush_tip_dialog(ctx, dialog)
            || self.process_canvas_and_transform_dialog(ctx, dialog)
            || self.process_adjustments_dialog(ctx, dialog)
            || self.process_blur_dialog(ctx, dialog)
            || self.process_distort_dialog(ctx, dialog)
            || self.process_noise_dialog(ctx, dialog)
            || self.process_stylize_dialog(ctx, dialog)
            || self.process_render_dialog(ctx, dialog)
            || self.process_glitch_and_artistic_dialog(ctx, dialog)
            || self.process_ai_and_color_selection_dialog(ctx, dialog)
    }

    /// Consume a "live preview turned off" request from `preview_controls` and
    /// restore the original pixels of the dialog's target layer. This makes the
    /// checkbox a before/after toggle: off shows the untouched image, on
    /// re-applies the current parameters (via the usual `DialogResult::Changed`
    /// path), so the two states can be compared in a loop.
    fn sync_preview_restore_request(&mut self, ctx: &egui::Context) {
        let request_id = crate::ui::dialogs::core::preview_restore_request_id();
        let requested = ctx.data_mut(|d| d.get_temp::<bool>(request_id).unwrap_or(false));
        if !requested {
            return;
        }
        ctx.data_mut(|d| {
            d.insert_temp(request_id, false);
        });

        // Discard any in-flight async preview job so its result cannot repaint
        // the effect over the restored original (mirrors the Cancel path).
        self.filter_cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let Some((layer_idx, original)) = preview_restore_target(&self.active_dialog) else {
            return;
        };
        let Some(original) = original else {
            return;
        };
        if let Some(project) = self.active_project_mut() {
            if let Some(layer) = project.canvas_state.layers.get_mut(layer_idx) {
                layer.pixels = original;
            }
            project.canvas_state.mark_dirty(None);
            ctx.request_repaint();
        }
    }
}

/// `(layer_idx, original pixels)` for every dialog whose live preview rewrites
/// a layer — the restore target for `sync_preview_restore_request`. Dialogs
/// without a previewable layer return `None`.
fn preview_restore_target(dialog: &ActiveDialog) -> Option<(usize, Option<TiledImage>)> {
    Some(match dialog {
        // --- Adjustments ---
        ActiveDialog::BrightnessContrast(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::HueSaturation(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Exposure(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::HighlightsShadows(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Levels(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Curves(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::TemperatureTint(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Threshold(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Posterize(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::ColorBalance(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::GradientMap(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::BlackAndWhite(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Vibrance(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Blur ---
        ActiveDialog::GaussianBlur(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::BokehBlur(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::MotionBlur(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::BoxBlur(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::ZoomBlur(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Distort ---
        ActiveDialog::Crystallize(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Dents(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Pixelate(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Bulge(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Twist(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Noise ---
        ActiveDialog::AddNoise(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::ReduceNoise(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Median(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Stylize ---
        ActiveDialog::Glow(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Sharpen(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Vignette(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Halftone(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Render ---
        ActiveDialog::Grid(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::DropShadow(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Outline(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::CanvasBorder(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::SeamlessTexture(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Glitch ---
        ActiveDialog::PixelDrag(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::RgbDisplace(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Artistic ---
        ActiveDialog::Ink(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::OilPainting(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::ColorFilter(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::ColorToAlpha(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::RecoverTransparency(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::Contours(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Transform ---
        ActiveDialog::LayerTransform(d) => (d.layer_idx, d.original_pixels.clone()),
        ActiveDialog::AlignLayer(d) => (d.layer_idx, d.original_pixels.clone()),
        // --- Paint.NET plugin host ---
        #[cfg(not(target_arch = "wasm32"))]
        ActiveDialog::PaintDotNetPlugin(d) => (d.layer_idx, Some(d.original_pixels.clone())),
        _ => return None,
    })
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
#[cfg(not(target_arch = "wasm32"))]
include!("dialogs/paintdotnet_plugin.rs");
