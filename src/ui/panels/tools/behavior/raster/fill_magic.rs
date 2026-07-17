impl ToolsPanel {
    fn reset_fill_preview_state(&mut self, clear_pending_clicks: bool) {
        self.fill_state.active_fill = None;
        self.fill_state.fill_color_u8 = None;
        self.fill_state.tolerance_changed_at = None;
        self.fill_state.recalc_pending = false;
        self.fill_state.async_rx = None;
        self.fill_state.preview_in_flight = false;
        self.fill_state.gpu_preview_region.clear();
        self.fill_state.defer_preview_clear = false;
        self.fill_state.preview_clear_at = None;
        if clear_pending_clicks {
            self.fill_state.pending_clicks.clear();
        }
        // Keep cached_flat_rgba alive to avoid re-fetch on next click.
    }

    fn clear_fill_preview_state(&mut self) {
        self.reset_fill_preview_state(true);
    }

    pub(crate) fn should_keep_fill_preview_overlay(&self) -> bool {
        self.fill_state.active_fill.is_some()
            || self.fill_state.preview_in_flight
            || self.fill_state.defer_preview_clear
            || self.fill_state.preview_clear_at.is_some()
            || !self.fill_state.pending_clicks.is_empty()
    }

    fn clear_magic_wand_async_state(&mut self) {
        self.magic_wand_state.operations.clear();
        self.magic_wand_state.session_before_mask = None;
        self.magic_wand_state.pending_operation = None;
        self.magic_wand_state.async_rx = None;
        self.magic_wand_state.computing = false;
        self.magic_wand_state.preview_pending = false;
        self.magic_wand_state.tolerance_changed_at = None;
        self.magic_wand_state.last_applied_tolerance = -1.0;
        // Force a fresh texture upload on the next click so stale GPU state
        // cannot cause the wand to seed from the wrong pixel.
        self.magic_wand_state.cached_flat_rgba = None;
    }

    fn magic_wand_has_session(&self) -> bool {
        self.magic_wand_state.computing
            || self.magic_wand_state.pending_operation.is_some()
            || !self.magic_wand_state.operations.is_empty()
    }

    fn begin_magic_wand_session_if_needed(&mut self, canvas_state: &CanvasState) {
        if self.magic_wand_state.session_before_mask.is_none() {
            self.magic_wand_state.session_before_mask = canvas_state.selection_mask.clone();
        }
    }

    fn finalize_magic_wand_session(&mut self, canvas_state: &mut CanvasState) {
        let before = self.magic_wand_state.session_before_mask.clone();
        let after = canvas_state.selection_mask.clone();
        if before != after {
            self.pending_history_commands
                .push(Box::new(SelectionCommand::new(
                    "Magic Wand Select",
                    before,
                    after,
                )));
        }
        self.clear_magic_wand_async_state();
    }

    fn cancel_magic_wand_session(&mut self, canvas_state: &mut CanvasState) {
        canvas_state.selection_mask = self.magic_wand_state.session_before_mask.clone();
        canvas_state.invalidate_selection_overlay();
        canvas_state.mark_dirty(None);
        self.clear_magic_wand_async_state();
    }

    #[inline]
    fn tolerance_threshold_u8(tolerance: f32) -> u8 {
        let normalized = (tolerance / 100.0).clamp(0.0, 1.0);
        (normalized * 255.0).round().clamp(0.0, 255.0) as u8
    }

    #[inline]
    fn srgb_to_linear(v: f32) -> f32 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    #[inline]
    fn perceptual_distance(flat_rgba: &[u8], idx: usize, target: &Rgba<u8>) -> u8 {
        let base = idx * 4;
        let r = flat_rgba[base] as f32 / 255.0;
        let g = flat_rgba[base + 1] as f32 / 255.0;
        let b = flat_rgba[base + 2] as f32 / 255.0;
        let a = flat_rgba[base + 3] as f32 / 255.0;

        let tr = target.0[0] as f32 / 255.0;
        let tg = target.0[1] as f32 / 255.0;
        let tb = target.0[2] as f32 / 255.0;
        let ta = target.0[3] as f32 / 255.0;

        if ta <= 0.0 && a <= 0.0 {
            return 0;
        }

        let r_lin = Self::srgb_to_linear(r) * a;
        let g_lin = Self::srgb_to_linear(g) * a;
        let b_lin = Self::srgb_to_linear(b) * a;
        let tr_lin = Self::srgb_to_linear(tr) * ta;
        let tg_lin = Self::srgb_to_linear(tg) * ta;
        let tb_lin = Self::srgb_to_linear(tb) * ta;

        let dr = r_lin - tr_lin;
        let dg = g_lin - tg_lin;
        let db = b_lin - tb_lin;

        let dluma = (0.2126 * dr + 0.7152 * dg + 0.0722 * db).abs();
        let dchroma = (0.5 * (dr - dg) * (dr - dg)
            + 0.5 * (dg - db) * (dg - db)
            + 0.5 * (db - dr) * (db - dr))
            .sqrt();
        let color_term = (dluma * 0.7 + dchroma * 0.8).clamp(0.0, 1.0);
        let alpha_term = (a - ta).abs();

        ((color_term.max(alpha_term) * 255.0).round()).clamp(0.0, 255.0) as u8
    }

    #[inline]
    fn selection_mode_to_gpu(mode: SelectionMode) -> u32 {
        match mode {
            SelectionMode::Replace => 0,
            SelectionMode::Add => 1,
            SelectionMode::Subtract => 2,
            SelectionMode::Intersect => 3,
        }
    }

    #[inline]
    fn union_bbox(
        a: Option<(u32, u32, u32, u32)>,
        b: Option<(u32, u32, u32, u32)>,
    ) -> Option<(u32, u32, u32, u32)> {
        match (a, b) {
            (Some((ax0, ay0, ax1, ay1)), Some((bx0, by0, bx1, by1))) => {
                Some((ax0.min(bx0), ay0.min(by0), ax1.max(bx1), ay1.max(by1)))
            }
            (Some(bbox), None) | (None, Some(bbox)) => Some(bbox),
            (None, None) => None,
        }
    }

    fn apply_fill_preview_patch(
        &mut self,
        canvas_state: &mut CanvasState,
        dirty_bbox: Option<(u32, u32, u32, u32)>,
        fill_bbox: Option<(u32, u32, u32, u32)>,
        preview_region: &[u8],
        preview_region_w: u32,
        preview_region_h: u32,
    ) {
        if self.fill_state.defer_preview_clear {
            // The previous preview stayed visible during reseed handoff; now
            // fully replace it so stale pixels from the old fill cannot leak
            // into the new overlay or the next commit.
            let preview_blend_mode = canvas_state.preview_blend_mode;
            let preview_force_composite = canvas_state.preview_force_composite;
            let preview_is_eraser = canvas_state.preview_is_eraser;
            let preview_replaces_layer = canvas_state.preview_replaces_layer;
            let preview_targets_mask = canvas_state.preview_targets_mask;
            let preview_mask_reveal = canvas_state.preview_mask_reveal;
            canvas_state.clear_preview_state();
            canvas_state.preview_blend_mode = preview_blend_mode;
            canvas_state.preview_force_composite = preview_force_composite;
            canvas_state.preview_is_eraser = preview_is_eraser;
            canvas_state.preview_replaces_layer = preview_replaces_layer;
            canvas_state.preview_targets_mask = preview_targets_mask;
            canvas_state.preview_mask_reveal = preview_mask_reveal;
            self.fill_state.defer_preview_clear = false;
        }

        if let Some(dirty_bbox) = dirty_bbox {
            if canvas_state.preview_layer.is_none() {
                canvas_state.preview_layer =
                    Some(TiledImage::new(canvas_state.width, canvas_state.height));
            }
            if let Some(preview) = canvas_state.preview_layer.as_mut() {
                // Use blit_rgba_at_replace so that pixels becoming transparent
                // (tolerance decreased, unfilled) properly clear the preview layer.
                preview.blit_rgba_at_replace(
                    dirty_bbox.0 as i32,
                    dirty_bbox.1 as i32,
                    preview_region_w,
                    preview_region_h,
                    preview_region,
                );
            }

            let dirty_rect = egui::Rect::from_min_max(
                egui::pos2(dirty_bbox.0 as f32, dirty_bbox.1 as f32),
                egui::pos2(
                    (dirty_bbox.2 + 1).min(canvas_state.width) as f32,
                    (dirty_bbox.3 + 1).min(canvas_state.height) as f32,
                ),
            );
            canvas_state.preview_dirty_rect = Some(match canvas_state.preview_dirty_rect {
                Some(existing) => existing.union(dirty_rect),
                None => dirty_rect,
            });
            canvas_state.preview_generation = canvas_state.preview_generation.wrapping_add(1);
        }

        if let Some(bbox) = fill_bbox {
            let fill_bounds = egui::Rect::from_min_max(
                egui::pos2(bbox.0 as f32, bbox.1 as f32),
                egui::pos2(
                    (bbox.2 + 1).min(canvas_state.width) as f32,
                    (bbox.3 + 1).min(canvas_state.height) as f32,
                ),
            );
            canvas_state.preview_stroke_bounds = Some(fill_bounds);
            self.stroke_tracker.expand_bounds(fill_bounds);
        } else {
            canvas_state.preview_stroke_bounds = None;
        }
    }

    fn render_fill_preview_gpu(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: &mut crate::gpu::GpuRenderer,
    ) {
        let Some(active_fill) = self.fill_state.active_fill.as_mut() else {
            return;
        };
        let Some(region_index) = active_fill.region_index.as_ref() else {
            return;
        };
        let Some(fill_color_u8) = self.fill_state.fill_color_u8 else {
            return;
        };
        let Some(flat_rgba) =
            Self::cached_active_layer_rgba(&mut self.fill_state.cached_flat_rgba, canvas_state)
        else {
            return;
        };

        let threshold = Self::tolerance_threshold_u8(self.fill_state.tolerance);
        let anti_aliased = self.fill_state.anti_aliased;
        let color_or_aa_changed = self.fill_state.last_preview_aa != anti_aliased
            || self.fill_state.last_preview_tolerance < 0.0
            || self.fill_state.fill_color_u8 != Some(fill_color_u8);
        let old_bbox = active_fill.fill_bbox;
        let new_bbox = region_index.threshold_bbox(threshold);
        let mut dirty_bbox = if active_fill.last_threshold != Some(threshold) || color_or_aa_changed
        {
            Self::union_bbox(old_bbox, new_bbox)
        } else {
            None
        };

        if let Some((x0, y0, x1, y1)) = dirty_bbox {
            let pad = if anti_aliased { 1 } else { 0 };
            dirty_bbox = Some((
                x0.saturating_sub(pad),
                y0.saturating_sub(pad),
                (x1 + pad).min(canvas_state.width.saturating_sub(1)),
                (y1 + pad).min(canvas_state.height.saturating_sub(1)),
            ));
        }

        active_fill.last_threshold = Some(threshold);
        active_fill.fill_bbox = new_bbox;

        if let Some((x0, y0, x1, y1)) = dirty_bbox {
            let region_w = x1.saturating_sub(x0) + 1;
            let region_h = y1.saturating_sub(y0) + 1;
            let selection_mask = canvas_state
                .selection_mask
                .as_ref()
                .map(|mask| (mask.as_raw().as_slice(), mask.as_raw().as_ptr() as usize));
            let distance_key = region_index.distances.as_ref().as_ptr() as usize;
            let background_key = flat_rgba.as_ref().as_ptr() as usize;

            gpu_renderer.fill_preview_pipeline.generate_into(
                &gpu_renderer.ctx,
                region_index.distances.as_ref(),
                distance_key,
                flat_rgba.as_ref(),
                background_key,
                selection_mask,
                canvas_state.width,
                canvas_state.height,
                x0,
                y0,
                region_w,
                region_h,
                threshold,
                anti_aliased,
                fill_color_u8.0,
                &mut self.fill_state.gpu_preview_region,
            );
            let preview_region = std::mem::take(&mut self.fill_state.gpu_preview_region);

            self.apply_fill_preview_patch(
                canvas_state,
                Some((x0, y0, x1, y1)),
                new_bbox,
                &preview_region,
                region_w,
                region_h,
            );
            self.fill_state.gpu_preview_region = preview_region;
        } else {
            canvas_state.preview_stroke_bounds = new_bbox.map(|bbox| {
                egui::Rect::from_min_max(
                    egui::pos2(bbox.0 as f32, bbox.1 as f32),
                    egui::pos2(
                        (bbox.2 + 1).min(canvas_state.width) as f32,
                        (bbox.3 + 1).min(canvas_state.height) as f32,
                    ),
                )
            });
        }

        self.fill_state.last_preview_tolerance = self.fill_state.tolerance;
        self.fill_state.last_preview_aa = anti_aliased;
        self.fill_state.tolerance_changed_at = None;
        self.fill_state.recalc_pending = false;
        self.fill_state.preview_in_flight = false;
    }

    fn poll_active_layer_rgba_prewarm(&mut self) {
        let Some(rx) = &self.active_layer_rgba_prewarm_rx else {
            return;
        };
        if let Ok(cache) = rx.try_recv() {
            let key = (
                cache.layer_index,
                cache.gpu_generation,
                cache.width,
                cache.height,
            );
            self.magic_wand_state.cached_flat_rgba = Some(cache.clone());
            self.fill_state.cached_flat_rgba = Some(cache);
            self.active_layer_rgba_prewarm_rx = None;
            self.active_layer_rgba_prewarm_key = Some(key);
        }
    }

    fn maybe_prewarm_active_layer_rgba(&mut self, canvas_state: &CanvasState) {
        if !matches!(self.active_tool, Tool::MagicWand) {
            return;
        }

        let layer_index = canvas_state.active_layer_index;
        let Some(layer) = canvas_state.layers.get(layer_index) else {
            return;
        };
        let key = (
            layer_index,
            layer.gpu_generation,
            canvas_state.width,
            canvas_state.height,
        );

        let already_cached = self
            .magic_wand_state
            .cached_flat_rgba
            .as_ref()
            .is_some_and(|entry| {
                (
                    entry.layer_index,
                    entry.gpu_generation,
                    entry.width,
                    entry.height,
                ) == key
            })
            || self
                .fill_state
                .cached_flat_rgba
                .as_ref()
                .is_some_and(|entry| {
                    (
                        entry.layer_index,
                        entry.gpu_generation,
                        entry.width,
                        entry.height,
                    ) == key
                });
        if already_cached || self.active_layer_rgba_prewarm_key == Some(key) {
            return;
        }

        let pixels = layer.pixels.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.active_layer_rgba_prewarm_rx = Some(rx);
        self.active_layer_rgba_prewarm_key = Some(key);

        crate::par_compat::spawn(move || {
            let data: Arc<[u8]> = Arc::from(pixels.to_rgba_image().into_raw().into_boxed_slice());
            let _ = tx.send(FlatLayerCache {
                layer_index: key.0,
                gpu_generation: key.1,
                width: key.2,
                height: key.3,
                data,
            });
        });
    }

    #[inline]
    fn threshold_alpha(distance: u8, threshold: u8, anti_aliased: bool) -> u8 {
        if !anti_aliased {
            return if distance <= threshold { 255 } else { 0 };
        }

        if distance <= threshold {
            255
        } else if distance == threshold.saturating_add(1) {
            128
        } else {
            0
        }
    }

    fn apply_threshold_delta<F>(
        index: &ThresholdRegionIndex,
        mask: &mut [u8],
        old_threshold: Option<u8>,
        old_anti_aliased: bool,
        new_threshold: u8,
        new_anti_aliased: bool,
        mut on_change: F,
    ) where
        F: FnMut(usize, u8),
    {
        let aa_band = if old_anti_aliased || new_anti_aliased { 1u8 } else { 0u8 };
        let (start_distance, end_distance) = match old_threshold {
            Some(previous) => (
                previous.min(new_threshold),
                previous.max(new_threshold).saturating_add(aa_band),
            ),
            None => (0, new_threshold.saturating_add(aa_band)),
        };

        for distance in start_distance..=end_distance {
            let old_alpha = old_threshold
                .map(|threshold| Self::threshold_alpha(distance, threshold, old_anti_aliased))
                .unwrap_or(0);
            let new_alpha = Self::threshold_alpha(distance, new_threshold, new_anti_aliased);
            if old_alpha == new_alpha {
                continue;
            }
            for &idx in index.buckets[distance as usize].iter() {
                let idx = idx as usize;
                if mask[idx] != new_alpha {
                    mask[idx] = new_alpha;
                    on_change(idx, new_alpha);
                }
            }
        }
    }

    fn build_threshold_mask(
        index: &ThresholdRegionIndex,
        threshold: u8,
        anti_aliased: bool,
    ) -> Vec<u8> {
        let mut mask = vec![0u8; (index.width * index.height) as usize];
        Self::apply_threshold_delta(
            index,
            &mut mask,
            None,
            false,
            threshold,
            anti_aliased,
            |_, _| {},
        );
        mask
    }

    #[inline]
    fn merge_magic_wand_masks(base_value: u8, raw_value: u8, combine_mode: SelectionMode) -> u8 {
        match combine_mode {
            SelectionMode::Replace => raw_value,
            SelectionMode::Add => base_value.max(raw_value),
            SelectionMode::Subtract => base_value.saturating_sub(raw_value),
            SelectionMode::Intersect => ((base_value as u16 * raw_value as u16) / 255) as u8,
        }
    }

    fn replay_magic_wand_selection(&self, threshold: u8, anti_aliased: bool) -> Option<GrayImage> {
        let first = self.magic_wand_state.operations.first()?;
        let n = (first.region_index.width * first.region_index.height) as usize;
        let mut final_mask = vec![0u8; n];

        for op in &self.magic_wand_state.operations {
            let raw_mask = Self::build_threshold_mask(&op.region_index, threshold, anti_aliased);
            for (idx, raw_value) in raw_mask.into_iter().enumerate() {
                final_mask[idx] =
                    Self::merge_magic_wand_masks(final_mask[idx], raw_value, op.combine_mode);
            }
        }

        GrayImage::from_raw(
            first.region_index.width,
            first.region_index.height,
            final_mask,
        )
    }

    fn apply_magic_wand_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        _gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        if self.magic_wand_state.operations.is_empty() {
            return;
        }
        let threshold = Self::tolerance_threshold_u8(self.magic_wand_state.tolerance);
        canvas_state.selection_mask =
            self.replay_magic_wand_selection(threshold, self.magic_wand_state.anti_aliased);
        canvas_state.invalidate_selection_overlay();
        canvas_state.mark_dirty(None);
        self.magic_wand_state.last_applied_tolerance = self.magic_wand_state.tolerance;
        self.magic_wand_state.last_applied_aa = self.magic_wand_state.anti_aliased;
        self.magic_wand_state.preview_pending = false;
        self.magic_wand_state.tolerance_changed_at = None;
    }

    fn maybe_spawn_magic_wand_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        if !self.magic_wand_state.preview_pending {
            return;
        }
        if let Some(changed_at) = self.magic_wand_state.tolerance_changed_at
            && changed_at.elapsed().as_millis() < 20
        {
            return;
        }
        self.apply_magic_wand_preview(canvas_state, gpu_renderer);
    }

    fn build_fill_preview_region(
        fill_mask: &[u8],
        fill_bbox: (u32, u32, u32, u32),
        fill_color: Rgba<u8>,
        selection_mask: Option<&GrayImage>,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let (bx0, by0, bx1, by1) = fill_bbox;
        let wu = width as usize;
        let region_w = bx1.saturating_sub(bx0) + 1;
        let region_h = by1.saturating_sub(by0) + 1;
        let mut region_buf = vec![0u8; region_w as usize * region_h as usize * 4];

        let spans =
            Self::collect_fill_preview_spans(fill_mask, fill_bbox, selection_mask, width, height);

        for span in spans {
            let row_offset = (span.y - by0) as usize * region_w as usize * 4;
            for x in span.x0..=span.x1 {
                let mask_idx = span.y as usize * wu + x as usize;
                let coverage = fill_mask[mask_idx];
                if coverage == 0 {
                    continue;
                }
                let local_idx = row_offset + (x - bx0) as usize * 4;
                region_buf[local_idx] = fill_color.0[0];
                region_buf[local_idx + 1] = fill_color.0[1];
                region_buf[local_idx + 2] = fill_color.0[2];
                region_buf[local_idx + 3] =
                    ((fill_color.0[3] as u16 * coverage as u16 + 127) / 255) as u8;
            }
        }

        region_buf
    }

    #[inline]
    fn fill_pixel_is_active(
        fill_mask: &[u8],
        selection_mask: Option<&GrayImage>,
        width: u32,
        height: u32,
        x: u32,
        y: u32,
    ) -> bool {
        if x >= width || y >= height {
            return false;
        }
        let idx = y as usize * width as usize + x as usize;
        fill_mask[idx] != 0
            && selection_mask
                .map(|mask| mask.get_pixel(x, y).0[0] > 0)
                .unwrap_or(true)
    }

    fn collect_fill_preview_spans(
        fill_mask: &[u8],
        fill_bbox: (u32, u32, u32, u32),
        selection_mask: Option<&GrayImage>,
        width: u32,
        height: u32,
    ) -> Vec<FillPreviewSpan> {
        let (bx0, by0, bx1, by1) = fill_bbox;
        let mut spans = Vec::new();
        let row_width = width as usize;

        for y in by0..=by1.min(height.saturating_sub(1)) {
            let row_offset = y as usize * row_width;
            let mut x = bx0;
            while x <= bx1.min(width.saturating_sub(1)) {
                let idx = row_offset + x as usize;
                if fill_mask[idx] == 0
                    || selection_mask
                        .map(|mask| mask.get_pixel(x, y).0[0] == 0)
                        .unwrap_or(false)
                {
                    x += 1;
                    continue;
                }

                let x0 = x;
                x += 1;
                while x <= bx1.min(width.saturating_sub(1)) {
                    let idx = row_offset + x as usize;
                    if fill_mask[idx] == 0
                        || selection_mask
                            .map(|mask| mask.get_pixel(x, y).0[0] == 0)
                            .unwrap_or(false)
                    {
                        break;
                    }
                    x += 1;
                }

                spans.push(FillPreviewSpan { y, x0, x1: x - 1 });
            }
        }

        spans
    }
    fn maybe_spawn_fill_preview(
        &mut self,
        canvas_state: &mut CanvasState,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        if !self.fill_state.recalc_pending || self.fill_state.preview_in_flight {
            return;
        }
        if let Some(changed_at) = self.fill_state.tolerance_changed_at
            && changed_at.elapsed().as_millis() < 50
        {
            return;
        }
        let Some(mut active_fill) = self.fill_state.active_fill.clone() else {
            return;
        };
        let Some(fill_color_u8) = self.fill_state.fill_color_u8 else {
            return;
        };
        let Some(flat_rgba) =
            Self::cached_active_layer_rgba(&mut self.fill_state.cached_flat_rgba, canvas_state)
        else {
            return;
        };
        let gpu_available = gpu_renderer.is_some();

        if let Some(gpu) = gpu_renderer
            && active_fill.region_index.is_some()
        {
            self.render_fill_preview_gpu(canvas_state, gpu);
            return;
        }

        let tolerance = self.fill_state.tolerance;
        let threshold = Self::tolerance_threshold_u8(tolerance);
        let anti_aliased = self.fill_state.anti_aliased;
        let selection_mask = canvas_state.selection_mask.clone();
        let width = canvas_state.width;
        let height = canvas_state.height;
        let distance_mode = self.fill_state.distance_mode;
        let connectivity = self.fill_state.connectivity;
        let color_or_aa_changed = self.fill_state.last_preview_aa != anti_aliased
            || self.fill_state.last_preview_tolerance < 0.0
            || self.fill_state.fill_color_u8 != Some(fill_color_u8);
        let mut dirty_bbox: Option<(u32, u32, u32, u32)> = None;

        if let Some(region_index) = active_fill.region_index.as_ref() {
            let previous_threshold = active_fill.last_threshold;
            if previous_threshold != Some(threshold) || active_fill.fill_mask.is_empty() {
                if active_fill.fill_mask.len() != (width * height) as usize {
                    active_fill.fill_mask = vec![0u8; (width * height) as usize];
                }
                Self::apply_threshold_delta(
                    region_index,
                    &mut active_fill.fill_mask,
                    previous_threshold,
                    false,
                    threshold,
                    false,
                    |idx, _| {
                        let x = (idx as u32) % width;
                        let y = (idx as u32) / width;
                        dirty_bbox = Some(match dirty_bbox {
                            Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                            None => (x, y, x, y),
                        });
                    },
                );
                active_fill.last_threshold = Some(threshold);
                active_fill.fill_bbox = region_index.threshold_bbox(threshold);
            }
            self.fill_state.active_fill = Some(active_fill.clone());
        }

        if dirty_bbox.is_none() && color_or_aa_changed {
            dirty_bbox = active_fill.fill_bbox;
        }

        if let Some((x0, y0, x1, y1)) = dirty_bbox {
            let pad = if anti_aliased { 1 } else { 0 };
            dirty_bbox = Some((
                x0.saturating_sub(pad),
                y0.saturating_sub(pad),
                (x1 + pad).min(width.saturating_sub(1)),
                (y1 + pad).min(height.saturating_sub(1)),
            ));
        }

        let request_id = self.fill_state.preview_request_id.wrapping_add(1);
        self.fill_state.preview_request_id = request_id;
        self.fill_state.preview_in_flight = true;
        self.fill_state.recalc_pending = false;
        let global_fill = self.fill_state.global_fill;

        let (tx, rx) = std::sync::mpsc::channel();
        self.fill_state.async_rx = Some(rx);
        crate::par_compat::spawn(move || {
            let (region_index, fill_mask, fill_bbox) =
                if let Some(region_index) = active_fill.region_index.clone() {
                    (
                        Some(region_index),
                        active_fill.fill_mask.clone(),
                        active_fill.fill_bbox,
                    )
                } else {
                    let region_index = if global_fill {
                        Self::compute_global_distance_map(
                            &flat_rgba,
                            &active_fill.target_color,
                            width,
                            height,
                            distance_mode,
                        )
                    } else {
                        Self::compute_flood_distance_map(
                            &flat_rgba,
                            (active_fill.start_x, active_fill.start_y),
                            &active_fill.target_color,
                            width,
                            height,
                            distance_mode,
                            connectivity,
                        )
                    };
                    let fill_mask = Self::build_threshold_mask(&region_index, threshold, false);
                    let fill_bbox = region_index.threshold_bbox(threshold);
                    (Some(region_index), fill_mask, fill_bbox)
                };

            let dirty_bbox = dirty_bbox.or(fill_bbox);

            let (preview_region, preview_region_w, preview_region_h) = if gpu_available {
                (Vec::new(), 0, 0)
            } else if let Some(bbox) = dirty_bbox {
                let region_w = bbox.2.saturating_sub(bbox.0) + 1;
                let region_h = bbox.3.saturating_sub(bbox.1) + 1;
                (
                    Self::build_fill_preview_region(
                        &fill_mask,
                        bbox,
                        fill_color_u8,
                        selection_mask.as_ref(),
                        width,
                        height,
                    ),
                    region_w,
                    region_h,
                )
            } else {
                (Vec::new(), 0, 0)
            };

            let _ = tx.send(FillPreviewResult {
                request_id,
                region_index,
                start_x: active_fill.start_x,
                start_y: active_fill.start_y,
                target_color: active_fill.target_color,
                threshold,
                fill_mask,
                fill_bbox,
                dirty_bbox,
                preview_region,
                preview_region_w,
                preview_region_h,
            });
        });
    }

    fn perform_magic_wand_selection(
        &mut self,
        canvas_state: &mut CanvasState,
        start_pos: (u32, u32),
        combine_mode: SelectionMode,
        scope: MagicWandScope,
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        // Bounds check
        if start_pos.0 >= canvas_state.width || start_pos.1 >= canvas_state.height {
            return;
        }

        // Get the color at the clicked position from the active layer
        let active_layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(layer) => layer,
            None => return,
        };

        let target_color = *active_layer.pixels.get_pixel(start_pos.0, start_pos.1);

        let Some(flat_rgba) = Self::cached_active_layer_rgba(
            &mut self.magic_wand_state.cached_flat_rgba,
            canvas_state,
        ) else {
            return;
        };

        let w = canvas_state.width;
        let h = canvas_state.height;
        let distance_mode = self.magic_wand_state.distance_mode;
        let connectivity = self.magic_wand_state.connectivity;
        let request_id = self.magic_wand_state.next_request_id.wrapping_add(1);
        self.magic_wand_state.next_request_id = request_id;
        self.magic_wand_state.pending_operation = Some(PendingMagicWandOperation {
            request_id,
            start_x: start_pos.0,
            start_y: start_pos.1,
            target_color,
            combine_mode,
            scope,
            distance_mode,
            connectivity,
        });
        self.magic_wand_state.async_rx = None;
        self.magic_wand_state.computing = false;

        // GPU path: compute distances synchronously on GPU
        if let Some(gpu) = gpu_renderer {
            let input_key = flat_rgba.as_ref().as_ptr() as usize;
            let mut distances = Vec::new();
            let ok = if scope == MagicWandScope::Global {
                gpu.flood_fill_pipeline.compute_global_distances(
                    &gpu.ctx,
                    flat_rgba.as_ref(),
                    input_key,
                    target_color.0,
                    w,
                    h,
                    distance_mode,
                    &mut distances,
                )
            } else {
                gpu.flood_fill_pipeline.compute_flood_distances(
                    &gpu.ctx,
                    flat_rgba.as_ref(),
                    input_key,
                    target_color.0,
                    start_pos.0,
                    start_pos.1,
                    w,
                    h,
                    distance_mode,
                    connectivity,
                    &mut distances,
                )
            };
            if ok {
                let region_index = ThresholdRegionIndex::from_distances(distances, w, h);
                self.magic_wand_state.operations.push(MagicWandOperation {
                    start_x: start_pos.0,
                    start_y: start_pos.1,
                    target_color,
                    combine_mode,
                    scope,
                    distance_mode,
                    connectivity,
                    region_index,
                });
                self.magic_wand_state.pending_operation = None;
                self.magic_wand_state.computing = false;
                self.magic_wand_state.preview_pending = true;
                self.magic_wand_state.last_applied_tolerance = -1.0;
                self.apply_magic_wand_preview(canvas_state, Some(gpu));
                return;
            }
            // GPU failed — fall through to CPU path
        }

        // CPU fallback: spawn async computation
        let (tx, rx) = std::sync::mpsc::channel();
        self.magic_wand_state.async_rx = Some(rx);
        self.magic_wand_state.computing = true;

        crate::par_compat::spawn(move || {
            let index = if scope == MagicWandScope::Global {
                Self::compute_global_distance_map(&flat_rgba, &target_color, w, h, distance_mode)
            } else {
                Self::compute_flood_distance_map(
                    &flat_rgba,
                    start_pos,
                    &target_color,
                    w,
                    h,
                    distance_mode,
                    connectivity,
                )
            };
            let _ = tx.send(MagicWandAsyncResult::Ready { request_id, index });
        });
    }

    /// Dijkstra minimax (bottleneck) distance map.
    ///
    /// `distances[y*w + x]` = minimum possible *maximum* per-step color distance
    /// along any 4-connected path from the seed to (x,y).  Thresholding at `t`
    /// selects all pixels reachable without any edge exceeding `t`, which is
    /// **monotone** — higher tolerances never remove already-selected pixels.
    ///
    /// Uses a 256-bucket queue because the bottleneck distance is bounded to 0..=255.
    fn compute_flood_distance_map(
        flat_rgba: &[u8],
        seed: (u32, u32),
        target_color: &Rgba<u8>,
        width: u32,
        height: u32,
        distance_mode: WandDistanceMode,
        connectivity: FloodConnectivity,
    ) -> ThresholdRegionIndex {
        let n = (width * height) as usize;
        let w = width as usize;

        let seed_idx = seed.1 as usize * w + seed.0 as usize;
        let seed_dist =
            Self::pixel_color_distance(flat_rgba, seed_idx, target_color, distance_mode);

        let mut distances = vec![u8::MAX; n];
        distances[seed_idx] = seed_dist;

        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 256];
        buckets[seed_dist as usize].push(seed_idx);
        let mut current_bucket = seed_dist as usize;

        while current_bucket < 256 {
            let Some(idx) = buckets[current_bucket].pop() else {
                current_bucket += 1;
                continue;
            };
            let cost = distances[idx] as usize;
            if cost != current_bucket {
                continue;
            }

            let x = (idx % w) as u32;
            let y = (idx / w) as u32;

            let neighbors_4: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            let neighbors_8: [(i32, i32); 8] = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
                (1, 1),
            ];
            let neighbors: &[(i32, i32)] = match connectivity {
                FloodConnectivity::Four => &neighbors_4,
                FloodConnectivity::Eight => &neighbors_8,
            };
            for &(dx, dy) in neighbors {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                let neighbor_dist =
                    Self::pixel_color_distance(flat_rgba, ni, target_color, distance_mode);
                let new_cost = (cost as u8).max(neighbor_dist);
                if new_cost < distances[ni] {
                    distances[ni] = new_cost;
                    buckets[new_cost as usize].push(ni);
                }
            }
        }

        ThresholdRegionIndex::from_distances(distances, width, height)
    }

    /// Global distance map: direct color distance from target for every pixel.
    /// No connectivity; monotone by construction (pure per-pixel metric).
    /// Parallelized for speed on large canvases.
    fn compute_global_distance_map(
        flat_rgba: &[u8],
        target_color: &Rgba<u8>,
        width: u32,
        height: u32,
        distance_mode: WandDistanceMode,
    ) -> ThresholdRegionIndex {
        let n = (width * height) as usize;
        let mut out = vec![0u8; n];
        let tc = *target_color;
        out.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * 4096;
                for (j, val) in chunk.iter_mut().enumerate() {
                    *val = Self::pixel_color_distance(flat_rgba, base + j, &tc, distance_mode);
                }
            });
        ThresholdRegionIndex::from_distances(out, width, height)
    }

    /// Max-component color distance between a flat RGBA pixel at `idx` and `target`.
    /// Returns a value in [0.0, 255.0].
    #[inline]
    fn pixel_color_distance(
        flat_rgba: &[u8],
        idx: usize,
        target: &Rgba<u8>,
        mode: WandDistanceMode,
    ) -> u8 {
        if mode == WandDistanceMode::Perceptual {
            return Self::perceptual_distance(flat_rgba, idx, target);
        }

        let base = idx * 4;
        let r = flat_rgba[base];
        let g = flat_rgba[base + 1];
        let b = flat_rgba[base + 2];
        let a = flat_rgba[base + 3];

        // Both transparent → zero distance
        if target.0[3] == 0 && a == 0 {
            return 0;
        }

        let dr = u8::abs_diff(r, target.0[0]);
        let dg = u8::abs_diff(g, target.0[1]);
        let db = u8::abs_diff(b, target.0[2]);
        let da = u8::abs_diff(a, target.0[3]);

        dr.max(dg).max(db).max(da)
    }

    fn handle_fill_click(
        &mut self,
        canvas_state: &mut CanvasState,
        pos: (u32, u32),
        use_secondary: bool,
        global_fill: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        self.apply_fill_click_direct(
            canvas_state,
            pos,
            use_secondary,
            global_fill,
            primary_color_f32,
            secondary_color_f32,
            gpu_renderer,
        );
    }

    fn apply_fill_click_direct(
        &mut self,
        canvas_state: &mut CanvasState,
        pos: (u32, u32),
        use_secondary: bool,
        global_fill: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        _gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        if pos.0 >= canvas_state.width || pos.1 >= canvas_state.height {
            return;
        }

        // Commit any existing fill preview before starting a new one
        self.commit_fill_preview(canvas_state);
        canvas_state.fill_commit_overlays.clear();

        let active_layer_index = canvas_state.active_layer_index;
        let Some(active_layer) = canvas_state.layers.get(active_layer_index) else {
            return;
        };

        let target_color = *active_layer.pixels.get_pixel(pos.0, pos.1);
        let flat_rgba = active_layer.pixels.to_rgba_image().into_raw();
        let width = canvas_state.width;
        let height = canvas_state.height;
        let distance_mode = WandDistanceMode::LegacyRgba;
        let connectivity = FloodConnectivity::Four;
        let threshold = Self::tolerance_threshold_u8(self.fill_state.tolerance);
        let anti_aliased = self.fill_state.anti_aliased;
        let fill_threshold = threshold;
        let selection_mask = canvas_state.selection_mask.clone();

        let fill_color = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };
        let fill_color_u8 = Rgba([
            (fill_color[0] * 255.0) as u8,
            (fill_color[1] * 255.0) as u8,
            (fill_color[2] * 255.0) as u8,
            (fill_color[3] * 255.0) as u8,
        ]);

        let region_index = if global_fill {
            Self::compute_global_distance_map(
                &flat_rgba,
                &target_color,
                width,
                height,
                distance_mode,
            )
        } else {
            Self::compute_flood_distance_map(
                &flat_rgba,
                pos,
                &target_color,
                width,
                height,
                distance_mode,
                connectivity,
            )
        };

        let fill_mask = Self::build_threshold_mask(&region_index, fill_threshold, anti_aliased);
        let Some(fill_bbox) = region_index.threshold_bbox(fill_threshold) else {
            return;
        };
        let preview_region = Self::build_fill_preview_region(
            &fill_mask,
            fill_bbox,
            fill_color_u8,
            selection_mask.as_ref(),
            width,
            height,
        );
        let region_w = fill_bbox.2.saturating_sub(fill_bbox.0) + 1;
        let region_h = fill_bbox.3.saturating_sub(fill_bbox.1) + 1;
        if preview_region.is_empty() || region_w == 0 || region_h == 0 {
            return;
        }

        let mut history_cmd = crate::components::history::SingleLayerSnapshotCommand::new_for_layer(
            "Fill".to_string(),
            canvas_state,
            active_layer_index,
        );
        let blend_mode = self.properties.blending_mode;
        let mut changed = false;

        if let Some(active_layer) = canvas_state.layers.get_mut(active_layer_index) {
            for row in 0..region_h as usize {
                let gy = fill_bbox.1 + row as u32;
                let src_row = row * region_w as usize * 4;
                for col in 0..region_w as usize {
                    let gx = fill_bbox.0 + col as u32;
                    let src_off = src_row + col * 4;
                    let src = Rgba([
                        preview_region[src_off],
                        preview_region[src_off + 1],
                        preview_region[src_off + 2],
                        preview_region[src_off + 3],
                    ]);
                    if src[3] == 0 {
                        continue;
                    }
                    let dst = *active_layer.pixels.get_pixel(gx, gy);
                    let blended = CanvasState::blend_pixel_static(dst, src, blend_mode, 1.0);
                    if blended != dst {
                        active_layer.pixels.put_pixel(gx, gy, blended);
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            return;
        }

        let dirty_rect = egui::Rect::from_min_max(
            egui::pos2(fill_bbox.0 as f32, fill_bbox.1 as f32),
            egui::pos2((fill_bbox.2 + 1) as f32, (fill_bbox.3 + 1) as f32),
        );
        canvas_state.mark_dirty(Some(dirty_rect));
        history_cmd.set_after(canvas_state);
        self.pending_history_commands.push(Box::new(history_cmd));
        self.pending_stroke_event = None;
    }

    /// Fill tool - flood fill with preview
    fn perform_flood_fill(
        &mut self,
        canvas_state: &mut CanvasState,
        start_pos: (u32, u32),
        use_secondary: bool,
        global_fill: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        gpu_renderer: Option<&mut crate::gpu::GpuRenderer>,
    ) {
        // Bounds check
        if start_pos.0 >= canvas_state.width || start_pos.1 >= canvas_state.height {
            return;
        }

        // Get the active layer to determine target color
        let active_layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(layer) => layer,
            None => return,
        };

        let target_color = *active_layer.pixels.get_pixel(start_pos.0, start_pos.1);

        // Determine fill color from primary or secondary
        let fill_color = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };
        let fill_color_u8 = Rgba([
            (fill_color[0] * 255.0) as u8,
            (fill_color[1] * 255.0) as u8,
            (fill_color[2] * 255.0) as u8,
            (fill_color[3] * 255.0) as u8,
        ]);

        // Fill tool uses simple max-component distance with 4-connected flood.
        // Perceptual distance (sRGB-to-linear + luminance/chroma weighting) can
        // produce non-zero distances for visually identical pixels, causing
        // 1-pixel gaps at fill boundaries. LegacyRgba is more predictable for
        // pixel art.
        self.fill_state.distance_mode = WandDistanceMode::LegacyRgba;
        self.fill_state.connectivity = FloodConnectivity::Four;

        self.fill_state.active_fill = Some(ActiveFillRegion {
            start_x: start_pos.0,
            start_y: start_pos.1,
            layer_idx: canvas_state.active_layer_index,
            target_color,
            region_index: None,
            fill_mask: Vec::new(),
            fill_bbox: None,
            last_threshold: None,
        });
        self.fill_state.last_preview_tolerance = -1.0;
        self.fill_state.last_preview_aa = !self.fill_state.anti_aliased;
        self.fill_state.fill_color_u8 = Some(fill_color_u8);
        self.fill_state.use_secondary_color = use_secondary;
        self.fill_state.global_fill = global_fill;
        self.fill_state.tolerance_changed_at = None;
        self.fill_state.recalc_pending = true;
        self.fill_state.preview_in_flight = false;

        // Keep existing preview visible until the first patch of the new seed
        // arrives to avoid an empty-frame flash between clicks.
        self.fill_state.defer_preview_clear = canvas_state.preview_layer.is_some();

        // Start stroke tracking for undo/redo
        if let Some(_layer) = canvas_state.layers.get(canvas_state.active_layer_index) {
            self.stroke_tracker
                .start_preview_tool(canvas_state.active_layer_index, "Fill");
        }

        canvas_state.preview_blend_mode = BlendMode::Normal;
        canvas_state.preview_force_composite = fill_color_u8.0[3] < 255;

        // A connected GPU fill needs up to width+height full-frame relaxation
        // passes plus blocking convergence readbacks. On 4K canvases that can
        // monopolize the UI thread for seconds. Keep the single-pass global
        // distance map on GPU, but run connected minimax flood asynchronously.
        if global_fill && let Some(gpu) = gpu_renderer {
            let flat_rgba =
                Self::cached_active_layer_rgba(&mut self.fill_state.cached_flat_rgba, canvas_state);
            if let Some(flat_rgba) = flat_rgba {
                let input_key = flat_rgba.as_ref().as_ptr() as usize;
                let mut distances = Vec::new();
                let ok = gpu.flood_fill_pipeline.compute_global_distances(
                    &gpu.ctx,
                    flat_rgba.as_ref(),
                    input_key,
                    target_color.0,
                    canvas_state.width,
                    canvas_state.height,
                    self.fill_state.distance_mode,
                    &mut distances,
                );
                if ok {
                    let region_index = ThresholdRegionIndex::from_distances(
                        distances,
                        canvas_state.width,
                        canvas_state.height,
                    );
                    if let Some(active_fill) = self.fill_state.active_fill.as_mut() {
                        active_fill.region_index = Some(region_index);
                    }
                    self.fill_state.recalc_pending = false;
                    self.render_fill_preview_gpu(canvas_state, gpu);
                    return;
                }
            }
        }

        // Connected fill (and GPU fallback): async bucketed minimax flood.
        self.maybe_spawn_fill_preview(canvas_state, None);
    }

    /// Commit the fill preview to the actual layer
    fn commit_fill_preview(&mut self, canvas_state: &mut CanvasState) {
        self.commit_fill_preview_impl(canvas_state, true);
    }

    /// Commit fill preview to the active layer.
    ///
    /// When `clear_preview_overlay` is false, the current preview remains
    /// visible until a subsequent preview patch replaces it.
    fn commit_fill_preview_impl(
        &mut self,
        canvas_state: &mut CanvasState,
        clear_preview_overlay: bool,
    ) {
        let Some(active_fill) = self.fill_state.active_fill.as_ref() else {
            return;
        };

        let target_layer_idx = active_fill.layer_idx;
        if target_layer_idx >= canvas_state.layers.len() {
            self.reset_fill_preview_state(clear_preview_overlay);
            if clear_preview_overlay {
                canvas_state.clear_preview_state();
            }
            return;
        }

        let dirty_rect = canvas_state.preview_stroke_bounds;
        let overlay_bounds = active_fill.fill_bbox.or_else(|| {
            dirty_rect.map(|bounds| {
                let x0 = bounds.min.x.floor().max(0.0) as u32;
                let y0 = bounds.min.y.floor().max(0.0) as u32;
                let x1 = bounds.max.x.ceil().max(0.0) as u32;
                let y1 = bounds.max.y.ceil().max(0.0) as u32;
                (
                    x0,
                    y0,
                    x1.saturating_sub(1),
                    y1.saturating_sub(1),
                )
            })
        });
        let commit_overlay =
            if let (Some(preview), Some((x0, y0, x1, y1))) =
                (canvas_state.preview_layer.as_ref(), overlay_bounds)
        {
            let width = x1.saturating_sub(x0) + 1;
            let height = y1.saturating_sub(y0) + 1;
            if width > 0 && height > 0 {
                let mut region = Vec::new();
                preview.extract_region_rgba_fast(x0, y0, width, height, &mut region);
                Some(((x0, y0, x1, y1), region))
            } else {
                None
            }
        } else {
            None
        };
        self.reset_fill_preview_state(clear_preview_overlay);

        let blend_mode = self.properties.blending_mode;

        // IMPORTANT: Capture "before" snapshot BEFORE modifying the layer
        // For preview-based tools, the layer is still unmodified at this point
        let stroke_event = self.stroke_tracker.finish(canvas_state);

        // Commit preview to actual layer with proper blend mode
        if let Some(ref preview) = canvas_state.preview_layer {
            // Collect populated chunk data before mutating the layer
            let chunk_data: Vec<(u32, u32, image::RgbaImage)> = preview
                .chunk_keys()
                .filter_map(|(cx, cy)| {
                    preview
                        .get_chunk(cx, cy)
                        .map(|chunk| (cx, cy, chunk.clone()))
                })
                .collect();

            let chunk_size = crate::canvas::CHUNK_SIZE;

            if let Some(active_layer) = canvas_state.layers.get_mut(target_layer_idx) {
                for (cx, cy, chunk) in &chunk_data {
                    let base_x = cx * chunk_size;
                    let base_y = cy * chunk_size;
                    let cw = chunk_size.min(canvas_state.width.saturating_sub(base_x));
                    let ch = chunk_size.min(canvas_state.height.saturating_sub(base_y));
                    for ly in 0..ch {
                        for lx in 0..cw {
                            let gx = base_x + lx;
                            let gy = base_y + ly;
                            let src = *chunk.get_pixel(lx, ly);
                            if src.0[3] == 0 {
                                continue;
                            }
                            let dst = active_layer.pixels.get_pixel_mut(gx, gy);
                            *dst = CanvasState::blend_pixel_static(*dst, src, blend_mode, 1.0);
                        }
                    }
                }
            }
        }

        if let Some((overlay_bounds, overlay_rgba)) = commit_overlay {
            self.fill_state.cached_flat_rgba = None;
            canvas_state.push_fill_commit_overlay(
                overlay_bounds,
                overlay_rgba,
                std::time::Duration::from_millis(100),
            );
        }

        // Mark dirty and clear preview (optional for reseed click path)
        if clear_preview_overlay {
            self.fill_state.preview_clear_at = Some(
                crate::time_compat::Instant::now() + std::time::Duration::from_millis(200),
            );
        }
        canvas_state.mark_dirty(dirty_rect);

        // Store event for history
        if stroke_event.is_some() {
            self.pending_stroke_event = stroke_event;
        }
    }

    /// Color picker - sample color from canvas
    fn pick_color_at_position(
        &mut self,
        canvas_state: &mut CanvasState,
        pos: (u32, u32),
        use_secondary: bool,
    ) {
        // Bounds check
        if pos.0 >= canvas_state.width || pos.1 >= canvas_state.height {
            return;
        }

        // Get the color from the active layer
        let active_layer = match canvas_state.layers.get(canvas_state.active_layer_index) {
            Some(layer) => layer,
            None => return,
        };

        let pixel = active_layer.pixels.get_pixel(pos.0, pos.1);
        let color_32 =
            Color32::from_rgba_unmultiplied(pixel.0[0], pixel.0[1], pixel.0[2], pixel.0[3]);

        // Store the picked color
        self.last_picked_color = Some((color_32, use_secondary));

        // Keep the active primary tool color unchanged when only the secondary swatch is sampled.
        if !use_secondary {
            self.properties.color = color_32;
        }

        // Note: The app.rs code will synchronize this to the ColorsPanel
        // We can't directly modify ColorsPanel from here due to borrow constraints
    }

    fn clip_preview_bounds(
        canvas_state: &CanvasState,
        off_x: i32,
        off_y: i32,
        buf_w: u32,
        buf_h: u32,
    ) -> Option<egui::Rect> {
        let x0 = off_x.max(0) as f32;
        let y0 = off_y.max(0) as f32;
        let x1 = (off_x + buf_w as i32).min(canvas_state.width as i32).max(0) as f32;
        let y1 = (off_y + buf_h as i32)
            .min(canvas_state.height as i32)
            .max(0) as f32;
        if x1 > x0 && y1 > y0 {
            Some(egui::Rect::from_min_max(
                egui::pos2(x0, y0),
                egui::pos2(x1, y1),
            ))
        } else {
            None
        }
    }

    /// Fast flood fill using a DFS Vec-stack on a pre-extracted flat RGBA buffer.
    /// Returns (flat_mask, bbox) where flat_mask is width*height bytes (255=filled)
    /// and bbox is (min_x, min_y, max_x, max_y).  Returns None bbox when nothing filled.
    fn flood_fill_fast(
        flat_pixels: &[u8],
        start_x: u32,
        start_y: u32,
        target_color: &Rgba<u8>,
        tolerance: f32,
        canvas_w: u32,
        canvas_h: u32,
    ) -> (Vec<u8>, Option<(u32, u32, u32, u32)>) {
        let wu = canvas_w as usize;
        // mask doubles as the visited array and the output
        let mut mask = vec![0u8; wu * canvas_h as usize];

        if start_x >= canvas_w || start_y >= canvas_h {
            return (mask, None);
        }

        let tc = target_color.0;
        let tol = tolerance;

        // Inline pixel fetch from flat RGBA buffer
        #[inline(always)]
        fn pix(flat: &[u8], idx: usize) -> [u8; 4] {
            let o = idx * 4;
            [flat[o], flat[o + 1], flat[o + 2], flat[o + 3]]
        }

        // Inline tight color match (same logic as colors_match but on raw arrays)
        #[inline(always)]
        fn matches(p: [u8; 4], tc: [u8; 4], tol: f32) -> bool {
            if tc[3] == 0 && p[3] == 0 {
                return true;
            }
            if tc[3] == 0 || p[3] == 0 {
                return (tc[3] as f32 - p[3] as f32).abs() <= tol;
            }
            let r = (tc[0] as f32 - p[0] as f32).abs();
            let g = (tc[1] as f32 - p[1] as f32).abs();
            let b = (tc[2] as f32 - p[2] as f32).abs();
            let a = (tc[3] as f32 - p[3] as f32).abs();
            r.max(g).max(b).max(a) <= tol
        }

        // Seed check
        let seed_idx = start_y as usize * wu + start_x as usize;
        if !matches(pix(flat_pixels, seed_idx), tc, tol) {
            return (mask, None);
        }

        // Bounding box
        let mut min_x = start_x;
        let mut min_y = start_y;
        let mut max_x = start_x;
        let mut max_y = start_y;

        // Scanline flood fill touches contiguous spans instead of individual pixels.
        let mut stack: Vec<(u32, u32)> = Vec::with_capacity(1024);
        stack.push((start_x, start_y));

        while let Some((seed_x, y)) = stack.pop() {
            let row = y as usize * wu;
            let seed_flat = row + seed_x as usize;
            if mask[seed_flat] != 0 || !matches(pix(flat_pixels, seed_flat), tc, tol) {
                continue;
            }

            let mut lx = seed_x;
            while lx > 0 {
                let ni = row + lx as usize - 1;
                if mask[ni] != 0 || !matches(pix(flat_pixels, ni), tc, tol) {
                    break;
                }
                lx -= 1;
            }

            let mut rx = seed_x;
            while rx + 1 < canvas_w {
                let ni = row + rx as usize + 1;
                if mask[ni] != 0 || !matches(pix(flat_pixels, ni), tc, tol) {
                    break;
                }
                rx += 1;
            }

            for x in lx..=rx {
                mask[row + x as usize] = 255;
            }

            min_x = min_x.min(lx);
            max_x = max_x.max(rx);
            min_y = min_y.min(y);
            max_y = max_y.max(y);

            for ny in [y.checked_sub(1), (y + 1 < canvas_h).then_some(y + 1)]
                .into_iter()
                .flatten()
            {
                let nrow = ny as usize * wu;
                let mut x = lx;
                while x <= rx {
                    let ni = nrow + x as usize;
                    if mask[ni] == 0 && matches(pix(flat_pixels, ni), tc, tol) {
                        stack.push((x, ny));
                        x += 1;
                        while x <= rx {
                            let seg_idx = nrow + x as usize;
                            if mask[seg_idx] != 0 || !matches(pix(flat_pixels, seg_idx), tc, tol) {
                                break;
                            }
                            x += 1;
                        }
                    } else {
                        x += 1;
                    }
                }
            }
        }

        let bbox = if max_x >= min_x {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        };
        (mask, bbox)
    }

    // ================================================================
    // Lasso: scanline polygon rasterization into selection mask
    // ================================================================
}

