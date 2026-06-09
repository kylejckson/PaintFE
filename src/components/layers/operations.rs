impl LayersPanel {
    fn folder_insert_index(&self, canvas_state: &CanvasState, folder_id: u64) -> usize {
        canvas_state
            .layers
            .iter()
            .rposition(|layer| layer.folder_id == Some(folder_id))
            .map(|idx| idx + 1)
            .or_else(|| {
                canvas_state
                    .layer_folder(folder_id)
                    .and_then(|folder| folder.insert_above_layer.map(|idx| idx + 1))
            })
            .unwrap_or_else(|| (canvas_state.active_layer_index + 1).min(canvas_state.layers.len()))
            .min(canvas_state.layers.len())
    }

    fn active_target_folder(&self, canvas_state: &CanvasState) -> Option<u64> {
        self.selected_folder
            .filter(|id| canvas_state.layer_folder(*id).is_some())
    }

    fn add_new_layer(&mut self, canvas_state: &mut CanvasState, history: &mut HistoryManager) {
        let layer_num = canvas_state.layers.len() + 1;
        let layer_name = format!("Layer {}", layer_num);
        let transparent = Rgba([0, 0, 0, 0]);
        let mut new_layer = Layer::new(
            layer_name.clone(),
            canvas_state.width,
            canvas_state.height,
            transparent,
        );
        new_layer.folder_id = self.active_target_folder(canvas_state).or_else(|| {
            canvas_state
                .layers
                .get(canvas_state.active_layer_index)
                .and_then(|layer| layer.folder_id)
        });

        // Insert above current active layer
        let insert_idx = new_layer
            .folder_id
            .map(|id| self.folder_insert_index(canvas_state, id))
            .unwrap_or_else(|| (canvas_state.active_layer_index + 1).min(canvas_state.layers.len()));
        canvas_state.layers.insert(insert_idx, new_layer);
        canvas_state.active_layer_index = insert_idx;
        if let Some(id) = canvas_state.layers[insert_idx].folder_id
            && let Some(folder) = canvas_state.layer_folder_mut(id)
        {
            folder.insert_above_layer = None;
        }

        // Record history
        history.push(Box::new(LayerOpCommand::new(LayerOperation::Add {
            index: insert_idx,
            name: layer_name,
            width: canvas_state.width,
            height: canvas_state.height,
            folder_id: canvas_state.layers[insert_idx].folder_id,
        })));

        self.thumbnail_cache.clear();
        self.mark_full_dirty(canvas_state);
    }

    fn add_new_text_layer(&mut self, canvas_state: &mut CanvasState, history: &mut HistoryManager) {
        let layer_num = canvas_state.layers.len() + 1;
        let layer_name = format!("Text Layer {}", layer_num);
        let mut new_layer =
            Layer::new_text(layer_name.clone(), canvas_state.width, canvas_state.height);
        new_layer.folder_id = self.active_target_folder(canvas_state).or_else(|| {
            canvas_state
                .layers
                .get(canvas_state.active_layer_index)
                .and_then(|layer| layer.folder_id)
        });

        let insert_idx = new_layer
            .folder_id
            .map(|id| self.folder_insert_index(canvas_state, id))
            .unwrap_or_else(|| (canvas_state.active_layer_index + 1).min(canvas_state.layers.len()));
        canvas_state.layers.insert(insert_idx, new_layer);
        canvas_state.active_layer_index = insert_idx;
        if let Some(id) = canvas_state.layers[insert_idx].folder_id
            && let Some(folder) = canvas_state.layer_folder_mut(id)
        {
            folder.insert_above_layer = None;
        }

        history.push(Box::new(LayerOpCommand::new(LayerOperation::Add {
            index: insert_idx,
            name: layer_name,
            width: canvas_state.width,
            height: canvas_state.height,
            folder_id: canvas_state.layers[insert_idx].folder_id,
        })));

        self.thumbnail_cache.clear();
        self.mark_full_dirty(canvas_state);
    }

    fn add_adjustment_layer(
        &mut self,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        let mut snap = SnapshotCommand::new("Add Adjustment Layer".to_string(), canvas_state);
        let layer_num = canvas_state.layers.len() + 1;
        let layer_name = format!("Adjustment Layer {}", layer_num);
        let mut new_layer = Layer::new_adjustment(
            layer_name,
            canvas_state.width,
            canvas_state.height,
            crate::canvas::AdjustmentKind::Exposure { ev: 0.5 },
        );
        new_layer.folder_id = canvas_state.layers[canvas_state.active_layer_index].folder_id;
        let insert_idx = canvas_state.active_layer_index + 1;
        canvas_state.layers.insert(insert_idx, new_layer);
        canvas_state.active_layer_index = insert_idx;
        snap.set_after(canvas_state);
        history.push(Box::new(snap));

        self.thumbnail_cache.clear();
        self.mark_full_dirty(canvas_state);
    }

    fn add_layer_folder(
        &mut self,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        let mut snap = SnapshotCommand::new("Add Layer Folder".to_string(), canvas_state);
        let id = canvas_state.next_layer_folder_id.max(1);
        canvas_state.next_layer_folder_id = id + 1;
        let name = format!("Folder {}", canvas_state.layer_folders.len() + 1);
        let insert_above_layer = canvas_state
            .layers
            .get(canvas_state.active_layer_index)
            .and_then(|layer| layer.folder_id)
            .and_then(|id| {
                canvas_state
                    .layers
                    .iter()
                    .rposition(|layer| layer.folder_id == Some(id))
            })
            .or(Some(canvas_state.active_layer_index));
        canvas_state.layer_folders.push(crate::canvas::LayerFolder {
            id,
            name,
            visible: true,
            collapsed: false,
            insert_above_layer,
            color_index: None,
        });
        snap.set_after(canvas_state);
        history.push(Box::new(snap));
    }

    fn folder_top_insert_index(&self, canvas_state: &CanvasState, folder_id: u64) -> usize {
        canvas_state
            .layers
            .iter()
            .rposition(|layer| layer.folder_id == Some(folder_id))
            .map(|idx| idx + 1)
            .or_else(|| {
                canvas_state
                    .layer_folder(folder_id)
                    .and_then(|folder| folder.insert_above_layer.map(|idx| idx + 1))
            })
            .unwrap_or(canvas_state.layers.len())
            .min(canvas_state.layers.len())
    }

    fn move_folder_block(
        &mut self,
        folder_id: u64,
        insert_before_idx: usize,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if canvas_state.layer_folder(folder_id).is_none() {
            return;
        }
        let member_indices: Vec<usize> = canvas_state
            .layers
            .iter()
            .enumerate()
            .filter_map(|(idx, layer)| (layer.folder_id == Some(folder_id)).then_some(idx))
            .collect();
        let mut snap = SnapshotCommand::new("Move Folder".to_string(), canvas_state);
        if member_indices.is_empty() {
            let adjusted = insert_before_idx.saturating_sub(1).min(canvas_state.layers.len());
            if let Some(folder) = canvas_state.layer_folder_mut(folder_id) {
                folder.insert_above_layer = Some(adjusted);
            }
        } else {
            let removed_before = member_indices
                .iter()
                .filter(|idx| **idx < insert_before_idx)
                .count();
            let mut moved = Vec::with_capacity(member_indices.len());
            for idx in member_indices.iter().rev() {
                moved.push(canvas_state.layers.remove(*idx));
            }
            moved.reverse();
            let insert_idx = insert_before_idx
                .saturating_sub(removed_before)
                .min(canvas_state.layers.len());
            for (offset, layer) in moved.into_iter().enumerate() {
                canvas_state.layers.insert(insert_idx + offset, layer);
            }
            if let Some(folder) = canvas_state.layer_folder_mut(folder_id) {
                folder.insert_above_layer = None;
            }
        }
        snap.set_after(canvas_state);
        history.push(Box::new(snap));
        self.thumbnail_cache.clear();
        self.pending_gpu_clear = true;
        self.mark_full_dirty(canvas_state);
    }

    fn add_layer_folder_above(
        &mut self,
        anchor_folder_id: u64,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        let Some(anchor) = canvas_state.layer_folder(anchor_folder_id) else {
            return;
        };
        let insert_above_layer = anchor.insert_above_layer;
        let mut snap = SnapshotCommand::new("Add Layer Folder".to_string(), canvas_state);
        let id = canvas_state.next_layer_folder_id.max(1);
        canvas_state.next_layer_folder_id = id + 1;
        let name = format!("Folder {}", canvas_state.layer_folders.len() + 1);
        canvas_state.layer_folders.push(crate::canvas::LayerFolder {
            id,
            name,
            visible: true,
            collapsed: false,
            insert_above_layer,
            color_index: None,
        });
        snap.set_after(canvas_state);
        history.push(Box::new(snap));
    }

    fn set_layer_folder(
        &mut self,
        layer_idx: usize,
        folder_id: Option<u64>,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if layer_idx >= canvas_state.layers.len()
            || canvas_state.layers[layer_idx].folder_id == folder_id
        {
            return;
        }
        if let Some(id) = folder_id
            && canvas_state.layer_folder(id).is_none()
        {
            return;
        }
        let mut snap = SnapshotCommand::new("Move Layer to Folder".to_string(), canvas_state);
        canvas_state.layers[layer_idx].folder_id = folder_id;
        if let Some(id) = folder_id
            && let Some(folder) = canvas_state.layer_folder_mut(id)
        {
            folder.insert_above_layer = None;
        }
        snap.set_after(canvas_state);
        history.push(Box::new(snap));
        self.mark_full_dirty(canvas_state);
    }

    fn rename_layer_folder(
        &mut self,
        folder_id: u64,
        new_name: String,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if new_name.trim().is_empty() {
            return;
        }
        let Some(folder) = canvas_state.layer_folder(folder_id) else {
            return;
        };
        if folder.name == new_name {
            return;
        }
        let mut snap = SnapshotCommand::new("Rename Layer Folder".to_string(), canvas_state);
        if let Some(folder) = canvas_state.layer_folder_mut(folder_id) {
            folder.name = new_name;
        }
        snap.set_after(canvas_state);
        history.push(Box::new(snap));
    }

    fn toggle_layer_folder_visibility(
        &mut self,
        folder_id: u64,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        let mut snap = SnapshotCommand::new("Toggle Layer Folder Visibility".to_string(), canvas_state);
        if let Some(folder) = canvas_state.layer_folder_mut(folder_id) {
            folder.visible = !folder.visible;
        }
        snap.set_after(canvas_state);
        history.push(Box::new(snap));
        self.mark_full_dirty(canvas_state);
    }

    fn set_layer_folder_collapsed(&mut self, folder_id: u64, collapsed: bool, canvas_state: &mut CanvasState) {
        if let Some(folder) = canvas_state.layer_folder_mut(folder_id) {
            folder.collapsed = collapsed;
        }
    }

    fn delete_layer_folder(
        &mut self,
        folder_id: u64,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if canvas_state.layer_folder(folder_id).is_none() {
            return;
        }
        let mut snap = SnapshotCommand::new("Delete Layer Folder".to_string(), canvas_state);
        canvas_state.layer_folders.retain(|folder| folder.id != folder_id);
        for layer in &mut canvas_state.layers {
            if layer.folder_id == Some(folder_id) {
                layer.folder_id = None;
            }
        }
        snap.set_after(canvas_state);
        history.push(Box::new(snap));
        self.mark_full_dirty(canvas_state);
    }

    /// Rasterize a text layer — converts it to a regular raster layer,
    /// losing editability but preserving the current pixel appearance.
    fn rasterize_text_layer(
        &mut self,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if layer_idx >= canvas_state.layers.len() {
            return;
        }
        if !matches!(
            canvas_state.layers[layer_idx].content,
            LayerContent::Text(_)
        ) {
            return;
        }

        // Snapshot before
        let mut snap = crate::components::history::SingleLayerSnapshotCommand::new_for_layer(
            t!("layer.rasterize_text_layer"),
            canvas_state,
            layer_idx,
        );

        // Ensure rasterized pixels are up to date
        canvas_state.ensure_all_text_layers_rasterized();

        // Convert to raster by simply changing content to Raster.
        // The pixels are already up-to-date from the rasterize call above.
        canvas_state.layers[layer_idx].content = LayerContent::Raster;

        // If this was the layer being tracked for text editing, clear the
        // canvas-level marker so ensure_text_layers_rasterized no longer
        // skips it and so the tools panel state can be cleaned up cleanly.
        if canvas_state.text_editing_layer == Some(layer_idx) {
            canvas_state.text_editing_layer = None;
            canvas_state.clear_preview_state();
        }

        snap.set_after(canvas_state);
        history.push(Box::new(snap));

        self.thumbnail_cache.clear();
        self.mark_full_dirty(canvas_state);
    }

    /// Public entry point for rasterizing a text layer from app-level code
    /// (e.g. via `LayerAppAction::RasterizeTextLayer`).
    pub fn rasterize_text_layer_from_app(
        &mut self,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        self.rasterize_text_layer(layer_idx, canvas_state, history);
    }

    /// Public entry point for deleting the current active layer from app-level code.
    pub fn delete_active_layer_from_app(
        &mut self,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        self.delete_active_layer(canvas_state, history);
    }

    fn delete_active_layer(
        &mut self,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        self.delete_layer(canvas_state.active_layer_index, canvas_state, history);
    }

    fn delete_layer(
        &mut self,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if canvas_state.layers.len() <= 1 {
            return;
        }

        // Capture layer data before deletion for undo
        let layer = &canvas_state.layers[layer_idx];
        let pixels = layer.pixels.clone();
        let mask = layer.mask.clone();
        let mask_enabled = layer.mask_enabled;
        let name = layer.name.clone();
        let visible = layer.visible;
        let folder_id = layer.folder_id;
        let opacity = layer.opacity;
        let content = layer.content.clone();
        let pixel_format = layer.pixel_format;
        let hdr_metadata = layer.hdr_metadata.clone();
        let source_metadata = layer.source_metadata.clone();
        let deep_pixels = layer.deep_pixels.clone();
        let clear_selection =
            canvas_state.active_layer_index == layer_idx && canvas_state.selection_mask.is_some();
        let snapshot_cmd = clear_selection
            .then(|| SnapshotCommand::new(format!("Delete Layer: {}", name), canvas_state));

        canvas_state.layers.remove(layer_idx);

        if canvas_state.text_editing_layer == Some(layer_idx) {
            canvas_state.text_editing_layer = None;
            canvas_state.clear_preview_state();
        } else if let Some(text_idx) = canvas_state.text_editing_layer
            && layer_idx < text_idx
        {
            canvas_state.text_editing_layer = Some(text_idx - 1);
        }

        if canvas_state.active_layer_index >= canvas_state.layers.len() {
            canvas_state.active_layer_index = canvas_state.layers.len() - 1;
        } else if canvas_state.active_layer_index > layer_idx {
            canvas_state.active_layer_index -= 1;
        }

        if clear_selection {
            canvas_state.clear_selection();
        }

        // Notify the deletion index so the UI can clean up GPU textures.
        self.pending_gpu_delete = Some(layer_idx);
        self.pending_deleted_layer = Some(layer_idx);

        // Record history
        if let Some(mut cmd) = snapshot_cmd {
            cmd.set_after(canvas_state);
            history.push(Box::new(cmd));
        } else {
            history.push(Box::new(LayerOpCommand::new(LayerOperation::Delete {
                index: layer_idx,
                pixels,
                mask,
                mask_enabled,
                name,
                visible,
                folder_id,
                opacity,
                content,
                pixel_format,
                hdr_metadata,
                source_metadata,
                deep_pixels,
            })));
        }

        self.thumbnail_cache.clear();
        self.mark_full_dirty(canvas_state);
    }

    fn duplicate_layer(
        &mut self,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if layer_idx >= canvas_state.layers.len() {
            return;
        }

        let source = &canvas_state.layers[layer_idx];
        let new_name = format!("{} copy", source.name);
        let mut new_layer = Layer::new(
            new_name.clone(),
            canvas_state.width,
            canvas_state.height,
            Rgba([0, 0, 0, 0]),
        );
        new_layer.pixels = source.pixels.clone();
        new_layer.visible = source.visible;
        new_layer.folder_id = source.folder_id;
        new_layer.opacity = source.opacity;
        new_layer.blend_mode = source.blend_mode;
        new_layer.content = source.content.clone();
        new_layer.mask = source.mask.clone();
        new_layer.mask_enabled = source.mask_enabled;
        new_layer.pixel_format = source.pixel_format;
        new_layer.hdr_metadata = source.hdr_metadata.clone();
        new_layer.source_metadata = source.source_metadata.clone();
        new_layer.deep_pixels = source.deep_pixels.clone();

        let new_index = layer_idx + 1;

        // Capture data for history before inserting
        let pixels = new_layer.pixels.clone();
        let mask = new_layer.mask.clone();
        let mask_enabled = new_layer.mask_enabled;
        let visible = new_layer.visible;
        let folder_id = new_layer.folder_id;
        let opacity = new_layer.opacity;
        let content = new_layer.content.clone();
        let pixel_format = new_layer.pixel_format;
        let hdr_metadata = new_layer.hdr_metadata.clone();
        let source_metadata = new_layer.source_metadata.clone();
        let deep_pixels = new_layer.deep_pixels.clone();

        // Insert above the duplicated layer
        canvas_state.layers.insert(new_index, new_layer);
        canvas_state.active_layer_index = new_index;

        // Record history
        history.push(Box::new(LayerOpCommand::new(LayerOperation::Duplicate {
            source_index: layer_idx,
            new_index,
            pixels,
            mask,
            mask_enabled,
            name: new_name,
            visible,
            folder_id,
            opacity,
            content,
            pixel_format,
            hdr_metadata,
            source_metadata,
            deep_pixels,
        })));

        self.thumbnail_cache.clear();
        self.mark_full_dirty(canvas_state);
    }

    fn move_layer(
        &mut self,
        from_idx: usize,
        to_idx: usize,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if from_idx == to_idx
            || from_idx >= canvas_state.layers.len()
            || to_idx >= canvas_state.layers.len()
        {
            return;
        }

        let layer = canvas_state.layers.remove(from_idx);
        canvas_state.layers.insert(to_idx, layer);

        // Update active index
        if canvas_state.active_layer_index == from_idx {
            canvas_state.active_layer_index = to_idx;
        } else if from_idx < canvas_state.active_layer_index
            && to_idx >= canvas_state.active_layer_index
        {
            canvas_state.active_layer_index -= 1;
        } else if from_idx > canvas_state.active_layer_index
            && to_idx <= canvas_state.active_layer_index
        {
            canvas_state.active_layer_index += 1;
        }

        // Record history
        history.push(Box::new(LayerOpCommand::new(LayerOperation::Move {
            from_index: from_idx,
            to_index: to_idx,
        })));

        self.thumbnail_cache.clear();
        self.pending_gpu_clear = true;
        self.mark_full_dirty(canvas_state);
    }

    fn move_layer_group(
        &mut self,
        mut indices: Vec<usize>,
        insert_before_idx: usize,
        folder_id: Option<u64>,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) -> Option<Vec<usize>> {
        indices.sort_unstable();
        indices.dedup();
        indices.retain(|idx| *idx < canvas_state.layers.len());
        if indices.is_empty() {
            return None;
        }
        if let Some(id) = folder_id
            && canvas_state.layer_folder(id).is_none()
        {
            return None;
        }

        let active_rank = indices
            .iter()
            .position(|idx| *idx == canvas_state.active_layer_index)
            .unwrap_or(0);
        let mut snap = SnapshotCommand::new("Move Layers".to_string(), canvas_state);
        let removed_before = indices
            .iter()
            .filter(|idx| **idx < insert_before_idx)
            .count();
        let mut moved = Vec::with_capacity(indices.len());
        for idx in indices.iter().rev() {
            moved.push(canvas_state.layers.remove(*idx));
        }
        moved.reverse();
        let insert_idx = insert_before_idx
            .saturating_sub(removed_before)
            .min(canvas_state.layers.len());
        for (offset, mut layer) in moved.into_iter().enumerate() {
            layer.folder_id = folder_id;
            canvas_state.layers.insert(insert_idx + offset, layer);
        }
        canvas_state.active_layer_index = (insert_idx + active_rank).min(canvas_state.layers.len() - 1);
        if let Some(id) = folder_id
            && let Some(folder) = canvas_state.layer_folder_mut(id)
        {
            folder.insert_above_layer = None;
        }
        snap.set_after(canvas_state);
        history.push(Box::new(snap));
        self.thumbnail_cache.clear();
        self.pending_gpu_clear = true;
        self.mark_full_dirty(canvas_state);
        Some((insert_idx..insert_idx + indices.len()).collect())
    }

    /// Start peeking at a layer (hide all others temporarily)
    fn start_peek(&mut self, layer_idx: usize, canvas_state: &mut CanvasState) {
        if !self.peek_state.is_peeking {
            self.peek_state.saved_visibility =
                canvas_state.layers.iter().map(|l| l.visible).collect();
            self.peek_state.is_peeking = true;
            self.peek_state.peek_layer_index = Some(layer_idx);

            for (i, layer) in canvas_state.layers.iter_mut().enumerate() {
                layer.visible = i == layer_idx;
            }
            self.mark_full_dirty(canvas_state);
        } else if self.peek_state.peek_layer_index != Some(layer_idx) {
            self.peek_state.peek_layer_index = Some(layer_idx);
            for (i, layer) in canvas_state.layers.iter_mut().enumerate() {
                layer.visible = i == layer_idx;
            }
            self.mark_full_dirty(canvas_state);
        }
    }

    fn update_peek_state(&mut self, ui: &egui::Ui, canvas_state: &mut CanvasState) {
        // Clear the one-frame suppression flag from the previous frame.
        self.peek_state.peek_just_ended = false;

        if self.peek_state.is_peeking {
            let any_button_held = ui.input(|i| i.pointer.any_down());
            if !any_button_held {
                // Restore visibility — if soloed, restore to solo state instead of saved
                if self.peek_state.is_soloed {
                    // Restore to solo state (only the soloed layer visible)
                    let solo_idx = self.peek_state.solo_layer_index;
                    for (i, layer) in canvas_state.layers.iter_mut().enumerate() {
                        layer.visible = solo_idx == Some(i);
                    }
                } else {
                    for (i, &was_visible) in self.peek_state.saved_visibility.iter().enumerate() {
                        if i < canvas_state.layers.len() {
                            canvas_state.layers[i].visible = was_visible;
                        }
                    }
                }
                self.peek_state.is_peeking = false;
                self.peek_state.peek_layer_index = None;
                self.peek_state.saved_visibility.clear();
                self.peek_state.peek_just_ended = true;
                self.mark_full_dirty(canvas_state);
            }
        }
    }

    /// Solo a layer — hide all others permanently until unsoloed
    fn solo_layer(&mut self, layer_idx: usize, canvas_state: &mut CanvasState) {
        if self.peek_state.is_soloed && self.peek_state.solo_layer_index == Some(layer_idx) {
            // Already soloed on this layer — unsolo
            self.show_all_layers(canvas_state);
            return;
        }

        // Save current visibility if not already soloed
        if !self.peek_state.is_soloed {
            self.peek_state.solo_saved_visibility =
                canvas_state.layers.iter().map(|l| l.visible).collect();
        }

        self.peek_state.is_soloed = true;
        self.peek_state.solo_layer_index = Some(layer_idx);

        for (i, layer) in canvas_state.layers.iter_mut().enumerate() {
            layer.visible = i == layer_idx;
        }
        self.mark_full_dirty(canvas_state);
    }

    /// Hide all layers
    fn hide_all_layers(&mut self, canvas_state: &mut CanvasState) {
        // Clear solo state if active
        if self.peek_state.is_soloed {
            self.peek_state.is_soloed = false;
            self.peek_state.solo_layer_index = None;
            self.peek_state.solo_saved_visibility.clear();
        }

        for layer in canvas_state.layers.iter_mut() {
            layer.visible = false;
        }
        self.mark_full_dirty(canvas_state);
    }

    /// Show all layers (also clears solo state)
    fn show_all_layers(&mut self, canvas_state: &mut CanvasState) {
        if self.peek_state.is_soloed {
            // Restore saved visibility from before solo
            for (i, layer) in canvas_state.layers.iter_mut().enumerate() {
                if i < self.peek_state.solo_saved_visibility.len() {
                    layer.visible = self.peek_state.solo_saved_visibility[i];
                } else {
                    layer.visible = true;
                }
            }
            self.peek_state.is_soloed = false;
            self.peek_state.solo_layer_index = None;
            self.peek_state.solo_saved_visibility.clear();
        } else {
            for layer in canvas_state.layers.iter_mut() {
                layer.visible = true;
            }
        }
        self.mark_full_dirty(canvas_state);
    }

    fn merge_down(
        &mut self,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
        history: &mut HistoryManager,
    ) {
        if layer_idx == 0 || layer_idx >= canvas_state.layers.len() {
            return;
        }

        // Snapshot before merge for undo (multi-layer op requires full snapshot)
        let mut snap_cmd = SnapshotCommand::new("Merge Down".to_string(), canvas_state);

        // Auto-rasterize text layers before merge (pixels must be up-to-date)
        for idx in [layer_idx, layer_idx - 1] {
            if canvas_state.layers[idx].is_text_layer() {
                canvas_state.ensure_all_text_layers_rasterized();
                canvas_state.layers[idx].content = LayerContent::Raster;
            }
        }

        let width = canvas_state.width;
        let height = canvas_state.height;

        let top_blend_mode = canvas_state.layers[layer_idx].blend_mode;
        let top_opacity = canvas_state.layers[layer_idx].opacity;
        let top_visible = canvas_state.layers[layer_idx].visible;

        if !top_visible {
            canvas_state.layers.remove(layer_idx);
            if canvas_state.active_layer_index >= layer_idx && canvas_state.active_layer_index > 0 {
                canvas_state.active_layer_index -= 1;
            }
            self.thumbnail_cache.clear();
            self.pending_gpu_clear = true;
            return;
        }

        let top_pixels: Vec<Rgba<u8>> = {
            let top_layer = &canvas_state.layers[layer_idx];
            (0..height)
                .flat_map(|y| (0..width).map(move |x| *top_layer.pixels.get_pixel(x, y)))
                .collect()
        };

        let bottom_layer = &mut canvas_state.layers[layer_idx - 1];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let top_pixel = top_pixels[idx];
                let base_pixel = *bottom_layer.pixels.get_pixel(x, y);

                let blended = CanvasState::blend_pixel_static(
                    base_pixel,
                    top_pixel,
                    top_blend_mode,
                    top_opacity,
                );
                bottom_layer.pixels.put_pixel(x, y, blended);
            }
        }

        canvas_state.layers.remove(layer_idx);
        if canvas_state.active_layer_index >= layer_idx && canvas_state.active_layer_index > 0 {
            canvas_state.active_layer_index -= 1;
        }

        self.thumbnail_cache.clear();
        self.pending_gpu_clear = true;
        self.mark_full_dirty(canvas_state);

        // Record undo after merge
        snap_cmd.set_after(canvas_state);
        history.push(Box::new(snap_cmd));
    }

    fn flatten_image(&mut self, canvas_state: &mut CanvasState, history: &mut HistoryManager) {
        if canvas_state.layers.len() <= 1 {
            return;
        }

        // Snapshot before flatten for undo (multi-layer op requires full snapshot)
        let mut snap_cmd = SnapshotCommand::new("Flatten Image".to_string(), canvas_state);

        canvas_state.ensure_all_text_layers_rasterized();
        let flattened = canvas_state.composite();

        let mut new_layer = Layer::new(
            "Background".to_string(),
            canvas_state.width,
            canvas_state.height,
            Rgba([255, 255, 255, 255]),
        );
        new_layer.pixels = TiledImage::from_rgba_image(&flattened);

        canvas_state.layers = vec![new_layer];
        canvas_state.active_layer_index = 0;

        self.thumbnail_cache.clear();
        self.pending_gpu_clear = true;
        self.mark_full_dirty(canvas_state);

        // Record undo after flatten
        snap_cmd.set_after(canvas_state);
        history.push(Box::new(snap_cmd));
    }

    fn mark_full_dirty(&self, canvas_state: &mut CanvasState) {
        canvas_state.dirty_rect = Some(Rect::from_min_max(
            Pos2::ZERO,
            Pos2::new(canvas_state.width as f32, canvas_state.height as f32),
        ));
    }
}
