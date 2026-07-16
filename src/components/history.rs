use eframe::egui;
use egui::Rect;
use image::Rgba;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::assets::{Assets, Icon};
use crate::canvas::{CanvasState, LayerContent, LayerFolder, TiledImage};

// ============================================================================
// COMMAND TRAIT
// ============================================================================

/// Trait for undoable/redoable commands.
pub trait Command: Send + Sync {
    fn undo(&self, canvas: &mut CanvasState);
    fn redo(&self, canvas: &mut CanvasState);
    fn description(&self) -> String;
    fn memory_size(&self) -> usize;
}

pub struct MarkerCommand {
    desc: String,
}

impl MarkerCommand {
    pub fn new(desc: impl Into<String>) -> Self {
        Self { desc: desc.into() }
    }
}

impl Command for MarkerCommand {
    fn undo(&self, _canvas: &mut CanvasState) {}
    fn redo(&self, _canvas: &mut CanvasState) {}
    fn description(&self) -> String {
        self.desc.clone()
    }
    fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.desc.len()
    }
}

// ============================================================================
// BRUSH COMMAND - Memory-efficient patch-based undo for drawing
// ============================================================================

/// A rectangular patch of pixel data for efficient undo/redo.
#[derive(Clone)]
pub struct PixelPatch {
    pub layer_index: usize,
    pub rect: Rect,
    pub pixels: Vec<Rgba<u8>>,
    pub width: u32,
    pub height: u32,
}

impl PixelPatch {
    pub fn capture(canvas: &CanvasState, layer_index: usize, rect: Rect) -> Self {
        let layer = match canvas.layers.get(layer_index) {
            Some(l) => l,
            None => {
                eprintln!(
                    "PixelPatch::capture: layer index {} out of bounds ({})",
                    layer_index,
                    canvas.layers.len()
                );
                return Self {
                    layer_index,
                    rect,
                    pixels: Vec::new(),
                    width: 0,
                    height: 0,
                };
            }
        };

        // Clamp rect to canvas bounds
        let min_x = (rect.min.x.floor() as u32).min(canvas.width);
        let min_y = (rect.min.y.floor() as u32).min(canvas.height);
        let max_x = (rect.max.x.ceil() as u32).min(canvas.width);
        let max_y = (rect.max.y.ceil() as u32).min(canvas.height);

        let width = max_x.saturating_sub(min_x);
        let height = max_y.saturating_sub(min_y);

        let mut pixels = Vec::with_capacity((width * height) as usize);

        for y in min_y..max_y {
            for x in min_x..max_x {
                pixels.push(*layer.pixels.get_pixel(x, y));
            }
        }

        Self {
            layer_index,
            rect: Rect::from_min_max(
                egui::pos2(min_x as f32, min_y as f32),
                egui::pos2(max_x as f32, max_y as f32),
            ),
            pixels,
            width,
            height,
        }
    }

    pub fn from_image(
        image: &TiledImage,
        layer_index: usize,
        rect: Rect,
        canvas_width: u32,
        canvas_height: u32,
    ) -> Self {
        // Clamp rect to image bounds
        let min_x = (rect.min.x.floor() as u32).min(canvas_width);
        let min_y = (rect.min.y.floor() as u32).min(canvas_height);
        let max_x = (rect.max.x.ceil() as u32).min(canvas_width);
        let max_y = (rect.max.y.ceil() as u32).min(canvas_height);

        let width = max_x.saturating_sub(min_x);
        let height = max_y.saturating_sub(min_y);

        let mut pixels = Vec::with_capacity((width * height) as usize);

        for y in min_y..max_y {
            for x in min_x..max_x {
                if x < image.width() && y < image.height() {
                    pixels.push(*image.get_pixel(x, y));
                } else {
                    pixels.push(Rgba([0, 0, 0, 0]));
                }
            }
        }

        Self {
            layer_index,
            rect: Rect::from_min_max(
                egui::pos2(min_x as f32, min_y as f32),
                egui::pos2(max_x as f32, max_y as f32),
            ),
            pixels,
            width,
            height,
        }
    }

    pub fn apply(&self, canvas: &mut CanvasState) {
        if self.layer_index >= canvas.layers.len() {
            eprintln!("PixelPatch: layer index {} out of bounds", self.layer_index);
            return;
        }

        let layer = &mut canvas.layers[self.layer_index];

        let min_x = self.rect.min.x as u32;
        let min_y = self.rect.min.y as u32;

        let mut idx = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                let canvas_x = min_x + x;
                let canvas_y = min_y + y;

                if canvas_x < canvas.width && canvas_y < canvas.height && idx < self.pixels.len() {
                    layer.pixels.put_pixel(canvas_x, canvas_y, self.pixels[idx]);
                }
                idx += 1;
            }
        }

        // Invalidate GPU texture cache so the renderer re-uploads the restored pixels.
        layer.sync_deep_pixels_from_preview_region(
            min_x,
            min_y,
            min_x + self.width,
            min_y + self.height,
        );
        layer.invalidate_lod();
        layer.gpu_generation += 1;

        // Mark the affected area as dirty for re-rendering
        canvas.mark_dirty(Some(self.rect));
    }

    pub fn memory_size(&self) -> usize {
        self.pixels.len() * 4 // 4 bytes per RGBA pixel
    }
}

pub struct BrushCommand {
    description: String,
    /// The pixels before the modification (for undo)
    before_patch: PixelPatch,
    /// The pixels after the modification (for redo) - optional, can be recalculated
    after_patch: Option<PixelPatch>,
}

impl BrushCommand {
    pub fn new(description: String, before_patch: PixelPatch, after_patch: PixelPatch) -> Self {
        Self {
            description,
            before_patch,
            after_patch: Some(after_patch),
        }
    }

    /// Create a brush command storing only the before patch (redo recaptures)
    pub fn with_before_only(description: String, before_patch: PixelPatch) -> Self {
        Self {
            description,
            before_patch,
            after_patch: None,
        }
    }
}

impl Command for BrushCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        self.before_patch.apply(canvas);
    }

    fn redo(&self, canvas: &mut CanvasState) {
        if let Some(ref after) = self.after_patch {
            after.apply(canvas);
        } else {
            // If no after patch stored, apply before_patch to maintain consistency
            // This can happen if capture failed — log and apply best effort
            eprintln!(
                "BrushCommand: no after_patch for redo, re-applying before_patch to maintain state"
            );
            self.before_patch.apply(canvas);
        }
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn memory_size(&self) -> usize {
        self.before_patch.memory_size() + self.after_patch.as_ref().map_or(0, |p| p.memory_size())
    }
}

/// Undo/redo for mask painting strokes (stores mask image before/after).
pub struct LayerMaskCommand {
    description: String,
    layer_index: usize,
    before_mask: Option<TiledImage>,
    after_mask: Option<TiledImage>,
    before_enabled: bool,
    after_enabled: bool,
}

impl LayerMaskCommand {
    pub fn new(
        description: String,
        layer_index: usize,
        before_mask: Option<TiledImage>,
        after_mask: Option<TiledImage>,
        before_enabled: bool,
        after_enabled: bool,
    ) -> Self {
        Self {
            description,
            layer_index,
            before_mask,
            after_mask,
            before_enabled,
            after_enabled,
        }
    }
}

impl Command for LayerMaskCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        if let Some(layer) = canvas.layers.get_mut(self.layer_index) {
            layer.mask = self.before_mask.clone();
            layer.mask_enabled = self.before_enabled;
        }
        canvas.mark_dirty(None);
    }

    fn redo(&self, canvas: &mut CanvasState) {
        if let Some(layer) = canvas.layers.get_mut(self.layer_index) {
            layer.mask = self.after_mask.clone();
            layer.mask_enabled = self.after_enabled;
        }
        canvas.mark_dirty(None);
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn memory_size(&self) -> usize {
        self.before_mask.as_ref().map_or(0, |m| m.memory_bytes())
            + self.after_mask.as_ref().map_or(0, |m| m.memory_bytes())
    }
}

// ============================================================================
// LAYER OPERATION COMMAND - For layer add/delete/reorder/opacity changes
// ============================================================================

/// Types of layer operations that can be undone/redone
#[derive(Clone)]
pub enum LayerOperation {
    /// A layer was added at the given index
    Add {
        index: usize,
        name: String,
        width: u32,
        height: u32,
        folder_id: Option<u64>,
    },
    /// A layer was deleted (stores the full layer data for restore)
    Delete {
        index: usize,
        pixels: TiledImage,
        mask: Option<TiledImage>,
        mask_enabled: bool,
        name: String,
        visible: bool,
        folder_id: Option<u64>,
        opacity: f32,
        content: LayerContent,
        pixel_format: crate::canvas::PixelFormat,
        hdr_metadata: crate::canvas::HdrMetadata,
        source_metadata: crate::canvas::ImageMetadata,
        webp_frame_compression: crate::canvas::WebpFrameCompression,
        deep_pixels: Option<crate::experimental::DeepRgbaBuffer>,
    },
    /// Layer was moved from one index to another
    Move { from_index: usize, to_index: usize },
    /// Layer opacity was changed
    Opacity {
        index: usize,
        old_opacity: f32,
        new_opacity: f32,
    },
    /// Layer visibility was toggled
    Visibility { index: usize, was_visible: bool },
    /// Layer was renamed
    Rename {
        index: usize,
        old_name: String,
        new_name: String,
    },
    /// Layer was duplicated (stores the new layer's data for undo)
    Duplicate {
        source_index: usize,
        new_index: usize,
        pixels: TiledImage,
        mask: Option<TiledImage>,
        mask_enabled: bool,
        name: String,
        visible: bool,
        folder_id: Option<u64>,
        opacity: f32,
        content: LayerContent,
        pixel_format: crate::canvas::PixelFormat,
        hdr_metadata: crate::canvas::HdrMetadata,
        source_metadata: crate::canvas::ImageMetadata,
        webp_frame_compression: crate::canvas::WebpFrameCompression,
        deep_pixels: Option<crate::experimental::DeepRgbaBuffer>,
    },
}

/// Command for layer structure operations
pub struct LayerOpCommand {
    operation: LayerOperation,
}

impl LayerOpCommand {
    pub fn new(operation: LayerOperation) -> Self {
        Self { operation }
    }
}

impl Command for LayerOpCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        use crate::canvas::Layer;

        match &self.operation {
            LayerOperation::Add { index, .. } => {
                // Undo add = remove the layer
                if *index < canvas.layers.len() {
                    canvas.layers.remove(*index);
                    if canvas.layers.is_empty() {
                        canvas.active_layer_index = 0;
                    } else if canvas.active_layer_index >= canvas.layers.len() {
                        canvas.active_layer_index = canvas.layers.len() - 1;
                    }
                }
            }
            LayerOperation::Delete {
                index,
                pixels,
                mask,
                mask_enabled,
                name,
                visible,
                opacity,
                content,
                pixel_format,
                hdr_metadata,
                source_metadata,
                webp_frame_compression,
                deep_pixels,
                folder_id,
            } => {
                // Undo delete = restore the layer
                let mut layer = Layer::new(
                    name.clone(),
                    pixels.width(),
                    pixels.height(),
                    Rgba([0, 0, 0, 0]),
                );
                layer.pixels = pixels.clone();
                layer.mask = mask.clone();
                layer.mask_enabled = *mask_enabled;
                layer.visible = *visible;
                layer.folder_id = *folder_id;
                layer.opacity = *opacity;
                layer.content = content.clone();
                layer.pixel_format = *pixel_format;
                layer.hdr_metadata = hdr_metadata.clone();
                layer.source_metadata = source_metadata.clone();
                layer.webp_frame_compression = *webp_frame_compression;
                layer.deep_pixels = deep_pixels.clone();

                let insert_idx = (*index).min(canvas.layers.len());
                canvas.layers.insert(insert_idx, layer);
            }
            LayerOperation::Move {
                from_index,
                to_index,
            } => {
                // Undo move = move back
                if *to_index < canvas.layers.len() {
                    let layer = canvas.layers.remove(*to_index);
                    let insert_idx = (*from_index).min(canvas.layers.len());
                    canvas.layers.insert(insert_idx, layer);
                }
            }
            LayerOperation::Opacity {
                index, old_opacity, ..
            } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].opacity = *old_opacity;
                }
            }
            LayerOperation::Visibility { index, was_visible } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].visible = *was_visible;
                }
            }
            LayerOperation::Rename {
                index, old_name, ..
            } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].name = old_name.clone();
                }
            }
            LayerOperation::Duplicate { new_index, .. } => {
                // Undo duplicate = remove the duplicated layer
                if *new_index < canvas.layers.len() {
                    canvas.layers.remove(*new_index);
                    if canvas.layers.is_empty() {
                        canvas.active_layer_index = 0;
                    } else if canvas.active_layer_index >= canvas.layers.len() {
                        canvas.active_layer_index = canvas.layers.len() - 1;
                    }
                }
            }
        }

        canvas.mark_dirty(None);
    }

    fn redo(&self, canvas: &mut CanvasState) {
        use crate::canvas::Layer;

        match &self.operation {
            LayerOperation::Add {
                index,
                name,
                width,
                height,
                folder_id,
            } => {
                // Redo add = add the layer again
                let mut layer = Layer::new(name.clone(), *width, *height, Rgba([0, 0, 0, 0]));
                layer.folder_id = *folder_id;
                let insert_idx = (*index).min(canvas.layers.len());
                canvas.layers.insert(insert_idx, layer);
            }
            LayerOperation::Delete { index, .. } => {
                // Redo delete = remove the layer again
                if *index < canvas.layers.len() {
                    canvas.layers.remove(*index);
                    if canvas.layers.is_empty() {
                        canvas.active_layer_index = 0;
                    } else if canvas.active_layer_index >= canvas.layers.len() {
                        canvas.active_layer_index = canvas.layers.len() - 1;
                    }
                }
            }
            LayerOperation::Move {
                from_index,
                to_index,
            } => {
                // Redo move = move again
                if *from_index < canvas.layers.len() {
                    let layer = canvas.layers.remove(*from_index);
                    let insert_idx = (*to_index).min(canvas.layers.len());
                    canvas.layers.insert(insert_idx, layer);
                }
            }
            LayerOperation::Opacity {
                index, new_opacity, ..
            } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].opacity = *new_opacity;
                }
            }
            LayerOperation::Visibility { index, was_visible } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].visible = !was_visible;
                }
            }
            LayerOperation::Rename {
                index, new_name, ..
            } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].name = new_name.clone();
                }
            }
            LayerOperation::Duplicate {
                new_index,
                pixels,
                mask,
                mask_enabled,
                name,
                visible,
                opacity,
                content,
                pixel_format,
                hdr_metadata,
                source_metadata,
                webp_frame_compression,
                deep_pixels,
                folder_id,
                ..
            } => {
                // Redo duplicate = restore the duplicated layer
                let mut layer = Layer::new(
                    name.clone(),
                    pixels.width(),
                    pixels.height(),
                    Rgba([0, 0, 0, 0]),
                );
                layer.pixels = pixels.clone();
                layer.mask = mask.clone();
                layer.mask_enabled = *mask_enabled;
                layer.visible = *visible;
                layer.folder_id = *folder_id;
                layer.opacity = *opacity;
                layer.content = content.clone();
                layer.pixel_format = *pixel_format;
                layer.hdr_metadata = hdr_metadata.clone();
                layer.source_metadata = source_metadata.clone();
                layer.webp_frame_compression = *webp_frame_compression;
                layer.deep_pixels = deep_pixels.clone();
                let insert_idx = (*new_index).min(canvas.layers.len());
                canvas.layers.insert(insert_idx, layer);
                canvas.active_layer_index = insert_idx;
            }
        }

        canvas.mark_dirty(None);
    }

    fn description(&self) -> String {
        match &self.operation {
            LayerOperation::Add { name, .. } => format!("Add Layer: {}", name),
            LayerOperation::Delete { name, .. } => format!("Delete Layer: {}", name),
            LayerOperation::Move {
                from_index,
                to_index,
            } => {
                format!("Move Layer {} → {}", from_index, to_index)
            }
            LayerOperation::Opacity {
                index, new_opacity, ..
            } => {
                format!("Layer {} Opacity: {:.0}%", index, new_opacity * 100.0)
            }
            LayerOperation::Visibility { index, was_visible } => {
                if *was_visible {
                    format!("Hide Layer {}", index)
                } else {
                    format!("Show Layer {}", index)
                }
            }
            LayerOperation::Rename {
                old_name, new_name, ..
            } => {
                format!("Rename: {} → {}", old_name, new_name)
            }
            LayerOperation::Duplicate { name, .. } => {
                format!("Duplicate: {}", name)
            }
        }
    }

    fn memory_size(&self) -> usize {
        match &self.operation {
            LayerOperation::Delete {
                pixels, mask, name, ..
            } => pixels.memory_bytes() + mask.as_ref().map_or(0, |m| m.memory_bytes()) + name.len(),
            LayerOperation::Duplicate {
                pixels, mask, name, ..
            } => pixels.memory_bytes() + mask.as_ref().map_or(0, |m| m.memory_bytes()) + name.len(),
            LayerOperation::Add { name, .. } => name.len(),
            LayerOperation::Rename {
                old_name, new_name, ..
            } => old_name.len() + new_name.len(),
            _ => std::mem::size_of::<LayerOperation>(),
        }
    }
}

// ============================================================================
// HISTORY MANAGER - Manages undo/redo stacks with memory limits
// ============================================================================

/// Undo/redo history manager with memory limits.
pub struct HistoryManager {
    undo_stack: VecDeque<Box<dyn Command>>,
    redo_stack: VecDeque<Box<dyn Command>>,
    max_history_size: usize,
    /// Optional memory cap in bytes.
    max_memory_bytes: Option<usize>,
    /// Running memory total across both stacks.
    total_memory: usize,
}

impl Default for HistoryManager {
    fn default() -> Self {
        Self::new(50)
    }
}

impl HistoryManager {
    pub fn new(max_history_size: usize) -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
            max_history_size,
            max_memory_bytes: Some(100 * 1024 * 1024), // 100 MB default limit
            total_memory: 0,
        }
    }

    pub fn push(&mut self, command: Box<dyn Command>) {
        // Clear redo stack when a new action is performed
        for cmd in self.redo_stack.drain(..) {
            self.total_memory = self.total_memory.saturating_sub(cmd.memory_size());
        }

        // Add the new command
        self.total_memory += command.memory_size();
        self.undo_stack.push_back(command);

        // Prune old commands if we exceed the limit
        self.prune();
    }

    pub fn undo(&mut self, canvas: &mut CanvasState) -> Option<String> {
        if let Some(command) = self.undo_stack.pop_back() {
            let description = command.description();
            command.undo(canvas);
            self.redo_stack.push_back(command);
            Some(description)
        } else {
            None
        }
    }

    pub fn redo(&mut self, canvas: &mut CanvasState) -> Option<String> {
        if let Some(command) = self.redo_stack.pop_back() {
            let description = command.description();
            command.redo(canvas);
            self.undo_stack.push_back(command);
            Some(description)
        } else {
            None
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo_description(&self) -> Option<String> {
        self.undo_stack.back().map(|c| c.description())
    }

    pub fn redo_description(&self) -> Option<String> {
        self.redo_stack.back().map(|c| c.description())
    }

    /// Get all undo descriptions (most recent first)
    pub fn undo_history(&self) -> Vec<String> {
        self.undo_stack
            .iter()
            .rev()
            .map(|c| c.description())
            .collect()
    }

    /// Get the current memory usage of the history (O(1) via cached total)
    pub fn memory_usage(&self) -> usize {
        self.total_memory
    }

    /// Prune old commands to stay within limits
    fn prune(&mut self) {
        // Prune by count
        while self.undo_stack.len() > self.max_history_size {
            if let Some(removed) = self.undo_stack.pop_front() {
                self.total_memory = self.total_memory.saturating_sub(removed.memory_size());
            }
        }

        // Prune by memory if limit is set
        if let Some(max_bytes) = self.max_memory_bytes {
            while self.total_memory > max_bytes && self.undo_stack.len() > 1 {
                if let Some(removed) = self.undo_stack.pop_front() {
                    self.total_memory = self.total_memory.saturating_sub(removed.memory_size());
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.total_memory = 0;
    }

    /// Undo to position `index` in undo_history() (0 = most recent).
    pub fn undo_to(&mut self, index: usize, canvas: &mut CanvasState) {
        // index is how many undos we need to do
        for _ in 0..index {
            if self.can_undo() {
                self.undo(canvas);
            } else {
                break;
            }
        }
    }

    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }
}

// ============================================================================
// SNAPSHOT COMMAND — full-canvas undo for heavy operations (resize, blur, etc.)
// ============================================================================

/// Stores a complete canvas snapshot for undo/redo of destructive operations.
pub struct SnapshotCommand {
    description: String,
    before: CanvasSnapshot,
    after: Option<CanvasSnapshot>,
}

/// A lightweight snapshot of the canvas state (layers + dimensions).
#[derive(Clone)]
pub struct CanvasSnapshot {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<LayerSnapshot>,
    pub layer_folders: Vec<LayerFolder>,
    pub next_layer_folder_id: u64,
    pub active_layer_index: usize,
    pub selection_mask: Option<image::GrayImage>,
    pub selection_all: bool,
}

#[derive(Clone)]
pub struct LayerSnapshot {
    pub name: String,
    pub visible: bool,
    pub folder_id: Option<u64>,
    pub opacity: f32,
    pub blend_mode: crate::canvas::BlendMode,
    pub pixels: TiledImage,
    pub mask: Option<TiledImage>,
    pub mask_enabled: bool,
    pub content: LayerContent,
    pub pixel_format: crate::canvas::PixelFormat,
    pub hdr_metadata: crate::canvas::HdrMetadata,
    pub source_metadata: crate::canvas::ImageMetadata,
    pub webp_frame_compression: crate::canvas::WebpFrameCompression,
    pub deep_pixels: Option<crate::experimental::DeepRgbaBuffer>,
}

impl CanvasSnapshot {
    pub fn capture(state: &CanvasState) -> Self {
        Self {
            width: state.width,
            height: state.height,
            active_layer_index: state.active_layer_index,
            selection_mask: state.selection_mask.clone(),
            selection_all: state.selection_all,
            layer_folders: state.layer_folders.clone(),
            next_layer_folder_id: state.next_layer_folder_id,
            layers: state
                .layers
                .iter()
                .map(|l| LayerSnapshot {
                    name: l.name.clone(),
                    visible: l.visible,
                    folder_id: l.folder_id,
                    opacity: l.opacity,
                    blend_mode: l.blend_mode,
                    pixels: l.pixels.clone(),
                    mask: l.mask.clone(),
                    mask_enabled: l.mask_enabled,
                    content: l.content.clone(),
                    pixel_format: l.pixel_format,
                    hdr_metadata: l.hdr_metadata.clone(),
                    source_metadata: l.source_metadata.clone(),
                    webp_frame_compression: l.webp_frame_compression,
                    deep_pixels: l.deep_pixels.clone(),
                })
                .collect(),
        }
    }

    pub fn restore_into(&self, state: &mut CanvasState) {
        state.width = self.width;
        state.height = self.height;
        state.active_layer_index = self.active_layer_index;
        state.layer_folders = self.layer_folders.clone();
        state.next_layer_folder_id = self.next_layer_folder_id;
        state.layers.clear();
        for snap in &self.layers {
            let mut layer = crate::canvas::Layer::new(
                snap.name.clone(),
                snap.pixels.width(),
                snap.pixels.height(),
                Rgba([0, 0, 0, 0]),
            );
            layer.pixels = snap.pixels.clone();
            layer.visible = snap.visible;
            layer.folder_id = snap.folder_id;
            layer.opacity = snap.opacity;
            layer.blend_mode = snap.blend_mode;
            layer.mask = snap.mask.clone();
            layer.mask_enabled = snap.mask_enabled;
            layer.content = snap.content.clone();
            layer.pixel_format = snap.pixel_format;
            layer.hdr_metadata = snap.hdr_metadata.clone();
            layer.source_metadata = snap.source_metadata.clone();
            layer.webp_frame_compression = snap.webp_frame_compression;
            layer.deep_pixels = snap.deep_pixels.clone();
            state.layers.push(layer);
        }
        state.selection_mask = self.selection_mask.clone();
        state.selection_all = self.selection_all;
        state.composite_cache = None;
        state.clear_preview_state();
        state.invalidate_selection_overlay();
        state.selection_overlay_texture = None;
        state.mark_dirty(None);
    }

    fn memory_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.pixels.memory_bytes() + l.name.len())
            .sum()
    }
}

impl SnapshotCommand {
    /// Create a snapshot command. Call BEFORE performing the operation.
    /// After the operation, call `set_after()`.
    pub fn new(description: String, state: &CanvasState) -> Self {
        Self {
            description,
            before: CanvasSnapshot::capture(state),
            after: None,
        }
    }

    /// Capture the "after" state. Call this AFTER the operation completes.
    pub fn set_after(&mut self, state: &CanvasState) {
        self.after = Some(CanvasSnapshot::capture(state));
    }

    pub fn from_snapshots(
        description: String,
        before: CanvasSnapshot,
        after: CanvasSnapshot,
    ) -> Self {
        Self {
            description,
            before,
            after: Some(after),
        }
    }
}

impl Command for SnapshotCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        self.before.restore_into(canvas);
    }

    fn redo(&self, canvas: &mut CanvasState) {
        if let Some(ref after) = self.after {
            after.restore_into(canvas);
        }
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn memory_size(&self) -> usize {
        self.before.memory_bytes() + self.after.as_ref().map_or(0, |a| a.memory_bytes())
    }
}

// ============================================================================
// SINGLE-LAYER SNAPSHOT — efficient undo for single-layer operations
// ============================================================================

/// Captures only the active layer's pixels and metadata before/after an
/// operation. For a 5-layer 4K image, this stores ~66MB instead of ~330MB.
pub struct SingleLayerSnapshotCommand {
    description: String,
    layer_index: usize,
    before_pixels: TiledImage,
    after_pixels: Option<TiledImage>,
    before_mask: Option<TiledImage>,
    after_mask: Option<TiledImage>,
    before_mask_enabled: bool,
    after_mask_enabled: bool,
    before_opacity: f32,
    after_opacity: f32,
    before_blend_mode: crate::canvas::BlendMode,
    after_blend_mode: crate::canvas::BlendMode,
    before_content: LayerContent,
    after_content: LayerContent,
    before_pixel_format: crate::canvas::PixelFormat,
    after_pixel_format: crate::canvas::PixelFormat,
    before_hdr_metadata: crate::canvas::HdrMetadata,
    after_hdr_metadata: crate::canvas::HdrMetadata,
    before_source_metadata: crate::canvas::ImageMetadata,
    after_source_metadata: crate::canvas::ImageMetadata,
    before_webp_frame_compression: crate::canvas::WebpFrameCompression,
    after_webp_frame_compression: crate::canvas::WebpFrameCompression,
    before_deep_pixels: Option<crate::experimental::DeepRgbaBuffer>,
    after_deep_pixels: Option<crate::experimental::DeepRgbaBuffer>,
}

impl SingleLayerSnapshotCommand {
    /// Create before performing the operation. Call `set_after()` when done.
    pub fn new(description: String, state: &CanvasState) -> Self {
        Self::new_for_layer(description, state, state.active_layer_index)
    }

    /// Create for a specific layer index (when the active layer may differ from
    /// the layer being modified, e.g., dialog commits store layer_idx at open time).
    pub fn new_for_layer(description: String, state: &CanvasState, layer_idx: usize) -> Self {
        let safe_idx = if state.layers.is_empty() {
            0
        } else {
            layer_idx.min(state.layers.len() - 1)
        };
        let (
            before_pixels,
            before_mask,
            before_mask_enabled,
            before_opacity,
            before_blend_mode,
            before_content,
            before_pixel_format,
            before_hdr_metadata,
            before_source_metadata,
            before_webp_frame_compression,
            before_deep_pixels,
        ) = if let Some(layer) = state.layers.get(safe_idx) {
            (
                layer.pixels.clone(),
                layer.mask.clone(),
                layer.mask_enabled,
                layer.opacity,
                layer.blend_mode,
                layer.content.clone(),
                layer.pixel_format,
                layer.hdr_metadata.clone(),
                layer.source_metadata.clone(),
                layer.webp_frame_compression,
                layer.deep_pixels.clone(),
            )
        } else {
            (
                TiledImage::new(1, 1),
                None,
                true,
                1.0,
                crate::canvas::BlendMode::Normal,
                LayerContent::Raster,
                crate::canvas::PixelFormat::RgbaU8,
                crate::canvas::HdrMetadata::default(),
                crate::canvas::ImageMetadata::default(),
                crate::canvas::WebpFrameCompression::default(),
                None,
            )
        };
        Self {
            description,
            layer_index: safe_idx,
            before_pixels,
            after_pixels: None,
            before_mask,
            after_mask: None,
            before_mask_enabled,
            after_mask_enabled: before_mask_enabled,
            before_opacity,
            after_opacity: before_opacity,
            before_blend_mode,
            after_blend_mode: before_blend_mode,
            before_content: before_content.clone(),
            after_content: before_content,
            before_pixel_format,
            after_pixel_format: before_pixel_format,
            before_hdr_metadata: before_hdr_metadata.clone(),
            after_hdr_metadata: before_hdr_metadata,
            before_source_metadata: before_source_metadata.clone(),
            after_source_metadata: before_source_metadata,
            before_webp_frame_compression,
            after_webp_frame_compression: before_webp_frame_compression,
            before_deep_pixels: before_deep_pixels.clone(),
            after_deep_pixels: before_deep_pixels,
        }
    }

    /// Capture the layer's state after the operation.
    pub fn set_after(&mut self, state: &CanvasState) {
        if let Some(layer) = state.layers.get(self.layer_index) {
            self.after_pixels = Some(layer.pixels.clone());
            self.after_mask = layer.mask.clone();
            self.after_mask_enabled = layer.mask_enabled;
            self.after_opacity = layer.opacity;
            self.after_blend_mode = layer.blend_mode;
            self.after_content = layer.content.clone();
            self.after_pixel_format = layer.pixel_format;
            self.after_hdr_metadata = layer.hdr_metadata.clone();
            self.after_source_metadata = layer.source_metadata.clone();
            self.after_webp_frame_compression = layer.webp_frame_compression;
            self.after_deep_pixels = layer.deep_pixels.clone();
        }
    }
}

impl Command for SingleLayerSnapshotCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        if let Some(layer) = canvas.layers.get_mut(self.layer_index) {
            layer.pixels = self.before_pixels.clone();
            layer.mask = self.before_mask.clone();
            layer.mask_enabled = self.before_mask_enabled;
            layer.opacity = self.before_opacity;
            layer.blend_mode = self.before_blend_mode;
            layer.content = self.before_content.clone();
            layer.pixel_format = self.before_pixel_format;
            layer.hdr_metadata = self.before_hdr_metadata.clone();
            layer.source_metadata = self.before_source_metadata.clone();
            layer.webp_frame_compression = self.before_webp_frame_compression;
            layer.deep_pixels = self.before_deep_pixels.clone();
        }
        canvas.mark_dirty(None);
    }

    fn redo(&self, canvas: &mut CanvasState) {
        if let Some(ref after) = self.after_pixels
            && let Some(layer) = canvas.layers.get_mut(self.layer_index)
        {
            layer.pixels = after.clone();
            layer.mask = self.after_mask.clone();
            layer.mask_enabled = self.after_mask_enabled;
            layer.opacity = self.after_opacity;
            layer.blend_mode = self.after_blend_mode;
            layer.content = self.after_content.clone();
            layer.pixel_format = self.after_pixel_format;
            layer.hdr_metadata = self.after_hdr_metadata.clone();
            layer.source_metadata = self.after_source_metadata.clone();
            layer.webp_frame_compression = self.after_webp_frame_compression;
            layer.deep_pixels = self.after_deep_pixels.clone();
        }
        canvas.mark_dirty(None);
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn memory_size(&self) -> usize {
        self.before_pixels.memory_bytes()
            + self.before_mask.as_ref().map_or(0, |m| m.memory_bytes())
            + self.after_pixels.as_ref().map_or(0, |p| p.memory_bytes())
            + self.after_mask.as_ref().map_or(0, |m| m.memory_bytes())
    }
}

// ============================================================================
// TEXT LAYER EDIT COMMAND — ultra-light undo for text layer vector data
// ============================================================================

/// Lightweight undo command for text layer edits.
/// Stores only vector data (TextLayerData, typically 1–50 KB) instead of
/// rasterized pixels (~66 MB for a single 4K layer). ~1000× more efficient.
pub struct TextLayerEditCommand {
    description: String,
    layer_index: usize,
    before: crate::ops::text_layer::TextLayerData,
    after: Option<crate::ops::text_layer::TextLayerData>,
}

impl TextLayerEditCommand {
    /// Create before the text edit operation. Call `set_after()` when done.
    pub fn new(description: String, layer_index: usize, state: &CanvasState) -> Self {
        let before = if let Some(layer) = state.layers.get(layer_index)
            && let LayerContent::Text(ref td) = layer.content
        {
            td.clone()
        } else {
            crate::ops::text_layer::TextLayerData::default()
        };
        Self {
            description,
            layer_index,
            before,
            after: None,
        }
    }

    /// Create from an already-captured TextLayerData snapshot.
    pub fn new_from(
        description: String,
        layer_index: usize,
        before: crate::ops::text_layer::TextLayerData,
    ) -> Self {
        Self {
            description,
            layer_index,
            before,
            after: None,
        }
    }

    /// Capture the text layer's state after the edit.
    pub fn set_after(&mut self, state: &CanvasState) {
        if let Some(layer) = state.layers.get(self.layer_index)
            && let LayerContent::Text(ref td) = layer.content
        {
            self.after = Some(td.clone());
        }
    }

    /// Set the "after" state from an already-captured TextLayerData.
    pub fn set_after_from(&mut self, after: crate::ops::text_layer::TextLayerData) {
        self.after = Some(after);
    }
}

impl Command for TextLayerEditCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        if let Some(layer) = canvas.layers.get_mut(self.layer_index) {
            layer.content = LayerContent::Text(self.before.clone());
            // Mark dirty so rasterization is triggered before next composite
            if let LayerContent::Text(ref mut td) = layer.content {
                td.mark_dirty();
            }
            layer.invalidate_lod();
            layer.gpu_generation += 1;
        }
        canvas.mark_dirty(None);
    }

    fn redo(&self, canvas: &mut CanvasState) {
        if let Some(ref after) = self.after
            && let Some(layer) = canvas.layers.get_mut(self.layer_index)
        {
            layer.content = LayerContent::Text(after.clone());
            if let LayerContent::Text(ref mut td) = layer.content {
                td.mark_dirty();
            }
            layer.invalidate_lod();
            layer.gpu_generation += 1;
        }
        canvas.mark_dirty(None);
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn memory_size(&self) -> usize {
        // Estimate: struct overhead + serialized sizes of TextLayerData
        // TextLayerData is mostly Vec<TextBlock> with string data — typically 1-50KB
        let before_size = self
            .before
            .blocks
            .iter()
            .map(|b| b.runs.iter().map(|r| r.text.len() + 100).sum::<usize>() + 200)
            .sum::<usize>()
            + 200;
        let after_size = self.after.as_ref().map_or(0, |a| {
            a.blocks
                .iter()
                .map(|b| b.runs.iter().map(|r| r.text.len() + 100).sum::<usize>() + 200)
                .sum::<usize>()
                + 200
        });
        before_size + after_size
    }
}

// ============================================================================
// SELECTION COMMAND - Undo/redo for selection mask changes
// ============================================================================

/// Command that stores the selection-mask state before and after an operation.
/// The `Arc<image::GrayImage>` wrapper lets before/after share heap storage cheaply.
pub struct SelectionCommand {
    description: String,
    before: Option<Arc<image::GrayImage>>,
    after: Option<Arc<image::GrayImage>>,
    before_all: bool,
    after_all: bool,
}

impl SelectionCommand {
    pub fn new(
        description: impl Into<String>,
        before: Option<image::GrayImage>,
        after: Option<image::GrayImage>,
    ) -> Self {
        Self::new_states(description, before, false, after, false)
    }

    pub fn new_states(
        description: impl Into<String>,
        before: Option<image::GrayImage>,
        before_all: bool,
        after: Option<image::GrayImage>,
        after_all: bool,
    ) -> Self {
        Self {
            description: description.into(),
            before: before.map(Arc::new),
            after: after.map(Arc::new),
            before_all,
            after_all,
        }
    }

    pub fn new_all(
        description: impl Into<String>,
        before: Option<image::GrayImage>,
        before_all: bool,
    ) -> Self {
        Self::new_states(description, before, before_all, None, true)
    }
}

impl Command for SelectionCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        canvas.selection_mask = self.before.as_ref().map(|a| (**a).clone());
        canvas.selection_all = self.before_all;
        canvas.invalidate_selection_overlay();
    }

    fn redo(&self, canvas: &mut CanvasState) {
        canvas.selection_mask = self.after.as_ref().map(|a| (**a).clone());
        canvas.selection_all = self.after_all;
        canvas.invalidate_selection_overlay();
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn memory_size(&self) -> usize {
        fn mask_size(m: &Option<Arc<image::GrayImage>>) -> usize {
            m.as_ref()
                .map(|a| (a.width() * a.height()) as usize)
                .unwrap_or(0)
        }
        mask_size(&self.before) + mask_size(&self.after)
    }
}

// ============================================================================
// CUT SELECTION COMMAND - one layer plus semantic selection state
// ============================================================================

pub struct CutSelectionCommand {
    layer_index: usize,
    before_pixels: TiledImage,
    after_pixels: Option<TiledImage>,
    before_content: LayerContent,
    after_content: Option<LayerContent>,
    before_selection: Option<image::GrayImage>,
    before_selection_all: bool,
}

impl CutSelectionCommand {
    pub fn new(state: &CanvasState) -> Option<Self> {
        let layer = state.layers.get(state.active_layer_index)?;
        Some(Self {
            layer_index: state.active_layer_index,
            before_pixels: layer.pixels.clone(),
            after_pixels: None,
            before_content: layer.content.clone(),
            after_content: None,
            before_selection: state.selection_mask.clone(),
            before_selection_all: state.selection_all,
        })
    }

    pub fn set_after(&mut self, state: &CanvasState) {
        if let Some(layer) = state.layers.get(self.layer_index) {
            self.after_pixels = Some(layer.pixels.clone());
            self.after_content = Some(layer.content.clone());
        }
    }
}

impl Command for CutSelectionCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        if let Some(layer) = canvas.layers.get_mut(self.layer_index) {
            layer.pixels = self.before_pixels.clone();
            layer.content = self.before_content.clone();
        }
        canvas.selection_mask = self.before_selection.clone();
        canvas.selection_all = self.before_selection_all;
        canvas.invalidate_selection_overlay();
        canvas.mark_dirty(None);
    }

    fn redo(&self, canvas: &mut CanvasState) {
        if let (Some(layer), Some(pixels)) = (
            canvas.layers.get_mut(self.layer_index),
            self.after_pixels.as_ref(),
        ) {
            layer.pixels = pixels.clone();
            if let Some(content) = self.after_content.as_ref() {
                layer.content = content.clone();
            }
        }
        canvas.clear_selection();
        canvas.mark_dirty(None);
    }

    fn description(&self) -> String {
        "Cut Selection".to_owned()
    }

    fn memory_size(&self) -> usize {
        self.before_pixels.memory_bytes()
            + self
                .after_pixels
                .as_ref()
                .map_or(0, TiledImage::memory_bytes)
            + self
                .before_selection
                .as_ref()
                .map_or(0, |mask| mask.as_raw().len())
    }
}

// ============================================================================
// HISTORY PANEL - UI for displaying history
// ============================================================================

#[derive(Default)]
pub struct HistoryPanel {
    show_memory_info: bool,
}

impl HistoryPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, history: &HistoryManager) {
        ui.horizontal(|ui| {
            ui.label(format!(
                "Undo: {} | Redo: {}",
                history.undo_count(),
                history.redo_count()
            ));

            if ui
                .small_button("ℹ")
                .on_hover_text("Show memory info")
                .clicked()
            {
                self.show_memory_info = !self.show_memory_info;
            }
        });

        if self.show_memory_info {
            let mem_usage = history.memory_usage();
            let mem_mb = mem_usage as f64 / (1024.0 * 1024.0);
            ui.label(format!("Memory: {:.2} MB", mem_mb));
        }

        // Show recent history entries
        egui::ScrollArea::vertical().show(ui, |ui| {
            let history_items = history.undo_history();

            if history_items.is_empty() {
                ui.label("No history");
            } else {
                for (i, desc) in history_items.iter().enumerate() {
                    let label = if i == 0 {
                        format!("▶ {}", desc) // Current state indicator
                    } else {
                        format!("  {}", desc)
                    };
                    ui.label(label);
                }
            }
        });
    }

    /// Show with mutable history (for interactive undo in the panel)
    pub fn show_interactive(
        &mut self,
        ui: &mut egui::Ui,
        history: &mut HistoryManager,
        canvas: &mut CanvasState,
        assets: &Assets,
    ) {
        // Show history list (no undo/redo buttons - they're in the toolbar)
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                let items = history.undo_history();
                if items.is_empty() {
                    ui.weak("No history yet");
                } else {
                    let mut revert_to: Option<usize> = None;

                    for (i, desc) in items.iter().enumerate() {
                        let is_current = i == 0;
                        let icon = Self::icon_for_action(desc);

                        let response = ui.horizontal(|ui| {
                            // Render actual tool icon (14x14)
                            let icon_size = egui::Vec2::splat(14.0);
                            if let Some(texture) = assets.get_texture(icon) {
                                let sized = egui::load::SizedTexture::from_handle(texture);
                                let img =
                                    egui::Image::from_texture(sized).fit_to_exact_size(icon_size);
                                ui.add(img);
                            } else {
                                ui.label(egui::RichText::new(icon.emoji()).size(12.0));
                            }

                            let text = if is_current {
                                egui::RichText::new(desc).strong().size(11.0)
                            } else {
                                egui::RichText::new(desc).weak().size(11.0)
                            };

                            ui.add(egui::Label::new(text).sense(egui::Sense::click()))
                        });

                        let label_response = response.inner;
                        if label_response.clicked() && i > 0 {
                            revert_to = Some(i);
                        }

                        if label_response.hovered() && i > 0 {
                            label_response.on_hover_text("Click to revert to this state");
                        }
                    }

                    // Process revert outside the iteration
                    if let Some(index) = revert_to {
                        history.undo_to(index, canvas);
                    }
                }
            });
    }

    fn icon_for_action(desc: &str) -> Icon {
        let desc_lower = desc.to_lowercase();
        if desc_lower.contains("brush") {
            Icon::Brush
        } else if desc_lower.contains("eraser") {
            Icon::Eraser
        } else if desc_lower.contains("line") {
            Icon::Line
        } else if desc_lower.contains("layer")
            || desc_lower.contains("duplicate")
            || desc_lower.contains("rename")
        {
            Icon::Layers
        } else if desc_lower.contains("opacity")
            || desc_lower.contains("visible")
            || desc_lower.contains("hide")
            || desc_lower.contains("show")
        {
            Icon::Visible
        } else if desc_lower.contains("delete") {
            Icon::Delete
        } else if desc_lower.contains("flatten") {
            Icon::Flatten
        } else if desc_lower.contains("merge") {
            Icon::MergeDown
        } else {
            Icon::Brush
        }
    }
}
