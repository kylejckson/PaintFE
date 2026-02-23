use eframe::egui;
use egui::Rect;
use image::Rgba;
use std::collections::VecDeque;

use crate::assets::{Assets, Icon};
use crate::canvas::{CanvasState, TiledImage};

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
                eprintln!("PixelPatch::capture: layer index {} out of bounds ({})", layer_index, canvas.layers.len());
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
        let min_x = (rect.min.x.floor() as u32).max(0).min(canvas.width);
        let min_y = (rect.min.y.floor() as u32).max(0).min(canvas.height);
        let max_x = (rect.max.x.ceil() as u32).max(0).min(canvas.width);
        let max_y = (rect.max.y.ceil() as u32).max(0).min(canvas.height);
        
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
    
    pub fn from_image(image: &TiledImage, layer_index: usize, rect: Rect, canvas_width: u32, canvas_height: u32) -> Self {
        // Clamp rect to image bounds
        let min_x = (rect.min.x.floor() as u32).max(0).min(canvas_width);
        let min_y = (rect.min.y.floor() as u32).max(0).min(canvas_height);
        let max_x = (rect.max.x.ceil() as u32).max(0).min(canvas_width);
        let max_y = (rect.max.y.ceil() as u32).max(0).min(canvas_height);
        
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
            eprintln!("BrushCommand: no after_patch for redo, re-applying before_patch to maintain state");
            self.before_patch.apply(canvas);
        }
    }
    
    fn description(&self) -> String {
        self.description.clone()
    }
    
    fn memory_size(&self) -> usize {
        self.before_patch.memory_size() 
            + self.after_patch.as_ref().map_or(0, |p| p.memory_size())
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
    },
    /// A layer was deleted (stores the full layer data for restore)
    Delete { 
        index: usize, 
        pixels: TiledImage,
        name: String,
        visible: bool,
        opacity: f32,
    },
    /// Layer was moved from one index to another
    Move { 
        from_index: usize, 
        to_index: usize,
    },
    /// Layer opacity was changed
    Opacity { 
        index: usize, 
        old_opacity: f32, 
        new_opacity: f32,
    },
    /// Layer visibility was toggled
    Visibility { 
        index: usize, 
        was_visible: bool,
    },
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
        name: String,
        visible: bool,
        opacity: f32,
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
                    if canvas.active_layer_index >= canvas.layers.len() && canvas.layers.len() > 0 {
                        canvas.active_layer_index = canvas.layers.len() - 1;
                    }
                }
            }
            LayerOperation::Delete { index, pixels, name, visible, opacity } => {
                // Undo delete = restore the layer
                let mut layer = Layer::new(name.clone(), pixels.width(), pixels.height(), Rgba([0, 0, 0, 0]));
                layer.pixels = pixels.clone();
                layer.visible = *visible;
                layer.opacity = *opacity;
                
                let insert_idx = (*index).min(canvas.layers.len());
                canvas.layers.insert(insert_idx, layer);
            }
            LayerOperation::Move { from_index, to_index } => {
                // Undo move = move back
                if *to_index < canvas.layers.len() {
                    let layer = canvas.layers.remove(*to_index);
                    let insert_idx = (*from_index).min(canvas.layers.len());
                    canvas.layers.insert(insert_idx, layer);
                }
            }
            LayerOperation::Opacity { index, old_opacity, .. } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].opacity = *old_opacity;
                }
            }
            LayerOperation::Visibility { index, was_visible } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].visible = *was_visible;
                }
            }
            LayerOperation::Rename { index, old_name, .. } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].name = old_name.clone();
                }
            }
            LayerOperation::Duplicate { new_index, .. } => {
                // Undo duplicate = remove the duplicated layer
                if *new_index < canvas.layers.len() {
                    canvas.layers.remove(*new_index);
                    if canvas.active_layer_index >= canvas.layers.len() && canvas.layers.len() > 0 {
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
            LayerOperation::Add { index, name, width, height } => {
                // Redo add = add the layer again
                let layer = Layer::new(name.clone(), *width, *height, Rgba([0, 0, 0, 0]));
                let insert_idx = (*index).min(canvas.layers.len());
                canvas.layers.insert(insert_idx, layer);
            }
            LayerOperation::Delete { index, .. } => {
                // Redo delete = remove the layer again
                if *index < canvas.layers.len() {
                    canvas.layers.remove(*index);
                    if canvas.active_layer_index >= canvas.layers.len() && canvas.layers.len() > 0 {
                        canvas.active_layer_index = canvas.layers.len() - 1;
                    }
                }
            }
            LayerOperation::Move { from_index, to_index } => {
                // Redo move = move again
                if *from_index < canvas.layers.len() {
                    let layer = canvas.layers.remove(*from_index);
                    let insert_idx = (*to_index).min(canvas.layers.len());
                    canvas.layers.insert(insert_idx, layer);
                }
            }
            LayerOperation::Opacity { index, new_opacity, .. } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].opacity = *new_opacity;
                }
            }
            LayerOperation::Visibility { index, was_visible } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].visible = !was_visible;
                }
            }
            LayerOperation::Rename { index, new_name, .. } => {
                if *index < canvas.layers.len() {
                    canvas.layers[*index].name = new_name.clone();
                }
            }
            LayerOperation::Duplicate { new_index, pixels, name, visible, opacity, .. } => {
                // Redo duplicate = restore the duplicated layer
                let mut layer = Layer::new(name.clone(), pixels.width(), pixels.height(), Rgba([0, 0, 0, 0]));
                layer.pixels = pixels.clone();
                layer.visible = *visible;
                layer.opacity = *opacity;
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
            LayerOperation::Move { from_index, to_index } => {
                format!("Move Layer {} → {}", from_index, to_index)
            }
            LayerOperation::Opacity { index, new_opacity, .. } => {
                format!("Layer {} Opacity: {:.0}%", index, new_opacity * 100.0)
            }
            LayerOperation::Visibility { index, was_visible } => {
                if *was_visible {
                    format!("Hide Layer {}", index)
                } else {
                    format!("Show Layer {}", index)
                }
            }
            LayerOperation::Rename { old_name, new_name, .. } => {
                format!("Rename: {} → {}", old_name, new_name)
            }
            LayerOperation::Duplicate { name, .. } => {
                format!("Duplicate: {}", name)
            }
        }
    }
    
    fn memory_size(&self) -> usize {
        match &self.operation {
            LayerOperation::Delete { pixels, name, .. } => {
                pixels.memory_bytes() + name.len()
            }
            LayerOperation::Duplicate { pixels, name, .. } => {
                pixels.memory_bytes() + name.len()
            }
            LayerOperation::Add { name, .. } => name.len(),
            LayerOperation::Rename { old_name, new_name, .. } => old_name.len() + new_name.len(),
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
        self.undo_stack.iter().rev().map(|c| c.description()).collect()
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
    pub active_layer_index: usize,
}

#[derive(Clone)]
pub struct LayerSnapshot {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend_mode: crate::canvas::BlendMode,
    pub pixels: TiledImage,
}

impl CanvasSnapshot {
    pub fn capture(state: &CanvasState) -> Self {
        Self {
            width: state.width,
            height: state.height,
            active_layer_index: state.active_layer_index,
            layers: state.layers.iter().map(|l| LayerSnapshot {
                name: l.name.clone(),
                visible: l.visible,
                opacity: l.opacity,
                blend_mode: l.blend_mode,
                pixels: l.pixels.clone(),
            }).collect(),
        }
    }

    pub fn restore_into(&self, state: &mut CanvasState) {
        state.width = self.width;
        state.height = self.height;
        state.active_layer_index = self.active_layer_index;
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
            layer.opacity = snap.opacity;
            layer.blend_mode = snap.blend_mode;
            state.layers.push(layer);
        }
        state.composite_cache = None;
        state.clear_preview_state();
        state.invalidate_selection_overlay();
        state.selection_overlay_texture = None;
        state.mark_dirty(None);
    }

    fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.pixels.memory_bytes() + l.name.len()).sum()
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
        self.before.memory_bytes()
            + self.after.as_ref().map_or(0, |a| a.memory_bytes())
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
    before_opacity: f32,
    after_opacity: f32,
    before_blend_mode: crate::canvas::BlendMode,
    after_blend_mode: crate::canvas::BlendMode,
}

impl SingleLayerSnapshotCommand {
    /// Create before performing the operation. Call `set_after()` when done.
    pub fn new(description: String, state: &CanvasState) -> Self {
        Self::new_for_layer(description, state, state.active_layer_index)
    }

    /// Create for a specific layer index (when the active layer may differ from
    /// the layer being modified, e.g., dialog commits store layer_idx at open time).
    pub fn new_for_layer(description: String, state: &CanvasState, layer_idx: usize) -> Self {
        let safe_idx = if state.layers.is_empty() { 0 } else { layer_idx.min(state.layers.len() - 1) };
        let (before_pixels, before_opacity, before_blend_mode) = if let Some(layer) = state.layers.get(safe_idx) {
            (layer.pixels.clone(), layer.opacity, layer.blend_mode)
        } else {
            (TiledImage::new(1, 1), 1.0, crate::canvas::BlendMode::Normal)
        };
        Self {
            description,
            layer_index: safe_idx,
            before_pixels,
            after_pixels: None,
            before_opacity,
            after_opacity: before_opacity,
            before_blend_mode,
            after_blend_mode: before_blend_mode,
        }
    }

    /// Capture the layer's state after the operation.
    pub fn set_after(&mut self, state: &CanvasState) {
        if let Some(layer) = state.layers.get(self.layer_index) {
            self.after_pixels = Some(layer.pixels.clone());
            self.after_opacity = layer.opacity;
            self.after_blend_mode = layer.blend_mode;
        }
    }
}

impl Command for SingleLayerSnapshotCommand {
    fn undo(&self, canvas: &mut CanvasState) {
        if let Some(layer) = canvas.layers.get_mut(self.layer_index) {
            layer.pixels = self.before_pixels.clone();
            layer.opacity = self.before_opacity;
            layer.blend_mode = self.before_blend_mode;
        }
        canvas.mark_dirty(None);
    }

    fn redo(&self, canvas: &mut CanvasState) {
        if let Some(ref after) = self.after_pixels {
            if let Some(layer) = canvas.layers.get_mut(self.layer_index) {
                layer.pixels = after.clone();
                layer.opacity = self.after_opacity;
                layer.blend_mode = self.after_blend_mode;
            }
        }
        canvas.mark_dirty(None);
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn memory_size(&self) -> usize {
        self.before_pixels.memory_bytes()
            + self.after_pixels.as_ref().map_or(0, |p| p.memory_bytes())
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
            ui.label(format!("Undo: {} | Redo: {}", history.undo_count(), history.redo_count()));
            
            if ui.small_button("ℹ").on_hover_text("Show memory info").clicked() {
                self.show_memory_info = !self.show_memory_info;
            }
        });
        
        if self.show_memory_info {
            let mem_usage = history.memory_usage();
            let mem_mb = mem_usage as f64 / (1024.0 * 1024.0);
            ui.label(format!("Memory: {:.2} MB", mem_mb));
        }
        
        // Show recent history entries
        egui::ScrollArea::vertical()
            .max_height(150.0)
            .show(ui, |ui| {
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
        let scroll_width = ui.available_width();
        egui::ScrollArea::vertical()
            .max_height(180.0)
            .min_scrolled_width(scroll_width)
            .show(ui, |ui| {
                ui.set_min_width(scroll_width);
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
                                let img = egui::Image::from_texture(sized).fit_to_exact_size(icon_size);
                                ui.add(img);
                            } else {
                                ui.label(egui::RichText::new(icon.emoji()).size(12.0));
                            }
                            
                            let text = if is_current {
                                egui::RichText::new(desc).strong().size(11.0)
                            } else {
                                egui::RichText::new(desc).weak().size(11.0)
                            };
                            
                            ui.add(
                                egui::Label::new(text).sense(egui::Sense::click())
                            )
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
        } else if desc_lower.contains("layer") || desc_lower.contains("duplicate") || desc_lower.contains("rename") {
            Icon::Layers
        } else if desc_lower.contains("opacity") || desc_lower.contains("visible") || desc_lower.contains("hide") || desc_lower.contains("show") {
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