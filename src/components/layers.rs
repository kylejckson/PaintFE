use crate::assets::{Assets, Icon};
use crate::canvas::{BlendMode, CanvasState, Layer, LayerContent, TiledImage};
use crate::components::history::{HistoryManager, LayerOpCommand, LayerOperation, SnapshotCommand};
use crate::ops::dialogs::DialogColors;
use crate::ops::text_layer::{
    EnvelopeWarp, InnerShadowEffect, OutlineEffect, OutlinePosition, ShadowEffect, TextEffects,
    TextWarp, TextureFillEffect,
};
use eframe::egui;
use egui::{
    Color32, ColorImage, CursorIcon, Id, Pos2, Rect, Sense, TextureHandle, TextureOptions, Vec2,
};
use image::Rgba;
use std::collections::HashMap;
use std::time::Instant;

const THUMBNAIL_SIZE: u32 = 40;
const MAX_RECOMMENDED_LAYERS: usize = 200;

struct ThumbnailCache {
    texture: Option<TextureHandle>,
    /// Canvas dirty_generation at last compute.
    last_generation: u64,
    last_update: Instant,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum LayerSettingsTab {
    #[default]
    General,
    Effects,
    Warp,
}

#[derive(Default)]
pub struct LayerSettingsState {
    pub editing_layer: Option<usize>,
    pub editing_name: String,
    pub editing_opacity: f32,
    pub editing_blend_mode: BlendMode,
    /// Active tab for the settings window.
    pub tab: LayerSettingsTab,
    /// Cloned text effects for editing (applied on "Apply").
    pub text_effects: TextEffects,
    /// Cloned text warp for editing (applied to ALL blocks live).
    pub text_warp: TextWarp,
    /// Receiver for async texture file loading.
    pub texture_load_rx: Option<std::sync::mpsc::Receiver<Vec<u8>>>,
}

/// State for peek functionality
#[derive(Default)]
pub struct PeekState {
    pub is_peeking: bool,
    pub peek_layer_index: Option<usize>,
    pub saved_visibility: Vec<bool>,
    /// True when a layer is permanently soloed (right-click peek).
    pub is_soloed: bool,
    pub solo_layer_index: Option<usize>,
    pub solo_saved_visibility: Vec<bool>,
    /// Set for one frame after a peek ends, so the `secondary_clicked` event
    /// on the same release frame doesn't accidentally toggle solo.
    peek_just_ended: bool,
}

/// State for inline rename
#[derive(Default)]
pub struct RenameState {
    pub renaming_layer: Option<usize>,
    pub rename_text: String,
    pub focus_requested: bool,
}

/// State for drag-and-drop layer reordering
#[derive(Default)]
struct DragState {
    /// Display index currently being dragged (0 = topmost in UI).
    dragging_display_idx: Option<usize>,
    drag_offset_y: f32,
    origin_display_idx: usize,
    /// Animated visual offsets per display index (elastic slide-out).
    anim_offsets: Vec<f32>,
}

/// Actions that need app-level handling (file dialogs, active dialogs, etc.)
#[derive(Debug, Clone)]
pub enum LayerAppAction {
    ImportFromFile,
    FlipHorizontal,
    FlipVertical,
    RotateScale,
    /// Merge the layer at `layer_idx` down as an alpha mask onto the layer below it.
    MergeDownAsMask(usize),
    /// Rasterize the text layer at `layer_idx`, closing the settings dialog.
    RasterizeTextLayer(usize),
}

#[derive(Default)]
pub struct LayersPanel {
    settings_state: LayerSettingsState,
    peek_state: PeekState,
    rename_state: RenameState,
    drag_state: DragState,
    thumbnail_cache: HashMap<usize, ThumbnailCache>,
    last_layer_count: usize,
    /// Layer index to remove from GPU texture cache.
    pub pending_gpu_delete: Option<usize>,
    /// When set, all GPU layer textures should be invalidated.
    pub pending_gpu_clear: bool,
    pub pending_app_action: Option<LayerAppAction>,
    search_query: String,
}

impl LayersPanel {
    /// Main show method - renders the entire layers panel
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        canvas_state: &mut CanvasState,
        assets: &Assets,
        history: &mut HistoryManager,
    ) {
        let layer_count = canvas_state.layers.len();

        // Invalidate cache if layer count changed
        if layer_count != self.last_layer_count {
            self.thumbnail_cache.clear();
            self.last_layer_count = layer_count;
        }

        // Check if we should end peek (no longer pressing)
        self.update_peek_state(ui, canvas_state);

        ui.vertical(|ui| {
            // Search bar — only shown when there are more than 2 layers
            if layer_count > 2 {
                self.show_search_bar(ui);
                ui.add_space(2.0);
            } else {
                // Clear search when layer count drops to 2 or less
                self.search_query.clear();
            }

            // Layer list with scroll area (takes most of the space)
            self.show_layer_list(ui, canvas_state, assets, history);

            ui.add_space(4.0);

            // Fixed footer toolbar
            self.show_footer_toolbar(ui, canvas_state, assets, history);
        });

        // Show layer settings popup if active
        self.show_layer_settings_popup(ui, canvas_state);
    }

    /// Show the layer search/filter bar
    fn show_search_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label("🔍");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.search_query)
                    .hint_text("Filter layers…")
                    .desired_width(ui.available_width() - 20.0)
                    .frame(true),
            );
            // Clear button — only show when there's text
            if !self.search_query.is_empty() && ui.small_button("×").clicked() {
                self.search_query.clear();
                resp.request_focus();
            }
        });
    }

    /// Render the scrollable layer list with drag-and-drop reordering
    fn show_layer_list(
        &mut self,
        ui: &mut egui::Ui,
        canvas_state: &mut CanvasState,
        assets: &Assets,
        history: &mut HistoryManager,
    ) {
        let layer_count = canvas_state.layers.len();
        if layer_count == 0 {
            return;
        }

        let row_height = 48.0;
        let row_gap = 3.0;
        let row_stride = row_height + row_gap;

        // Build filtered display list: display_idx → layer_idx mapping
        // When searching, only include layers whose name matches the query
        let is_filtering = !self.search_query.is_empty();
        let query_lower = self.search_query.to_lowercase();

        let visible_entries: Vec<(usize, usize)> = (0..layer_count)
            .map(|display_idx| (display_idx, layer_count - 1 - display_idx))
            .filter(|&(_, layer_idx)| {
                if !is_filtering {
                    return true;
                }
                canvas_state.layers[layer_idx]
                    .name
                    .to_lowercase()
                    .contains(&query_lower)
            })
            .collect();

        let visible_count = visible_entries.len();
        if visible_count == 0 {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.label(egui::RichText::new("No layers match filter").weak());
            });
            return;
        }

        // Ensure anim_offsets vec is the right size
        self.drag_state.anim_offsets.resize(layer_count, 0.0);

        // Disable drag-and-drop when filtering (reordering filtered list is confusing)
        if is_filtering {
            self.drag_state.dragging_display_idx = None;
            self.drag_state.drag_offset_y = 0.0;
        }

        // Scroll area — pinned to a fixed height that leaves room for the
        // footer toolbar, spacing, and item_spacing gaps that follow.
        //
        // We measure available_height() HERE (after the search bar was drawn)
        // and subtract a generous 60px for the footer (~36px) + add_space(4)
        // + egui item_spacing between elements (~20px combined).
        //
        // auto_shrink([false, false]) makes the scroll area always fill its
        // max_height regardless of content, so the window stays the same size
        // whether there are 1 or 100 layers.  Because the deduction (60px)
        // is larger than the actual overhead (~44px), total content is always
        // ~16px LESS than the window — a resizable egui::Window only grows
        // when content EXCEEDS stored size, so the window never grows.
        let scroll_h = (ui.available_height() - 60.0).max(80.0);
        egui::ScrollArea::vertical()
            .id_source("layer_scroll")
            .max_height(scroll_h)
            .auto_shrink([false, false])
            .show(ui, |ui: &mut egui::Ui| {
                // Reserve total space for visible rows only
                let total_h = visible_count as f32 * row_stride;
                let available_w = ui.available_width();
                let (total_rect, _) =
                    ui.allocate_exact_size(Vec2::new(available_w, total_h), Sense::hover());

                // --- Drag logic (frame-level) ---
                let pointer_pos = ui.input(|i| i.pointer.latest_pos());
                let pointer_down = ui.input(|i| i.pointer.primary_down());
                let drag_delta_y = ui.input(|i| i.pointer.delta().y);

                // Determine which display_idx the pointer would be over
                let _hover_display_idx: Option<usize> = pointer_pos.map(|p| {
                    let relative_y = p.y - total_rect.top();
                    ((relative_y / row_stride).floor() as usize).min(layer_count.saturating_sub(1))
                });

                // If dragging, update offset and compute target slot
                let mut drop_target: Option<usize> = None; // display index to drop before/at

                if let Some(drag_didx) = self.drag_state.dragging_display_idx {
                    if pointer_down {
                        // Accumulate drag offset
                        self.drag_state.drag_offset_y += drag_delta_y;

                        // Compute target display index from current drag position
                        let origin_center_y = self.drag_state.origin_display_idx as f32
                            * row_stride
                            + row_stride * 0.5;
                        let dragged_center_y = origin_center_y + self.drag_state.drag_offset_y;
                        let target = ((dragged_center_y / row_stride).floor() as usize)
                            .min(layer_count.saturating_sub(1));
                        drop_target = Some(target);

                        // Set grab cursor
                        ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                        // Keep repainting during drag for smooth animation
                        ui.ctx().request_repaint();
                    } else {
                        // Mouse released — commit the reorder
                        let origin_center_y = self.drag_state.origin_display_idx as f32
                            * row_stride
                            + row_stride * 0.5;
                        let dragged_center_y = origin_center_y + self.drag_state.drag_offset_y;
                        let target = ((dragged_center_y / row_stride).floor() as usize)
                            .min(layer_count.saturating_sub(1));

                        if target != drag_didx {
                            // Convert display indices to layer indices (display is reversed)
                            let from_layer_idx = layer_count - 1 - drag_didx;
                            let to_layer_idx = layer_count - 1 - target;
                            self.move_layer(from_layer_idx, to_layer_idx, canvas_state, history);
                        }

                        // Reset drag state
                        self.drag_state.dragging_display_idx = None;
                        self.drag_state.drag_offset_y = 0.0;
                        for v in self.drag_state.anim_offsets.iter_mut() {
                            *v = 0.0;
                        }
                    }
                }

                // Compute animated offsets for non-dragged rows (elastic slide)
                if let (Some(drag_didx), Some(target)) =
                    (self.drag_state.dragging_display_idx, drop_target)
                {
                    let direction = if target > drag_didx { 1 } else { -1i32 };
                    let range_lo = drag_didx.min(target);
                    let range_hi = drag_didx.max(target);

                    for i in 0..layer_count {
                        let target_offset = if i == drag_didx {
                            0.0 // dragged row uses raw drag_offset_y
                        } else if i > range_lo && i <= range_hi && direction == 1 {
                            // Rows between origin and drop target slide up
                            -row_stride
                        } else if i >= range_lo && i < range_hi && direction == -1 {
                            // Rows between drop target and origin slide down
                            row_stride
                        } else {
                            0.0
                        };
                        // Smooth interpolation toward target
                        let speed = 0.25;
                        self.drag_state.anim_offsets[i] +=
                            (target_offset - self.drag_state.anim_offsets[i]) * speed;
                        // Snap when close
                        if (self.drag_state.anim_offsets[i] - target_offset).abs() < 0.5 {
                            self.drag_state.anim_offsets[i] = target_offset;
                        }
                    }
                } else {
                    // No drag — decay all offsets to 0
                    for v in self.drag_state.anim_offsets.iter_mut() {
                        *v *= 0.7;
                        if v.abs() < 0.5 {
                            *v = 0.0;
                        }
                    }
                }

                // --- Collect deferred actions ---
                let mut layer_to_merge: Option<usize> = None;
                let mut layer_to_flatten = false;
                let mut layer_to_add = false;
                let mut layer_to_add_text = false;
                let mut layer_to_duplicate: Option<usize> = None;
                let mut layer_to_delete: Option<usize> = None;
                let mut layer_to_rasterize: Option<usize> = None;
                let mut new_active: Option<usize> = None;
                let mut swap_layers: Option<(usize, usize)> = None;

                // --- Draw rows ---
                let is_dragging = self.drag_state.dragging_display_idx.is_some();

                for (vis_idx, &(display_idx, layer_idx)) in visible_entries.iter().enumerate() {
                    let is_dragged = self.drag_state.dragging_display_idx == Some(display_idx);

                    // Compute visual Y position — use vis_idx for layout when filtering
                    let base_y = total_rect.top() + vis_idx as f32 * row_stride;
                    let visual_y = if is_dragged {
                        base_y + self.drag_state.drag_offset_y
                    } else if !is_filtering {
                        base_y + self.drag_state.anim_offsets[display_idx]
                    } else {
                        base_y
                    };

                    let row_rect = Rect::from_min_size(
                        Pos2::new(total_rect.left(), visual_y),
                        Vec2::new(available_w, row_height),
                    );

                    // Draw the row with an overlay for the dragged item
                    let paint_layer = if is_dragged {
                        // Paint dragged row on a higher layer so it renders on top
                        ui.painter().clone().with_layer_id(egui::LayerId::new(
                            egui::Order::Tooltip,
                            Id::new("drag_layer"),
                        ))
                    } else {
                        ui.painter().clone()
                    };

                    let (action, context_action) = self.show_layer_row_at(
                        ui,
                        &paint_layer,
                        row_rect,
                        layer_idx,
                        display_idx,
                        canvas_state,
                        assets,
                        is_dragged,
                        is_dragging,
                    );

                    // Handle row click actions
                    if let Some(act) = action {
                        match act {
                            LayerAction::Select => new_active = Some(layer_idx),
                            LayerAction::StartRename => {
                                self.rename_state.renaming_layer = Some(layer_idx);
                                self.rename_state.rename_text =
                                    canvas_state.layers[layer_idx].name.clone();
                                self.rename_state.focus_requested = true;
                            }
                            LayerAction::FinishRename => {
                                if let Some(rename_idx) = self.rename_state.renaming_layer
                                    && !self.rename_state.rename_text.is_empty()
                                {
                                    let old_name = canvas_state.layers[rename_idx].name.clone();
                                    let new_name = self.rename_state.rename_text.clone();
                                    if old_name != new_name {
                                        canvas_state.layers[rename_idx].name = new_name.clone();
                                        history.push(Box::new(LayerOpCommand::new(
                                            LayerOperation::Rename {
                                                index: rename_idx,
                                                old_name,
                                                new_name,
                                            },
                                        )));
                                    }
                                }
                                self.rename_state.renaming_layer = None;
                            }
                            LayerAction::CancelRename => {
                                self.rename_state.renaming_layer = None;
                            }
                            LayerAction::ToggleVisibility => {
                                let was_visible = canvas_state.layers[layer_idx].visible;
                                canvas_state.layers[layer_idx].visible = !was_visible;
                                history.push(Box::new(LayerOpCommand::new(
                                    LayerOperation::Visibility {
                                        index: layer_idx,
                                        was_visible,
                                    },
                                )));
                                self.mark_full_dirty(canvas_state);
                            }
                            LayerAction::BeginDrag => {
                                self.drag_state.dragging_display_idx = Some(display_idx);
                                self.drag_state.origin_display_idx = display_idx;
                                self.drag_state.drag_offset_y = 0.0;
                                for v in self.drag_state.anim_offsets.iter_mut() {
                                    *v = 0.0;
                                }
                            }
                        }
                    }

                    // Handle context menu actions
                    if let Some(ctx_act) = context_action {
                        match ctx_act {
                            ContextAction::AddNew => layer_to_add = true,
                            ContextAction::MergeDown => layer_to_merge = Some(layer_idx),
                            ContextAction::MergeDownAsMask => {
                                self.pending_app_action =
                                    Some(LayerAppAction::MergeDownAsMask(layer_idx));
                            }
                            ContextAction::FlattenImage => layer_to_flatten = true,
                            ContextAction::Duplicate => layer_to_duplicate = Some(layer_idx),
                            ContextAction::Delete => layer_to_delete = Some(layer_idx),
                            ContextAction::OpenSettings => {
                                self.open_settings_for_layer(
                                    layer_idx,
                                    canvas_state,
                                    LayerSettingsTab::General,
                                );
                            }
                            ContextAction::MoveToTop => {
                                let top = canvas_state.layers.len().saturating_sub(1);
                                if layer_idx < top {
                                    // Use move_layer which handles remove+insert, active index, and history
                                    swap_layers = Some((layer_idx, top));
                                }
                            }
                            ContextAction::MoveUp => {
                                if layer_idx + 1 < canvas_state.layers.len() {
                                    swap_layers = Some((layer_idx, layer_idx + 1));
                                }
                            }
                            ContextAction::MoveDown => {
                                if layer_idx > 0 {
                                    swap_layers = Some((layer_idx, layer_idx - 1));
                                }
                            }
                            ContextAction::MoveToBottom => {
                                if layer_idx > 0 {
                                    // Use move_layer which handles remove+insert, active index, and history
                                    swap_layers = Some((layer_idx, 0));
                                }
                            }
                            ContextAction::Rename => {
                                self.rename_state.renaming_layer = Some(layer_idx);
                                self.rename_state.rename_text =
                                    canvas_state.layers[layer_idx].name.clone();
                                self.rename_state.focus_requested = true;
                            }
                            ContextAction::ImportFromFile => {
                                self.pending_app_action = Some(LayerAppAction::ImportFromFile);
                            }
                            ContextAction::FlipHorizontal => {
                                self.pending_app_action = Some(LayerAppAction::FlipHorizontal);
                            }
                            ContextAction::FlipVertical => {
                                self.pending_app_action = Some(LayerAppAction::FlipVertical);
                            }
                            ContextAction::RotateScale => {
                                self.pending_app_action = Some(LayerAppAction::RotateScale);
                            }
                            ContextAction::SoloLayer => {
                                self.solo_layer(layer_idx, canvas_state);
                            }
                            ContextAction::HideAll => {
                                self.hide_all_layers(canvas_state);
                            }
                            ContextAction::ShowAll => {
                                self.show_all_layers(canvas_state);
                            }
                            ContextAction::AddNewTextLayer => {
                                layer_to_add_text = true;
                            }
                            ContextAction::RasterizeTextLayer => {
                                layer_to_rasterize = Some(layer_idx);
                            }
                            ContextAction::TextLayerEffects => {
                                self.open_settings_for_layer(
                                    layer_idx,
                                    canvas_state,
                                    LayerSettingsTab::Effects,
                                );
                            }
                            ContextAction::TextLayerWarp => {
                                self.open_settings_for_layer(
                                    layer_idx,
                                    canvas_state,
                                    LayerSettingsTab::Warp,
                                );
                            }
                        }
                    }
                }

                // Draw drop indicator line
                if let (Some(_drag_didx), Some(target)) =
                    (self.drag_state.dragging_display_idx, drop_target)
                {
                    let indicator_y =
                        total_rect.top() + target as f32 * row_stride + row_stride * 0.5;
                    let accent = ui.visuals().selection.bg_fill;
                    ui.painter().line_segment(
                        [
                            Pos2::new(total_rect.left() + 4.0, indicator_y),
                            Pos2::new(total_rect.right() - 4.0, indicator_y),
                        ],
                        egui::Stroke::new(2.0, accent),
                    );
                }

                // Process deferred actions
                if let Some(idx) = new_active {
                    canvas_state.active_layer_index = idx;
                }
                if let Some(merge_idx) = layer_to_merge {
                    self.merge_down(merge_idx, canvas_state, history);
                }
                if layer_to_flatten {
                    self.flatten_image(canvas_state, history);
                }
                if layer_to_add {
                    self.add_new_layer(canvas_state, history);
                }
                if layer_to_add_text {
                    self.add_new_text_layer(canvas_state, history);
                }
                if let Some(dup_idx) = layer_to_duplicate {
                    self.duplicate_layer(dup_idx, canvas_state, history);
                }
                if let Some(del_idx) = layer_to_delete {
                    self.delete_layer(del_idx, canvas_state, history);
                }
                if let Some(rast_idx) = layer_to_rasterize {
                    self.rasterize_text_layer(rast_idx, canvas_state, history);
                }
                if let Some((from, to)) = swap_layers {
                    self.move_layer(from, to, canvas_state, history);
                }
            });
    }

    /// Render a single layer row at an explicit rect (supports drag offset)
    fn show_layer_row_at(
        &mut self,
        ui: &mut egui::Ui,
        painter: &egui::Painter,
        row_rect: Rect,
        layer_idx: usize,
        display_idx: usize,
        canvas_state: &mut CanvasState,
        assets: &Assets,
        is_dragged: bool,
        is_any_dragging: bool,
    ) -> (Option<LayerAction>, Option<ContextAction>) {
        // Copy the values we need from the layer to avoid borrow conflicts
        let layer_visible = canvas_state.layers[layer_idx].visible;
        let layer_name = canvas_state.layers[layer_idx].name.clone();
        let is_active = layer_idx == canvas_state.active_layer_index;
        let is_renaming = self.rename_state.renaming_layer == Some(layer_idx);

        let mut action: Option<LayerAction> = None;
        let mut context_action: Option<ContextAction> = None;
        let mut should_peek = false;

        let row_height = row_rect.height();
        let selection_color = ui.visuals().selection.bg_fill;
        let row_bg = if is_active {
            selection_color
        } else {
            Color32::TRANSPARENT
        };

        // Interact with the row at its visual position
        let row_id = Id::new("layer_row").with(display_idx);
        let row_response = ui.interact(row_rect, row_id, Sense::click_and_drag());

        // Drag initiation — only when not already dragging and not renaming
        if !is_any_dragging && !is_renaming && row_response.drag_started() {
            action = Some(LayerAction::BeginDrag);
        }

        // Hover cursor: show grab hand when hovering a row (but not when dragging)
        if !is_any_dragging && row_response.hovered() {
            ui.ctx().set_cursor_icon(CursorIcon::Grab);
        }

        // Paint row background
        if ui.is_rect_visible(row_rect) {
            // Row background - shrink vertically to avoid overlap
            let shrink = row_height * 0.09;
            let bg_rect = row_rect.shrink2(Vec2::new(0.0, shrink));

            if is_dragged {
                // Dragged item: elevated shadow + accent border
                let shadow_color = Color32::from_black_alpha(60);
                let shadow_rect = bg_rect.translate(Vec2::new(0.0, 2.0));
                painter.rect_filled(shadow_rect, 6.0, shadow_color);

                // Slightly brighter background for dragged item
                let drag_bg = if is_active {
                    selection_color
                } else {
                    ui.visuals().widgets.active.bg_fill
                };
                painter.rect_filled(bg_rect, 4.0, drag_bg);
                painter.rect_stroke(bg_rect, 4.0, egui::Stroke::new(1.5, selection_color));
            } else {
                painter.rect_filled(bg_rect, 4.0, row_bg);
            }

            // Layout: [Eye] [Thumbnail] [Name]
            let mut x = row_rect.left() + 4.0;
            let center_y = row_rect.center().y;

            // Pre-calculate all rects
            let eye_rect = Rect::from_center_size(Pos2::new(x + 10.0, center_y), Vec2::splat(20.0));
            x += 24.0;

            let thumb_size = 36.0;
            let thumb_rect = Rect::from_min_size(
                Pos2::new(x + 2.0, center_y - thumb_size / 2.0),
                Vec2::splat(thumb_size),
            );
            x += thumb_size + 8.0;

            let is_text_layer = matches!(
                canvas_state.layers[layer_idx].content,
                LayerContent::Text(_)
            );
            let gear_width = if is_text_layer { 20.0 } else { 0.0 };
            let name_rect = Rect::from_min_max(
                Pos2::new(x, row_rect.top() + 4.0),
                Pos2::new(row_rect.right() - 6.0 - gear_width, row_rect.bottom() - 4.0),
            );

            // Small gear icon for text layer settings (right side of row)
            if is_text_layer {
                let gear_rect = Rect::from_center_size(
                    Pos2::new(row_rect.right() - 6.0 - gear_width / 2.0, center_y),
                    Vec2::splat(16.0),
                );
                let gear_color = if gear_rect
                    .contains(ui.input(|i| i.pointer.hover_pos().unwrap_or_default()))
                {
                    ui.visuals().strong_text_color()
                } else {
                    ui.visuals().text_color()
                };
                ui.painter().text(
                    gear_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "\u{2699}",
                    egui::FontId::proportional(14.0),
                    gear_color,
                );
                let gear_resp =
                    ui.interact(gear_rect, Id::new(("text_gear", layer_idx)), Sense::click());
                if gear_resp.clicked() {
                    context_action = Some(ContextAction::TextLayerEffects);
                }
                gear_resp.on_hover_text(t!("layer.text_settings"));
            }

            // Draw thumbnail (needs mutable self for cache)
            self.draw_thumbnail(ui, thumb_rect, layer_idx, canvas_state, 1.0);

            // Draw eye icon (left-click: toggle visibility, right-click hold: peek layer)
            let icon_color = ui.visuals().strong_text_color();
            let muted_color = ui.visuals().text_color();
            let is_this_soloed =
                self.peek_state.is_soloed && self.peek_state.solo_layer_index == Some(layer_idx);
            let eye_icon = if layer_visible {
                Icon::Visible
            } else {
                Icon::Hidden
            };
            let eye_tint = if is_this_soloed {
                Color32::from_rgb(255, 180, 60) // Orange tint when soloed
            } else if layer_visible {
                icon_color
            } else {
                muted_color
            };
            let eye_response = assets.icon_in_rect(ui, eye_icon, eye_rect, eye_tint);
            if eye_response.clicked() {
                action = Some(LayerAction::ToggleVisibility);
            }
            // Right-click hold: temporary peek (show only this layer while held)
            if eye_response.is_pointer_button_down_on()
                && ui.input(|i| i.pointer.button_down(egui::PointerButton::Secondary))
            {
                should_peek = true;
            }
            // Right-click release (secondary_clicked): toggle solo —
            // but only if we weren't just peeking (peek_just_ended suppresses
            // the solo toggle on the same frame the hold was released).
            if eye_response.secondary_clicked() && !self.peek_state.peek_just_ended {
                if is_this_soloed {
                    context_action = Some(ContextAction::ShowAll);
                } else {
                    context_action = Some(ContextAction::SoloLayer);
                }
            }
            if is_this_soloed {
                eye_response.on_hover_text("Soloed — right-click to unsolo");
            } else {
                eye_response.on_hover_text(if layer_visible {
                    "Hide layer · Right-click hold to peek"
                } else {
                    "Show layer · Right-click hold to peek"
                });
            }

            // Draw name or rename field
            if is_renaming {
                let text_edit = egui::TextEdit::singleline(&mut self.rename_state.rename_text)
                    .font(egui::TextStyle::Body)
                    .desired_width(name_rect.width());

                let response = ui.put(name_rect, text_edit);

                if self.rename_state.focus_requested {
                    response.request_focus();
                    self.rename_state.focus_requested = false;
                }

                if response.lost_focus() {
                    if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        action = Some(LayerAction::FinishRename);
                    } else if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        action = Some(LayerAction::CancelRename);
                    } else {
                        action = Some(LayerAction::FinishRename);
                    }
                }
            } else {
                let is_text = matches!(
                    canvas_state.layers[layer_idx].content,
                    LayerContent::Text(_)
                );
                let mut name_text =
                    egui::RichText::new(&layer_name)
                        .size(13.0)
                        .color(if is_active {
                            ui.visuals().strong_text_color()
                        } else {
                            icon_color
                        });
                if is_text {
                    name_text = name_text.strong();
                }

                let mut child_ui =
                    ui.child_ui(name_rect, egui::Layout::top_down(egui::Align::LEFT));
                child_ui.spacing_mut().item_spacing.y = 0.0;
                // Vertically center the content block within the row
                let content_h = if is_text { 13.0 + 9.0 + 1.0 } else { 13.0 };
                let pad = ((name_rect.height() - content_h) / 2.0).max(0.0);
                if pad > 0.0 {
                    child_ui.add_space(pad);
                }
                child_ui.add(egui::Label::new(name_text).truncate(true));
                if is_text {
                    let accent = child_ui.visuals().selection.stroke.color;
                    child_ui.add(egui::Label::new(
                        egui::RichText::new("TEXT LAYER")
                            .size(9.0)
                            .strong()
                            .color(accent),
                    ));
                }
            }

            // Now handle peek (after all layer borrows are done)
            if should_peek {
                self.start_peek(layer_idx, canvas_state);
            }
        }

        // Row click handling (select layer) — only when not dragging
        if !is_any_dragging && row_response.clicked() && action.is_none() {
            action = Some(LayerAction::Select);
        }

        // Double-click to rename
        if !is_any_dragging && row_response.double_clicked() {
            action = Some(LayerAction::StartRename);
        }

        // Right-click context menu
        row_response.context_menu(|ui| {
            if assets
                .menu_item(ui, Icon::LayerAdd, &t!("layer.add_layer"))
                .clicked()
            {
                context_action = Some(ContextAction::AddNew);
                ui.close_menu();
            }
            if assets
                .menu_item(ui, Icon::Rename, &t!("layer.add_text_layer"))
                .clicked()
            {
                context_action = Some(ContextAction::AddNewTextLayer);
                ui.close_menu();
            }
            if assets
                .menu_item(ui, Icon::LayerDuplicate, &t!("layer.duplicate_layer"))
                .clicked()
            {
                context_action = Some(ContextAction::Duplicate);
                ui.close_menu();
            }
            if canvas_state.layers.len() > 1
                && assets
                    .menu_item(ui, Icon::LayerDelete, &t!("layer.delete_layer"))
                    .clicked()
            {
                context_action = Some(ContextAction::Delete);
                ui.close_menu();
            }
            ui.separator();
            if layer_idx > 0 {
                if assets
                    .menu_item(ui, Icon::MergeDown, &t!("layer.merge_down"))
                    .clicked()
                {
                    context_action = Some(ContextAction::MergeDown);
                    ui.close_menu();
                }
                if assets
                    .menu_item(ui, Icon::MergeDownAsMask, &t!("layer.merge_down_as_mask"))
                    .clicked()
                {
                    context_action = Some(ContextAction::MergeDownAsMask);
                    ui.close_menu();
                }
            }
            if canvas_state.layers.len() > 1
                && assets
                    .menu_item(ui, Icon::Flatten, &t!("layer.flatten_all"))
                    .clicked()
            {
                context_action = Some(ContextAction::FlattenImage);
                ui.close_menu();
            }
            ui.separator();
            // Move submenu
            let layer_count = canvas_state.layers.len();
            let can_up = layer_idx + 1 < layer_count;
            let can_down = layer_idx > 0;
            if assets
                .menu_item_enabled(ui, Icon::MoveTop, &t!("layer.move_to_top"), can_up)
                .clicked()
            {
                context_action = Some(ContextAction::MoveToTop);
                ui.close_menu();
            }
            if assets
                .menu_item_enabled(ui, Icon::MoveUp, &t!("layer.move_up"), can_up)
                .clicked()
            {
                context_action = Some(ContextAction::MoveUp);
                ui.close_menu();
            }
            if assets
                .menu_item_enabled(ui, Icon::MoveDown, &t!("layer.move_down"), can_down)
                .clicked()
            {
                context_action = Some(ContextAction::MoveDown);
                ui.close_menu();
            }
            if assets
                .menu_item_enabled(ui, Icon::MoveBottom, &t!("layer.move_to_bottom"), can_down)
                .clicked()
            {
                context_action = Some(ContextAction::MoveToBottom);
                ui.close_menu();
            }
            ui.separator();
            if assets
                .menu_item(ui, Icon::ImportLayer, &t!("layer.import_from_file"))
                .clicked()
            {
                context_action = Some(ContextAction::ImportFromFile);
                ui.close_menu();
            }
            ui.separator();
            ui.menu_button(t!("layer.transform"), |ui| {
                if assets
                    .menu_item(ui, Icon::LayerFlipH, &t!("layer.transform.flip_horizontal"))
                    .clicked()
                {
                    context_action = Some(ContextAction::FlipHorizontal);
                    ui.close_menu();
                }
                if assets
                    .menu_item(ui, Icon::LayerFlipV, &t!("layer.transform.flip_vertical"))
                    .clicked()
                {
                    context_action = Some(ContextAction::FlipVertical);
                    ui.close_menu();
                }
                ui.separator();
                if assets
                    .menu_item(ui, Icon::LayerRotate, &t!("layer.transform.rotate_scale"))
                    .clicked()
                {
                    context_action = Some(ContextAction::RotateScale);
                    ui.close_menu();
                }
            });
            ui.separator();
            // Visibility group: Solo / Hide All / Show All
            {
                let is_this_soloed = self.peek_state.is_soloed
                    && self.peek_state.solo_layer_index == Some(layer_idx);
                let solo_label = if is_this_soloed {
                    t!("layer.unsolo_layer")
                } else {
                    t!("layer.solo_layer")
                };
                if assets.menu_item(ui, Icon::SoloLayer, &solo_label).clicked() {
                    if is_this_soloed {
                        context_action = Some(ContextAction::ShowAll);
                    } else {
                        context_action = Some(ContextAction::SoloLayer);
                    }
                    ui.close_menu();
                }
                if assets
                    .menu_item(ui, Icon::HideAll, &t!("layer.hide_all"))
                    .clicked()
                {
                    context_action = Some(ContextAction::HideAll);
                    ui.close_menu();
                }
                if assets
                    .menu_item(ui, Icon::ShowAll, &t!("layer.show_all"))
                    .clicked()
                {
                    context_action = Some(ContextAction::ShowAll);
                    ui.close_menu();
                }
            }
            ui.separator();
            if assets
                .menu_item(ui, Icon::Rename, &t!("layer.rename_layer"))
                .clicked()
            {
                context_action = Some(ContextAction::Rename);
                ui.close_menu();
            }
            if assets
                .menu_item(ui, Icon::LayerProperties, &t!("layer.layer_properties"))
                .clicked()
            {
                context_action = Some(ContextAction::OpenSettings);
                ui.close_menu();
            }
            // Rasterize option for text layers + effects/warp
            if matches!(
                canvas_state.layers[layer_idx].content,
                LayerContent::Text(_)
            ) {
                ui.separator();
                if assets
                    .menu_item(ui, Icon::LayerProperties, &t!("layer.text_effects"))
                    .clicked()
                {
                    context_action = Some(ContextAction::TextLayerEffects);
                    ui.close_menu();
                }
                if assets
                    .menu_item(ui, Icon::LayerProperties, &t!("layer.text_warp"))
                    .clicked()
                {
                    context_action = Some(ContextAction::TextLayerWarp);
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .add(egui::Button::new(t!("layer.rasterize_text_layer")))
                    .clicked()
                {
                    context_action = Some(ContextAction::RasterizeTextLayer);
                    ui.close_menu();
                }
            }
        });

        (action, context_action)
    }

    /// Draw layer thumbnail with checkerboard background
    fn draw_thumbnail(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        layer_idx: usize,
        canvas_state: &CanvasState,
        alpha: f32,
    ) {
        // Get or create cached thumbnail texture first (requires mutable ui borrow)
        let texture = self.get_or_create_thumbnail(ui, layer_idx, canvas_state);

        // Now get painter for drawing (immutable borrow)
        let painter = ui.painter();

        // Draw checkerboard background for transparency
        let is_dark = ui.visuals().dark_mode;
        let grid_size = 6.0;
        let (light, dark) = if is_dark {
            (Color32::from_gray(60), Color32::from_gray(40))
        } else {
            (Color32::from_gray(240), Color32::from_gray(200))
        };

        let mut y = rect.top();
        let mut row = 0;
        while y < rect.bottom() {
            let mut x = rect.left();
            let mut col = 0;
            while x < rect.right() {
                let cell_rect = Rect::from_min_size(
                    Pos2::new(x, y),
                    Vec2::new(
                        (rect.right() - x).min(grid_size),
                        (rect.bottom() - y).min(grid_size),
                    ),
                );
                let color = if (row + col) % 2 == 0 { light } else { dark };
                painter.rect_filled(cell_rect, 0.0, color);
                x += grid_size;
                col += 1;
            }
            y += grid_size;
            row += 1;
        }

        // Draw border
        let border_color = if is_dark {
            Color32::from_gray(80)
        } else {
            Color32::from_gray(180)
        };
        painter.rect_stroke(rect, 2.0, egui::Stroke::new(1.0, border_color));

        // Draw cached thumbnail texture
        if let Some(texture) = texture {
            let tint = Color32::from_rgba_unmultiplied(255, 255, 255, (alpha * 255.0) as u8);
            painter.image(
                texture.id(),
                rect.shrink(1.0),
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                tint,
            );
        }
    }

    /// Get or create a cached thumbnail texture for a layer
    /// Uses canvas dirty_generation + time throttle (500ms) to avoid excessive recomputation
    fn get_or_create_thumbnail(
        &mut self,
        ui: &mut egui::Ui,
        layer_idx: usize,
        canvas_state: &CanvasState,
    ) -> Option<TextureHandle> {
        let current_gen = canvas_state.dirty_generation;
        let now = Instant::now();

        // Check cache: reuse if generation hasn't changed, or if changed but too recently updated
        if let Some(cache) = self.thumbnail_cache.get(&layer_idx) {
            if cache.last_generation == current_gen {
                // Nothing changed since last thumbnail
                return cache.texture.clone();
            }
            // Generation changed — only refresh if at least 500ms since last update
            if now.duration_since(cache.last_update).as_millis() < 500 {
                // Too soon, request a repaint later and return stale thumbnail
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(500));
                return cache.texture.clone();
            }
        }

        // Generate new thumbnail
        let layer = &canvas_state.layers[layer_idx];

        // Include preview layer if this is the active layer and preview exists
        let is_active = layer_idx == canvas_state.active_layer_index;
        let thumbnail_image = if is_active && canvas_state.preview_layer.is_some() {
            self.generate_thumbnail_with_preview(
                &layer.pixels,
                canvas_state.preview_layer.as_ref().unwrap(),
                canvas_state.preview_blend_mode,
            )
        } else {
            self.generate_thumbnail(&layer.pixels)
        };

        let color_image = ColorImage::from_rgba_unmultiplied(
            [THUMBNAIL_SIZE as usize, THUMBNAIL_SIZE as usize],
            &thumbnail_image,
        );

        let texture = ui.ctx().load_texture(
            format!("layer_thumb_{}", layer_idx),
            color_image,
            TextureOptions::LINEAR,
        );

        // Cache it
        self.thumbnail_cache.insert(
            layer_idx,
            ThumbnailCache {
                texture: Some(texture.clone()),
                last_generation: current_gen,
                last_update: now,
            },
        );

        Some(texture)
    }

    /// Generate a downscaled thumbnail from full-size layer.
    /// Uses uniform scaling to preserve aspect ratio (letterbox/pillarbox).
    fn generate_thumbnail(&self, source: &TiledImage) -> Vec<u8> {
        let src_w = source.width() as f32;
        let src_h = source.height() as f32;
        let dst_size = THUMBNAIL_SIZE as f32;

        // Uniform scale: fit the largest dimension into THUMBNAIL_SIZE
        let scale = dst_size / src_w.max(src_h);
        let fit_w = (src_w * scale).round() as u32;
        let fit_h = (src_h * scale).round() as u32;
        // Offset to center the image within the square thumbnail
        let off_x = (THUMBNAIL_SIZE.saturating_sub(fit_w)) / 2;
        let off_y = (THUMBNAIL_SIZE.saturating_sub(fit_h)) / 2;

        let mut result = vec![0u8; (THUMBNAIL_SIZE * THUMBNAIL_SIZE * 4) as usize];

        for y in 0..THUMBNAIL_SIZE {
            for x in 0..THUMBNAIL_SIZE {
                // Only sample within the fitted region
                if x >= off_x && x < off_x + fit_w && y >= off_y && y < off_y + fit_h {
                    let local_x = x - off_x;
                    let local_y = y - off_y;
                    let src_x = ((local_x as f32 / fit_w as f32) * src_w) as u32;
                    let src_y = ((local_y as f32 / fit_h as f32) * src_h) as u32;

                    let src_x = src_x.min(source.width() - 1);
                    let src_y = src_y.min(source.height() - 1);

                    let pixel = source.get_pixel(src_x, src_y);
                    let idx = ((y * THUMBNAIL_SIZE + x) * 4) as usize;
                    result[idx] = pixel[0];
                    result[idx + 1] = pixel[1];
                    result[idx + 2] = pixel[2];
                    result[idx + 3] = pixel[3];
                }
                // Outside fitted region: stays transparent (0,0,0,0)
            }
        }

        result
    }

    /// Generate a thumbnail with preview layer composited on top
    /// Used for showing fill/brush previews in the active layer thumbnail
    fn generate_thumbnail_with_preview(
        &self,
        base_layer: &TiledImage,
        preview_layer: &TiledImage,
        blend_mode: BlendMode,
    ) -> Vec<u8> {
        let src_w = base_layer.width() as f32;
        let src_h = base_layer.height() as f32;
        let dst_size = THUMBNAIL_SIZE as f32;

        // Uniform scale: fit the largest dimension into THUMBNAIL_SIZE
        let scale = dst_size / src_w.max(src_h);
        let fit_w = (src_w * scale).round() as u32;
        let fit_h = (src_h * scale).round() as u32;
        // Offset to center the image within the square thumbnail
        let off_x = (THUMBNAIL_SIZE.saturating_sub(fit_w)) / 2;
        let off_y = (THUMBNAIL_SIZE.saturating_sub(fit_h)) / 2;

        let mut result = vec![0u8; (THUMBNAIL_SIZE * THUMBNAIL_SIZE * 4) as usize];

        for y in 0..THUMBNAIL_SIZE {
            for x in 0..THUMBNAIL_SIZE {
                // Only sample within the fitted region
                if x >= off_x && x < off_x + fit_w && y >= off_y && y < off_y + fit_h {
                    let local_x = x - off_x;
                    let local_y = y - off_y;
                    let src_x = ((local_x as f32 / fit_w as f32) * src_w) as u32;
                    let src_y = ((local_y as f32 / fit_h as f32) * src_h) as u32;

                    let src_x = src_x.min(base_layer.width() - 1);
                    let src_y = src_y.min(base_layer.height() - 1);

                    // Get base pixel
                    let base = base_layer.get_pixel(src_x, src_y);
                    // Get preview pixel
                    let preview = preview_layer.get_pixel(src_x, src_y);

                    // Composite preview on top of base using the blend mode
                    let composited = self.blend_pixels(*base, *preview, blend_mode);

                    let idx = ((y * THUMBNAIL_SIZE + x) * 4) as usize;
                    result[idx] = composited[0];
                    result[idx + 1] = composited[1];
                    result[idx + 2] = composited[2];
                    result[idx + 3] = composited[3];
                }
                // Outside fitted region: stays transparent (0,0,0,0)
            }
        }

        result
    }

    /// Simple alpha blend for thumbnail preview (Normal blend mode only for simplicity)
    /// Both inputs are in straight alpha format (RGB not premultiplied)
    fn blend_pixels(&self, base: Rgba<u8>, overlay: Rgba<u8>, _blend_mode: BlendMode) -> Rgba<u8> {
        // If overlay is fully transparent, return base
        if overlay[3] == 0 {
            return base;
        }

        // If base is fully transparent, return overlay
        if base[3] == 0 {
            return overlay;
        }

        // Convert to float for blending
        let base_a = base[3] as f32 / 255.0;
        let overlay_a = overlay[3] as f32 / 255.0;

        // Straight alpha "over" compositing formula:
        // result_rgb = overlay_rgb * overlay_a + base_rgb * (1 - overlay_a)
        // result_a = overlay_a + base_a * (1 - overlay_a)
        let one_minus_overlay_a = 1.0 - overlay_a;

        let result_r = (overlay[0] as f32 * overlay_a + base[0] as f32 * one_minus_overlay_a)
            .clamp(0.0, 255.0) as u8;
        let result_g = (overlay[1] as f32 * overlay_a + base[1] as f32 * one_minus_overlay_a)
            .clamp(0.0, 255.0) as u8;
        let result_b = (overlay[2] as f32 * overlay_a + base[2] as f32 * one_minus_overlay_a)
            .clamp(0.0, 255.0) as u8;
        let result_a = (overlay_a + base_a * one_minus_overlay_a).clamp(0.0, 1.0);
        let result_a_u8 = (result_a * 255.0) as u8;

        Rgba([result_r, result_g, result_b, result_a_u8])
    }

    /// Show the footer toolbar with layer action buttons
    fn show_footer_toolbar(
        &mut self,
        ui: &mut egui::Ui,
        canvas_state: &mut CanvasState,
        assets: &Assets,
        history: &mut HistoryManager,
    ) {
        ui.separator();

        // All actions in one row: New, Delete, Duplicate, Merge, Flatten, Options + count
        ui.horizontal(|ui| {
            // New Layer
            if assets.small_icon_button(ui, Icon::NewLayer).clicked() {
                self.add_new_layer(canvas_state, history);
            }

            // Delete Layer
            let can_delete = canvas_state.layers.len() > 1;
            if assets
                .icon_button_enabled(ui, Icon::Delete, can_delete)
                .clicked()
                && can_delete
            {
                self.delete_active_layer(canvas_state, history);
            }

            // Duplicate Layer
            if assets.small_icon_button(ui, Icon::Duplicate).clicked() {
                self.duplicate_layer(canvas_state.active_layer_index, canvas_state, history);
            }

            // Merge Down
            let can_merge = canvas_state.active_layer_index > 0;
            if assets
                .icon_button_enabled(ui, Icon::MergeDown, can_merge)
                .clicked()
                && can_merge
            {
                self.merge_down(canvas_state.active_layer_index, canvas_state, history);
            }

            // Flatten Image
            let can_flatten = canvas_state.layers.len() > 1;
            if assets
                .icon_button_enabled(ui, Icon::Flatten, can_flatten)
                .clicked()
                && can_flatten
            {
                self.flatten_image(canvas_state, history);
            }

            // Layer Options (settings)
            if assets.small_icon_button(ui, Icon::Settings).clicked() {
                let idx = canvas_state.active_layer_index;
                self.open_settings_for_layer(idx, canvas_state, LayerSettingsTab::General);
            }
        });
    }

    /// Open the layer settings dialog for a given layer, starting on the given tab.
    fn open_settings_for_layer(
        &mut self,
        idx: usize,
        canvas_state: &CanvasState,
        tab: LayerSettingsTab,
    ) {
        if idx >= canvas_state.layers.len() {
            return;
        }
        let layer = &canvas_state.layers[idx];
        self.settings_state.editing_layer = Some(idx);
        self.settings_state.editing_name = layer.name.clone();
        self.settings_state.editing_opacity = layer.opacity;
        self.settings_state.editing_blend_mode = layer.blend_mode;
        self.settings_state.tab = tab;
        self.settings_state.texture_load_rx = None;

        // Load text effects / warp if this is a text layer
        if let LayerContent::Text(ref td) = layer.content {
            self.settings_state.text_effects = td.effects.clone();
            // Load warp from the first block (or default)
            if let Some(block) = td.blocks.first() {
                self.settings_state.text_warp = block.warp.clone();
            } else {
                self.settings_state.text_warp = TextWarp::None;
            }
        } else {
            self.settings_state.text_effects = TextEffects::default();
            self.settings_state.text_warp = TextWarp::None;
        }
    }

    /// Show the layer settings popup window (Options menu).
    /// For text layers this includes Effects and Warp tabs.
    fn show_layer_settings_popup(&mut self, ui: &mut egui::Ui, canvas_state: &mut CanvasState) {
        if let Some(layer_idx) = self.settings_state.editing_layer {
            if layer_idx >= canvas_state.layers.len() {
                self.settings_state.editing_layer = None;
                return;
            }

            let is_text = matches!(
                canvas_state.layers[layer_idx].content,
                LayerContent::Text(_)
            );

            let mut open = true;
            let title = if is_text {
                t!("layer.text_options_title")
            } else {
                t!("layer.options_title")
            };
            let win_width = if is_text { 340.0 } else { 280.0 };

            egui::Window::new("layer_settings_popup_win")
                .id(Id::new("layer_settings_popup"))
                .title_bar(false)
                .collapsible(false)
                .resizable(true)
                .default_width(win_width)
                .show(ui.ctx(), |ui| {
                    // Header with close button
                    let colors = DialogColors::from_ctx(ui.ctx());
                    let available_w = ui.available_width();
                    let header_height = 32.0;
                    let (header_rect, _) = ui
                        .allocate_exact_size(Vec2::new(available_w, header_height), Sense::hover());
                    let painter = ui.painter();
                    painter.rect_filled(
                        header_rect,
                        egui::Rounding::same(4.0),
                        colors.accent_faint,
                    );
                    painter.rect_filled(
                        Rect::from_min_size(header_rect.min, Vec2::new(3.0, header_height)),
                        egui::Rounding::ZERO,
                        colors.accent,
                    );
                    painter.text(
                        Pos2::new(header_rect.min.x + 12.0, header_rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        format!("\u{2699} {title}"),
                        egui::FontId::proportional(14.0),
                        colors.accent_strong,
                    );
                    // Close button (×) — matches panel_header ghost-style
                    let close_size = 18.0;
                    let close_rect = Rect::from_center_size(
                        Pos2::new(header_rect.right() - 14.0, header_rect.center().y),
                        Vec2::splat(close_size),
                    );
                    let close_resp =
                        ui.interact(close_rect, Id::new("ls_close_btn"), Sense::click());
                    if ui.is_rect_visible(close_rect) {
                        let hovered = close_resp.hovered();
                        if hovered {
                            painter.rect_filled(
                                close_rect,
                                egui::Rounding::same(4.0),
                                colors.accent_faint,
                            );
                        }
                        let color = if hovered {
                            colors.accent_strong
                        } else {
                            colors.text_muted
                        };
                        let font = egui::FontId::proportional(13.0);
                        let galley = painter.layout_no_wrap("\u{00D7}".to_string(), font, color);
                        let gpos = close_rect.center() - galley.size() / 2.0;
                        painter.galley(Pos2::new(gpos.x, gpos.y), galley);
                    }
                    if close_resp.clicked() {
                        open = false;
                    }
                    ui.add_space(4.0);
                    // --- Tab bar for text layers ---
                    if is_text {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            let tabs = [
                                (LayerSettingsTab::General, t!("layer.tab.general")),
                                (LayerSettingsTab::Effects, t!("layer.tab.effects")),
                                (LayerSettingsTab::Warp, t!("layer.tab.warp")),
                            ];
                            for (tab, label) in &tabs {
                                let selected = self.settings_state.tab == *tab;
                                let btn = egui::Button::new(
                                    egui::RichText::new(label.as_str())
                                        .strong()
                                        .size(13.0)
                                        .color(if selected {
                                            ui.visuals().strong_text_color()
                                        } else {
                                            ui.visuals().text_color()
                                        }),
                                )
                                .fill(if selected {
                                    ui.visuals().selection.bg_fill
                                } else {
                                    Color32::TRANSPARENT
                                })
                                .rounding(egui::Rounding {
                                    nw: 4.0,
                                    ne: 4.0,
                                    sw: 0.0,
                                    se: 0.0,
                                })
                                .min_size(Vec2::new(80.0, 24.0));
                                if ui.add(btn).clicked() {
                                    self.settings_state.tab = *tab;
                                }
                            }
                        });
                        // Underline for current tab
                        let accent = ui.visuals().selection.bg_fill;
                        let r = ui.available_rect_before_wrap();
                        ui.painter().line_segment(
                            [Pos2::new(r.left(), r.top()), Pos2::new(r.right(), r.top())],
                            egui::Stroke::new(1.0, accent.linear_multiply(0.3)),
                        );
                        ui.add_space(4.0);
                    }

                    match self.settings_state.tab {
                        LayerSettingsTab::General => {
                            self.show_settings_general_tab(ui, layer_idx, canvas_state);
                        }
                        LayerSettingsTab::Effects if is_text => {
                            self.show_settings_effects_tab(ui, layer_idx, canvas_state);
                        }
                        LayerSettingsTab::Warp if is_text => {
                            self.show_settings_warp_tab(ui, layer_idx, canvas_state);
                        }
                        _ => {
                            // Non-text layers only have General
                            self.show_settings_general_tab(ui, layer_idx, canvas_state);
                        }
                    }

                    // Rasterize button for text layers
                    if is_text {
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(4.0);
                        if ui
                            .button(
                                egui::RichText::new(t!("layer.rasterize_text_layer")).size(13.0),
                            )
                            .clicked()
                        {
                            self.pending_app_action =
                                Some(LayerAppAction::RasterizeTextLayer(layer_idx));
                            open = false;
                        }
                    }
                });

            if !open {
                self.settings_state.editing_layer = None;
            }
        }
    }

    /// General tab: name, opacity, blend mode.
    fn show_settings_general_tab(
        &mut self,
        ui: &mut egui::Ui,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
    ) {
        ui.horizontal(|ui| {
            ui.label(t!("layer.name"));
            if ui
                .text_edit_singleline(&mut self.settings_state.editing_name)
                .changed()
                && !self.settings_state.editing_name.is_empty()
            {
                canvas_state.layers[layer_idx].name = self.settings_state.editing_name.clone();
            }
        });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(t!("layer.opacity"));
            if ui
                .add(
                    egui::Slider::new(&mut self.settings_state.editing_opacity, 0.0..=1.0)
                        .fixed_decimals(2)
                        .show_value(true),
                )
                .changed()
            {
                canvas_state.layers[layer_idx].opacity = self.settings_state.editing_opacity;
                self.mark_full_dirty(canvas_state);
            }
        });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label(t!("layer.blend"));
            egui::ComboBox::from_id_source("blend_mode_combo_ls")
                .selected_text(self.settings_state.editing_blend_mode.display_name())
                .width(120.0)
                .show_ui(ui, |ui: &mut egui::Ui| {
                    for &mode in BlendMode::all() {
                        if ui
                            .selectable_label(
                                mode == self.settings_state.editing_blend_mode,
                                mode.display_name(),
                            )
                            .clicked()
                        {
                            self.settings_state.editing_blend_mode = mode;
                            canvas_state.layers[layer_idx].blend_mode = mode;
                            self.mark_full_dirty(canvas_state);
                        }
                    }
                });
        });
    }

    /// Effects tab (text layers only): outline, shadow, inner shadow, texture fill.
    fn show_settings_effects_tab(
        &mut self,
        ui: &mut egui::Ui,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
    ) {
        let mut changed = false;

        // --- Outline ---
        {
            let has_outline = self.settings_state.text_effects.outline.is_some();
            let mut outline_on = has_outline;
            if ui
                .checkbox(&mut outline_on, t!("ctx.text.effects.outline"))
                .changed()
            {
                changed = true;
            }
            if outline_on && !has_outline {
                self.settings_state.text_effects.outline = Some(OutlineEffect::default());
                changed = true;
            } else if !outline_on && has_outline {
                self.settings_state.text_effects.outline = None;
                changed = true;
            }
            if let Some(ref mut outline) = self.settings_state.text_effects.outline {
                ui.indent("ls_outline", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.outline.color"));
                        let mut c = Color32::from_rgba_unmultiplied(
                            outline.color[0],
                            outline.color[1],
                            outline.color[2],
                            outline.color[3],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            outline.color = [c.r(), c.g(), c.b(), c.a()];
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.outline.width"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut outline.width)
                                    .speed(0.1)
                                    .clamp_range(0.5..=50.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.outline.position"));
                        let positions = [
                            (
                                OutlinePosition::Outside,
                                t!("ctx.text.effects.outline.outside"),
                            ),
                            (
                                OutlinePosition::Inside,
                                t!("ctx.text.effects.outline.inside"),
                            ),
                            (
                                OutlinePosition::Center,
                                t!("ctx.text.effects.outline.center"),
                            ),
                        ];
                        for (pos, label) in &positions {
                            if ui
                                .selectable_label(outline.position == *pos, label.as_str())
                                .clicked()
                            {
                                outline.position = *pos;
                                changed = true;
                            }
                        }
                    });
                });
            }
        }

        ui.add_space(4.0);

        // --- Drop Shadow ---
        {
            let has_shadow = self.settings_state.text_effects.shadow.is_some();
            let mut shadow_on = has_shadow;
            if ui
                .checkbox(&mut shadow_on, t!("ctx.text.effects.shadow"))
                .changed()
            {
                changed = true;
            }
            if shadow_on && !has_shadow {
                self.settings_state.text_effects.shadow = Some(ShadowEffect::default());
                changed = true;
            } else if !shadow_on && has_shadow {
                self.settings_state.text_effects.shadow = None;
                changed = true;
            }
            if let Some(ref mut shadow) = self.settings_state.text_effects.shadow {
                ui.indent("ls_shadow", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.shadow.color"));
                        let mut c = Color32::from_rgba_unmultiplied(
                            shadow.color[0],
                            shadow.color[1],
                            shadow.color[2],
                            shadow.color[3],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            shadow.color = [c.r(), c.g(), c.b(), c.a()];
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.shadow.offset_x"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut shadow.offset_x)
                                    .speed(0.5)
                                    .clamp_range(-100.0..=100.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        ui.label(t!("ctx.text.effects.shadow.offset_y"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut shadow.offset_y)
                                    .speed(0.5)
                                    .clamp_range(-100.0..=100.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.shadow.blur"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut shadow.blur_radius)
                                    .speed(0.2)
                                    .clamp_range(0.0..=50.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        ui.label(t!("ctx.text.effects.shadow.spread"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut shadow.spread)
                                    .speed(0.2)
                                    .clamp_range(0.0..=30.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                });
            }
        }

        ui.add_space(4.0);

        // --- Inner Shadow ---
        {
            let has_inner = self.settings_state.text_effects.inner_shadow.is_some();
            let mut inner_on = has_inner;
            if ui
                .checkbox(&mut inner_on, t!("ctx.text.effects.inner_shadow"))
                .changed()
            {
                changed = true;
            }
            if inner_on && !has_inner {
                self.settings_state.text_effects.inner_shadow = Some(InnerShadowEffect::default());
                changed = true;
            } else if !inner_on && has_inner {
                self.settings_state.text_effects.inner_shadow = None;
                changed = true;
            }
            if let Some(ref mut inner) = self.settings_state.text_effects.inner_shadow {
                ui.indent("ls_inner_shadow", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.inner_shadow.color"));
                        let mut c = Color32::from_rgba_unmultiplied(
                            inner.color[0],
                            inner.color[1],
                            inner.color[2],
                            inner.color[3],
                        );
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            inner.color = [c.r(), c.g(), c.b(), c.a()];
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.inner_shadow.offset_x"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut inner.offset_x)
                                    .speed(0.5)
                                    .clamp_range(-100.0..=100.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        ui.label(t!("ctx.text.effects.inner_shadow.offset_y"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut inner.offset_y)
                                    .speed(0.5)
                                    .clamp_range(-100.0..=100.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(t!("ctx.text.effects.inner_shadow.blur"));
                        if ui
                            .add(
                                egui::DragValue::new(&mut inner.blur_radius)
                                    .speed(0.2)
                                    .clamp_range(0.0..=50.0)
                                    .suffix("px"),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                });
            }
        }

        ui.add_space(4.0);

        // --- Texture Fill ---
        {
            let has_texture = self.settings_state.text_effects.texture_fill.is_some();
            let mut texture_on = has_texture;
            if ui
                .checkbox(&mut texture_on, t!("ctx.text.effects.texture"))
                .changed()
            {
                changed = true;
            }
            if texture_on && !has_texture {
                self.settings_state.text_effects.texture_fill = Some(TextureFillEffect::default());
                changed = true;
            } else if !texture_on && has_texture {
                self.settings_state.text_effects.texture_fill = None;
                changed = true;
            }

            // Poll for async texture load
            if let Some(ref rx) = self.settings_state.texture_load_rx
                && let Ok(data) = rx.try_recv()
            {
                if let Ok(img) = image::load_from_memory(&data)
                    && let Some(ref mut tex) = self.settings_state.text_effects.texture_fill
                {
                    tex.texture_data = data;
                    tex.texture_width = img.width();
                    tex.texture_height = img.height();
                    changed = true;
                }
                self.settings_state.texture_load_rx = None;
            }

            let mut remove_texture = false;
            let mut spawn_texture_dialog = false;
            if let Some(ref mut tex) = self.settings_state.text_effects.texture_fill {
                ui.indent("ls_texture", |ui| {
                    ui.horizontal(|ui| {
                        if ui.button(t!("ctx.text.effects.texture.load")).clicked() {
                            spawn_texture_dialog = true;
                        }
                        if tex.texture_width > 0 {
                            ui.label(format!("{}×{}", tex.texture_width, tex.texture_height));
                        }
                    });
                    if tex.texture_width > 0 {
                        ui.horizontal(|ui| {
                            ui.label(t!("ctx.text.effects.texture.scale"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut tex.scale)
                                        .speed(0.01)
                                        .clamp_range(0.1..=10.0),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label(t!("ctx.text.effects.texture.offset_x"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut tex.offset[0])
                                        .speed(0.5)
                                        .clamp_range(-1000.0..=1000.0),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                            ui.label(t!("ctx.text.effects.texture.offset_y"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut tex.offset[1])
                                        .speed(0.5)
                                        .clamp_range(-1000.0..=1000.0),
                                )
                                .changed()
                            {
                                changed = true;
                            }
                        });
                        if ui.button(t!("ctx.text.effects.texture.remove")).clicked() {
                            remove_texture = true;
                            changed = true;
                        }
                    }
                });
            }
            if remove_texture {
                self.settings_state.text_effects.texture_fill = None;
            }
            if spawn_texture_dialog {
                let (tx, rx) = std::sync::mpsc::channel();
                self.settings_state.texture_load_rx = Some(rx);
                std::thread::spawn(move || {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
                        .pick_file()
                        && let Ok(data) = std::fs::read(&path)
                    {
                        let _ = tx.send(data);
                    }
                });
            }
        }

        // Live-commit: write effects to TextLayerData and rasterize immediately
        if changed {
            if let Some(layer) = canvas_state.layers.get_mut(layer_idx)
                && let LayerContent::Text(ref mut td) = layer.content
            {
                td.effects = self.settings_state.text_effects.clone();
                td.mark_effects_dirty();
            }
            canvas_state.force_rasterize_text_layer(layer_idx);
            canvas_state.mark_dirty(None);
        }
    }

    /// Warp tab (text layers only): warp type selector and per-type controls.
    fn show_settings_warp_tab(
        &mut self,
        ui: &mut egui::Ui,
        layer_idx: usize,
        canvas_state: &mut CanvasState,
    ) {
        let mut changed = false;

        let current_name = self.settings_state.text_warp.name().to_string();
        ui.horizontal(|ui| {
            ui.label(t!("ctx.text.warp.type"));
            egui::ComboBox::from_id_source("ls_text_warp_type")
                .selected_text(&current_name)
                .width(130.0)
                .show_ui(ui, |ui| {
                    for name in TextWarp::all_names() {
                        if *name == "Path Follow" {
                            continue; // hidden for now
                        }
                        if ui.selectable_label(current_name == *name, *name).clicked() {
                            let new_warp = TextWarp::from_name(name);
                            if self.settings_state.text_warp != new_warp {
                                self.settings_state.text_warp = new_warp;
                                changed = true;
                            }
                        }
                    }
                });
        });

        ui.add_space(4.0);

        match &mut self.settings_state.text_warp {
            TextWarp::None => {}
            TextWarp::Arc(arc) => {
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.arc.bend"));
                    if ui
                        .add(
                            egui::DragValue::new(&mut arc.bend)
                                .speed(0.01)
                                .clamp_range(-1.0..=1.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.arc.hdist"));
                    if ui
                        .add(
                            egui::DragValue::new(&mut arc.horizontal_distortion)
                                .speed(0.01)
                                .clamp_range(-1.0..=1.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.arc.vdist"));
                    if ui
                        .add(
                            egui::DragValue::new(&mut arc.vertical_distortion)
                                .speed(0.01)
                                .clamp_range(-1.0..=1.0),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
            }
            TextWarp::Circular(circ) => {
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.circular.radius"));
                    if ui
                        .add(
                            egui::DragValue::new(&mut circ.radius)
                                .speed(1.0)
                                .clamp_range(20.0..=2000.0)
                                .suffix("px"),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.circular.start_angle"));
                    let mut degrees = circ.start_angle.to_degrees();
                    if ui
                        .add(
                            egui::DragValue::new(&mut degrees)
                                .speed(1.0)
                                .clamp_range(-360.0..=360.0)
                                .suffix("°"),
                        )
                        .changed()
                    {
                        circ.start_angle = degrees.to_radians();
                        changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(t!("ctx.text.warp.circular.clockwise"));
                    if ui.checkbox(&mut circ.clockwise, "").changed() {
                        changed = true;
                    }
                });
            }
            TextWarp::PathFollow(_) => {
                // Hidden from UI for now; no controls shown.
            }
            TextWarp::Envelope(env) => {
                ui.label(t!("ctx.text.warp.envelope.top"));
                let mut any_changed = false;
                for (i, pt) in env.top_curve.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("T{i}:"));
                        if ui
                            .add(egui::DragValue::new(&mut pt[0]).speed(1.0).prefix("x: "))
                            .changed()
                        {
                            any_changed = true;
                        }
                        if ui
                            .add(egui::DragValue::new(&mut pt[1]).speed(1.0).prefix("y: "))
                            .changed()
                        {
                            any_changed = true;
                        }
                    });
                }
                ui.add_space(2.0);
                ui.label(t!("ctx.text.warp.envelope.bottom"));
                for (i, pt) in env.bottom_curve.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("B{i}:"));
                        if ui
                            .add(egui::DragValue::new(&mut pt[0]).speed(1.0).prefix("x: "))
                            .changed()
                        {
                            any_changed = true;
                        }
                        if ui
                            .add(egui::DragValue::new(&mut pt[1]).speed(1.0).prefix("y: "))
                            .changed()
                        {
                            any_changed = true;
                        }
                    });
                }
                if any_changed {
                    changed = true;
                }
                if ui.button(t!("ctx.text.warp.reset")).clicked() {
                    *env = EnvelopeWarp::default();
                    changed = true;
                }
            }
        }

        // Live-commit: write warp to ALL blocks and rasterize immediately
        if changed {
            if let Some(layer) = canvas_state.layers.get_mut(layer_idx)
                && let LayerContent::Text(ref mut td) = layer.content
            {
                for block in &mut td.blocks {
                    block.warp = self.settings_state.text_warp.clone();
                }
                td.mark_dirty();
            }
            canvas_state.force_rasterize_text_layer(layer_idx);
            canvas_state.mark_dirty(None);
        }
    }

    // === Layer Operations ===

    fn add_new_layer(&mut self, canvas_state: &mut CanvasState, history: &mut HistoryManager) {
        let layer_num = canvas_state.layers.len() + 1;
        let layer_name = format!("Layer {}", layer_num);
        let transparent = Rgba([0, 0, 0, 0]);
        let new_layer = Layer::new(
            layer_name.clone(),
            canvas_state.width,
            canvas_state.height,
            transparent,
        );

        // Insert above current active layer
        let insert_idx = canvas_state.active_layer_index + 1;
        canvas_state.layers.insert(insert_idx, new_layer);
        canvas_state.active_layer_index = insert_idx;

        // Record history
        history.push(Box::new(LayerOpCommand::new(LayerOperation::Add {
            index: insert_idx,
            name: layer_name,
            width: canvas_state.width,
            height: canvas_state.height,
        })));

        self.thumbnail_cache.clear();
        self.mark_full_dirty(canvas_state);
    }

    fn add_new_text_layer(&mut self, canvas_state: &mut CanvasState, history: &mut HistoryManager) {
        let layer_num = canvas_state.layers.len() + 1;
        let layer_name = format!("Text Layer {}", layer_num);
        let new_layer =
            Layer::new_text(layer_name.clone(), canvas_state.width, canvas_state.height);

        let insert_idx = canvas_state.active_layer_index + 1;
        canvas_state.layers.insert(insert_idx, new_layer);
        canvas_state.active_layer_index = insert_idx;

        history.push(Box::new(LayerOpCommand::new(LayerOperation::Add {
            index: insert_idx,
            name: layer_name,
            width: canvas_state.width,
            height: canvas_state.height,
        })));

        self.thumbnail_cache.clear();
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
        let name = layer.name.clone();
        let visible = layer.visible;
        let opacity = layer.opacity;
        let content = layer.content.clone();

        canvas_state.layers.remove(layer_idx);

        if canvas_state.active_layer_index >= canvas_state.layers.len() {
            canvas_state.active_layer_index = canvas_state.layers.len() - 1;
        } else if canvas_state.active_layer_index > layer_idx {
            canvas_state.active_layer_index -= 1;
        }

        // Notify the deletion index so the UI can clean up GPU textures.
        self.pending_gpu_delete = Some(layer_idx);

        // Record history
        history.push(Box::new(LayerOpCommand::new(LayerOperation::Delete {
            index: layer_idx,
            pixels,
            name,
            visible,
            opacity,
            content,
        })));

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
        new_layer.opacity = source.opacity;
        new_layer.blend_mode = source.blend_mode;
        new_layer.content = source.content.clone();

        let new_index = layer_idx + 1;

        // Capture data for history before inserting
        let pixels = new_layer.pixels.clone();
        let visible = new_layer.visible;
        let opacity = new_layer.opacity;
        let content = new_layer.content.clone();

        // Insert above the duplicated layer
        canvas_state.layers.insert(new_index, new_layer);
        canvas_state.active_layer_index = new_index;

        // Record history
        history.push(Box::new(LayerOpCommand::new(LayerOperation::Duplicate {
            source_index: layer_idx,
            new_index,
            pixels,
            name: new_name,
            visible,
            opacity,
            content,
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

/// Actions from layer row interactions
enum LayerAction {
    Select,
    StartRename,
    FinishRename,
    CancelRename,
    ToggleVisibility,
    BeginDrag,
}

/// Actions from context menu
enum ContextAction {
    AddNew,
    AddNewTextLayer,
    MergeDown,
    MergeDownAsMask,
    FlattenImage,
    Duplicate,
    Delete,
    OpenSettings,
    MoveToTop,
    MoveUp,
    MoveDown,
    MoveToBottom,
    Rename,
    ImportFromFile,
    FlipHorizontal,
    FlipVertical,
    RotateScale,
    SoloLayer,
    HideAll,
    ShowAll,
    RasterizeTextLayer,
    TextLayerEffects,
    TextLayerWarp,
}
