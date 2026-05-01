impl ToolsPanel {
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn handle_surface_transform_input(
        &mut self,
        ui: &egui::Ui,
        canvas_state: &mut CanvasState,
        canvas_pos: Option<(u32, u32)>,
        canvas_pos_f32: Option<(f32, f32)>,
        canvas_pos_f32_clamped: Option<(f32, f32)>,
        canvas_pos_unclamped: Option<(f32, f32)>,
        raw_motion_events: &[(f32, f32)],
        painter: &egui::Painter,
        canvas_rect: Rect,
        zoom: f32,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        gpu_renderer: &mut Option<&mut crate::gpu::GpuRenderer>,
        stroke_event: &mut Option<StrokeEvent>,
        is_primary_down: bool,
        is_primary_released: bool,
        is_primary_clicked: bool,
        is_primary_pressed: bool,
        is_secondary_down: bool,
        is_secondary_pressed: bool,
        is_secondary_released: bool,
        is_secondary_clicked: bool,
        shift_held: bool,
        enter_pressed: bool,
        escape_pressed_global: bool,
    ) {
        match self.active_tool {
            // LIQUIFY TOOL - click+drag to push/pull pixels
            // ================================================================
            Tool::Liquify => {
                // Guard: auto-rasterize text layers before destructive liquify
                if is_primary_pressed
                    && let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }

                let escape_pressed = escape_pressed_global;

                // Escape: reset
                if escape_pressed && self.liquify_state.is_active {
                    self.liquify_state.displacement = None;
                    self.liquify_state.is_active = false;
                    self.liquify_state.source_snapshot = None;
                    self.liquify_state.source_layer_index = None;
                    self.liquify_state.warp_buffer.clear();
                    self.liquify_state.dirty_rect = None;
                    canvas_state.clear_preview_state();
                    canvas_state.mark_dirty(None);
                }

                // Enter: commit
                if enter_pressed && self.liquify_state.is_active {
                    self.liquify_state.commit_pending = true;
                    self.liquify_state.commit_pending_frame = 0;
                }

                // Mouse drag: apply displacement
                if is_primary_down {
                    if let Some(pos_f) = canvas_pos_f32 {
                        // Initialize on first use
                        if !self.liquify_state.is_active {
                            self.liquify_state.is_active = true;
                            if let Some(layer) =
                                canvas_state.layers.get(canvas_state.active_layer_index)
                            {
                                self.liquify_state.source_snapshot =
                                    Some(layer.pixels.to_rgba_image());
                                self.liquify_state.source_layer_index =
                                    Some(canvas_state.active_layer_index);
                            }
                            self.liquify_state.displacement =
                                Some(crate::ops::transform::DisplacementField::new(
                                    canvas_state.width,
                                    canvas_state.height,
                                ));
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Liquify");
                            // Tell GPU pipeline the source snapshot changed
                            if let Some(gpu) = gpu_renderer {
                                gpu.liquify_pipeline.invalidate_source();
                            }
                        }

                        if let Some(ref mut disp) = self.liquify_state.displacement {
                            let radius = self.properties.size;
                            let cx = pos_f.0;
                            let cy = pos_f.1;

                            match self.liquify_state.mode {
                                LiquifyMode::Push => {
                                    // Get delta from last position
                                    if let Some(last) = self.liquify_state.last_pos {
                                        let dx = cx - last[0];
                                        let dy = cy - last[1];
                                        if dx.abs() > 0.1 || dy.abs() > 0.1 {
                                            disp.apply_push(
                                                cx,
                                                cy,
                                                dx * self.liquify_state.strength,
                                                dy * self.liquify_state.strength,
                                                radius,
                                                self.liquify_state.strength,
                                            );
                                        }
                                    }
                                }
                                LiquifyMode::Expand => {
                                    disp.apply_expand(
                                        cx,
                                        cy,
                                        radius,
                                        self.liquify_state.strength * 2.0,
                                    );
                                }
                                LiquifyMode::Contract => {
                                    disp.apply_contract(
                                        cx,
                                        cy,
                                        radius,
                                        self.liquify_state.strength * 2.0,
                                    );
                                }
                                LiquifyMode::TwirlCW => {
                                    disp.apply_twirl(
                                        cx,
                                        cy,
                                        radius,
                                        self.liquify_state.strength * 0.05,
                                        true,
                                    );
                                }
                                LiquifyMode::TwirlCCW => {
                                    disp.apply_twirl(
                                        cx,
                                        cy,
                                        radius,
                                        self.liquify_state.strength * 0.05,
                                        false,
                                    );
                                }
                            }

                            // Update dirty rect
                            let r = radius as i32 + 2;
                            let new_dirty =
                                [cx as i32 - r, cy as i32 - r, cx as i32 + r, cy as i32 + r];
                            self.liquify_state.dirty_rect =
                                Some(match self.liquify_state.dirty_rect {
                                    Some(d) => [
                                        d[0].min(new_dirty[0]),
                                        d[1].min(new_dirty[1]),
                                        d[2].max(new_dirty[2]),
                                        d[3].max(new_dirty[3]),
                                    ],
                                    None => new_dirty,
                                });

                            self.liquify_state.last_pos = Some([cx, cy]);
                            self.render_liquify_preview(canvas_state, gpu_renderer.as_deref_mut());
                        }
                    }
                } else if self.liquify_state.last_pos.is_some() {
                    self.liquify_state.last_pos = None;
                    self.liquify_state.dirty_rect = None;
                }
            }

            // ================================================================
            // MESH WARP TOOL - drag control points to warp image
            // ================================================================
            Tool::MeshWarp => {
                // Guard: auto-rasterize text layers before destructive mesh warp
                if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }

                let escape_pressed = escape_pressed_global;

                // Escape: reset to original
                if escape_pressed && self.mesh_warp_state.is_active {
                    self.mesh_warp_state.points = self.mesh_warp_state.original_points.clone();
                }

                // Enter: commit
                if enter_pressed && self.mesh_warp_state.is_active {
                    self.mesh_warp_state.commit_pending = true;
                    self.mesh_warp_state.commit_pending_frame = 0;
                }

                // Auto-initialize grid as soon as tool is selected
                if !self.mesh_warp_state.is_active {
                    self.init_mesh_warp_grid(canvas_state);
                    self.stroke_tracker
                        .start_preview_tool(canvas_state.active_layer_index, "Mesh Warp");
                } else {
                    // Detect stale snapshot (layer changed since snapshot was taken)
                    let cur_layer_idx = canvas_state.active_layer_index;
                    let cur_gen = canvas_state
                        .layers
                        .get(cur_layer_idx)
                        .map(|l| l.gpu_generation)
                        .unwrap_or(0);
                    if cur_layer_idx != self.mesh_warp_state.snapshot_layer_index
                        || cur_gen != self.mesh_warp_state.snapshot_generation
                    {
                        // Re-snapshot: layer content or active layer changed
                        self.init_mesh_warp_grid(canvas_state);
                        self.stroke_tracker
                            .start_preview_tool(cur_layer_idx, "Mesh Warp");
                    }
                }
                if self.mesh_warp_state.is_active {
                    // Find nearest control point for hover/drag
                    // Use clamped coordinates so edge/corner handles are reachable
                    let mesh_pos = canvas_pos_f32_clamped.or(canvas_pos_f32);
                    if let Some(pos_f) = mesh_pos {
                        let hit_radius = 12.0 / zoom; // screen pixels -> canvas pixels

                        if is_primary_pressed {
                            // Start drag
                            let mut best_idx = None;
                            let mut best_dist = f32::MAX;
                            for (i, pt) in self.mesh_warp_state.points.iter().enumerate() {
                                let dx = pos_f.0 - pt[0];
                                let dy = pos_f.1 - pt[1];
                                let dist = (dx * dx + dy * dy).sqrt();
                                if dist < hit_radius && dist < best_dist {
                                    best_dist = dist;
                                    best_idx = Some(i);
                                }
                            }
                            self.mesh_warp_state.dragging_index = best_idx;
                        }

                        if is_primary_down && let Some(idx) = self.mesh_warp_state.dragging_index {
                            self.mesh_warp_state.points[idx] = [pos_f.0, pos_f.1];
                        }

                        if is_primary_released {
                            self.mesh_warp_state.dragging_index = None;
                        }

                        // Hover detection
                        if !is_primary_down {
                            let mut hover = None;
                            let mut best_dist = f32::MAX;
                            for (i, pt) in self.mesh_warp_state.points.iter().enumerate() {
                                let dx = pos_f.0 - pt[0];
                                let dy = pos_f.1 - pt[1];
                                let dist = (dx * dx + dy * dy).sqrt();
                                if dist < hit_radius && dist < best_dist {
                                    best_dist = dist;
                                    hover = Some(i);
                                }
                            }
                            self.mesh_warp_state.hover_index = hover;
                        }
                    }

                    // Draw overlay (grid + handles) - no live preview, warp only on commit
                    self.draw_mesh_warp_overlay(ui, painter, canvas_rect, zoom, canvas_state);
                }
            }

            // ================================================================
            // COLOR REMOVER TOOL - click to remove color
            // ================================================================
            Tool::ColorRemover => {
                // Guard: auto-rasterize text layers before destructive color removal
                if is_primary_clicked
                    && let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }
                if is_primary_clicked && let Some(pos) = canvas_pos {
                    self.commit_color_removal(canvas_state, pos.0, pos.1);
                }
            }

            // ================================================================
            // SMUDGE TOOL - drag to smear/blend canvas pixels
            // ================================================================
            Tool::Smudge => {
                if is_primary_pressed {
                    // Guard: auto-rasterize text layers before destructive smudge
                    if let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                        && layer.is_text_layer()
                    {
                        self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                        return;
                    }

                    // Stroke start: pick up the color at the start position
                    if let Some(pos) = canvas_pos {
                        if let Some(layer) =
                            canvas_state.layers.get(canvas_state.active_layer_index)
                        {
                            let px = layer.pixels.get_pixel(
                                pos.0.min(canvas_state.width.saturating_sub(1)),
                                pos.1.min(canvas_state.height.saturating_sub(1)),
                            );
                            self.smudge_state.pickup_color =
                                [px[0] as f32, px[1] as f32, px[2] as f32, px[3] as f32];
                        }
                        self.smudge_state.is_stroking = true;
                        let layer_pixels = canvas_state
                            .layers
                            .get(canvas_state.active_layer_index)
                            .map(|l| l.pixels.clone());
                        if let Some(pixels) = layer_pixels {
                            self.stroke_tracker.start_direct_tool(
                                canvas_state.active_layer_index,
                                "Smudge",
                                &pixels,
                            );
                        }
                    }
                }
                if is_primary_down
                    && self.smudge_state.is_stroking
                    && let Some(pos) = canvas_pos
                {
                    let radius = (self.properties.size * 0.5).max(1.0);
                    self.draw_smudge_no_dirty(canvas_state, pos.0, pos.1);
                    let dirty = egui::Rect::from_min_max(
                        egui::pos2(
                            (pos.0 as f32 - radius - 2.0).max(0.0),
                            (pos.1 as f32 - radius - 2.0).max(0.0),
                        ),
                        egui::pos2(pos.0 as f32 + radius + 2.0, pos.1 as f32 + radius + 2.0),
                    );
                    self.stroke_tracker.expand_bounds(dirty);
                    canvas_state.mark_dirty(Some(dirty));
                }
                if is_primary_released && self.smudge_state.is_stroking {
                    self.smudge_state.is_stroking = false;
                    if let Some(se) = self.stroke_tracker.finish(canvas_state) {
                        self.pending_stroke_event = Some(se);
                    }
                }
            }

            // ================================================================
            // SHAPES TOOL - click+drag to draw, then adjust
            // ================================================================
            Tool::Shapes => {
                // Guard: auto-rasterize text layers before destructive shape drawing
                if is_primary_pressed
                    && let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }

                let escape_pressed = escape_pressed_global;

                // Escape: cancel
                if escape_pressed {
                    if self.shapes_state.placed.is_some() {
                        self.shapes_state.placed = None;
                        self.shapes_state.source_layer_index = None;
                        canvas_state.clear_preview_state();
                        canvas_state.mark_dirty(None);
                    } else if self.shapes_state.is_drawing {
                        self.shapes_state.is_drawing = false;
                        self.shapes_state.draw_start = None;
                        self.shapes_state.draw_end = None;
                        self.shapes_state.source_layer_index = None;
                        canvas_state.clear_preview_state();
                        canvas_state.mark_dirty(None);
                    }
                }

                // Enter: commit placed shape
                if enter_pressed && self.shapes_state.placed.is_some() {
                    self.stroke_tracker
                        .start_preview_tool(canvas_state.active_layer_index, "Shape");
                    self.commit_shape(canvas_state);
                }

                if let Some(pos_f) = canvas_pos_unclamped {
                    if self.shapes_state.placed.is_some() {
                        // Shape is placed - handle move/resize/rotate
                        let hit_radius = 10.0 / zoom;

                        // Determine what was hit on press
                        let mut should_commit = false;
                        let mut need_preview = false;

                        if is_primary_pressed {
                            use crate::ops::shapes::ShapeHandle;
                            let placed = self.shapes_state.placed.as_ref().unwrap();
                            let cos_r = placed.rotation.cos();
                            let sin_r = placed.rotation.sin();

                            let corners_local = [
                                (-placed.hw, -placed.hh),
                                (placed.hw, -placed.hh),
                                (placed.hw, placed.hh),
                                (-placed.hw, placed.hh),
                            ];
                            let corners_canvas: Vec<(f32, f32)> = corners_local
                                .iter()
                                .map(|(cx, cy)| {
                                    (
                                        cx * cos_r - cy * sin_r + placed.cx,
                                        cx * sin_r + cy * cos_r + placed.cy,
                                    )
                                })
                                .collect();

                            let top_mid_x = (corners_canvas[0].0 + corners_canvas[1].0) * 0.5;
                            let top_mid_y = (corners_canvas[0].1 + corners_canvas[1].1) * 0.5;
                            let right_mid_x = (corners_canvas[1].0 + corners_canvas[2].0) * 0.5;
                            let right_mid_y = (corners_canvas[1].1 + corners_canvas[2].1) * 0.5;
                            let bottom_mid_x = (corners_canvas[2].0 + corners_canvas[3].0) * 0.5;
                            let bottom_mid_y = (corners_canvas[2].1 + corners_canvas[3].1) * 0.5;
                            let left_mid_x = (corners_canvas[3].0 + corners_canvas[0].0) * 0.5;
                            let left_mid_y = (corners_canvas[3].1 + corners_canvas[0].1) * 0.5;
                            let rot_offset = 20.0 / zoom;
                            let dir_x = top_mid_x - placed.cx;
                            let dir_y = top_mid_y - placed.cy;
                            let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(0.001);
                            let rot_handle_x = top_mid_x + (dir_x / dir_len) * rot_offset;
                            let rot_handle_y = top_mid_y + (dir_y / dir_len) * rot_offset;

                            let handle_names = [
                                ShapeHandle::TopLeft,
                                ShapeHandle::TopRight,
                                ShapeHandle::BottomRight,
                                ShapeHandle::BottomLeft,
                            ];
                            let mut hit: Option<ShapeHandle> = None;

                            // Check rotation handle
                            let dx = pos_f.0 - rot_handle_x;
                            let dy = pos_f.1 - rot_handle_y;
                            if (dx * dx + dy * dy).sqrt() < hit_radius {
                                hit = Some(ShapeHandle::Rotate);
                            }

                            // Check corners
                            if hit.is_none() {
                                for (i, &(cx, cy)) in corners_canvas.iter().enumerate() {
                                    let dx = pos_f.0 - cx;
                                    let dy = pos_f.1 - cy;
                                    if (dx * dx + dy * dy).sqrt() < hit_radius {
                                        hit = Some(handle_names[i]);
                                        break;
                                    }
                                }
                            }

                            // Check edge-midpoint handles
                            if hit.is_none() {
                                let edge_handles = [
                                    (ShapeHandle::Top, top_mid_x, top_mid_y),
                                    (ShapeHandle::Right, right_mid_x, right_mid_y),
                                    (ShapeHandle::Bottom, bottom_mid_x, bottom_mid_y),
                                    (ShapeHandle::Left, left_mid_x, left_mid_y),
                                ];
                                for (edge_handle, cx, cy) in edge_handles {
                                    let dx = pos_f.0 - cx;
                                    let dy = pos_f.1 - cy;
                                    if (dx * dx + dy * dy).sqrt() < hit_radius {
                                        hit = Some(edge_handle);
                                        break;
                                    }
                                }
                            }

                            // Check inside shape for move
                            if hit.is_none() {
                                let dx = pos_f.0 - placed.cx;
                                let dy = pos_f.1 - placed.cy;
                                let lx = (dx * cos_r + dy * sin_r).abs();
                                let ly = (-dx * sin_r + dy * cos_r).abs();
                                if lx <= placed.hw + hit_radius && ly <= placed.hh + hit_radius {
                                    hit = Some(ShapeHandle::Move);
                                }
                            }

                            let pcx = placed.cx;
                            let pcy = placed.cy;
                            let prot = placed.rotation;
                            // Drop the immutable borrow before mutable access below
                            let _ = placed;

                            if let Some(h) = hit {
                                let p = self.shapes_state.placed.as_mut().unwrap();
                                p.handle_dragging = Some(h);
                                p.drag_offset = [pos_f.0 - pcx, pos_f.1 - pcy];
                                if h == ShapeHandle::Rotate {
                                    p.rotate_start_angle = (pos_f.1 - pcy).atan2(pos_f.0 - pcx);
                                    p.rotate_start_rotation = prot;
                                }
                                // For corner resize: compute anchor = opposite corner in canvas coords
                                let cos_r = prot.cos();
                                let sin_r = prot.sin();
                                let (anchor_lx, anchor_ly) = match h {
                                    ShapeHandle::TopLeft => (p.hw, p.hh),
                                    ShapeHandle::TopRight => (-p.hw, p.hh),
                                    ShapeHandle::BottomRight => (-p.hw, -p.hh),
                                    ShapeHandle::BottomLeft => (p.hw, -p.hh),
                                    ShapeHandle::Top => (0.0, p.hh),
                                    ShapeHandle::Right => (-p.hw, 0.0),
                                    ShapeHandle::Bottom => (0.0, -p.hh),
                                    ShapeHandle::Left => (p.hw, 0.0),
                                    _ => (0.0, 0.0),
                                };
                                p.drag_anchor = [
                                    anchor_lx * cos_r - anchor_ly * sin_r + pcx,
                                    anchor_lx * sin_r + anchor_ly * cos_r + pcy,
                                ];
                            } else {
                                should_commit = true;
                            }
                        }

                        if should_commit {
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Shape");
                            self.commit_shape(canvas_state);
                            if self.pending_stroke_event.is_none()
                                && let Some(evt) = stroke_event.take()
                            {
                                self.pending_stroke_event = Some(evt);
                            }
                        }

                        if is_primary_down && !should_commit {
                            use crate::ops::shapes::ShapeHandle;
                            if let Some(ref mut p) = self.shapes_state.placed {
                                match p.handle_dragging {
                                    Some(ShapeHandle::Move) => {
                                        p.cx = (pos_f.0 - p.drag_offset[0]).round();
                                        p.cy = (pos_f.1 - p.drag_offset[1]).round();
                                        need_preview = true;
                                    }
                                    Some(ShapeHandle::Rotate) => {
                                        let angle = (pos_f.1 - p.cy).atan2(pos_f.0 - p.cx);
                                        let mut new_rot = p.rotate_start_rotation
                                            + (angle - p.rotate_start_angle);
                                        // Shift: snap to 45┬░ increments
                                        if shift_held {
                                            let snap = std::f32::consts::FRAC_PI_4; // 45┬░
                                            new_rot = (new_rot / snap).round() * snap;
                                        }
                                        p.rotation = new_rot;
                                        need_preview = true;
                                    }
                                    Some(_handle) => {
                                        // Resize: corners are two-axis, edge handles are one-axis.
                                        let ax = p.drag_anchor[0];
                                        let ay = p.drag_anchor[1];
                                        let mx = pos_f.0;
                                        let my = pos_f.1;
                                        // Half-sizes in local (rotated) space
                                        let cos_r = p.rotation.cos();
                                        let sin_r = p.rotation.sin();
                                        let dx = (mx - ax) * 0.5;
                                        let dy = (my - ay) * 0.5;
                                        let local_dx = dx * cos_r + dy * sin_r;
                                        let local_dy = -dx * sin_r + dy * cos_r;

                                        if matches!(
                                            _handle,
                                            ShapeHandle::Top
                                                | ShapeHandle::Right
                                                | ShapeHandle::Bottom
                                                | ShapeHandle::Left
                                        ) {
                                            // Opposite edge midpoint is fixed anchor. For edge handles,
                                            // only one local axis changes to avoid perpendicular drift.
                                            match _handle {
                                                ShapeHandle::Top | ShapeHandle::Bottom => {
                                                    let new_hh = local_dy.abs().max(0.5);
                                                    let center_local = match _handle {
                                                        ShapeHandle::Top => (0.0, -new_hh),
                                                        ShapeHandle::Bottom => (0.0, new_hh),
                                                        _ => (0.0, 0.0),
                                                    };
                                                    p.cx = ax + center_local.0 * cos_r
                                                        - center_local.1 * sin_r;
                                                    p.cy = ay
                                                        + center_local.0 * sin_r
                                                        + center_local.1 * cos_r;
                                                    p.cx = p.cx.round();
                                                    p.cy = p.cy.round();
                                                    p.hh = new_hh;
                                                }
                                                ShapeHandle::Left | ShapeHandle::Right => {
                                                    let new_hw = local_dx.abs().max(0.5);
                                                    let center_local = match _handle {
                                                        ShapeHandle::Right => (new_hw, 0.0),
                                                        ShapeHandle::Left => (-new_hw, 0.0),
                                                        _ => (0.0, 0.0),
                                                    };
                                                    p.cx = ax + center_local.0 * cos_r
                                                        - center_local.1 * sin_r;
                                                    p.cy = ay
                                                        + center_local.0 * sin_r
                                                        + center_local.1 * cos_r;
                                                    p.cx = p.cx.round();
                                                    p.cy = p.cy.round();
                                                    p.hw = new_hw;
                                                }
                                                _ => {}
                                            }
                                        } else {
                                            let mut lx = local_dx.abs();
                                            let mut ly = local_dy.abs();
                                            // Shift: constrain to 1:1 aspect ratio
                                            if shift_held {
                                                let side = lx.max(ly);
                                                lx = side;
                                                ly = side;
                                            }
                                            // Recompute center from anchor + constrained size
                                            // The dragged corner in local space is the negation of the anchor's local offset
                                            let (anchor_lx, anchor_ly) = match _handle {
                                                ShapeHandle::TopLeft => (lx, ly),
                                                ShapeHandle::TopRight => (-lx, ly),
                                                ShapeHandle::BottomRight => (-lx, -ly),
                                                ShapeHandle::BottomLeft => (lx, -ly),
                                                _ => (0.0, 0.0),
                                            };
                                            let new_cx =
                                                ax + (-anchor_lx) * cos_r - (-anchor_ly) * sin_r;
                                            let new_cy =
                                                ay + (-anchor_lx) * sin_r + (-anchor_ly) * cos_r;
                                            p.cx = new_cx.round();
                                            p.cy = new_cy.round();
                                            p.hw = lx.max(0.5).round().max(0.5);
                                            p.hh = ly.max(0.5).round().max(0.5);
                                        }
                                        need_preview = true;
                                    }
                                    None => {}
                                }
                            }
                        }

                        if need_preview {
                            self.render_shape_preview(
                                canvas_state,
                                primary_color_f32,
                                secondary_color_f32,
                            );
                        }

                        if is_primary_released && let Some(ref mut p) = self.shapes_state.placed {
                            p.handle_dragging = None;
                        }
                    } else {
                        // Drawing mode
                        if is_primary_pressed && !self.shapes_state.is_drawing {
                            self.shapes_state.is_drawing = true;
                            self.shapes_state.draw_start = Some([pos_f.0, pos_f.1]);
                            self.shapes_state.draw_end = Some([pos_f.0, pos_f.1]);
                            self.shapes_state.source_layer_index =
                                Some(canvas_state.active_layer_index);
                            self.stroke_tracker
                                .start_preview_tool(canvas_state.active_layer_index, "Shape");
                        }

                        if is_primary_down && self.shapes_state.is_drawing {
                            let mut end = [pos_f.0, pos_f.1];
                            // Shift: constrain to square/circle
                            if shift_held && let Some(start) = self.shapes_state.draw_start {
                                let dx = (end[0] - start[0]).abs();
                                let dy = (end[1] - start[1]).abs();
                                let side = dx.max(dy);
                                end[0] = start[0] + side * (end[0] - start[0]).signum();
                                end[1] = start[1] + side * (end[1] - start[1]).signum();
                            }
                            self.shapes_state.draw_end = Some([end[0].round(), end[1].round()]);
                            self.render_shape_preview(
                                canvas_state,
                                primary_color_f32,
                                secondary_color_f32,
                            );
                        }

                        if is_primary_released && self.shapes_state.is_drawing {
                            self.shapes_state.is_drawing = false;
                            // Convert to placed shape for manipulation
                            if let (Some(start), Some(end)) =
                                (self.shapes_state.draw_start, self.shapes_state.draw_end)
                            {
                                let cx = ((start[0] + end[0]) * 0.5).round();
                                let cy = ((start[1] + end[1]) * 0.5).round();
                                let hw = ((end[0] - start[0]) * 0.5).abs();
                                let hh = ((end[1] - start[1]) * 0.5).abs();
                                if hw >= 0.5 && hh >= 0.5 {
                                    let primary = [
                                        (primary_color_f32[0] * 255.0) as u8,
                                        (primary_color_f32[1] * 255.0) as u8,
                                        (primary_color_f32[2] * 255.0) as u8,
                                        (primary_color_f32[3] * 255.0) as u8,
                                    ];
                                    let secondary = [
                                        (secondary_color_f32[0] * 255.0) as u8,
                                        (secondary_color_f32[1] * 255.0) as u8,
                                        (secondary_color_f32[2] * 255.0) as u8,
                                        (secondary_color_f32[3] * 255.0) as u8,
                                    ];
                                    self.shapes_state.placed =
                                        Some(crate::ops::shapes::PlacedShape {
                                            cx,
                                            cy,
                                            hw,
                                            hh,
                                            rotation: 0.0,
                                            kind: self.shapes_state.selected_shape,
                                            custom_shape: self.shapes_state.selected_custom_shape.clone(),
                                            custom_shape_data: self.shapes_state.selected_custom_shape_data.clone(),
                                            fill_mode: self.shapes_state.fill_mode,
                                            outline_width: self.properties.size,
                                            primary_color: primary,
                                            secondary_color: secondary,
                                            anti_alias: self.shapes_state.anti_alias,
                                            corner_radius: self.shapes_state.corner_radius,
                                            handle_dragging: None,
                                            drag_offset: [0.0, 0.0],
                                            drag_anchor: [0.0, 0.0],
                                            rotate_start_angle: 0.0,
                                            rotate_start_rotation: 0.0,
                                        });
                                    self.render_shape_preview(
                                        canvas_state,
                                        primary_color_f32,
                                        secondary_color_f32,
                                    );
                                } else {
                                    self.shapes_state.draw_start = None;
                                    self.shapes_state.draw_end = None;
                                    self.shapes_state.source_layer_index = None;
                                    canvas_state.clear_preview_state();
                                }
                            }
                        }
                    }
                }

                // Property sync is handled by update_shape_if_dirty() called from canvas update loop

                // Draw shape overlay (bounding box, handles)
                if self.shapes_state.placed.is_some() {
                    self.draw_shape_overlay(painter, canvas_rect, zoom);
                }
            }

            // ================================================================
            // GRADIENT TOOL - click+drag to define gradient direction
            // ================================================================
            Tool::Gradient => {
                let escape_pressed = escape_pressed_global;

                // Guard: auto-rasterize text layers before destructive gradient
                if is_primary_pressed
                    && let Some(layer) = canvas_state.layers.get(canvas_state.active_layer_index)
                    && layer.is_text_layer()
                {
                    self.pending_auto_rasterize = Some(canvas_state.active_layer_index);
                    return;
                }

                // Escape cancels active gradient
                if escape_pressed && self.gradient_state.drag_start.is_some() {
                    self.cancel_gradient(canvas_state);
                }

                // Enter commits active gradient  (deferred for loading bar)
                if enter_pressed
                    && self.gradient_state.drag_start.is_some()
                    && !self.gradient_state.dragging
                {
                    self.gradient_state.commit_pending = true;
                    self.gradient_state.commit_pending_frame = 0;
                }

                // Update stops from primary/secondary if using PrimarySecondary preset
                // and colors have changed
                if self.gradient_state.preset == GradientPreset::PrimarySecondary {
                    let p = [
                        (primary_color_f32[0] * 255.0) as u8,
                        (primary_color_f32[1] * 255.0) as u8,
                        (primary_color_f32[2] * 255.0) as u8,
                        (primary_color_f32[3] * 255.0) as u8,
                    ];
                    let s = [
                        (secondary_color_f32[0] * 255.0) as u8,
                        (secondary_color_f32[1] * 255.0) as u8,
                        (secondary_color_f32[2] * 255.0) as u8,
                        (secondary_color_f32[3] * 255.0) as u8,
                    ];
                    if self.gradient_state.stops.len() >= 2
                        && (self.gradient_state.stops[0].color != p
                            || self.gradient_state.stops[self.gradient_state.stops.len() - 1].color
                                != s)
                    {
                        self.gradient_state.stops[0] =
                            GradientStop::new(self.gradient_state.stops[0].position, p);
                        let last = self.gradient_state.stops.len() - 1;
                        self.gradient_state.stops[last] =
                            GradientStop::new(self.gradient_state.stops[last].position, s);
                        self.gradient_state.lut_dirty = true;
                    }
                } else if self.gradient_state.preset == GradientPreset::ForegroundTransparent {
                    let p = [
                        (primary_color_f32[0] * 255.0) as u8,
                        (primary_color_f32[1] * 255.0) as u8,
                        (primary_color_f32[2] * 255.0) as u8,
                        (primary_color_f32[3] * 255.0) as u8,
                    ];
                    if self.gradient_state.stops.len() >= 2
                        && self.gradient_state.stops[0].color != p
                    {
                        self.gradient_state.stops[0] =
                            GradientStop::new(self.gradient_state.stops[0].position, p);
                        let last = self.gradient_state.stops.len() - 1;
                        self.gradient_state.stops[last] = GradientStop::new(
                            self.gradient_state.stops[last].position,
                            [p[0], p[1], p[2], 0],
                        );
                        self.gradient_state.lut_dirty = true;
                    }
                }

                // Start/grab gradient - allow clicking outside canvas so handles
                // can be dragged off-edge (e.g. gradient extends past border)
                if is_primary_pressed && let Some(pos_f) = canvas_pos_unclamped {
                    let click_pos = Pos2::new(pos_f.0, pos_f.1);

                    // Check if clicking near an existing handle first
                    let handle_grab_radius = (12.0 / zoom).max(6.0); // screen-space ~12px
                    let mut grabbed_handle: Option<usize> = None;

                    if let Some(start) = self.gradient_state.drag_start
                        && (click_pos - start).length() < handle_grab_radius
                    {
                        grabbed_handle = Some(0);
                    }
                    if let Some(end) = self.gradient_state.drag_end
                        && (click_pos - end).length() < handle_grab_radius
                    {
                        grabbed_handle = Some(1);
                    }

                    if let Some(handle_idx) = grabbed_handle {
                        // Grab existing handle for repositioning
                        self.gradient_state.dragging = true;
                        self.gradient_state.dragging_handle = Some(handle_idx);
                    } else {
                        // No handle hit - commit previous and start new gradient.
                        // Immediate commit here (no defer) because the new
                        // gradient's drag_start is set on the same frame and
                        // commit_gradient clears it.  The old preview was
                        // already rendered at full res on mouse release.
                        if self.gradient_state.drag_start.is_some() {
                            self.commit_gradient(canvas_state);
                        }

                        self.gradient_state.drag_start = Some(click_pos);
                        self.gradient_state.drag_end = Some(click_pos);
                        self.gradient_state.dragging = true;
                        self.gradient_state.dragging_handle = None;
                        self.gradient_state.source_layer_index =
                            Some(canvas_state.active_layer_index);

                        // Start stroke tracking
                        self.stroke_tracker
                            .start_preview_tool(canvas_state.active_layer_index, "Gradient");
                        self.stroke_tracker.expand_bounds(egui::Rect::from_min_max(
                            egui::pos2(0.0, 0.0),
                            egui::pos2(canvas_state.width as f32, canvas_state.height as f32),
                        ));

                        if self.gradient_state.lut_dirty {
                            self.gradient_state.rebuild_lut();
                        }
                    }
                }

                if is_primary_down
                    && self.gradient_state.dragging
                    && let Some(pos_f) = canvas_pos_unclamped
                {
                    let new_pos = Pos2::new(pos_f.0, pos_f.1);
                    match self.gradient_state.dragging_handle {
                        Some(0) => self.gradient_state.drag_start = Some(new_pos),
                        Some(1) => self.gradient_state.drag_end = Some(new_pos),
                        _ => self.gradient_state.drag_end = Some(new_pos),
                    }
                    self.render_gradient_to_preview(canvas_state, gpu_renderer.as_deref_mut());
                    ui.ctx().request_repaint();
                }

                if is_primary_released && self.gradient_state.dragging {
                    // Save handle index BEFORE clearing it so we update the correct handle
                    let released_handle = self.gradient_state.dragging_handle;
                    self.gradient_state.dragging = false;
                    self.gradient_state.dragging_handle = None;
                    if let Some(pos_f) = canvas_pos_unclamped {
                        let new_pos = Pos2::new(pos_f.0, pos_f.1);
                        match released_handle {
                            Some(0) => self.gradient_state.drag_start = Some(new_pos),
                            Some(1) => self.gradient_state.drag_end = Some(new_pos),
                            _ => self.gradient_state.drag_end = Some(new_pos),
                        }
                        self.render_gradient_to_preview(canvas_state, gpu_renderer.as_deref_mut());
                    }
                }

                // If gradient is active (not dragging) and LUT changed, re-render
                if !self.gradient_state.dragging
                    && self.gradient_state.drag_start.is_some()
                    && self.gradient_state.lut_dirty
                {
                    self.render_gradient_to_preview(canvas_state, gpu_renderer.as_deref_mut());
                    ui.ctx().request_repaint();
                }
                // preview_dirty is handled separately via update_gradient_if_dirty()
                // so it works even when handle_input is blocked by UI interactions

                // Draw overlay handles
                self.draw_gradient_overlay(painter, canvas_rect, zoom, canvas_state);
            }

            // ================================================================

            _ => {}
        }
    }
}

