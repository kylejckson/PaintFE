impl ToolsPanel {
    #[allow(clippy::too_many_arguments)]
    fn draw_loaded_contact(
        &mut self,
        target: &mut TiledImage,
        width: u32,
        height: u32,
        center: (f32, f32),
        previous: Option<((f32, f32), f32)>,
        color: [u8; 3],
        source_alpha: f32,
        eraser: bool,
        selection: Option<&GrayImage>,
    ) {
        let Some(_tip) = self.brush_paintbrush else {
            return;
        };
        let settings = self.properties.brush_engine.sanitized();
        let shape = self.brush_dab_transform;
        let half = self.brush_tip_mask_size.max(1) as f32 * 0.5;
        let (sin, cos) = self.active_tip_rotation_deg.to_radians().sin_cos();
        let lx = half * shape.scale_x * 0.96 * (1. + shape.shear.abs() * settings.roundness);
        let ly = half * shape.scale_y * settings.roundness * 0.96;
        let extent_x = cos.abs() * lx + sin.abs() * ly + 1.;
        let extent_y = sin.abs() * lx + cos.abs() * ly + 1.;
        let start = previous.map_or(center, |p| p.0);
        let x0 = (start.0.min(center.0) - extent_x).floor().max(0.) as u32;
        let y0 = (start.1.min(center.1) - extent_y).floor().max(0.) as u32;
        let x1 = ((start.0.max(center.0) + extent_x).ceil().max(0.) as u32).min(width);
        let y1 = ((start.1.max(center.1) + extent_y).ceil().max(0.) as u32).min(height);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let mut writer = self.brush_pixel_writer();
        let dx = center.0 - start.0;
        let dy = center.1 - start.1;
        let length = dx * dx + dy * dy;
        let distance = length.sqrt();
        let exposure = if previous.is_some() {
            distance / (self.properties.size * self.properties.spacing).max(0.5)
        } else {
            1.
        };
        writer.flow = 1. - (1. - writer.flow.clamp(0., 1.)).powf(exposure);
        let paint = settings.paint_load
            * (-self.brush_travel[self.brush_motion_arm.min(3)] / self.properties.size.max(1.)
                * settings.depletion
                * 0.3)
                .exp();

        if eraser {
            writer.flow *= paint;
        } else {
            writer.opacity *= paint;
        }
        let cs = crate::canvas::CHUNK_SIZE;
        for cy in y0 / cs..=(y1 - 1) / cs {
            for cx in x0 / cs..=(x1 - 1) / cs {
                let bx = cx * cs;
                let by = cy * cs;
                let mut precise = (!eraser
                    && matches!(writer.mode, BrushMode::Normal | BrushMode::BuildUp))
                .then(|| {
                    self.brush_coverage
                        .entry((cx, cy))
                        .or_insert_with(|| vec![0; (cs * cs) as usize])
                });
                let pixels = target.ensure_chunk_mut(cx, cy).as_mut();
                for y in y0.saturating_sub(by)..(y1 - by).min(cs) {
                    for x in x0.saturating_sub(bx)..(x1 - bx).min(cs) {
                        let offset = (y * cs + x) as usize;
                        if !eraser
                            && writer.mode == BrushMode::Normal
                            && source_alpha * writer.opacity >= 1.
                            && precise
                                .as_deref()
                                .is_some_and(|coverage| coverage[offset] == u16::MAX)
                        {
                            continue;
                        }
                        let gx = bx + x;
                        let gy = by + y;
                        if selection.is_some_and(|m| {
                            gx >= m.width() || gy >= m.height() || m.get_pixel(gx, gy)[0] == 0
                        }) {
                            continue;
                        }
                        let fx = gx as f32 + 0.5 - start.0;
                        let fy = gy as f32 + 0.5 - start.1;
                        let t = if length > 0.00001 {
                            (fx * dx + fy * dy) / length
                        } else {
                            1.
                        }
                        .clamp(0., 1.);
                        let px = fx - dx * t;
                        let py = fy - dy * t;
                        let v = (-px * sin + py * cos) / (half * shape.scale_y);
                        let u = (px * cos + py * sin) / (half * shape.scale_x) - shape.shear * v;
                        let mx = ((u + 1.) * half).floor() as i32;
                        let my = ((v + 1.) * half).floor() as i32;
                        let size = self.brush_tip_mask_size as i32;
                        if mx < 0 || my < 0 || mx >= size || my >= size {
                            continue;
                        }
                        let alpha = self.brush_tip_mask[(my * size + mx) as usize] as f32 / 255.;
                        if alpha <= 0. {
                            continue;
                        }
                        writer.write(
                            pixels,
                            offset * 4,
                            alpha,
                            source_alpha,
                            color[0],
                            color[1],
                            color[2],
                            eraser,
                            precise.as_deref_mut().map(|coverage| &mut coverage[offset]),
                        );
                    }
                }
            }
        }
    }

    fn draw_bristle_dab(
        &mut self,
        target: &mut TiledImage,
        width: u32,
        height: u32,
        center: (f32, f32),
        rotation: f32,
        color: [u8; 3],
        source_alpha: f32,
        eraser: bool,
        selection: Option<&GrayImage>,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        let arm = self.brush_motion_arm.min(3);
        let previous = self.brush_sweeps[arm];
        let distance = previous.map_or(0., |(point, _)| {
            (center.0 - point.0).hypot(center.1 - point.1)
        });
        if previous.is_some() && distance < 0.0001 {
            return;
        }
        self.brush_sweeps[arm] = Some((center, rotation));
        self.brush_travel[arm] += distance;
        if self.brush_paintbrush.is_some_and(|tip| tip.loaded()) {
            self.draw_loaded_contact(
                target,
                width,
                height,
                center,
                previous,
                color,
                source_alpha,
                eraser,
                selection,
            );
            return;
        }
        let transform = self.brush_dab_transform;
        let half = self.brush_tip_mask_size as f32 * 0.5;
        if !half.is_finite() || half <= 0. {
            return;
        }
        let (sin, cos) = rotation.to_radians().sin_cos();
        let mut writer = self.brush_pixel_writer();
        let settings = self.properties.brush_engine.sanitized();
        let paint = settings.paint_load
            * (-self.brush_travel[arm] / self.properties.size.max(1.) * settings.depletion * 0.3)
                .exp();
        if eraser {
            writer.flow *= paint;
        } else {
            writer.opacity *= paint;
        }
        let unit = (self.properties.size * self.properties.spacing).max(0.5);
        let exposure = if previous.is_some() {
            distance / unit
        } else {
            1.
        };
        writer.flow = 1. - (1. - writer.flow.clamp(0., 1.)).powf(exposure);
        let chunk_size = crate::canvas::CHUNK_SIZE;
        for fiber in &self.brush_fibers {
            let local_x = (fiber.x + transform.shear * fiber.y) * transform.scale_x * half;
            let local_y = fiber.y * transform.scale_y * half;
            let cx = center.0 + local_x * cos - local_y * sin;
            let cy = center.1 + local_x * sin + local_y * cos;
            let original_rx = fiber.rx * transform.scale_x * half;
            let original_ry = fiber.ry * transform.scale_y * half;
            // Subpixel hairs retain their paint energy instead of disappearing
            // between pixel centres or turning into fully opaque one-pixel dots.
            let rx = original_rx.max(0.5);
            let ry = original_ry.max(0.5);
            let energy = (original_rx * original_ry / (rx * ry)).clamp(0., 1.);
            let shear = transform.shear * transform.scale_x / transform.scale_y;
            let (ux, uy) = (cos * rx, sin * rx);
            let (vx, vy) = ((cos * shear - sin) * ry, (sin * shear + cos) * ry);
            let determinant = ux * vy - uy * vx;
            let inverse = 1. / determinant;
            let extent_x = ux.hypot(vx);
            let extent_y = uy.hypot(vy);
            let (px, py) = previous.map_or((cx, cy), |(point, angle)| {
                let (sin, cos) = angle.to_radians().sin_cos();
                (
                    point.0 + local_x * cos - local_y * sin,
                    point.1 + local_x * sin + local_y * cos,
                )
            });
            let min_x = ((cx.min(px) - extent_x - 1.).floor().max(0.) as u32).min(width);
            let min_y = ((cy.min(py) - extent_y - 1.).floor().max(0.) as u32).min(height);
            let max_x = ((cx.max(px) + extent_x + 1.).ceil().max(0.) as u32).min(width);
            let max_y = ((cy.max(py) + extent_y + 1.).ceil().max(0.) as u32).min(height);
            if min_x >= max_x || min_y >= max_y {
                continue;
            }
            for chunk_y in min_y / chunk_size..=(max_y - 1) / chunk_size {
                for chunk_x in min_x / chunk_size..=(max_x - 1) / chunk_size {
                    let base_x = chunk_x * chunk_size;
                    let base_y = chunk_y * chunk_size;
                    let x0 = min_x.saturating_sub(base_x);
                    let y0 = min_y.saturating_sub(base_y);
                    let x1 = (max_x - base_x).min(chunk_size);
                    let y1 = (max_y - base_y).min(chunk_size);
                    let mut precise = (!eraser
                        && matches!(writer.mode, BrushMode::Normal | BrushMode::BuildUp))
                    .then(|| {
                        self.brush_coverage
                            .entry((chunk_x, chunk_y))
                            .or_insert_with(|| vec![0u16; (chunk_size * chunk_size) as usize])
                    });
                    let pixels = target.ensure_chunk_mut(chunk_x, chunk_y).as_mut();
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let gx = base_x + x;
                            let gy = base_y + y;
                            if selection.is_some_and(|mask| {
                                gx >= mask.width()
                                    || gy >= mask.height()
                                    || mask.get_pixel(gx, gy)[0] == 0
                            }) {
                                continue;
                            }
                            let fx = gx as f32 + 0.5;
                            let fy = gy as f32 + 0.5;
                            let sx = cx - px;
                            let sy = cy - py;
                            let length = sx * sx + sy * sy;
                            let t = if length > 0.00001 {
                                ((fx - px) * sx + (fy - py) * sy) / length
                            } else {
                                1.
                            }
                            .clamp(0., 1.);
                            let dx = fx - (px + sx * t);
                            let dy = fy - (py + sy * t);
                            let u = (dx * vy - dy * vx) * inverse;
                            let v = (dy * ux - dx * uy) * inverse;
                            let alpha = (1. - u * u - v * v).clamp(0., 1.) * fiber.weight * energy;
                            if alpha <= 0. {
                                continue;
                            }
                            let offset = (y * chunk_size + x) as usize;
                            writer.write(
                                pixels,
                                offset * 4,
                                alpha,
                                source_alpha,
                                color[0],
                                color[1],
                                color[2],
                                eraser,
                                precise.as_deref_mut().map(|coverage| &mut coverage[offset]),
                            );
                        }
                    }
                }
            }
        }
    }
}
