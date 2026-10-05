impl ToolsPanel {
    pub fn pressure_size(&self) -> f32 {
        if self.properties.pressure_size {
            let p = self.tool_state.current_pressure;
            let min = self.properties.pressure_min_size;
            self.properties.size * (min + (1.0 - min) * p)
        } else {
            self.properties.size
        }
    }

    /// Compute effective flow accounting for pen pressure.
    /// Returns `self.properties.flow` scaled by pressure when pressure_opacity is enabled.
    fn pressure_flow(&self) -> f32 {
        if self.properties.pressure_opacity {
            let p = self.tool_state.current_pressure;
            let min = self.properties.pressure_min_opacity;
            self.properties.flow * (min + (1.0 - min) * p)
        } else {
            self.properties.flow
        }
    }

    /// B6: Rebuild brush alpha LUT when brush properties change.
    /// The LUT maps squared-distance ratio (0..255 → 0.0..1.0 of `dist_sq/radius_sq`)
    /// to alpha (0..255).  Eliminates per-pixel `sqrt` + `smoothstep`.
    pub fn rebuild_brush_lut(&mut self) {
        let params = (
            self.properties.size,
            self.properties.hardness,
            self.properties.anti_aliased,
        );
        if params == self.lut_params {
            return;
        }
        self.lut_params = params;

        let radius = self.properties.size / 2.0;
        if radius < 0.001 {
            self.brush_alpha_lut = [0u8; 256];
            return;
        }

        for i in 0..256 {
            let t_sq = i as f32 / 255.0; // squared distance ratio
            let dist = t_sq.sqrt() * radius; // linear distance
            let alpha = self.compute_brush_alpha(dist, radius);
            self.brush_alpha_lut[i] = (alpha * 255.0).round().min(255.0) as u8;
        }
    }

    /// Compute brush alpha as material falloff multiplied by geometric coverage.
    ///
    /// Hardness is the fraction of the radius that stays fully opaque; the
    /// remainder fades smoothly to **zero** at the rim (Photoshop-style). The
    /// previous profile ended at `alpha == hardness` at the rim, so every pass
    /// left a faint hard-edged ring — overlapping soft passes stacked rims
    /// instead of blending into each other.
    fn compute_brush_alpha(&self, dist: f32, radius: f32) -> f32 {
        Self::brush_alpha_profile(
            dist,
            radius,
            self.properties.hardness,
            self.properties.anti_aliased,
        )
    }

    fn brush_alpha_profile(dist: f32, radius: f32, hardness: f32, anti_aliased: bool) -> f32 {
        if radius <= 0.0 {
            return 0.0;
        }

        let safe_hardness = hardness.clamp(0.0, 1.0);
        let t = (dist / radius).clamp(0.0, 1.0);
        let material_alpha = if t <= safe_hardness {
            1.0
        } else {
            let span = 1.0 - safe_hardness;
            if span <= 1e-4 {
                1.0
            } else {
                let u = ((t - safe_hardness) / span).clamp(0.0, 1.0);
                1.0 - u * u * (3.0 - 2.0 * u) // smoothstep out to zero
            }
        };

        let coverage = if anti_aliased {
            let edge0 = radius + 0.5;
            let edge1 = radius - 0.5;
            if dist <= edge1 {
                1.0
            } else if dist >= edge0 {
                0.0
            } else {
                let x = ((dist - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            }
        } else if dist <= radius {
            1.0
        } else {
            0.0
        };

        material_alpha * coverage
    }

    /// Compute alpha specifically for line tool with forced soft edges
    fn compute_line_alpha(
        &self,
        dist: f32,
        radius: f32,
        forced_hardness: f32,
        anti_alias: bool,
    ) -> f32 {
        // Hard-edge (no AA): binary cutoff at exact radius
        if !anti_alias {
            return if dist < radius { 1.0 } else { 0.0 };
        }

        // Anti-aliased: smoothstep fade
        let safe_hardness = forced_hardness.clamp(0.0, 0.99);

        // For very small radii (< 1.5px, i.e. size < 3px), keep the core at the
        // actual radius and add a 1px AA feather outside.  This prevents inflating
        // a 1px line into a ~5px blob while still giving smooth edges.
        let (effective_radius, fade_width) = if radius < 1.5 {
            // Core at actual radius, 1px AA feather outside
            (radius + 1.0, 1.0)
        } else if radius < 3.0 {
            // For tiny brushes, ensure at least 1.5px of AA range
            let aa_extend = 1.5;
            let extended_radius = radius + aa_extend;
            let fade = aa_extend + (radius * (1.0 - safe_hardness));
            (extended_radius, fade)
        } else {
            // Normal brushes: fade within the brush radius, but with larger fade for softness
            let fade = (radius * (1.0 - safe_hardness)).max(2.0); // Min 2px fade for softness
            (radius, fade)
        };

        let solid_radius = effective_radius - fade_width;

        if dist <= solid_radius {
            return 1.0;
        } else if dist >= effective_radius {
            return 0.0;
        }

        // Normalize distance within the fade region (0.0 to 1.0)
        let t = (dist - solid_radius) / fade_width;

        // Apply Smoothstep: t * t * (3 - 2t)
        // Invert t first because we want 1.0 at the inner edge and 0.0 at outer
        let x = 1.0 - t.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }

    /// Write one brush pixel with the current BrushMode semantics.
    /// Shared by the capsule segment sweep and the discrete stamp path.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn brush_pixel_writer(&self) -> BrushPixelWriter {
        BrushPixelWriter {
            mode: self.properties.brush_mode,
            flow: self.pressure_flow(),
            opacity: self.properties.opacity,
        }
    }

    #[cfg(test)]
    fn write_brush_pixel(
        &self,
        chunk_raw: &mut [u8],
        px_off: usize,
        geom_alpha: f32,
        src_a: f32,
        src_r8: u8,
        src_g8: u8,
        src_b8: u8,
        is_eraser: bool,
    ) {
        self.brush_pixel_writer().write(
            chunk_raw, px_off, geom_alpha, src_a, src_r8, src_g8, src_b8, is_eraser, None,
        );
    }

    /// Draw one drag segment as a swept capsule: every pixel within the brush
    /// radius of the segment `[start, end]` receives its falloff alpha exactly
    /// once. This is the true swept-disc coverage — unlike interpolated stamps
    /// it leaves no notches at direction changes and re-tracing behaves
    /// consistently. One pixel pass per segment (cheaper than N stamps).
    #[allow(clippy::too_many_arguments)]
    fn draw_capsule_no_dirty(
        &mut self,
        target_image: &mut TiledImage,
        width: u32,
        height: u32,
        start: (f32, f32),
        end: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        selection_mask: Option<&GrayImage>,
    ) {
        let radius = self.pressure_size() / 2.0;
        let radius_sq = radius * radius;
        if radius_sq < 0.001 {
            return;
        }
        let draw_radius = if self.properties.anti_aliased {
            radius + 0.5
        } else {
            radius
        };
        let draw_radius_sq = draw_radius * draw_radius;
        let use_direct_alpha = draw_radius > radius;
        let aa_inner_radius_sq = (radius - 0.5).max(0.0).powi(2);
        let inv_radius_sq = 1.0 / radius_sq;

        // Segment geometry for point-to-segment distance.
        let (dx, dy) = (end.0 - start.0, end.1 - start.1);
        let len_sq = dx * dx + dy * dy;
        let inv_len_sq = if len_sq > 1e-6 { 1.0 / len_sq } else { 0.0 };

        // Bounding box of the swept capsule.
        let min_x = ((start.0.min(end.0) - draw_radius).floor().max(0.0)) as u32;
        let max_x = ((start.0.max(end.0) + draw_radius).ceil().max(0.0)) as u32;
        let min_y = ((start.1.min(end.1) - draw_radius).floor().max(0.0)) as u32;
        let max_y = ((start.1.max(end.1) + draw_radius).ceil().max(0.0)) as u32;
        let max_x = max_x.min(width.saturating_sub(1));
        let max_y = max_y.min(height.saturating_sub(1));
        if min_x > max_x || min_y > max_y {
            return;
        }

        let brush_color_f32 = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };
        let [src_r, src_g, src_b, src_a] = brush_color_f32;
        let src_r8 = (src_r * 255.0) as u8;
        let src_g8 = (src_g * 255.0) as u8;
        let src_b8 = (src_b * 255.0) as u8;

        let writer = self.brush_pixel_writer();
        let lut = &self.brush_alpha_lut;
        let cs = crate::canvas::CHUNK_SIZE;

        let chunk_x0 = min_x / cs;
        let chunk_y0 = min_y / cs;
        let chunk_x1 = max_x / cs;
        let chunk_y1 = max_y / cs;

        for chunk_cy in chunk_y0..=chunk_y1 {
            for chunk_cx in chunk_x0..=chunk_x1 {
                let chunk_base_x = chunk_cx * cs;
                let chunk_base_y = chunk_cy * cs;

                let lx0 = min_x.saturating_sub(chunk_base_x);
                let ly0 = min_y.saturating_sub(chunk_base_y);
                let lx1 = (max_x + 1 - chunk_base_x).min(cs).min(width - chunk_base_x);
                let ly1 = (max_y + 1 - chunk_base_y)
                    .min(cs)
                    .min(height - chunk_base_y);
                if lx0 >= lx1 || ly0 >= ly1 {
                    continue;
                }

                let mut precise = (!is_eraser
                    && matches!(writer.mode, BrushMode::Normal | BrushMode::BuildUp))
                .then(|| {
                    self.brush_coverage
                        .entry((chunk_cx, chunk_cy))
                        .or_insert_with(|| vec![0u16; (cs * cs) as usize])
                });
                let chunk = target_image.ensure_chunk_mut(chunk_cx, chunk_cy);
                let chunk_raw = chunk.as_mut();
                let chunk_stride = cs as usize * 4;

                for ly in ly0..ly1 {
                    let global_y = chunk_base_y + ly;
                    let row_off = ly as usize * chunk_stride;

                    for lx in lx0..lx1 {
                        let global_x = chunk_base_x + lx;

                        if let Some(mask) = selection_mask {
                            if global_x < mask.width() && global_y < mask.height() {
                                if mask.get_pixel(global_x, global_y).0[0] == 0 {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        // Distance from the pixel to the segment.
                        let px = global_x as f32;
                        let py = global_y as f32;
                        let t = if inv_len_sq > 0.0 {
                            (((px - start.0) * dx + (py - start.1) * dy) * inv_len_sq)
                                .clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let proj_x = start.0 + dx * t;
                        let proj_y = start.1 + dy * t;
                        let ddx = px - proj_x;
                        let ddy = py - proj_y;
                        let dist_sq = ddx * ddx + ddy * ddy;
                        if dist_sq > draw_radius_sq {
                            continue;
                        }

                        let geom_alpha_u8 = if use_direct_alpha && dist_sq > aa_inner_radius_sq {
                            (Self::brush_alpha_profile(
                                dist_sq.sqrt(),
                                radius,
                                self.properties.hardness,
                                self.properties.anti_aliased,
                            ) * 255.0)
                                .round()
                                .min(255.0) as u8
                        } else {
                            let lut_idx = (dist_sq * inv_radius_sq * 255.0).min(255.0) as usize;
                            lut[lut_idx]
                        };
                        if geom_alpha_u8 == 0 {
                            continue;
                        }

                        let px_off = row_off + lx as usize * 4;
                        writer.write(
                            chunk_raw,
                            px_off,
                            geom_alpha_u8 as f32 / 255.0,
                            src_a,
                            src_r8,
                            src_g8,
                            src_b8,
                            is_eraser,
                            precise.as_deref_mut().map(|a| &mut a[px_off / 4]),
                        );
                    }
                }
            }
        }
    }

    pub fn draw_circle_no_dirty(
        &mut self,
        target_image: &mut TiledImage,
        width: u32,
        height: u32,
        pos: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        selection_mask: Option<&GrayImage>,
    ) {
        self.draw_circle_no_dirty_impl(
            target_image,
            width,
            height,
            pos,
            is_eraser,
            use_secondary,
            primary_color_f32,
            secondary_color_f32,
            selection_mask,
            true,
        );
    }

    fn draw_circle_no_dirty_impl(
        &mut self,
        target_image: &mut TiledImage,
        width: u32,
        height: u32,
        pos: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        selection_mask: Option<&GrayImage>,
        parallel: bool,
    ) {
        // Dispatch to image tip path if active
        if !self.properties.brush_tip.is_circle() {
            // Compute rotation angle for this stamp
            let rotation_deg = if self.properties.tip_random_rotation {
                // Hash position to get a deterministic-but-random angle
                let (lo, hi) = self.properties.tip_rotation_range;
                let range = hi - lo;
                if range.abs() < 0.01 {
                    lo
                } else {
                    // Simple hash of position for pseudorandom per-stamp rotation
                    let hash = Self::stamp_hash(pos.0, pos.1, self.stamp_counter);
                    lo + (hash % 10000) as f32 / 10000.0 * range
                }
            } else {
                self.properties.tip_rotation
            };
            self.draw_image_tip_no_dirty(
                target_image,
                width,
                height,
                pos,
                is_eraser,
                use_secondary,
                primary_color_f32,
                secondary_color_f32,
                selection_mask,
                rotation_deg,
            );
            return;
        }

        // Scatter: randomize stamp position by up to scatter*diameter
        let (cx, cy) = {
            let (px, py) = pos;
            if self.properties.scatter > 0.01 {
                let diam = self.pressure_size();
                let h1 = Self::stamp_hash(px, py, self.stamp_counter) as f32 / u32::MAX as f32;
                let h2 = Self::stamp_hash(py, px, self.stamp_counter.wrapping_add(99991)) as f32
                    / u32::MAX as f32;
                let ox = (h1 * 2.0 - 1.0) * self.properties.scatter * diam;
                let oy = (h2 * 2.0 - 1.0) * self.properties.scatter * diam;
                (px + ox, py + oy)
            } else {
                (px, py)
            }
        };
        let radius = self.pressure_size() / 2.0;
        let radius_sq = radius * radius;
        if radius_sq < 0.001 {
            return;
        }
        let draw_radius = if self.properties.anti_aliased {
            radius + 0.5
        } else {
            radius
        };
        let draw_radius_sq = draw_radius * draw_radius;
        let use_direct_alpha = draw_radius > radius;
        let aa_inner_radius_sq = (radius - 0.5).max(0.0).powi(2);
        let inv_radius_sq = 1.0 / radius_sq;

        let min_x = ((cx - draw_radius).floor().max(0.0)) as u32;
        let max_x = ((cx + draw_radius).ceil() as u32).min(width.saturating_sub(1));
        let min_y = ((cy - draw_radius).floor().max(0.0)) as u32;
        let max_y = ((cy + draw_radius).ceil() as u32).min(height.saturating_sub(1));
        if min_x > max_x || min_y > max_y {
            return;
        }

        // Determine brush color (high-precision unmultiplied)
        let brush_color_f32 = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };
        let [src_r, src_g, src_b, src_a] = brush_color_f32;
        let base_r8 = (src_r * 255.0) as u8;
        let base_g8 = (src_g * 255.0) as u8;
        let base_b8 = (src_b * 255.0) as u8;
        // Color jitter: per-stamp HSL perturbation
        let (src_r8, src_g8, src_b8) =
            if self.properties.hue_jitter > 0.01 || self.properties.brightness_jitter > 0.01 {
                let (mut h, s, mut l) = crate::ops::adjustments::rgb_to_hsl(src_r, src_g, src_b);
                if self.properties.hue_jitter > 0.01 {
                    let hh = Self::stamp_hash(
                        pos.0 + 0.1,
                        pos.1 + 0.2,
                        self.stamp_counter.wrapping_add(777),
                    ) as f32
                        / u32::MAX as f32;
                    h = (h + (hh * 2.0 - 1.0) * self.properties.hue_jitter * 0.5).fract();
                    if h < 0.0 {
                        h += 1.0;
                    }
                }
                if self.properties.brightness_jitter > 0.01 {
                    let bh = Self::stamp_hash(
                        pos.0 + 0.3,
                        pos.1 + 0.4,
                        self.stamp_counter.wrapping_add(555),
                    ) as f32
                        / u32::MAX as f32;
                    l = (l + (bh * 2.0 - 1.0) * self.properties.brightness_jitter * 0.5)
                        .clamp(0.0, 1.0);
                }
                let (nr, ng, nb) = crate::ops::adjustments::hsl_to_rgb(h, s, l);
                ((nr * 255.0) as u8, (ng * 255.0) as u8, (nb * 255.0) as u8)
            } else {
                (base_r8, base_g8, base_b8)
            };

        let writer = self.brush_pixel_writer();
        let lut = &self.brush_alpha_lut;
        let cs = crate::canvas::CHUNK_SIZE;

        // Determine which chunks overlap the brush bounding box
        let chunk_x0 = min_x / cs;
        let chunk_y0 = min_y / cs;
        let chunk_x1 = max_x / cs;
        let chunk_y1 = max_y / cs;

        // Parallelize independent chunks only for large dabs. Each dab finishes
        // before the next one starts, preserving per-pixel accumulation order.
        if parallel && (max_x - min_x + 1) as u64 * (max_y - min_y + 1) as u64 >= 65_536 {
            use crate::par_compat::*;
            let hardness = self.properties.hardness;
            let anti_aliased = self.properties.anti_aliased;
            let needs_coverage =
                !is_eraser && matches!(writer.mode, BrushMode::Normal | BrushMode::BuildUp);
            let mut jobs: Vec<_> = target_image
                .chunks_in_rect_mut(chunk_x0, chunk_y0, chunk_x1, chunk_y1)
                .map(|(x, y, chunk)| {
                    let coverage = needs_coverage.then(|| {
                        self.brush_coverage
                            .remove(&(x, y))
                            .unwrap_or_else(|| vec![0; (cs * cs) as usize])
                    });
                    (x, y, chunk, coverage)
                })
                .collect();
            jobs.par_iter_mut()
                .for_each(|(chunk_cx, chunk_cy, chunk, precise)| {
                    let chunk_cx = *chunk_cx;
                    let chunk_cy = *chunk_cy;
                    let chunk_base_x = chunk_cx * cs;
                    let chunk_base_y = chunk_cy * cs;

                    // Local pixel range within this chunk (clamped to brush bbox & canvas)
                    let lx0 = min_x.saturating_sub(chunk_base_x);
                    let ly0 = min_y.saturating_sub(chunk_base_y);
                    let lx1 = (max_x + 1 - chunk_base_x).min(cs).min(width - chunk_base_x);
                    let ly1 = (max_y + 1 - chunk_base_y)
                        .min(cs)
                        .min(height - chunk_base_y);
                    if lx0 >= lx1 || ly0 >= ly1 {
                        return;
                    }

                    // Quick check: does ANY pixel in this chunk-local range fall within the circle?
                    // Test the closest point of the local rect to the circle center
                    let near_x = (cx).clamp(
                        chunk_base_x as f32 + lx0 as f32,
                        chunk_base_x as f32 + lx1 as f32 - 1.0,
                    );
                    let near_y = (cy).clamp(
                        chunk_base_y as f32 + ly0 as f32,
                        chunk_base_y as f32 + ly1 as f32 - 1.0,
                    );
                    let nd = (near_x - cx) * (near_x - cx) + (near_y - cy) * (near_y - cy);
                    if nd > draw_radius_sq {
                        return;
                    }

                    let chunk_raw = chunk.as_mut();
                    let chunk_stride = cs as usize * 4;

                    for ly in ly0..ly1 {
                        let global_y = chunk_base_y + ly;
                        let dy = global_y as f32 - cy;
                        let dy_sq = dy * dy;
                        let row_off = ly as usize * chunk_stride;

                        for lx in lx0..lx1 {
                            let global_x = chunk_base_x + lx;

                            // Selection mask check
                            if let Some(mask) = selection_mask {
                                if global_x < mask.width() && global_y < mask.height() {
                                    if mask.get_pixel(global_x, global_y).0[0] == 0 {
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                            }

                            let dx = global_x as f32 - cx;
                            let dist_sq = dx * dx + dy_sq;
                            if dist_sq > draw_radius_sq {
                                continue;
                            }

                            let geom_alpha_u8 = if use_direct_alpha && dist_sq > aa_inner_radius_sq
                            {
                                (Self::brush_alpha_profile(
                                    dist_sq.sqrt(),
                                    radius,
                                    hardness,
                                    anti_aliased,
                                ) * 255.0)
                                    .round()
                                    .min(255.0) as u8
                            } else {
                                // B6: LUT lookup — replaces sqrt + smoothstep
                                let lut_idx = (dist_sq * inv_radius_sq * 255.0).min(255.0) as usize;
                                lut[lut_idx]
                            };
                            if geom_alpha_u8 == 0 {
                                continue;
                            }
                            let geom_alpha = geom_alpha_u8 as f32 / 255.0;

                            let px_off = row_off + lx as usize * 4;

                            writer.write(
                                chunk_raw,
                                px_off,
                                geom_alpha,
                                src_a,
                                src_r8,
                                src_g8,
                                src_b8,
                                is_eraser,
                                precise.as_deref_mut().map(|a| &mut a[px_off / 4]),
                            );
                        }
                    }
                });
            for (x, y, _, coverage) in jobs {
                if let Some(coverage) = coverage {
                    self.brush_coverage.insert((x, y), coverage);
                }
            }
            return;
        }

        for chunk_cy in chunk_y0..=chunk_y1 {
            for chunk_cx in chunk_x0..=chunk_x1 {
                let chunk_base_x = chunk_cx * cs;
                let chunk_base_y = chunk_cy * cs;

                // Local pixel range within this chunk (clamped to brush bbox & canvas)
                let lx0 = min_x.saturating_sub(chunk_base_x);
                let ly0 = min_y.saturating_sub(chunk_base_y);
                let lx1 = (max_x + 1 - chunk_base_x).min(cs).min(width - chunk_base_x);
                let ly1 = (max_y + 1 - chunk_base_y)
                    .min(cs)
                    .min(height - chunk_base_y);
                if lx0 >= lx1 || ly0 >= ly1 {
                    continue;
                }

                // Quick check: does ANY pixel in this chunk-local range fall within the circle?
                // Test the closest point of the local rect to the circle center
                let near_x = (cx).clamp(
                    chunk_base_x as f32 + lx0 as f32,
                    chunk_base_x as f32 + lx1 as f32 - 1.0,
                );
                let near_y = (cy).clamp(
                    chunk_base_y as f32 + ly0 as f32,
                    chunk_base_y as f32 + ly1 as f32 - 1.0,
                );
                let nd = (near_x - cx) * (near_x - cx) + (near_y - cy) * (near_y - cy);
                if nd > draw_radius_sq {
                    continue;
                }

                // Get or create chunk (COW-safe via ensure_chunk_mut)
                let mut precise = (!is_eraser
                    && matches!(writer.mode, BrushMode::Normal | BrushMode::BuildUp))
                .then(|| {
                    self.brush_coverage
                        .entry((chunk_cx, chunk_cy))
                        .or_insert_with(|| vec![0u16; (cs * cs) as usize])
                });
                let chunk = target_image.ensure_chunk_mut(chunk_cx, chunk_cy);
                let chunk_raw = chunk.as_mut();
                let chunk_stride = cs as usize * 4;

                for ly in ly0..ly1 {
                    let global_y = chunk_base_y + ly;
                    let dy = global_y as f32 - cy;
                    let dy_sq = dy * dy;
                    let row_off = ly as usize * chunk_stride;

                    for lx in lx0..lx1 {
                        let global_x = chunk_base_x + lx;

                        // Selection mask check
                        if let Some(mask) = selection_mask {
                            if global_x < mask.width() && global_y < mask.height() {
                                if mask.get_pixel(global_x, global_y).0[0] == 0 {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        let dx = global_x as f32 - cx;
                        let dist_sq = dx * dx + dy_sq;
                        if dist_sq > draw_radius_sq {
                            continue;
                        }

                        let geom_alpha_u8 = if use_direct_alpha && dist_sq > aa_inner_radius_sq {
                            (Self::brush_alpha_profile(
                                dist_sq.sqrt(),
                                radius,
                                self.properties.hardness,
                                self.properties.anti_aliased,
                            ) * 255.0)
                                .round()
                                .min(255.0) as u8
                        } else {
                            // B6: LUT lookup — replaces sqrt + smoothstep
                            let lut_idx = (dist_sq * inv_radius_sq * 255.0).min(255.0) as usize;
                            lut[lut_idx]
                        };
                        if geom_alpha_u8 == 0 {
                            continue;
                        }
                        let geom_alpha = geom_alpha_u8 as f32 / 255.0;

                        let px_off = row_off + lx as usize * 4;

                        writer.write(
                            chunk_raw,
                            px_off,
                            geom_alpha,
                            src_a,
                            src_r8,
                            src_g8,
                            src_b8,
                            is_eraser,
                            precise.as_deref_mut().map(|a| &mut a[px_off / 4]),
                        );
                    }
                }
            }
        }
    }

    /// Rescale the brush tip mask to the current brush size.
    /// Only rebuilds if tip or size changed. Called alongside rebuild_brush_lut.
    fn rebuild_tip_mask(&mut self, assets: &Assets) {
        if self.properties.brush_tip.is_circle() {
            return;
        }
        let tip_name = match &self.properties.brush_tip {
            BrushTip::Image(name) => name.clone(),
            BrushTip::Circle => return,
        };
        let target_size = (self.properties.size.ceil() as u32).max(1);
        let hardness_key = (self.properties.hardness * 100.0).round() as u32;
        let key = (tip_name.clone(), target_size, hardness_key);
        if key == self.brush_tip_cache_key {
            return;
        }
        self.brush_tip_cache_key = key;

        if let Some(data) = assets.get_brush_tip_data(&tip_name) {
            let src = &data.mask;
            let src_size = data.mask_size;
            if src_size == 0 {
                return;
            }

            let dst_size = target_size;
            self.brush_tip_mask
                .resize((dst_size * dst_size) as usize, 0);
            self.brush_tip_mask_size = dst_size;

            // Bilinear interpolation from source mask to target size
            let scale = src_size as f32 / dst_size as f32;
            for dy in 0..dst_size {
                for dx in 0..dst_size {
                    let sx = dx as f32 * scale;
                    let sy = dy as f32 * scale;
                    let sx0 = sx.floor() as u32;
                    let sy0 = sy.floor() as u32;
                    let sx1 = (sx0 + 1).min(src_size - 1);
                    let sy1 = (sy0 + 1).min(src_size - 1);
                    let fx = sx - sx0 as f32;
                    let fy = sy - sy0 as f32;

                    let v00 = src[(sy0 * src_size + sx0) as usize] as f32;
                    let v10 = src[(sy0 * src_size + sx1) as usize] as f32;
                    let v01 = src[(sy1 * src_size + sx0) as usize] as f32;
                    let v11 = src[(sy1 * src_size + sx1) as usize] as f32;

                    let top = v00 * (1.0 - fx) + v10 * fx;
                    let bot = v01 * (1.0 - fx) + v11 * fx;
                    let val = top * (1.0 - fy) + bot * fy;
                    self.brush_tip_mask[(dy * dst_size + dx) as usize] =
                        val.round().min(255.0) as u8;
                }
            }

            // Apply hardness as contrast modifier:
            // hardness 1.0 → use as-is
            // hardness 0.0 → heavily feathered (only brightest survive)
            let h = self.properties.hardness;
            if h < 0.99 {
                let threshold = (1.0 - h) * 0.6; // 0..0.6 range
                let range = 1.0 - threshold;
                for v in self.brush_tip_mask.iter_mut() {
                    let norm = *v as f32 / 255.0;
                    let adj = ((norm - threshold) / range).clamp(0.0, 1.0);
                    *v = (adj * 255.0).round() as u8;
                }
            }

            // Anti-alias pass: when downscaling significantly (src >> dst),
            // bilinear alone can't capture enough of the source detail and edges
            // look blocky. Apply a small box blur whose radius scales with the
            // downscale ratio. This smooths staircased edges while preserving shape.
            if dst_size < src_size && dst_size >= 3 {
                let ratio = src_size as f32 / dst_size as f32;
                // blur radius: 1 pass at 2-4× downscale, 2 at higher ratios
                let passes: usize = if ratio > 4.0 {
                    2
                } else if ratio > 1.5 {
                    1
                } else {
                    0
                };
                for _ in 0..passes {
                    // Horizontal pass
                    let mut tmp = self.brush_tip_mask.clone();
                    for y in 0..dst_size {
                        for x in 0..dst_size {
                            let idx = (y * dst_size + x) as usize;
                            let mut sum = self.brush_tip_mask[idx] as u32;
                            let mut count = 1u32;
                            if x > 0 {
                                sum += self.brush_tip_mask[(y * dst_size + x - 1) as usize] as u32;
                                count += 1;
                            }
                            if x + 1 < dst_size {
                                sum += self.brush_tip_mask[(y * dst_size + x + 1) as usize] as u32;
                                count += 1;
                            }
                            tmp[idx] = (sum / count) as u8;
                        }
                    }
                    // Vertical pass
                    for y in 0..dst_size {
                        for x in 0..dst_size {
                            let idx = (y * dst_size + x) as usize;
                            let mut sum = tmp[idx] as u32;
                            let mut count = 1u32;
                            if y > 0 {
                                sum += tmp[((y - 1) * dst_size + x) as usize] as u32;
                                count += 1;
                            }
                            if y + 1 < dst_size {
                                sum += tmp[((y + 1) * dst_size + x) as usize] as u32;
                                count += 1;
                            }
                            self.brush_tip_mask[idx] = (sum / count) as u8;
                        }
                    }
                }
            }
        } else {
            self.brush_tip_mask.clear();
            self.brush_tip_mask_size = 0;
        }
    }

    /// Stamp an image-based brush tip at the given position.
    /// Uses the pre-scaled tip mask from `brush_tip_mask`.
    /// `rotation_deg` applies rotation (degrees) to the mask sampling.
    fn draw_image_tip_no_dirty(
        &mut self,
        target_image: &mut TiledImage,
        width: u32,
        height: u32,
        pos: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        selection_mask: Option<&GrayImage>,
        rotation_deg: f32,
    ) {
        let mask_size = self.brush_tip_mask_size;
        if mask_size == 0 || self.brush_tip_mask.is_empty() {
            return;
        }

        // Scatter: randomize stamp position
        let (cx, cy) = {
            let (px, py) = pos;
            if self.properties.scatter > 0.01 {
                let diam = self.pressure_size();
                let h1 = Self::stamp_hash(px, py, self.stamp_counter) as f32 / u32::MAX as f32;
                let h2 = Self::stamp_hash(py, px, self.stamp_counter.wrapping_add(99991)) as f32
                    / u32::MAX as f32;
                let ox = (h1 * 2.0 - 1.0) * self.properties.scatter * diam;
                let oy = (h2 * 2.0 - 1.0) * self.properties.scatter * diam;
                (px + ox, py + oy)
            } else {
                (px, py)
            }
        };
        let half = mask_size as f32 / 2.0;

        // When rotated, the bounding box of the stamp expands.
        // The diagonal of the original square mask is half*sqrt(2) from center.
        let rotated = rotation_deg.abs() > 0.01;
        let (cos_a, sin_a) = if rotated {
            let rad = -rotation_deg.to_radians(); // negative = inverse rotation for sampling
            (rad.cos(), rad.sin())
        } else {
            (1.0, 0.0)
        };
        let effective_half = if rotated {
            half * std::f32::consts::SQRT_2
        } else {
            half
        };

        // Bounding box of the stamp in canvas coordinates
        let stamp_min_x = (cx - effective_half).max(0.0) as u32;
        let stamp_min_y = (cy - effective_half).max(0.0) as u32;
        let stamp_max_x = ((cx + effective_half) as u32).min(width.saturating_sub(1));
        let stamp_max_y = ((cy + effective_half) as u32).min(height.saturating_sub(1));
        if stamp_min_x > stamp_max_x || stamp_min_y > stamp_max_y {
            return;
        }

        // Brush color
        let brush_color_f32 = if use_secondary {
            secondary_color_f32
        } else {
            primary_color_f32
        };
        let [src_r, src_g, src_b, src_a] = brush_color_f32;
        let base_r8 = (src_r * 255.0) as u8;
        let base_g8 = (src_g * 255.0) as u8;
        let base_b8 = (src_b * 255.0) as u8;
        // Color jitter
        let (src_r8, src_g8, src_b8) =
            if self.properties.hue_jitter > 0.01 || self.properties.brightness_jitter > 0.01 {
                let (mut h, s, mut l) = crate::ops::adjustments::rgb_to_hsl(src_r, src_g, src_b);
                if self.properties.hue_jitter > 0.01 {
                    let hh = Self::stamp_hash(
                        pos.0 + 0.1,
                        pos.1 + 0.2,
                        self.stamp_counter.wrapping_add(777),
                    ) as f32
                        / u32::MAX as f32;
                    h = (h + (hh * 2.0 - 1.0) * self.properties.hue_jitter * 0.5).fract();
                    if h < 0.0 {
                        h += 1.0;
                    }
                }
                if self.properties.brightness_jitter > 0.01 {
                    let bh = Self::stamp_hash(
                        pos.0 + 0.3,
                        pos.1 + 0.4,
                        self.stamp_counter.wrapping_add(555),
                    ) as f32
                        / u32::MAX as f32;
                    l = (l + (bh * 2.0 - 1.0) * self.properties.brightness_jitter * 0.5)
                        .clamp(0.0, 1.0);
                }
                let (nr, ng, nb) = crate::ops::adjustments::hsl_to_rgb(h, s, l);
                ((nr * 255.0) as u8, (ng * 255.0) as u8, (nb * 255.0) as u8)
            } else {
                (base_r8, base_g8, base_b8)
            };

        let cs = crate::canvas::CHUNK_SIZE;
        let writer = self.brush_pixel_writer();
        let mask = &self.brush_tip_mask;

        // Chunk iteration (same pattern as draw_circle_no_dirty)
        let chunk_x0 = stamp_min_x / cs;
        let chunk_y0 = stamp_min_y / cs;
        let chunk_x1 = stamp_max_x / cs;
        let chunk_y1 = stamp_max_y / cs;

        for chunk_cy in chunk_y0..=chunk_y1 {
            for chunk_cx in chunk_x0..=chunk_x1 {
                let chunk_base_x = chunk_cx * cs;
                let chunk_base_y = chunk_cy * cs;

                let lx0 = stamp_min_x.saturating_sub(chunk_base_x);
                let ly0 = stamp_min_y.saturating_sub(chunk_base_y);
                let lx1 = (stamp_max_x + 1 - chunk_base_x)
                    .min(cs)
                    .min(width - chunk_base_x);
                let ly1 = (stamp_max_y + 1 - chunk_base_y)
                    .min(cs)
                    .min(height - chunk_base_y);
                if lx0 >= lx1 || ly0 >= ly1 {
                    continue;
                }

                let mut precise = (!is_eraser
                    && matches!(writer.mode, BrushMode::Normal | BrushMode::BuildUp))
                .then(|| {
                    self.brush_coverage
                        .entry((chunk_cx, chunk_cy))
                        .or_insert_with(|| vec![0u16; (cs * cs) as usize])
                });
                let chunk = target_image.ensure_chunk_mut(chunk_cx, chunk_cy);
                let chunk_raw = chunk.as_mut();
                let chunk_stride = cs as usize * 4;

                for ly in ly0..ly1 {
                    let global_y = chunk_base_y + ly;
                    let row_off = ly as usize * chunk_stride;

                    for lx in lx0..lx1 {
                        let global_x = chunk_base_x + lx;

                        // Selection mask check
                        if let Some(mask_img) = selection_mask {
                            if global_x < mask_img.width() && global_y < mask_img.height() {
                                if mask_img.get_pixel(global_x, global_y).0[0] == 0 {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }

                        // Map global pixel to mask coordinate, applying inverse rotation
                        let rel_x = global_x as f32 - cx;
                        let rel_y = global_y as f32 - cy;

                        let geom_alpha_u8 = if rotated {
                            // Inverse-rotate to find source mask position
                            let rot_x = rel_x * cos_a - rel_y * sin_a + half;
                            let rot_y = rel_x * sin_a + rel_y * cos_a + half;
                            // Bilinear sample from the unrotated mask
                            if rot_x < -0.5
                                || rot_y < -0.5
                                || rot_x >= mask_size as f32 - 0.5
                                || rot_y >= mask_size as f32 - 0.5
                            {
                                continue;
                            }
                            let sx = rot_x.max(0.0);
                            let sy = rot_y.max(0.0);
                            let sx0 = sx.floor() as u32;
                            let sy0 = sy.floor() as u32;
                            let sx1 = (sx0 + 1).min(mask_size - 1);
                            let sy1 = (sy0 + 1).min(mask_size - 1);
                            let fx = sx - sx0 as f32;
                            let fy = sy - sy0 as f32;
                            let v00 = mask[(sy0 * mask_size + sx0) as usize] as f32;
                            let v10 = mask[(sy0 * mask_size + sx1) as usize] as f32;
                            let v01 = mask[(sy1 * mask_size + sx0) as usize] as f32;
                            let v11 = mask[(sy1 * mask_size + sx1) as usize] as f32;
                            let top = v00 * (1.0 - fx) + v10 * fx;
                            let bot = v01 * (1.0 - fx) + v11 * fx;
                            let val = top * (1.0 - fy) + bot * fy;
                            val.round().min(255.0) as u8
                        } else {
                            let mask_x = (rel_x + half).round() as i32;
                            let mask_y = (rel_y + half).round() as i32;
                            if mask_x < 0
                                || mask_y < 0
                                || mask_x >= mask_size as i32
                                || mask_y >= mask_size as i32
                            {
                                continue;
                            }
                            mask[(mask_y as u32 * mask_size + mask_x as u32) as usize]
                        };
                        if geom_alpha_u8 == 0 {
                            continue;
                        }
                        let geom_alpha = geom_alpha_u8 as f32 / 255.0;

                        let px_off = row_off + lx as usize * 4;

                        writer.write(
                            chunk_raw,
                            px_off,
                            geom_alpha,
                            src_a,
                            src_r8,
                            src_g8,
                            src_b8,
                            is_eraser,
                            precise.as_deref_mut().map(|a| &mut a[px_off / 4]),
                        );
                    }
                }
            }
        }
    }

    pub fn draw_line_no_dirty(
        &mut self,
        target_image: &mut TiledImage,
        width: u32,
        height: u32,
        start: (f32, f32),
        end: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
        selection_mask: Option<&GrayImage>,
    ) {
        // Dense sub-pixel stepping for smooth lines
        let x0 = start.0;
        let y0 = start.1;
        let x1 = end.0;
        let y1 = end.1;

        let dx = x1 - x0;
        let dy = y1 - y0;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance < 0.1 {
            // Just draw one circle at start
            if start.0 >= 0.0
                && (start.0 as u32) < width
                && start.1 >= 0.0
                && (start.1 as u32) < height
            {
                self.draw_circle_no_dirty(
                    target_image,
                    width,
                    height,
                    start,
                    is_eraser,
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                    selection_mask,
                );
            }
            return;
        }

        // Clean circle tips: draw the segment as one swept capsule — the exact
        // swept-disc coverage. Interpolated stamps leave notches of soft pixels
        // at direction changes that later passes cannot fill (max-alpha); the
        // capsule sweep has no such gaps and re-traces consistently. Scatter
        // and colour jitter need discrete stamps, image tips are stamped by
        // design.
        let jitter = self.properties.hue_jitter > 0.01 || self.properties.brightness_jitter > 0.01;
        if self.properties.brush_tip.is_circle() && self.properties.scatter <= 0.01 && !jitter {
            self.draw_capsule_no_dirty(
                target_image,
                width,
                height,
                start,
                end,
                is_eraser,
                use_secondary,
                primary_color_f32,
                secondary_color_f32,
                selection_mask,
            );
            return;
        }

        // For image tips, use spacing-based stepping; for circle tips, dense per-pixel stepping
        let step = if self.properties.brush_tip.is_circle() {
            1.0
        } else {
            (self.pressure_size() * self.properties.spacing).max(1.0)
        };
        let steps = (distance / step).ceil() as usize;

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = x0 + dx * t;
            let y = y0 + dy * t;

            // Pass float position directly — no rounding — for sub-pixel smooth circles
            if x >= 0.0 && (x as u32) < width && y >= 0.0 && (y as u32) < height {
                self.draw_circle_no_dirty(
                    target_image,
                    width,
                    height,
                    (x, y),
                    is_eraser,
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                    selection_mask,
                );
            }
        }
    }

    fn mark_full_dirty(&self, canvas_state: &mut CanvasState) {
        canvas_state.dirty_rect = Some(Rect::from_min_max(
            Pos2::ZERO,
            Pos2::new(canvas_state.width as f32, canvas_state.height as f32),
        ));
    }

    /// Simple positional hash for pseudorandom per-stamp rotation.
    /// Produces a deterministic u32 from floating-point position + counter.
    fn stamp_hash(x: f32, y: f32, counter: u32) -> u32 {
        let ix = (x * 100.0) as u32;
        let iy = (y * 100.0) as u32;
        let mut h = ix
            .wrapping_mul(374761393)
            .wrapping_add(iy.wrapping_mul(668265263))
            .wrapping_add(counter.wrapping_mul(1013904223));
        h ^= h >> 13;
        h = h.wrapping_mul(1274126177);
        h ^= h >> 16;
        h
    }

    /// Draw a circle immediately to pixels and return its bounding box
    fn draw_circle_and_get_bounds(
        &mut self,
        canvas_state: &mut CanvasState,
        pos: (f32, f32),
        is_eraser: bool,
        use_secondary: bool,
        primary_color_f32: [f32; 4],
        secondary_color_f32: [f32; 4],
    ) -> Rect {
        // B6: Ensure brush alpha LUT is up-to-date
        self.rebuild_brush_lut();
        // Increment stamp counter for random rotation seeding
        self.stamp_counter = self.stamp_counter.wrapping_add(1);

        let (cx, cy) = pos;
        let radius = self.pressure_size() / 2.0;

        // Calculate bounds - expanded by max scatter offset so the dirty rect
        // covers stamps that landed far from the nominal position.
        let width = canvas_state.width;
        let height = canvas_state.height;
        // scatter moves center by up to scatter * size per axis
        let scatter_pad = self.properties.scatter * self.pressure_size();

        let min_x = (cx - radius - scatter_pad).max(0.0) as u32;
        let max_x = ((cx + radius + scatter_pad) as u32).min(width - 1);
        let min_y = (cy - radius - scatter_pad).max(0.0) as u32;
        let max_y = ((cy + radius + scatter_pad) as u32).min(height - 1);

        // All tools (Brush, Pencil, Eraser) write to the preview layer.
        // The eraser writes an erase-strength mask; the compositor handles the rest.
        {
            let mask_ptr = canvas_state
                .selection_mask
                .as_ref()
                .map(|m| m as *const GrayImage);
            if let Some(ref mut preview) = canvas_state.preview_layer {
                let mask_ref = mask_ptr.map(|p| unsafe { &*p });
                self.draw_circle_no_dirty(
                    preview,
                    width,
                    height,
                    pos,
                    is_eraser,
                    use_secondary,
                    primary_color_f32,
                    secondary_color_f32,
                    mask_ref,
                );
            }
        }

        // Return the bounding box (add 1 pixel padding)
        Rect::from_min_max(
            Pos2::new(
                min_x.saturating_sub(1) as f32,
                min_y.saturating_sub(1) as f32,
            ),
            Pos2::new(
                (max_x + 2).min(width) as f32,
                (max_y + 2).min(height) as f32,
            ),
        )
    }

    // ================================================================
}

/// Resample by travelled distance, independent of frame/event frequency.
fn sample_brush_path(
    mut previous: Option<(f32, f32)>,
    points: &[(f32, f32)],
    step: f32,
    remainder: &mut f32,
) -> Vec<(f32, f32)> {
    let mut samples = Vec::new();
    let step = step.max(0.5);
    *remainder = remainder.clamp(0.0, step);
    for &point in points {
        if let Some(start) = previous {
            let dx = point.0 - start.0;
            let dy = point.1 - start.1;
            let length = dx.hypot(dy);
            if length > 1e-6 {
                let mut distance = step - *remainder;
                while distance <= length + 1e-5 {
                    let travelled = distance.min(length);
                    samples.push((
                        start.0 + (dx / length) * travelled,
                        start.1 + (dy / length) * travelled,
                    ));
                    distance += step;
                }
                *remainder = (*remainder + length) % step;
                if *remainder < 1e-5 || step - *remainder < 1e-5 {
                    *remainder = 0.0;
                }
            }
        } else {
            samples.push(point);
            *remainder = 0.0;
        }
        previous = Some(point);
    }
    samples
}

#[cfg(test)]
mod mega_pass_brush_tests {
    use super::*;

    #[test]
    fn stationary_hold_has_no_extra_deposits() {
        let mut remainder = 0.0;
        assert_eq!(
            sample_brush_path(None, &[(4.0, 5.0)], 10.0, &mut remainder).len(),
            1
        );
        assert!(
            sample_brush_path(Some((4.0, 5.0)), &[(4.0, 5.0); 100], 10.0, &mut remainder)
                .is_empty()
        );
    }

    #[test]
    fn event_batches_produce_identical_deposits() {
        let mut r = 0.0;
        let whole = sample_brush_path(None, &[(0.0, 0.0), (35.0, 0.0), (35.0, 27.0)], 10.0, &mut r);
        let mut r = 0.0;
        let mut batches = sample_brush_path(None, &[(0.0, 0.0), (7.0, 0.0)], 10.0, &mut r);
        batches.extend(sample_brush_path(
            Some((7.0, 0.0)),
            &[(35.0, 0.0)],
            10.0,
            &mut r,
        ));
        batches.extend(sample_brush_path(
            Some((35.0, 0.0)),
            &[(35.0, 8.0), (35.0, 27.0)],
            10.0,
            &mut r,
        ));
        assert_eq!(whole, batches);
    }

    #[test]
    fn soft_overlap_builds_and_center_fills_without_exceeding_opacity() {
        let mut tools = ToolsPanel::default();
        tools.properties.opacity = 0.6;
        let mut pixel = [0u8; 4];
        tools.write_brush_pixel(&mut pixel, 0, 0.3, 1.0, 80, 40, 20, false);
        let first = pixel[3];
        tools.write_brush_pixel(&mut pixel, 0, 0.3, 1.0, 80, 40, 20, false);
        assert!(pixel[3] > first);
        tools.write_brush_pixel(&mut pixel, 0, 1.0, 1.0, 80, 40, 20, false);
        assert_eq!(pixel, [80, 40, 20, 153]);
        for _ in 0..100 {
            tools.write_brush_pixel(&mut pixel, 0, 0.5, 1.0, 80, 40, 20, false);
        }
        assert_eq!(pixel[3], 153);
    }

    #[test]
    fn large_soft_raster_crossing_fills_center_and_commits_identically() {
        let mut tools = ToolsPanel::default();
        tools.properties.size = 200.0;
        tools.properties.hardness = 0.0;
        tools.rebuild_brush_lut();
        let mut canvas = CanvasState::new(256, 256);
        canvas.layers[0].pixels = TiledImage::new(256, 256);
        let mut preview = TiledImage::new(256, 256);
        let color = [0.3, 0.2, 0.1, 1.0];
        tools.draw_circle_no_dirty(
            &mut preview,
            256,
            256,
            (128.0, 60.0),
            false,
            false,
            color,
            color,
            None,
        );
        let faint = preview.get_pixel(128, 128)[3];
        assert!(faint > 0 && faint < 200);
        tools.draw_circle_no_dirty(
            &mut preview,
            256,
            256,
            (128.0, 128.0),
            false,
            false,
            color,
            color,
            None,
        );
        assert_eq!(preview.get_pixel(128, 128)[3], 255);
        let expected = *preview.get_pixel(128, 100);
        canvas.preview_layer = Some(preview);
        tools.commit_bezier_to_layer(&mut canvas, color);
        assert_eq!(*canvas.layers[0].pixels.get_pixel(128, 100), expected);
    }

    #[test]
    fn brush_respects_selection_and_pressure_limit() {
        let mut tools = ToolsPanel::default();
        tools.properties.size = 30.0;
        tools.properties.hardness = 0.0;
        tools.properties.pressure_opacity = true;
        tools.properties.pressure_min_opacity = 0.0;
        tools.tool_state.current_pressure = 0.5;
        tools.rebuild_brush_lut();
        let mut preview = TiledImage::new(40, 40);
        let mut selection = GrayImage::new(40, 40);
        selection.put_pixel(20, 20, image::Luma([255]));
        let color = [0.0, 0.0, 0.0, 0.4];
        tools.draw_circle_no_dirty(
            &mut preview,
            40,
            40,
            (20.0, 20.0),
            false,
            false,
            color,
            color,
            Some(&selection),
        );
        assert_eq!(preview.get_pixel(20, 20)[3], 51);
        assert_eq!(preview.get_pixel(21, 20)[3], 0);
    }

    #[test]
    fn low_flow_keeps_accumulating_past_eight_bit_rounding_limit() {
        let writer = BrushPixelWriter {
            mode: BrushMode::Normal,
            flow: 0.01,
            opacity: 1.0,
        };
        let mut pixel = [0u8; 4];
        let mut precise = 0u16;
        for _ in 0..1000 {
            writer.write(&mut pixel, 0, 1.0, 1.0, 0, 0, 0, false, Some(&mut precise));
        }
        assert_eq!(pixel[3], 255);
    }

    #[test]
    fn uniform_center_replaces_faint_coverage() {
        let mut tools = ToolsPanel::default();
        tools.properties.brush_mode = BrushMode::Uniform;
        let mut pixel = [0u8; 4];
        tools.write_brush_pixel(&mut pixel, 0, 0.2, 1.0, 0, 0, 0, false);
        tools.write_brush_pixel(&mut pixel, 0, 1.0, 1.0, 0, 0, 0, false);
        assert_eq!(pixel[3], 255);
    }
}

struct BrushPixelWriter {
    mode: BrushMode,
    flow: f32,
    opacity: f32,
}
impl BrushPixelWriter {
    fn write(
        &self,
        chunk_raw: &mut [u8],
        px_off: usize,
        geom_alpha: f32,
        src_a: f32,
        src_r8: u8,
        src_g8: u8,
        src_b8: u8,
        is_eraser: bool,
        precise: Option<&mut u16>,
    ) {
        if is_eraser {
            let erase_strength = geom_alpha * src_a * self.flow;
            if erase_strength < 0.01 {
                return;
            }
            let old_mask = chunk_raw[px_off + 3] as f32 / 255.0;
            if erase_strength > old_mask {
                chunk_raw[px_off] = 0;
                chunk_raw[px_off + 1] = 0;
                chunk_raw[px_off + 2] = 0;
                chunk_raw[px_off + 3] = (erase_strength * 255.0) as u8;
            }
            return;
        }
        let brush_alpha = geom_alpha * src_a * self.flow;
        if brush_alpha <= 0.0 {
            return;
        }
        match self.mode {
            BrushMode::Uniform => {
                let brush_alpha_u8 = (brush_alpha * self.opacity * 255.0).round() as u8;
                let old_alpha = chunk_raw[px_off + 3];
                // Max-alpha stamping: only update if increasing opacity
                if brush_alpha_u8 >= old_alpha {
                    chunk_raw[px_off] = src_r8;
                    chunk_raw[px_off + 1] = src_g8;
                    chunk_raw[px_off + 2] = src_b8;
                    chunk_raw[px_off + 3] = brush_alpha_u8;
                }
            }
            BrushMode::Normal | BrushMode::BuildUp => {
                // Paint-like accumulation: each pass adds coverage over what is
                // already there (`1 - (1-a)(1-b)`), so painting over an area
                // reliably builds toward full opacity.
                let old_a = precise
                    .as_deref()
                    .map_or(chunk_raw[px_off + 3] as f32 / 255.0, |a| {
                        *a as f32 / 65535.0
                    });
                let cap = (src_a * self.opacity).clamp(0.0, 1.0);
                let add = (cap - old_a).max(0.0) * (geom_alpha * self.flow).clamp(0.0, 1.0);
                if add == 0.0 {
                    return;
                }
                let new_a = old_a + add;
                if new_a <= 0.0 {
                    return;
                }
                if let Some(a) = precise {
                    *a = (new_a * 65535.0).round() as u16;
                }
                if old_a == 0.0 {
                    chunk_raw[px_off..px_off + 3].copy_from_slice(&[src_r8, src_g8, src_b8]);
                } else if chunk_raw[px_off..px_off + 3] != [src_r8, src_g8, src_b8] {
                    let w_old = old_a / new_a;
                    let w_new = add / new_a;
                    chunk_raw[px_off] =
                        (chunk_raw[px_off] as f32 * w_old + src_r8 as f32 * w_new).round() as u8;
                    chunk_raw[px_off + 1] = (chunk_raw[px_off + 1] as f32 * w_old
                        + src_g8 as f32 * w_new)
                        .round() as u8;
                    chunk_raw[px_off + 2] = (chunk_raw[px_off + 2] as f32 * w_old
                        + src_b8 as f32 * w_new)
                        .round() as u8;
                }
                chunk_raw[px_off + 3] = (new_a * 255.0).round().min(255.0) as u8;
            }
            BrushMode::Dodge | BrushMode::Burn | BrushMode::Sponge => {
                // Read existing pixel, modify in HSL space, write back
                let old_r = chunk_raw[px_off] as f32 / 255.0;
                let old_g = chunk_raw[px_off + 1] as f32 / 255.0;
                let old_b = chunk_raw[px_off + 2] as f32 / 255.0;
                let (h, mut s, mut l) = crate::ops::adjustments::rgb_to_hsl(old_r, old_g, old_b);
                let strength = brush_alpha * 0.5;
                match self.mode {
                    BrushMode::Dodge => l = (l + strength).clamp(0.0, 1.0),
                    BrushMode::Burn => l = (l - strength).clamp(0.0, 1.0),
                    BrushMode::Sponge => s = (s - strength).clamp(0.0, 1.0),
                    _ => {}
                }
                let (nr, ng, nb) = crate::ops::adjustments::hsl_to_rgb(h, s, l);
                chunk_raw[px_off] = (nr * 255.0) as u8;
                chunk_raw[px_off + 1] = (ng * 255.0) as u8;
                chunk_raw[px_off + 2] = (nb * 255.0) as u8;
                // alpha unchanged
            }
        }
    }
}

#[cfg(test)]
mod brush_optimization_tests {
    use super::*;

    fn reference_write(
        pixel: &mut [u8; 4],
        coverage: &mut u16,
        flow: f32,
        opacity: f32,
        geom: f32,
        source: [u8; 3],
        source_alpha: f32,
    ) {
        if geom * source_alpha * flow <= 0.0 {
            return;
        }
        let old = *coverage as f32 / 65535.0;
        let cap = (source_alpha * opacity).clamp(0.0, 1.0);
        let add = (cap - old).max(0.0) * (geom * flow).clamp(0.0, 1.0);
        let next = old + add;
        if next <= 0.0 {
            return;
        }
        *coverage = (next * 65535.0).round() as u16;
        let a = old / next;
        let b = add / next;
        for c in 0..3 {
            pixel[c] = (pixel[c] as f32 * a + source[c] as f32 * b).round() as u8;
        }
        pixel[3] = (next * 255.0).round().min(255.0) as u8;
    }

    #[test]
    fn pixel_fast_path_matches_previous_formula() {
        for flow in [0.01, 0.3, 1.0] {
            for opacity in [0.0, 0.4, 1.0] {
                let writer = BrushPixelWriter {
                    mode: BrushMode::Normal,
                    flow,
                    opacity,
                };
                let (mut actual, mut expected) = ([0; 4], [0; 4]);
                let (mut coverage, mut reference) = (0, 0);
                for i in 0..2000 {
                    let source = if i < 1000 {
                        [80, 40, 20]
                    } else {
                        [20, 80, 150]
                    };
                    let alpha = if i < 1500 { 0.8 } else { 0.2 };
                    let geom = (i % 256) as f32 / 255.0;
                    writer.write(
                        &mut actual,
                        0,
                        geom,
                        alpha,
                        source[0],
                        source[1],
                        source[2],
                        false,
                        Some(&mut coverage),
                    );
                    reference_write(
                        &mut expected,
                        &mut reference,
                        flow,
                        opacity,
                        geom,
                        source,
                        alpha,
                    );
                    assert_eq!(
                        (actual, coverage),
                        (expected, reference),
                        "flow={flow} opacity={opacity} dab={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn parallel_large_dabs_match_serial_pixels_and_coverage() {
        for (mode, eraser, masked) in [
            (BrushMode::Normal, false, false),
            (BrushMode::BuildUp, false, true),
            (BrushMode::Uniform, false, false),
            (BrushMode::Normal, true, true),
        ] {
            let mut serial = ToolsPanel::default();
            serial.properties.size = 300.0;
            serial.properties.hardness = 0.0;
            serial.properties.flow = 0.07;
            serial.properties.opacity = 0.63;
            serial.properties.brush_mode = mode;
            serial.properties.hue_jitter = 0.2;
            serial.rebuild_brush_lut();
            let mut parallel = ToolsPanel {
                properties: serial.properties.clone(),
                ..Default::default()
            };
            parallel.rebuild_brush_lut();
            let mut a = TiledImage::new(389, 311);
            let mut b = TiledImage::new(389, 311);
            let mask = GrayImage::from_fn(389, 311, |x, y| {
                image::Luma([if (x + y) % 7 == 0 { 0 } else { 255 }])
            });
            let color = [0.3, 0.1, 0.7, 0.8];
            for (i, pos) in [
                (12.4, 10.8),
                (171.2, 158.9),
                (240.7, 154.3),
                (171.2, 158.9),
                (380.0, 300.0),
            ]
            .into_iter()
            .enumerate()
            {
                serial.stamp_counter = i as u32;
                parallel.stamp_counter = i as u32;
                serial.draw_circle_no_dirty_impl(
                    &mut a,
                    389,
                    311,
                    pos,
                    eraser,
                    false,
                    color,
                    color,
                    masked.then_some(&mask),
                    false,
                );
                parallel.draw_circle_no_dirty_impl(
                    &mut b,
                    389,
                    311,
                    pos,
                    eraser,
                    false,
                    color,
                    color,
                    masked.then_some(&mask),
                    true,
                );
            }
            for y in 0..311 {
                for x in 0..389 {
                    assert_eq!(a.get_pixel(x, y), b.get_pixel(x, y));
                }
            }
            for (key, values) in &serial.brush_coverage {
                assert_eq!(Some(values), parallel.brush_coverage.get(key));
            }
        }
    }

    #[test]
    #[ignore = "manual development-profile timing comparison"]
    fn large_brush_timing() {
        for use_parallel in [false, true] {
            let mut tools = ToolsPanel::default();
            tools.properties.size = 512.0;
            tools.properties.hardness = 0.0;
            tools.rebuild_brush_lut();
            let mut image = TiledImage::new(1536, 1024);
            let start = std::time::Instant::now();
            for i in 0..120 {
                let pos = (
                    256.0 + (i % 40) as f32 * 24.0,
                    350.0 + (i % 15) as f32 * 16.0,
                );
                tools.draw_circle_no_dirty_impl(
                    &mut image,
                    1536,
                    1024,
                    pos,
                    false,
                    false,
                    [0.3, 0.2, 0.1, 1.0],
                    [0.0; 4],
                    None,
                    use_parallel,
                );
            }
            eprintln!(
                "512px 120 dabs parallel={use_parallel}: {:?}",
                start.elapsed()
            );
            std::hint::black_box(image);
        }
    }
}
