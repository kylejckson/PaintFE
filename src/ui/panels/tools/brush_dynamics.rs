impl ToolsPanel {
    pub(crate) fn select_brush_tip(&mut self, tip: BrushTip, assets: &Assets) {
        if tip == self.properties.brush_tip {
            self.rebuild_tip_mask(assets);
            return;
        }
        self.remember_paintbrush();
        let paintbrush = assets
            .get_brush_tip_data(tip.display_name())
            .and_then(|tip| tip.paintbrush);
        if paintbrush.is_some() && self.brush_paintbrush.is_none() {
            self.brush_before_paintbrush = Some(Box::new(self.properties.clone()));
        }
        self.properties.brush_tip = tip;
        if let Some(paintbrush) = paintbrush {
            self.apply_paintbrush(&crate::brushes::SavedPreset::for_tip(paintbrush));
        }
        if paintbrush.is_none() && self.brush_paintbrush.is_some() {
            let previous = self
                .brush_before_paintbrush
                .take()
                .map(|settings| *settings)
                .unwrap_or_default();
            let size = self.properties.size;
            let color = self.properties.color;
            let blend = self.properties.blending_mode;
            let tip = self.properties.brush_tip.clone();
            self.properties = previous;
            self.properties.size = size;
            self.properties.color = color;
            self.properties.blending_mode = blend;
            self.properties.brush_tip = tip;
        }
        if paintbrush.is_some()
            && let Some(saved) = self
                .brush_saved_presets
                .get(&self.paint_preset_key())
                .cloned()
        {
            self.apply_paintbrush(&saved);
        }
        self.brush_sweeps = [None; 4];
        self.brush_travel = [0.; 4];
        let name = self.properties.brush_tip.display_name().to_owned();
        self.brush_browser.recent.retain(|item| item != &name);
        self.brush_browser.recent.insert(0, name);
        self.brush_browser.recent.truncate(12);
        self.brush_motion = Default::default();
        self.brush_motion_arm = 0;
        self.rebuild_tip_mask(assets);
    }

    pub(crate) fn restore_brush_browser(
        &mut self,
        settings: &crate::config::settings::AppSettings,
    ) {
        self.brush_browser = settings.brush_browser.clone();
        self.brush_tip_picker_category = self.brush_browser.category.clone();
        self.brush_saved_presets = settings.brush_saved_presets.clone();
        self.painting_tool_settings = settings.painting_tool_settings.clone();
        self.restore_painting_tool(self.active_tool);
    }
    pub(crate) fn store_brush_browser(
        &mut self,
        settings: &mut crate::config::settings::AppSettings,
    ) {
        self.remember_paintbrush();
        self.remember_painting_tool();
        settings.painting_tool_settings = self.painting_tool_settings.clone();
        self.brush_browser.category = self.brush_tip_picker_category.clone();
        settings.brush_browser.category = self.brush_browser.category.clone();
        settings.brush_browser.favourites = self.brush_browser.favourites.clone();
        settings.brush_browser.recent = self.brush_browser.recent.clone();
        if let Some(compact) = self.brush_browser_compact_request.take() {
            settings.brush_browser.compact = compact;
        }
        settings.brush_saved_presets = self.brush_saved_presets.clone();
    }
    fn reset_paintbrush(&mut self) {
        let Some(tip) = self.brush_paintbrush else {
            return;
        };
        self.brush_saved_presets.remove(&self.paint_preset_key());
        self.apply_paintbrush(&crate::brushes::SavedPreset::for_tip(tip));
        self.brush_preview_cache.clear();
    }
    fn capture_paint_preset(&self) -> crate::brushes::SavedPreset {
        crate::brushes::SavedPreset {
            engine: self.properties.brush_engine.sanitized(),
            flow: self.properties.flow,
            opacity: self.properties.opacity,
            spacing: self.properties.spacing,
            hardness: self.properties.hardness,
            rotation: self.properties.tip_rotation,
            random_rotation: self.properties.tip_random_rotation,
            mode: self.properties.brush_mode as u8,
            pressure_size: self.properties.pressure_size,
            pressure_flow: self.properties.pressure_opacity,
            minimum_size: self.properties.pressure_min_size,
            minimum_flow: self.properties.pressure_min_opacity,
            anti_aliased: self.properties.anti_aliased,
            scatter: self.properties.scatter,
            hue_jitter: self.properties.hue_jitter,
            brightness_jitter: self.properties.brightness_jitter,
            rotation_range: self.properties.tip_rotation_range,
        }
    }
    fn remember_paintbrush(&mut self) {
        if self.brush_paintbrush.is_some() {
            self.brush_saved_presets
                .insert(self.paint_preset_key(), self.capture_paint_preset());
        }
    }
    fn apply_paintbrush(&mut self, saved: &crate::brushes::SavedPreset) {
        self.properties.brush_engine = saved.engine.sanitized();
        self.properties.flow = saved.flow.clamp(0., 1.);
        self.properties.opacity = saved.opacity.clamp(0., 1.);
        self.properties.spacing = saved.spacing.clamp(0.01, 2.);
        self.properties.hardness = saved.hardness.clamp(0., 1.);
        self.properties.tip_rotation = saved.rotation.rem_euclid(360.);
        self.properties.tip_random_rotation = saved.random_rotation;
        self.properties.brush_mode = BrushMode::all()
            .get(saved.mode as usize)
            .copied()
            .unwrap_or(BrushMode::Normal);
        self.properties.pressure_size = saved.pressure_size;
        self.properties.pressure_opacity = saved.pressure_flow;
        self.properties.pressure_min_size = saved.minimum_size.clamp(0., 1.);
        self.properties.pressure_min_opacity = saved.minimum_flow.clamp(0., 1.);
        self.properties.anti_aliased = saved.anti_aliased;
        self.properties.scatter = saved.scatter.clamp(0., 1.);
        self.properties.hue_jitter = saved.hue_jitter.clamp(0., 1.);
        self.properties.brightness_jitter = saved.brightness_jitter.clamp(0., 1.);
        self.properties.tip_rotation_range = (
            saved.rotation_range.0.clamp(0., 360.),
            saved
                .rotation_range
                .1
                .clamp(saved.rotation_range.0.clamp(0., 360.), 360.),
        );
    }

    fn prepare_brush_dab(&mut self, position: (f32, f32)) {
        let settings = self.properties.brush_engine.sanitized();
        let motion = &mut self.brush_motion[self.brush_motion_arm.min(3)];
        let angle = motion.advance(
            position,
            self.properties.size,
            settings.smoothing * (1.5 - settings.stiffness),
        );
        let mut transform =
            self.brush_paintbrush
                .map_or_else(crate::brushes::DabTransform::default, |tip| {
                    tip.transform(
                        settings,
                        self.tool_state.current_pressure,
                        self.brush_speed,
                        motion.lag,
                    )
                });
        transform.angle = if settings.follow_direction {
            angle.to_degrees() + 90.
        } else {
            0.
        };
        if !self.properties.brush_tip.is_circle() && self.brush_tip_geometry.is_none() {
            let ratio = self.pressure_size() / self.brush_tip_mask_size.max(1) as f32;
            transform.scale_x *= ratio;
            transform.scale_y *= ratio;
        }
        self.brush_dab_transform = transform;
    }
    pub(crate) fn brush_cursor_transform(&self) -> crate::brushes::DabTransform {
        let mut transform =
            self.brush_paintbrush
                .map_or_else(crate::brushes::DabTransform::default, |tip| {
                    tip.transform(
                        self.properties.brush_engine,
                        self.tool_state.current_pressure,
                        self.brush_speed,
                        self.brush_motion[0].lag,
                    )
                });
        transform.angle = if self.properties.tip_random_rotation {
            0.
        } else {
            self.properties.tip_rotation
                + if self.properties.brush_engine.follow_direction {
                    self.brush_motion[0].angle().to_degrees() + 90.
                } else {
                    0.
                }
        };
        transform
    }

    fn show_inline_brush_controls(&self, settings: &crate::config::settings::AppSettings) -> bool {
        matches!(self.active_tool, Tool::Brush | Tool::Eraser)
            && (settings.show_advanced_brush_controls
                || (settings.tablet_pressure_enabled
                    && (self.properties.pressure_size || self.properties.pressure_opacity))
                || (settings.brush_preset_controls
                    && self.brush_paintbrush.is_some_and(|tip| tip.has_bristles())))
    }

    fn maximum_brush_radius(&self) -> f32 {
        if self.properties.brush_tip.is_circle() {
            return self.pressure_size() * 0.5;
        }
        let mut scale = 1f32;
        if let Some(tip) = self.brush_paintbrush {
            for pressure in [0., 1.] {
                let shape = tip.transform(
                    self.properties.brush_engine,
                    pressure,
                    0.,
                    std::f32::consts::FRAC_PI_2,
                );
                scale = scale.max(shape.scale_y.max(shape.scale_x * (1. + shape.shear.abs())));
            }
        }
        self.pressure_size() * 0.5 * std::f32::consts::SQRT_2 * scale
    }

    fn show_brush_engine_controls(&mut self, ui: &mut egui::Ui, tablet: bool) {
        if !matches!(self.active_tool, Tool::Brush | Tool::Eraser) {
            return;
        }
        if !self.properties.brush_tip.is_circle() {
            ui.separator();
            ui.checkbox(
                &mut self.properties.brush_engine.follow_direction,
                crate::t!("tools.brush.follow_direction"),
            );
            ui.add_enabled_ui(self.properties.brush_engine.follow_direction, |ui| {
                egui::Grid::new("brush_direction_controls")
                    .num_columns(2)
                    .show(ui, |ui| {
                        Self::brush_engine_percent(
                            ui,
                            &mut self.properties.brush_engine.smoothing,
                            "tools.brush.smoothing",
                            0.,
                            0.35,
                        );
                    });
            });
        }

        if let Some(paintbrush) = self.brush_paintbrush {
            egui::Grid::new("paintbrush_controls")
                .num_columns(2)
                .show(ui, |ui| {
                    Self::brush_engine_percent(
                        ui,
                        &mut self.properties.brush_engine.roundness,
                        "tools.brush.roundness",
                        8.,
                        paintbrush.settings().roundness,
                    );
                    if paintbrush.has_bristles() {
                        for (key, value, default) in [
                            (
                                "tools.brush.paint_load",
                                &mut self.properties.brush_engine.paint_load,
                                paintbrush.settings().paint_load,
                            ),
                            (
                                "tools.brush.texture",
                                &mut self.properties.brush_engine.texture,
                                paintbrush.settings().texture,
                            ),
                            (
                                "tools.brush.thickness",
                                &mut self.properties.brush_engine.bristle_thickness,
                                paintbrush.settings().bristle_thickness,
                            ),
                            (
                                "tools.brush.stiffness",
                                &mut self.properties.brush_engine.stiffness,
                                paintbrush.settings().stiffness,
                            ),
                            (
                                "tools.brush.depletion",
                                &mut self.properties.brush_engine.depletion,
                                0.,
                            ),
                        ] {
                            Self::brush_engine_percent(ui, value, key, 0., default);
                        }
                        Self::brush_engine_percent(
                            ui,
                            &mut self.properties.brush_engine.density,
                            "tools.brush.density",
                            0.,
                            0.65,
                        );
                        Self::brush_engine_percent(
                            ui,
                            &mut self.properties.brush_engine.spread,
                            "tools.brush.spread",
                            0.,
                            0.35,
                        );
                        Self::brush_engine_percent(
                            ui,
                            &mut self.properties.brush_engine.speed_response,
                            "tools.brush.speed_response",
                            0.,
                            0.,
                        );
                    }
                });
            if paintbrush.has_bristles() && tablet {
                ui.checkbox(
                    &mut self.properties.brush_engine.pressure_spread,
                    crate::t!("tools.brush.pressure_spread"),
                );
            }
        }
        if tablet {
            egui::Grid::new("brush_pressure_response")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label(crate::t!("tools.brush.pressure_response"));
                    ui.add(
                        crate::ui::controls::Slider::new(
                            &mut self.properties.brush_engine.pressure_gamma,
                            0.25..=4.,
                        )
                        .id_salt("brush_pressure_gamma")
                        .double_click_reset_to(1.)
                        .max_decimals(2),
                    );
                });
        }
    }

    fn brush_engine_percent(
        ui: &mut egui::Ui,
        value: &mut f32,
        key: &str,
        minimum: f32,
        default: f32,
    ) {
        ui.label(crate::i18n::translate(key));
        let mut percent = *value * 100.;
        if ui
            .add(
                crate::ui::controls::Slider::new(&mut percent, minimum..=100.)
                    .id_salt(key)
                    .double_click_reset_to(default * 100.)
                    .suffix("%")
                    .max_decimals(0),
            )
            .changed()
        {
            *value = percent / 100.;
        }
        ui.end_row();
    }
}

#[cfg(test)]
mod paintbrush_tests {
    use super::*;

    fn review_stroke(tip: crate::brushes::PaintTip, load: f32) -> image::RgbaImage {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        let mut tools = ToolsPanel::default();
        tools.properties.size = 48.;
        tools.select_brush_tip(BrushTip::Image(tip.name().into()), &assets);
        tools.properties.brush_engine.paint_load = load;
        let mut pixels = TiledImage::new(192, 128);
        for x in 20..172 {
            tools.draw_circle_no_dirty(
                &mut pixels,
                192,
                128,
                (x as f32, 64.),
                false,
                false,
                [0., 0., 0., 1.],
                [0.; 4],
                None,
            );
        }
        pixels.to_rgba_image()
    }

    #[test]
    fn loaded_paint_has_a_full_body_and_load_controls_deposition() {
        use crate::brushes::PaintTip;
        for tip in [
            PaintTip::RoundPaint,
            PaintTip::FlatPaint,
            PaintTip::Filbert,
            PaintTip::DenseFan,
        ] {
            let image = review_stroke(tip, 1.);
            assert!(image.get_pixel(96, 64)[3] >= 242, "{} centre", tip.name());
            let empty = review_stroke(tip, 0.);
            assert!(empty.pixels().all(|p| p[3] == 0));
        }
        let loaded = review_stroke(PaintTip::RoundPaint, 1.);
        let light = review_stroke(PaintTip::RoundPaint, 0.25);
        assert!(
            (55..=65).contains(&light.get_pixel(96, 64)[3]),
            "paint load is a material ceiling"
        );
        let dry = review_stroke(PaintTip::Bristle, 1.);
        let sum = |image: &image::RgbaImage| image.pixels().map(|p| u64::from(p[3])).sum::<u64>();
        assert!(
            sum(&loaded) > sum(&dry) * 5 / 4,
            "loaded coverage {}, dry {}",
            sum(&loaded),
            sum(&dry)
        );
    }

    #[test]
    fn connected_fibres_cover_the_path_without_redepositing_on_a_stationary_sample() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        let mut tools = ToolsPanel::default();
        tools.properties.size = 32.;
        tools.select_brush_tip(BrushTip::Image("Rake".into()), &assets);
        let mut image = TiledImage::new(192, 128);
        for point in [(30., 64.), (150., 64.)] {
            tools.draw_circle_no_dirty(
                &mut image,
                192,
                128,
                point,
                false,
                false,
                [0., 0., 0., 1.],
                [0.; 4],
                None,
            );
        }
        assert!((40..140).all(|x| (40..90).any(|y| image.get_pixel(x, y)[3] > 0)));
        let before = image.to_rgba_image();
        tools.draw_circle_no_dirty(
            &mut image,
            192,
            128,
            (150., 64.),
            false,
            false,
            [0., 0., 0., 1.],
            [0.; 4],
            None,
        );
        assert_eq!(before, image.to_rgba_image());
    }

    #[test]
    fn paintbrush_edits_are_per_preset_and_survive_settings_roundtrip() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        let mut tools = ToolsPanel::default();
        tools.select_brush_tip(BrushTip::Image("Round Paint".into()), &assets);
        tools.properties.brush_engine.paint_load = 0.42;
        tools.properties.flow = 0.6;
        tools.properties.scatter = 0.3;
        tools.properties.pressure_min_size = 0.4;
        tools.properties.anti_aliased = false;
        tools.select_brush_tip(BrushTip::Image("Flat Paint".into()), &assets);
        assert_eq!(tools.properties.flow, 1.);
        tools.select_brush_tip(BrushTip::Image("Round Paint".into()), &assets);
        assert_eq!(tools.properties.brush_engine.paint_load, 0.42);
        assert_eq!(tools.properties.flow, 0.6);
        let mut settings = crate::config::settings::AppSettings::default();
        tools.store_brush_browser(&mut settings);
        let encoded = serde_json::to_string(&settings.brush_saved_presets).unwrap();
        settings.brush_saved_presets = serde_json::from_str(&encoded).unwrap();
        let mut restored = ToolsPanel::default();
        restored.restore_brush_browser(&settings);
        restored.select_brush_tip(BrushTip::Image("Round Paint".into()), &assets);
        assert_eq!(restored.properties.flow, 0.6);
        assert_eq!(restored.properties.scatter, 0.3);
        assert_eq!(restored.properties.pressure_min_size, 0.4);
        assert!(!restored.properties.anti_aliased);
        restored.reset_paintbrush();
        assert_eq!(restored.properties.flow, 1.);
        assert_eq!(restored.properties.brush_engine.paint_load, 1.);
    }

    fn assets(ctx: &egui::Context) -> Assets {
        let mut assets = Assets::new();
        assets.init(ctx);
        assets
    }

    #[test]
    fn thickness_texture_and_stiffness_have_effective_controls() {
        use crate::brushes::PaintTip;
        for tip in [PaintTip::RoundPaint, PaintTip::Bristle, PaintTip::Rake] {
            let mut settings = tip.settings();
            let before = tip.mask(128, settings);
            settings.bristle_thickness = 0.;
            assert_ne!(before, tip.mask(128, settings), "{} thickness", tip.name());
            settings = tip.settings();
            settings.texture = 0.;
            assert_ne!(before, tip.mask(128, settings), "{} texture", tip.name());
        }
        let mut soft = PaintTip::Bristle.settings();
        soft.stiffness = 0.;
        let mut hard = soft;
        hard.stiffness = 1.;
        assert!(
            PaintTip::Bristle.transform(soft, 1., 0., 1.).shear.abs()
                > PaintTip::Bristle.transform(hard, 1., 0., 1.).shear.abs()
        );
    }

    #[test]
    #[ignore = "manual all-preset raster timing"]
    fn profile_paintbrush_presets() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        for tip in crate::brushes::PaintTip::ALL {
            for size in [32., 128., 512.] {
                let mut times = Vec::new();
                for sample in 0..4 {
                    let mut tools = ToolsPanel::default();
                    tools.properties.size = size;
                    tools.select_brush_tip(BrushTip::Image(tip.name().into()), &assets);
                    let mut pixels = TiledImage::new(1024, 1024);
                    let begin = std::time::Instant::now();
                    for step in 0..30 {
                        tools.draw_circle_no_dirty(
                            &mut pixels,
                            1024,
                            1024,
                            (350. + step as f32 * size * 0.025, 512.),
                            false,
                            false,
                            [0., 0., 0., 1.],
                            [0.; 4],
                            None,
                        );
                    }
                    if sample > 0 {
                        times.push(begin.elapsed().as_secs_f64() * 1000.);
                    }
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"tip":tip.name(),"size":size,"dabs":30,"median_ms":times[1]})
                );
            }
        }
    }

    #[test]
    fn catalogue_contains_every_tip_and_bounds_picker_textures() {
        let ctx = egui::Context::default();
        let mut assets = assets(&ctx);
        let count: usize = assets
            .brush_tip_categories()
            .iter()
            .map(|category| category.tips.len())
            .sum();
        assert_eq!(count, 44 + crate::brushes::PaintTip::ALL.len());
        for category in assets.brush_tip_categories() {
            for name in &category.tips {
                let tip = assets.get_brush_tip_data(name).unwrap();
                assert!(tip.icon_size[0] <= 64 && tip.icon_size[1] <= 64);
                assert!(tip.mask.iter().any(|&value| value > 0), "{name}");
                assert!(tip.builtin);
            }
        }
        assert!(!assets.remove_brush_tip("Chalk"));
        assert!(!assets.remove_brush_tip("Fan Bristle"));
    }

    #[test]
    fn paintbrush_selection_restores_previous_stamp_settings() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        let mut tools = ToolsPanel::default();
        tools.properties.flow = 0.6;
        tools.properties.opacity = 0.4;
        tools.properties.size = 64.;
        tools.properties.pressure_size = true;
        tools.properties.pressure_min_size = 0.25;
        tools.properties.anti_aliased = false;
        tools.properties.scatter = 0.4;
        tools.select_brush_tip(BrushTip::Image("Round Bristle".into()), &assets);
        assert_eq!(
            tools.properties.flow,
            crate::brushes::PaintTip::Bristle.flow()
        );
        assert!(tools.properties.brush_engine.follow_direction);
        tools.select_brush_tip(BrushTip::Image("Fan Bristle".into()), &assets);
        tools.properties.size = 72.;
        tools.properties.pressure_size = false;
        tools.properties.anti_aliased = true;
        tools.properties.scatter = 0.;
        tools.select_brush_tip(BrushTip::Circle, &assets);
        assert!(tools.properties.pressure_size);
        assert_eq!(tools.properties.pressure_min_size, 0.25);
        assert!(!tools.properties.anti_aliased);
        assert_eq!(tools.properties.scatter, 0.4);
        assert_eq!(tools.properties.flow, 0.6);
        assert_eq!(tools.properties.opacity, 0.4);
        assert_eq!(tools.properties.size, 72.);
    }

    #[test]
    fn pressure_changes_actual_pixels_and_dirty_bounds_cover_the_dab() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        let mut counts = Vec::new();
        for force in [0.1, 1.] {
            let mut tools = ToolsPanel::default();
            tools.properties.size = 64.;
            tools.select_brush_tip(BrushTip::Image("Fan Bristle".into()), &assets);
            tools.properties.pressure_size = true;
            tools.properties.brush_engine.spread = 1.;
            tools.tool_state.current_pressure = force;
            let mut state = CanvasState::new(192, 192);
            state.preview_layer = Some(TiledImage::new(192, 192));
            let bounds = tools.draw_circle_and_get_bounds(
                &mut state,
                (96., 96.),
                false,
                false,
                [1., 0., 0., 1.],
                [0.; 4],
            );
            let pixels = state.preview_layer.unwrap().to_rgba_image();
            let mut count = 0;
            for (x, y, pixel) in pixels.enumerate_pixels() {
                if pixel[3] > 0 {
                    count += 1;
                    assert!(bounds.contains(egui::pos2(x as f32, y as f32)));
                }
            }
            assert!(count > 0);
            counts.push(count);
        }
        assert!(counts[1] > counts[0]);
    }

    #[test]
    fn mirror_arms_keep_independent_direction_state() {
        let mut tools = ToolsPanel::default();
        tools.properties.brush_tip = BrushTip::Image("Fan Bristle".into());
        tools.properties.brush_engine.follow_direction = true;
        tools.properties.brush_engine.smoothing = 0.;
        tools.brush_tip_mask_size = 32;
        tools.prepare_brush_dab((10., 10.));
        tools.prepare_brush_dab((20., 10.));
        let first = tools.brush_dab_transform.angle;
        let cursor = tools.brush_cursor_transform();
        tools.brush_motion_arm = 1;
        tools.prepare_brush_dab((100., 10.));
        tools.prepare_brush_dab((90., 10.));
        assert!((tools.brush_dab_transform.angle - first).abs() > 170.);
        assert_eq!(tools.brush_motion[0].last, Some((20., 10.)));
        assert_eq!(tools.brush_cursor_transform().angle, cursor.angle);
        assert_eq!(tools.brush_cursor_transform().shear, cursor.shear);
        tools.reset_brush_pointer_state();
        assert!(
            tools
                .brush_motion
                .iter()
                .all(|motion| motion.last.is_none())
        );
    }

    #[test]
    fn custom_overrides_restore_bundled_tips_and_invalidate_masks() {
        let ctx = egui::Context::default();
        let mut assets = assets(&ctx);
        let mut tools = ToolsPanel::default();
        tools.properties.size = 16.;
        tools.properties.hardness = 1.;
        tools.properties.anti_aliased = false;
        tools.select_brush_tip(BrushTip::Image("Square".into()), &assets);
        let original_revision = assets.get_brush_tip_data("Square").unwrap().revision;
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageLuma8(image::GrayImage::from_pixel(16, 16, image::Luma([128])))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        assets.load_brush_tip(&ctx, "Square", "Custom", png.get_ref());
        tools.rebuild_tip_mask(&assets);
        let custom = assets.get_brush_tip_data("Square").unwrap();
        assert!(
            !custom.builtin && custom.geometry.is_none() && custom.revision > original_revision
        );
        assert!(tools.brush_tip_mask.iter().all(|&value| value == 128));
        assert_eq!(
            assets
                .brush_tip_categories()
                .iter()
                .flat_map(|category| &category.tips)
                .filter(|name| *name == "Square")
                .count(),
            1
        );
        assert!(assets.remove_brush_tip("Square"));
        tools.rebuild_tip_mask(&assets);
        let restored = assets.get_brush_tip_data("Square").unwrap();
        assert!(
            restored.builtin
                && restored.geometry.is_some()
                && restored.revision > original_revision
        );
        assert!(tools.brush_tip_mask.iter().all(|&value| value == 255));
        assert!(!assets.remove_brush_tip("Square"));
    }

    #[test]
    fn generated_solid_tips_respect_aa_off_and_pixel_alignment() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        for name in ["Oval Brush", "Flat Brush"] {
            let mut tools = ToolsPanel::default();
            tools.properties.size = 16.;
            tools.select_brush_tip(BrushTip::Image(name.into()), &assets);
            tools.properties.anti_aliased = false;
            tools.properties.tip_rotation = 0.;
            tools.rebuild_tip_mask(&assets);
            assert!(
                tools
                    .brush_tip_mask
                    .iter()
                    .all(|&value| value == 0 || value == 255)
            );
            let mut images = Vec::new();
            for point in [(20.1, 20.1), (20.9, 20.9)] {
                tools.brush_coverage.clear();
                let mut pixels = TiledImage::new(48, 48);
                tools.draw_circle_no_dirty(
                    &mut pixels,
                    48,
                    48,
                    point,
                    false,
                    false,
                    [0., 0., 0., 1.],
                    [0.; 4],
                    None,
                );
                let image = pixels.to_rgba_image();
                assert!(image.pixels().any(|pixel| pixel[3] > 0));
                assert!(image.pixels().all(|pixel| pixel[3] == 0 || pixel[3] == 255));
                images.push(image);
            }
            assert_eq!(images[0], images[1], "{name}");
        }
    }

    #[test]
    fn bristles_keep_opacity_caps_and_selection_boundaries() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        let mut tools = ToolsPanel::default();
        tools.properties.size = 64.;
        tools.select_brush_tip(BrushTip::Image("Fan Bristle".into()), &assets);
        tools.properties.opacity = 0.4;
        tools.properties.flow = 0.1;
        let selection =
            image::GrayImage::from_fn(96, 96, |x, _| image::Luma([if x < 48 { 255 } else { 0 }]));
        let mut pixels = TiledImage::new(96, 96);
        for _ in 0..100 {
            tools.draw_circle_no_dirty(
                &mut pixels,
                96,
                96,
                (48., 48.),
                false,
                false,
                [0.2, 0.5, 0.8, 0.5],
                [0.; 4],
                Some(&selection),
            );
        }
        let image = pixels.to_rgba_image();
        assert!(image.pixels().any(|pixel| pixel[3] > 0));
        for (x, _, pixel) in image.enumerate_pixels() {
            assert!(pixel[3] <= 51);
            if x >= 48 {
                assert_eq!(pixel[3], 0);
            }
        }
    }

    #[test]
    fn swept_paint_and_erase_bounds_include_spread_and_rotation() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        for eraser in [false, true] {
            let mut tools = ToolsPanel::default();
            tools.properties.size = 64.;
            tools.select_brush_tip(BrushTip::Image("Fan Bristle".into()), &assets);
            tools.properties.brush_engine.spread = 1.;
            tools.properties.brush_engine.speed_response = 1.;
            let mut state = CanvasState::new(256, 192);
            state.preview_layer = Some(TiledImage::new(256, 192));
            let bounds = tools.draw_line_and_get_bounds(
                &mut state,
                (64., 96.),
                (192., 96.),
                eraser,
                false,
                [0.2, 0.3, 0.7, 1.],
                [0.; 4],
            );
            let pixels = state.preview_layer.unwrap().to_rgba_image();
            assert!(pixels.pixels().any(|pixel| pixel[3] > 0));
            for (x, y, pixel) in pixels.enumerate_pixels() {
                if pixel[3] > 0 {
                    assert!(bounds.contains(egui::pos2(x as f32, y as f32)));
                }
            }
        }
    }

    #[test]
    fn automatic_controls_only_expand_for_bristle_presets() {
        let mut tools = ToolsPanel::default();
        let mut settings = crate::config::settings::AppSettings {
            show_advanced_brush_controls: false,
            brush_preset_controls: true,
            ..Default::default()
        };
        assert!(!tools.show_inline_brush_controls(&settings));
        tools.brush_paintbrush = Some(crate::brushes::PaintTip::Oval);
        assert!(!tools.show_inline_brush_controls(&settings));
        tools.brush_paintbrush = Some(crate::brushes::PaintTip::Fan);
        assert!(tools.show_inline_brush_controls(&settings));
        settings.brush_preset_controls = false;
        assert!(!tools.show_inline_brush_controls(&settings));
        settings.show_advanced_brush_controls = true;
        assert!(tools.show_inline_brush_controls(&settings));
        tools.active_tool = Tool::Pencil;
        assert!(!tools.show_inline_brush_controls(&settings));
        tools.active_tool = Tool::Eraser;
        assert!(tools.show_inline_brush_controls(&settings));
    }

    #[test]
    #[ignore = "manual brush raster gallery"]
    fn render_brush_raster_review_images() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/compact-brush-review/raster");
        std::fs::create_dir_all(&output).unwrap();
        let mut catalogue = vec![("Basic".to_owned(), BrushTip::Circle)];
        for category in assets.brush_tip_categories() {
            for name in &category.tips {
                catalogue.push((category.name.clone(), BrushTip::Image(name.clone())));
            }
        }
        let mut manifest = Vec::new();
        for (category, tip) in catalogue {
            let mut tools = ToolsPanel::default();
            tools.properties.size = 64.;
            tools.select_brush_tip(tip.clone(), &assets);
            tools.properties.flow = 1.;
            tools.properties.opacity = 1.;
            tools.properties.hardness = 1.;
            tools.rebuild_tip_mask(&assets);
            tools.rebuild_brush_lut();
            let mut pixels = TiledImage::new(128, 128);
            tools.draw_circle_no_dirty(
                &mut pixels,
                128,
                128,
                (64., 64.),
                false,
                false,
                [0.1, 0.35, 0.7, 1.],
                [0.; 4],
                None,
            );
            let path = tip.display_name().replace(' ', "_") + ".png";
            assert!(
                pixels.to_rgba_image().pixels().any(|pixel| pixel[3] > 0),
                "{} stamp",
                tip.display_name()
            );
            pixels.to_rgba_image().save(output.join(&path)).unwrap();
            manifest.push(
                serde_json::json!({"name":tip.display_name(),"category":category,"path":path}),
            );
        }
        std::fs::write(
            output.join("catalogue.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        for tip in crate::brushes::PaintTip::ALL {
            let mut tools = ToolsPanel::default();
            tools.properties.size = 64.;
            tools.select_brush_tip(BrushTip::Image(tip.name().into()), &assets);
            let mut pixels = TiledImage::new(512, 192);
            let raw: Vec<_> = (0..150)
                .map(|index| {
                    (
                        45. + index as f32 * 2.8,
                        95. + (index as f32 * 0.055).sin() * 45.,
                    )
                })
                .collect();
            let mut remainder = 0.;
            let points = sample_brush_path(
                None,
                &raw,
                (tools.pressure_size() * tools.properties.spacing).max(0.5),
                &mut remainder,
            );
            for (index, point) in points.iter().enumerate() {
                tools.tool_state.current_pressure =
                    0.15 + index as f32 / points.len() as f32 * 0.85;
                tools.draw_circle_no_dirty(
                    &mut pixels,
                    512,
                    192,
                    *point,
                    false,
                    false,
                    [0.05, 0.3, 0.65, 1.],
                    [0.; 4],
                    None,
                );
            }
            pixels
                .to_rgba_image()
                .save(output.join(format!("stroke-{}.png", tip.name().replace(' ', "_"))))
                .unwrap();
        }
    }

    #[test]
    #[ignore = "manual paintbrush CPU profile"]
    fn profile_paintbrush_dabs() {
        let ctx = egui::Context::default();
        let assets = assets(&ctx);
        for bitmap in [true, false] {
            for size in [32., 64., 128.] {
                let mut samples = Vec::new();
                for iteration in 0..4 {
                    let mut tools = ToolsPanel::default();
                    tools.properties.size = size;
                    tools.bristle_bitmap_probe = bitmap;
                    tools.select_brush_tip(BrushTip::Image("Fan Bristle".into()), &assets);
                    let mut pixels = TiledImage::new(512, 512);
                    let start = std::time::Instant::now();
                    for step in 0..50 {
                        tools.draw_circle_no_dirty(
                            &mut pixels,
                            512,
                            512,
                            (
                                150. + step as f32 * 2.,
                                200. + (step as f32 * 0.15).sin() * 30.,
                            ),
                            false,
                            false,
                            [0.3, 0.5, 0.8, 1.],
                            [0.; 4],
                            None,
                        );
                    }
                    if iteration > 0 {
                        samples.push(start.elapsed().as_secs_f64() * 1000.);
                    }
                    assert!(pixels.to_rgba_image().pixels().any(|pixel| pixel[3] > 0));
                }
                samples.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"size":size,"dabs":50,"median_ms":samples[1],"algorithm":if bitmap {"bitmap"} else {"fibers"}})
                );
            }
        }
    }
}

impl ToolsPanel {
    fn painting_tool_key(tool: Tool) -> Option<&'static str> {
        match tool {
            Tool::Brush => Some("Brush"),
            Tool::Pencil => Some("Pencil"),
            Tool::Eraser => Some("Eraser"),
            _ => None,
        }
    }
    fn paint_preset_key(&self) -> String {
        let name = self.properties.brush_tip.display_name();
        if self.active_tool == Tool::Eraser {
            format!("eraser:{name}")
        } else {
            name.into()
        }
    }
    fn remember_painting_tool(&mut self) {
        if let Some(key) = Self::painting_tool_key(self.active_tool) {
            self.painting_tool_settings.insert(
                key.into(),
                crate::brushes::PaintingToolSettings {
                    size: self.properties.size,
                    tip: self.properties.brush_tip.display_name().into(),
                    blend: format!("{:?}", self.properties.blending_mode),
                    preset: self.capture_paint_preset(),
                },
            );
            self.painting_tool_backups
                .insert(key.into(), self.brush_before_paintbrush.clone());
        }
    }
    fn restore_painting_tool(&mut self, tool: Tool) {
        let Some(key) = Self::painting_tool_key(tool) else {
            return;
        };
        if let Some(saved) = self.painting_tool_settings.get(key).cloned() {
            self.apply_paintbrush(&saved.preset);
            self.properties.size = if saved.size.is_finite() {
                saved.size.clamp(1., 999.)
            } else {
                10.
            };
            self.properties.brush_tip = if saved.tip == "Circle" {
                BrushTip::Circle
            } else {
                BrushTip::Image(saved.tip)
            };
            self.properties.blending_mode = BlendMode::all()
                .iter()
                .copied()
                .find(|blend| format!("{blend:?}") == saved.blend)
                .unwrap_or(BlendMode::Normal);
        } else if tool != Tool::Brush {
            let colour = self.properties.color;
            self.properties = ToolProperties::default();
            self.properties.color = colour;
        }
        self.brush_before_paintbrush = self.painting_tool_backups.remove(key).flatten();
        self.brush_paintbrush = None;
        self.brush_tip_cache_key.0.clear();
    }
}
