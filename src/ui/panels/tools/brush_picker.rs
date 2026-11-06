impl ToolsPanel {
    fn browser_label(category: &str) -> String {
        match category {
            "All" => crate::t!("tools.brush.all"),
            "Favourites" => crate::t!("tools.browser.favourites"),
            "Recent" => crate::t!("tools.browser.recent"),
            _ => crate::i18n::translate_english(category),
        }
    }

    fn show_brush_tip_picker(&mut self, ui: &mut egui::Ui, assets: &Assets) {
        ui.label(crate::t!("ui.tip"));
        let popup_id = ui.make_persistent_id("brush_tip_grid_popup");
        let name = self.properties.brush_tip.display_name();
        let title = Self::brush_tip_label(&self.properties.brush_tip, assets);
        let response = if let Some(texture) = assets.get_brush_tip_texture(name) {
            ui.add(egui::Button::image_and_text(
                egui::Image::from_texture(texture).fit_to_exact_size(egui::vec2(16., 16.)),
                title,
            ))
        } else {
            ui.button(title)
        };
        if response.clicked() {
            self.remember_paintbrush();
            self.brush_browser_open = !self.brush_browser_open;
        }
        let mut frame = egui::Frame::popup(ui.style());
        if let Some(background) = self.brush_browser.background {
            frame.fill = background;
        }
        if let Some(radius) = self.brush_browser.rounding {
            frame.corner_radius = egui::CornerRadius::same(radius as u8);
        }
        let was_open = self.brush_browser_open;
        let mut open = self.brush_browser_open;
        let category_key = popup_id.with("category_widget");
        let category_id = ui
            .ctx()
            .data(|data| data.get_temp::<egui::Id>(category_key));
        let category_open = category_id.is_some_and(|id| egui::ComboBox::is_open(ui.ctx(), id));
        egui::Popup::new(
            popup_id,
            ui.ctx().clone(),
            egui::PopupAnchor::from(&response),
            ui.layer_id(),
        )
        .open_bool(&mut open)
        .close_behavior(if category_open {
            egui::PopupCloseBehavior::IgnoreClicks
        } else {
            egui::PopupCloseBehavior::CloseOnClickOutside
        })
        .frame(frame)
        .show(|ui| {
            if let Some(selection) = self.brush_browser.selection {
                ui.visuals_mut().selection.bg_fill = selection;
            }
            if let Some(hover) = self.brush_browser.hover {
                ui.visuals_mut().widgets.hovered.bg_fill = hover;
                ui.visuals_mut().widgets.hovered.weak_bg_fill = hover;
            }
            let available = ui.ctx().content_rect().size();
            let width = (available.x - 32.).clamp(220., 560.);
            ui.set_width(width);
            let colors = crate::ui::dialogs::DialogColors::from_ctx(ui.ctx());
            ui.horizontal(|ui| {
                ui.label(crate::ui::dialogs::title_text(
                    ui.ctx(),
                    &colors,
                    &crate::t!("tools.browser.brushes"),
                ));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(crate::t!("menu.file.new")).clicked() {
                        self.pending_open_add_brush_tip = true;
                        ui.close();
                    }
                    ui.menu_button("...", |ui| {
                        if ui
                            .checkbox(
                                &mut self.brush_browser.compact,
                                crate::t!("tools.browser.compact"),
                            )
                            .changed()
                        {
                            self.brush_browser_compact_request = Some(self.brush_browser.compact);
                        }
                        let current = self.properties.brush_tip.display_name().to_owned();
                        if ui
                            .button(crate::t!("tools.browser.favourite_toggle"))
                            .clicked()
                        {
                            self.toggle_brush_favourite(&current);
                        }
                        if self.brush_paintbrush.is_some()
                            && ui.button(crate::t!("tools.browser.reset_preset")).clicked()
                        {
                            self.reset_paintbrush();
                        }
                    });
                });
            });
            ui.horizontal(|ui| {
                let search = ui.add(
                    egui::TextEdit::singleline(&mut self.brush_tip_picker_search)
                        .id_salt("brush_tip_search")
                        .hint_text(crate::t!("tools.brush.search"))
                        .desired_width((ui.available_width() - 30.).max(80.)),
                );
                if ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::F)) {
                    search.request_focus();
                }
                if ui
                    .small_button("×")
                    .on_hover_text(crate::t!("tools.browser.clear_search"))
                    .clicked()
                {
                    self.brush_tip_picker_search.clear();
                    search.request_focus();
                }
            });
            ui.separator();
            let mut categories = vec![
                "All".to_owned(),
                "Favourites".into(),
                "Recent".into(),
                "Basic".into(),
                "Paintbrushes".into(),
            ];
            for category in assets.brush_tip_categories() {
                if !categories.contains(&category.name) {
                    categories.push(category.name.clone());
                }
            }
            if !categories.contains(&self.brush_tip_picker_category) {
                self.brush_tip_picker_category = "All".into();
            }
            let height = (available.y - 150.).clamp(100., 310.);
            if width < 440. {
                let selector = egui::ComboBox::from_id_salt("brush_category_select")
                    .selected_text(Self::browser_label(&self.brush_tip_picker_category))
                    .show_ui(ui, |ui| {
                        for category in &categories {
                            ui.push_id(category, |ui| {
                                if ui
                                    .selectable_label(
                                        self.brush_tip_picker_category == *category,
                                        Self::browser_label(category),
                                    )
                                    .clicked()
                                {
                                    self.brush_tip_picker_category = category.clone();
                                }
                            });
                        }
                    });
                ui.ctx()
                    .data_mut(|data| data.insert_temp(category_key, selector.response.id));
                self.show_browser_items(ui, assets, width, height - 30.);
            } else {
                let nav = (width * 0.25).clamp(110., 145.);
                ui.horizontal_top(|ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(nav, height),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("brush_categories_scroll")
                                .max_height(height)
                                .show(ui, |ui| {
                                    for category in &categories {
                                        ui.push_id(("brush_category", category), |ui| {
                                            if ui
                                                .add_sized(
                                                    egui::vec2(nav, ui.spacing().interact_size.y),
                                                    egui::Button::selectable(
                                                        self.brush_tip_picker_category == *category,
                                                        Self::browser_label(category),
                                                    )
                                                    .truncate(),
                                                )
                                                .clicked()
                                            {
                                                self.brush_tip_picker_category = category.clone();
                                            }
                                        });
                                    }
                                });
                        },
                    );
                    ui.separator();
                    self.show_browser_items(ui, assets, (width - nav - 24.).max(100.), height);
                });
            }
        });
        self.brush_browser_open = open;
        let frame_number = ui.ctx().cumulative_frame_nr();
        ui.ctx().data_mut(|data| {
            data.insert_temp(
                egui::Id::new("brush_browser_input"),
                (frame_number, open, was_open && !open),
            )
        });
    }

    fn toggle_brush_favourite(&mut self, name: &str) {
        if self
            .brush_browser
            .favourites
            .iter()
            .any(|item| item == name)
        {
            self.brush_browser.favourites.retain(|item| item != name);
        } else if self.brush_browser.favourites.len() < 128 {
            self.brush_browser.favourites.push(name.to_owned());
        }
    }

    fn browser_tips(&self, assets: &Assets) -> Vec<BrushTip> {
        let query = self.brush_tip_picker_search.to_lowercase();
        let mut tips = vec![BrushTip::Circle];
        for category in assets.brush_tip_categories() {
            for name in &category.tips {
                tips.push(BrushTip::Image(name.clone()));
            }
        }
        tips.retain(|tip| {
            let name = tip.display_name();
            if !query.is_empty() {
                return name.to_lowercase().contains(&query)
                    || Self::brush_tip_label(tip, assets)
                        .to_lowercase()
                        .contains(&query);
            }
            match self.brush_tip_picker_category.as_str() {
                "All" => true,
                "Favourites" => self
                    .brush_browser
                    .favourites
                    .iter()
                    .any(|item| item == name),
                "Recent" => self.brush_browser.recent.iter().any(|item| item == name),
                category => assets
                    .get_brush_tip_data(name)
                    .map_or(category == "Basic", |tip| tip.category == category),
            }
        });
        if self.brush_tip_picker_category == "Recent" && query.is_empty() {
            tips.sort_by_key(|tip| {
                self.brush_browser
                    .recent
                    .iter()
                    .position(|name| name == tip.display_name())
                    .unwrap_or(usize::MAX)
            });
        }
        tips
    }

    fn show_browser_items(&mut self, ui: &mut egui::Ui, assets: &Assets, width: f32, height: f32) {
        ui.allocate_ui_with_layout(
            egui::vec2(width, height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                ui.set_width(width);
                ui.set_min_height(height);
                let tips = self.browser_tips(assets);
                if tips.is_empty() {
                    ui.weak(crate::t!("tools.brush.no_matches"));
                    return;
                }
                let paint = |tip: &BrushTip| {
                    assets
                        .get_brush_tip_data(tip.display_name())
                        .is_some_and(|data| data.paintbrush.is_some_and(|tip| tip.has_bristles()))
                };
                let mut responses = Vec::new();
                let gap = self.brush_browser.spacing;
                let category = self.brush_tip_picker_category.clone();
                egui::ScrollArea::vertical()
                    .id_salt(("brush_tip_scroll", &category))
                    .max_height(height)
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(gap, gap);
                        let recent =
                            category == "Recent" && self.brush_tip_picker_search.is_empty();
                        for is_paint in [true, false] {
                            if recent && !is_paint {
                                continue;
                            }
                            let mut group: Vec<_> = tips
                                .iter()
                                .filter(|tip| recent || paint(tip) == is_paint)
                                .cloned()
                                .collect();
                            if is_paint && !recent {
                                group.sort_by_key(|tip| {
                                    assets
                                        .get_brush_tip_data(tip.display_name())
                                        .and_then(|data| data.paintbrush)
                                        .and_then(|tip| {
                                            crate::brushes::PaintTip::ALL
                                                .iter()
                                                .position(|p| *p == tip)
                                        })
                                        .unwrap_or(usize::MAX)
                                });
                            }
                            if group.is_empty() {
                                continue;
                            }
                            if !recent && tips.iter().any(|tip| paint(tip) != is_paint) {
                                ui.weak(if is_paint {
                                    crate::i18n::translate_english("Paintbrushes")
                                } else {
                                    crate::t!("tools.browser.stamps")
                                });
                            }
                            let columns = if is_paint {
                                if width >= 320. { 2 } else { 1 }
                            } else {
                                (width / (self.brush_browser.preview_size + 34.))
                                    .floor()
                                    .max(1.) as usize
                            };
                            let tile = (width - gap * (columns - 1) as f32) / columns as f32;
                            ui.push_id(is_paint, |ui| {
                                egui::Grid::new("brush_tip_tiles")
                                    .num_columns(columns)
                                    .spacing(egui::vec2(gap, gap))
                                    .show(ui, |ui| {
                                        for (index, tip) in group.into_iter().enumerate() {
                                            let actual_paint = paint(&tip);
                                            let response = self.show_browser_tile(
                                                ui,
                                                assets,
                                                tip,
                                                tile,
                                                actual_paint,
                                            );
                                            responses.push((response.id, columns));
                                            if (index + 1) % columns == 0 {
                                                ui.end_row();
                                            }
                                        }
                                    });
                            });
                        }
                    });
                if let Some(index) = responses
                    .iter()
                    .position(|(id, _)| ui.memory(|m| m.has_focus(*id)))
                {
                    let columns = responses[index].1;
                    let delta = ui.input_mut(|input| {
                        if input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight) {
                            1
                        } else if input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft) {
                            -1
                        } else if input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) {
                            columns as isize
                        } else if input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) {
                            -(columns as isize)
                        } else {
                            0
                        }
                    });
                    if delta != 0 {
                        let next = (index as isize + delta).clamp(0, responses.len() as isize - 1)
                            as usize;
                        self.brush_browser_scroll_focus = Some(responses[next].0);
                        ui.memory_mut(|m| m.request_focus(responses[next].0));
                    }
                }
            },
        );
    }

    fn brush_tip_label(tip: &BrushTip, assets: &Assets) -> String {
        if tip.is_circle() {
            crate::t!("ui.circle")
        } else if let Some(paint) = assets
            .get_brush_tip_data(tip.display_name())
            .and_then(|data| data.paintbrush)
        {
            crate::i18n::translate_english(paint.label())
        } else {
            tip.display_name().to_owned()
        }
    }

    fn brush_stroke_preview(
        &mut self,
        ui: &egui::Ui,
        assets: &Assets,
        tip: &BrushTip,
    ) -> egui::TextureHandle {
        let color = ui.visuals().text_color();
        let saved = self.brush_saved_presets.get(tip.display_name()).cloned();
        let key = format!(
            "{}:{:?}:{}",
            tip.display_name(),
            color,
            serde_json::to_string(&saved).unwrap_or_default()
        );
        if let Some(texture) = self.brush_preview_cache.get(&key) {
            return texture.clone();
        }
        let frame = ui.ctx().cumulative_frame_nr();
        let budget_id = egui::Id::new("brush_preview_budget");
        let busy = ui
            .ctx()
            .data(|data| data.get_temp::<u64>(budget_id) == Some(frame));
        if busy {
            ui.ctx().request_repaint();
            return assets
                .get_brush_tip_texture(tip.display_name())
                .expect("paintbrush thumbnail")
                .clone();
        }
        ui.ctx().data_mut(|data| data.insert_temp(budget_id, frame));
        let mut tools = ToolsPanel::default();
        tools.properties.size = 20.;
        tools.select_brush_tip(tip.clone(), assets);
        if let Some(saved) = saved {
            tools.apply_paintbrush(&saved);
            tools.rebuild_tip_mask(assets);
        }
        let mut image = TiledImage::new(192, 36);
        for step in 0..=336 {
            let x = 12. + step as f32 * 0.5;
            let y = 18. + (step as f32 / 336. * std::f32::consts::TAU).sin() * 6.;
            tools.draw_circle_no_dirty(
                &mut image,
                192,
                36,
                (x, y),
                false,
                false,
                [
                    color.r() as f32 / 255.,
                    color.g() as f32 / 255.,
                    color.b() as f32 / 255.,
                    1.,
                ],
                [0.; 4],
                None,
            );
        }
        let rgba = image.to_rgba_image();
        let texture = ui.ctx().load_texture(
            "brush_stroke_preview",
            egui::ColorImage::from_rgba_unmultiplied([192, 36], rgba.as_raw()),
            egui::TextureOptions::LINEAR,
        );
        if self.brush_preview_cache.len() >= 32 {
            self.brush_preview_cache.clear();
        }
        self.brush_preview_cache.insert(key, texture.clone());
        texture
    }

    fn show_browser_tile(
        &mut self,
        ui: &mut egui::Ui,
        assets: &Assets,
        tip: BrushTip,
        width: f32,
        paint: bool,
    ) -> egui::Response {
        let label = Self::brush_tip_label(&tip, assets);
        let preview = if paint {
            Some(self.brush_stroke_preview(ui, assets, &tip))
        } else {
            None
        };
        let font = egui::TextStyle::Small.resolve(ui.style());
        let icon_size = self.brush_browser.preview_size;
        let preview_height = if self.brush_browser.compact { 28. } else { 46. };
        let height = if paint {
            preview_height + 8. + ui.fonts_mut(|f| f.row_height(&font)) * 2.
        } else {
            icon_size + 6. + ui.fonts_mut(|f| f.row_height(&font)) * 2.
        };
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
        let response = ui.interact(
            rect,
            egui::Id::new(("brush_browser_item", tip.display_name())),
            egui::Sense::click(),
        );
        let selected = self.properties.brush_tip == tip;
        let radius = self
            .brush_browser
            .rounding
            .unwrap_or_else(|| crate::theme::Theme::tool_button_rounding_for(ui));
        if selected {
            ui.painter().rect_filled(
                rect,
                radius,
                self.brush_browser
                    .selection
                    .unwrap_or(ui.visuals().selection.bg_fill),
            );
        } else if response.hovered() {
            ui.painter().rect_filled(
                rect,
                radius,
                self.brush_browser
                    .hover
                    .unwrap_or(ui.visuals().widgets.hovered.weak_bg_fill),
            );
        }
        let text_color = if selected {
            ui.visuals().selection.stroke.color
        } else {
            ui.visuals().text_color()
        };
        let icon = if paint {
            egui::Rect::from_min_size(
                rect.min + egui::vec2(5., 4.),
                egui::vec2((width - 10.).max(1.), preview_height),
            )
        } else {
            egui::Rect::from_center_size(
                egui::pos2(rect.center().x, rect.top() + icon_size * 0.5 + 3.),
                egui::vec2(icon_size, icon_size),
            )
        };
        let texture = preview
            .as_ref()
            .or_else(|| assets.get_brush_tip_texture(tip.display_name()));
        if let Some(texture) = texture {
            ui.painter().image(
                texture.id(),
                icon,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                egui::Color32::WHITE,
            );
        } else {
            ui.painter()
                .circle_filled(icon.center(), icon_size * 0.4, text_color);
        }
        let galley = ui
            .painter()
            .layout(label.clone(), font, text_color, width - 6.);
        let text_y = if paint {
            rect.top() + preview_height + 8.
        } else {
            rect.top() + icon_size + 6.
        };
        ui.painter()
            .with_clip_rect(rect.intersect(ui.clip_rect()))
            .galley(egui::pos2(rect.left() + 3., text_y), galley, text_color);
        if response.has_focus() {
            ui.memory_mut(|memory| {
                memory.set_focus_lock_filter(
                    response.id,
                    egui::EventFilter {
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        ..Default::default()
                    },
                )
            });
        }
        if self.brush_browser_scroll_focus == Some(response.id) || response.gained_focus() {
            response.scroll_to_me(None);
            self.brush_browser_scroll_focus = None;
        }
        crate::ui::polish::focus(ui, &response, radius);
        response.widget_info(|| {
            egui::WidgetInfo::selected(
                egui::WidgetType::Button,
                ui.is_enabled(),
                selected,
                label.clone(),
            )
        });
        let activate = response.has_focus()
            && ui.input_mut(|i| {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
                    || i.consume_key(egui::Modifiers::NONE, egui::Key::Space)
            });
        if response.clicked() || activate {
            self.select_brush_tip(tip.clone(), assets);
            ui.close();
        }
        crate::ui::menus::context_menu(&response, |ui| {
            if ui
                .button(crate::t!("tools.browser.favourite_toggle"))
                .clicked()
            {
                self.toggle_brush_favourite(tip.display_name());
                ui.close();
            }
            if assets
                .get_brush_tip_data(tip.display_name())
                .is_some_and(|data| !data.builtin)
                && ui.button(crate::t!("common.delete")).clicked()
            {
                self.pending_delete_brush_tip = Some(tip.display_name().to_owned());
                ui.close();
            }
        });
        response.on_hover_text(label)
    }
}

#[cfg(test)]
pub(crate) mod brush_picker_tests {
    use super::*;

    #[test]
    fn browser_filters_favourites_recent_and_search_without_losing_category() {
        let ctx = review_context();
        let mut assets = Assets::new();
        assets.init(&ctx);
        let mut tools = ToolsPanel::default();
        tools.toggle_brush_favourite("Round Paint");
        tools.brush_tip_picker_category = "Favourites".into();
        assert_eq!(
            tools.browser_tips(&assets),
            [BrushTip::Image("Round Paint".into())]
        );
        tools.brush_tip_picker_search = "dry fan".into();
        assert!(
            tools
                .browser_tips(&assets)
                .contains(&BrushTip::Image("Fan Bristle".into()))
        );
        tools.brush_tip_picker_search.clear();
        assert_eq!(tools.brush_tip_picker_category, "Favourites");
        tools.select_brush_tip(BrushTip::Image("Rake".into()), &assets);
        tools.select_brush_tip(BrushTip::Image("Flat Paint".into()), &assets);
        tools.brush_tip_picker_category = "Recent".into();
        assert_eq!(
            tools.browser_tips(&assets)[0],
            BrushTip::Image("Flat Paint".into())
        );
        let mut settings = crate::config::settings::AppSettings::default();
        settings.brush_browser.preview_size = 38.;
        tools.store_brush_browser(&mut settings);
        assert_eq!(settings.brush_browser.preview_size, 38.);
        let mut restored = ToolsPanel::default();
        restored.restore_brush_browser(&settings);
        assert_eq!(restored.brush_tip_picker_category, "Recent");
        assert_eq!(restored.brush_browser.favourites, ["Round Paint"]);
    }

    fn review_context() -> egui::Context {
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        fonts.families.insert(
            egui::FontFamily::Name("WidgetTitle".into()),
            fonts.families[&egui::FontFamily::Proportional].clone(),
        );
        ctx.set_fonts(fonts);
        ctx
    }

    #[test]
    fn narrow_category_selector_keeps_the_browser_alive() {
        crate::i18n::init();
        let ctx = review_context();
        let mut assets = Assets::new();
        assets.init(&ctx);
        let mut tools = ToolsPanel::default();
        let size = egui::vec2(360., 480.);
        let mut last = None;
        for step in 0..4 {
            last = Some(
                frame(
                    &mut tools,
                    &assets,
                    &ctx,
                    size,
                    step as f64 * 0.05,
                    vec![],
                    true,
                )
                .0,
            );
        }
        let point = last
            .as_ref()
            .unwrap()
            .shapes
            .iter()
            .find_map(|s| find_text(&s.shape, &crate::t!("tools.brush.all")))
            .expect("category selector");
        let mut time = 0.3;
        for pressed in [true, false] {
            let _ = (frame(
                &mut tools,
                &assets,
                &ctx,
                size,
                time,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                false,
            )
            .0,);
            time += 0.05;
        }
        assert!(tools.brush_browser_open);
        last = Some(frame(&mut tools, &assets, &ctx, size, time, vec![], false).0);
        time += 0.05;
        let point = last
            .as_ref()
            .unwrap()
            .shapes
            .iter()
            .find_map(|s| find_text(&s.shape, &crate::i18n::translate_english("Texture")))
            .unwrap_or_else(|| {
                fn collect(shape: &egui::Shape, out: &mut Vec<String>) {
                    match shape {
                        egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
                        egui::Shape::Vec(shapes) => {
                            for shape in shapes {
                                collect(shape, out)
                            }
                        }
                        _ => {}
                    }
                }
                let mut texts = Vec::new();
                for shape in &last.as_ref().unwrap().shapes {
                    collect(&shape.shape, &mut texts);
                }
                panic!("texture category missing: {texts:?}")
            });
        for pressed in [true, false] {
            frame(
                &mut tools,
                &assets,
                &ctx,
                size,
                time,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                false,
            );
            time += 0.05;
        }
        assert!(tools.brush_browser_open);
        assert_eq!(tools.brush_tip_picker_category, "Texture");
    }

    #[test]
    fn keyboard_navigation_selects_only_on_enter_and_popup_guard_expires() {
        crate::i18n::init();
        let ctx = review_context();
        let mut assets = Assets::new();
        assets.init(&ctx);
        let mut tools = ToolsPanel {
            brush_tip_picker_category: "Paintbrushes".into(),
            ..Default::default()
        };
        let size = egui::vec2(900., 700.);
        for step in 0..4 {
            frame(
                &mut tools,
                &assets,
                &ctx,
                size,
                step as f64 * 0.05,
                vec![],
                true,
            );
        }
        ctx.memory_mut(|memory| {
            memory.request_focus(egui::Id::new(("brush_browser_item", "Round Paint")))
        });
        frame(&mut tools, &assets, &ctx, size, 0.25, vec![], false);
        frame(
            &mut tools,
            &assets,
            &ctx,
            size,
            0.3,
            vec![egui::Event::Key {
                key: egui::Key::ArrowDown,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            false,
        );
        assert!(tools.properties.brush_tip.is_circle());
        assert!(ctx.memory(|m| m.has_focus(egui::Id::new(("brush_browser_item", "Filbert")))));
        frame(
            &mut tools,
            &assets,
            &ctx,
            size,
            0.35,
            vec![egui::Event::Key {
                key: egui::Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            false,
        );
        assert_eq!(
            tools.properties.brush_tip,
            BrushTip::Image("Filbert".into())
        );
        assert!(!tools.brush_browser_open);

        let _ = ctx.run_ui(egui::RawInput::default(), |_ui| {
            crate::ui::polish::remember_popup_input(&ctx, false);
            assert!(
                !crate::ui::polish::popup_blocks_input(&ctx),
                "the next canvas interaction must not be swallowed"
            );
        });
    }

    #[test]
    fn view_menu_keeps_the_browser_visible_and_changes_density() {
        crate::i18n::init();
        let ctx = review_context();
        let mut assets = Assets::new();
        assets.init(&ctx);
        let mut tools = ToolsPanel::default();
        let size = egui::vec2(900., 700.);
        let mut last = None;
        for step in 0..4 {
            last = Some(
                frame(
                    &mut tools,
                    &assets,
                    &ctx,
                    size,
                    step as f64 * 0.05,
                    vec![],
                    true,
                )
                .0,
            );
        }
        let point = last
            .as_ref()
            .unwrap()
            .shapes
            .iter()
            .find_map(|s| find_text(&s.shape, "..."))
            .expect("view menu");
        let mut time = 0.3;
        for pressed in [true, false] {
            frame(
                &mut tools,
                &assets,
                &ctx,
                size,
                time,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                false,
            );
            time += 0.05;
        }
        last = Some(frame(&mut tools, &assets, &ctx, size, time, vec![], false).0);
        time += 0.05;
        assert!(tools.brush_browser_open);
        let point = last
            .as_ref()
            .unwrap()
            .shapes
            .iter()
            .find_map(|s| find_text(&s.shape, &crate::t!("tools.browser.compact")))
            .expect("view option");
        for pressed in [true, false] {
            frame(
                &mut tools,
                &assets,
                &ctx,
                size,
                time,
                vec![
                    egui::Event::PointerMoved(point),
                    egui::Event::PointerButton {
                        pos: point,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                false,
            );
            time += 0.05;
        }
        assert!(tools.brush_browser_open);
        assert!(!tools.brush_browser.compact);
    }

    fn frame(
        tools: &mut ToolsPanel,
        assets: &Assets,
        ctx: &egui::Context,
        size: egui::Vec2,
        time: f64,
        events: Vec<egui::Event>,
        open: bool,
    ) -> (egui::FullOutput, egui::Id) {
        let mut popup_id = egui::Id::NULL;
        let output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
                time: Some(time),
                events,
                ..Default::default()
            },
            |ui| {
                popup_id = ui.make_persistent_id("brush_tip_grid_popup");
                if open {
                    tools.brush_browser_open = true;
                }
                let was_open = tools.brush_browser_open;
                tools.show_brush_tip_picker(ui, assets);
                if was_open && !tools.brush_browser_open {
                    assert!(
                        crate::ui::polish::popup_blocks_input(ctx),
                        "dismissal is blocked in its own frame"
                    );
                }
            },
        );
        (output, popup_id)
    }

    fn find_text(shape: &egui::Shape, query: &str) -> Option<egui::Pos2> {
        match shape {
            egui::Shape::Text(text) if text.galley.text() == query => {
                Some(text.pos + text.galley.size() * 0.5)
            }
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|shape| find_text(shape, query)),
            _ => None,
        }
    }

    #[test]
    fn pencil_does_not_show_paintbrush_engine_options() {
        let ctx = review_context();
        let mut tools = ToolsPanel {
            brush_paintbrush: Some(crate::brushes::PaintTip::Fan),
            ..Default::default()
        };
        tools.properties.brush_tip = BrushTip::Image("Fan Bristle".into());
        tools.active_tool = Tool::Pencil;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let before = ui.min_rect();
            tools.show_brush_engine_controls(ui, true);
            assert_eq!(ui.min_rect(), before);
        });
    }

    #[test]
    fn brush_and_eraser_settings_keep_wheel_for_scrolling() {
        crate::i18n::init();
        for tool in [Tool::Brush, Tool::Eraser] {
            for dark in [false, true] {
                let ctx = review_context();
                ctx.set_visuals(if dark {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
                let mut assets = Assets::new();
                assets.init(&ctx);
                let mut tools = ToolsPanel {
                    active_tool: tool,
                    ..Default::default()
                };
                tools.properties.flow = 0.4;
                tools.properties.opacity = 0.6;
                let settings = crate::config::settings::AppSettings {
                    tablet_pressure_enabled: true,
                    ..Default::default()
                };
                let mut point = egui::Pos2::ZERO;
                for step in 0..6 {
                    let output = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(800., 600.),
                            )),
                            time: Some(step as f64 * 0.05),
                            events: if step == 5 {
                                vec![
                                    egui::Event::PointerMoved(point),
                                    egui::Event::MouseWheel {
                                        phase: egui::TouchPhase::Move,
                                        unit: egui::MouseWheelUnit::Line,
                                        delta: egui::vec2(0., -2.),
                                        modifiers: egui::Modifiers::NONE,
                                    },
                                ]
                            } else {
                                vec![]
                            },
                            ..Default::default()
                        },
                        |ui| {
                            ui.horizontal(|ui| {
                                let id = ui.make_persistent_id(if tool == Tool::Brush {
                                    "brush_dyn_popup"
                                } else {
                                    "eraser_dyn_popup"
                                });
                                egui::Popup::open_id(&ctx, id);
                                if tool == Tool::Brush {
                                    tools.show_brush_options(ui, &assets, &settings);
                                } else {
                                    tools.show_eraser_options(ui, &assets, &settings);
                                }
                            });
                        },
                    );
                    if step == 4 {
                        point = output
                            .shapes
                            .iter()
                            .find_map(|shape| find_text(&shape.shape, &crate::t!("ui.flow")))
                            .expect("visible Flow label")
                            + egui::vec2(85., 0.);
                    }
                }
                assert_eq!(tools.properties.flow, 0.4);
                assert_eq!(tools.properties.opacity, 0.6);
            }
        }
    }

    #[test]
    fn picker_fits_small_and_large_windows_in_both_themes() {
        crate::i18n::init();
        for dark in [false, true] {
            for scale in [1., 1.4] {
                for size in [
                    egui::vec2(360., 480.),
                    egui::vec2(720., 600.),
                    egui::vec2(1200., 800.),
                ] {
                    let ctx = review_context();
                    ctx.set_visuals(if dark {
                        egui::Visuals::dark()
                    } else {
                        egui::Visuals::light()
                    });
                    ctx.all_styles_mut(|style| {
                        for font in style.text_styles.values_mut() {
                            font.size *= scale;
                        }
                    });
                    ctx.data_mut(|data| {
                        data.insert_temp(
                            egui::Id::new("paintfe_polish"),
                            crate::ui::polish::PolishSettings {
                                text_scale: scale,
                                ..Default::default()
                            },
                        )
                    });
                    let mut assets = Assets::new();
                    assets.init(&ctx);
                    let mut png = std::io::Cursor::new(Vec::new());
                    image::DynamicImage::ImageLuma8(image::GrayImage::from_pixel(
                        8,
                        8,
                        image::Luma([128]),
                    ))
                    .write_to(&mut png, image::ImageFormat::Png)
                    .unwrap();
                    assets.load_brush_tip(
                        &ctx,
                        "Custom test tip",
                        "A long custom brush category for testing",
                        png.get_ref(),
                    );
                    let mut tools = ToolsPanel {
                        brush_tip_picker_category: "All".into(),
                        ..Default::default()
                    };
                    let mut popup_id = egui::Id::NULL;
                    for step in 0..4 {
                        popup_id = frame(
                            &mut tools,
                            &assets,
                            &ctx,
                            size,
                            step as f64 * 0.016,
                            vec![],
                            true,
                        )
                        .1;
                    }
                    let popup = ctx
                        .memory(|memory| memory.area_rect(popup_id))
                        .expect("picker area");
                    assert!(
                        popup.width() <= size.x - 8.,
                        "picker width {popup:?}, screen {size:?}, scale {scale}"
                    );
                    assert!(
                        popup.height() <= size.y - 8.,
                        "picker height {popup:?}, screen {size:?}, scale {scale}"
                    );
                    assert!(popup.min.x >= -1. && popup.min.y >= -1.);
                    assert!(popup.max.x <= size.x + 1. && popup.max.y <= size.y + 1.);
                }
            }
        }
    }

    #[test]
    fn searched_preset_selects_once_and_secondary_click_does_not_select() {
        crate::i18n::init();
        let ctx = review_context();
        let mut assets = Assets::new();
        assets.init(&ctx);
        let mut tools = ToolsPanel {
            brush_tip_picker_search: "Fan Bristle".into(),
            ..Default::default()
        };
        let size = egui::vec2(900., 700.);
        let mut output = None;
        for step in 0..4 {
            output = Some(frame(
                &mut tools,
                &assets,
                &ctx,
                size,
                step as f64 * 0.016,
                vec![],
                true,
            ));
        }
        let (last, _id) = output.unwrap();
        let label = ToolsPanel::brush_tip_label(&BrushTip::Image("Fan Bristle".into()), &assets);
        let point = last
            .shapes
            .iter()
            .filter_map(|shape| find_text(&shape.shape, &label))
            .max_by(|a, b| a.y.total_cmp(&b.y))
            .unwrap();
        let button = |kind, pressed| egui::Event::PointerButton {
            pos: point,
            button: kind,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        frame(
            &mut tools,
            &assets,
            &ctx,
            size,
            0.08,
            vec![
                egui::Event::PointerMoved(point),
                button(egui::PointerButton::Secondary, true),
            ],
            false,
        );
        frame(
            &mut tools,
            &assets,
            &ctx,
            size,
            0.096,
            vec![button(egui::PointerButton::Secondary, false)],
            false,
        );
        assert!(tools.properties.brush_tip.is_circle());
        assert!(tools.brush_browser_open);
        frame(
            &mut tools,
            &assets,
            &ctx,
            size,
            0.10,
            vec![egui::Event::Key {
                key: egui::Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            false,
        );
        frame(&mut tools, &assets, &ctx, size, 0.105, vec![], true);
        frame(
            &mut tools,
            &assets,
            &ctx,
            size,
            0.112,
            vec![button(egui::PointerButton::Primary, true)],
            false,
        );
        frame(
            &mut tools,
            &assets,
            &ctx,
            size,
            0.128,
            vec![button(egui::PointerButton::Primary, false)],
            false,
        );
        assert_eq!(
            tools.properties.brush_tip,
            BrushTip::Image("Fan Bristle".into())
        );
        assert_eq!(tools.properties.flow, crate::brushes::PaintTip::Fan.flow());
        assert!(!tools.brush_browser_open);
    }
    #[test]
    #[ignore = "manual offscreen brush picker captures"]
    fn render_brush_picker_review_images() {
        crate::i18n::init();
        let gpu = crate::gpu::context::GpuContext::new("").expect("offscreen UI adapter");
        for dark in [false, true] {
            for (large, category) in [
                (false, "All"),
                (false, "Favourites"),
                (false, "Recent"),
                (false, "Basic"),
                (false, "Paintbrushes"),
                (false, "Artistic"),
                (false, "Effects"),
                (false, "Texture"),
                (false, "Vegetation"),
                (true, "Paintbrushes"),
            ] {
                let ctx = review_context();
                ctx.set_visuals(if dark {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
                if large {
                    ctx.data_mut(|data| {
                        data.insert_temp(
                            egui::Id::new("paintfe_polish"),
                            crate::ui::polish::PolishSettings {
                                text_scale: 1.4,
                                ..Default::default()
                            },
                        )
                    });
                    ctx.all_styles_mut(|style| {
                        for font in style.text_styles.values_mut() {
                            font.size *= 1.4;
                        }
                    });
                }
                let mut assets = Assets::new();
                assets.init(&ctx);
                assets.update_theme(&ctx, dark);
                let mut tools = ToolsPanel {
                    brush_tip_picker_category: category.into(),
                    ..Default::default()
                };
                tools.brush_browser.favourites =
                    vec!["Circle".into(), "Round Paint".into(), "Fan Bristle".into()];
                tools.brush_browser.recent =
                    vec!["Filbert".into(), "Flat Paint".into(), "Circle".into()];
                let size = egui::vec2(1000., 760.);
                let mut textures = egui::TexturesDelta::default();
                let mut final_output = None;
                for step in 0..12 {
                    let (mut output, _) = frame(
                        &mut tools,
                        &assets,
                        &ctx,
                        size,
                        step as f64 * 0.05,
                        vec![],
                        true,
                    );
                    textures.append(std::mem::take(&mut output.textures_delta));
                    final_output = Some(output);
                }
                let mut output = final_output.unwrap();
                output.textures_delta = textures;
                let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("target/compact-brush-review/screenshots")
                    .join(format!(
                        "picker-{}-{}-{}.png",
                        if dark { "dark" } else { "light" },
                        if large { "large" } else { "normal" },
                        category.to_lowercase()
                    ));
                save_ui_capture(&gpu, &ctx, output, [1000, 760], &path);
            }
        }
    }

    pub(crate) fn save_ui_capture(
        gpu: &crate::gpu::context::GpuContext,
        ctx: &egui::Context,
        output: egui::FullOutput,
        size: [u32; 2],
        path: &std::path::Path,
    ) {
        let device = &gpu.device;
        let queue = &gpu.queue;
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut renderer = eframe::egui_wgpu::Renderer::new(
            device,
            format,
            eframe::egui_wgpu::RendererOptions::default(),
        );
        for (id, delta) in output.textures_delta.set {
            renderer.update_texture(device, queue, id, &delta);
        }
        let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
        let screen = eframe::egui_wgpu::ScreenDescriptor {
            size_in_pixels: size,
            pixels_per_point: output.pixels_per_point,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brush_picker_capture"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut commands = renderer.update_buffers(device, queue, &mut encoder, &jobs, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("brush_picker_capture"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.15,
                                g: 0.15,
                                b: 0.15,
                                a: 1.,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            renderer.render(&mut pass, &jobs, &screen);
        }
        let stride = (size[0] * 4).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush_picker_capture"),
            size: u64::from(stride) * u64::from(size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
        );
        commands.push(encoder.finish());
        queue.submit(commands);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(20)),
            })
            .unwrap();
        rx.recv_timeout(std::time::Duration::from_secs(20))
            .unwrap()
            .unwrap();
        let mapped = buffer.slice(..).get_mapped_range();
        let mut pixels = Vec::with_capacity((size[0] * size[1] * 4) as usize);
        for y in 0..size[1] as usize {
            pixels.extend_from_slice(
                &mapped[y * stride as usize..y * stride as usize + size[0] as usize * 4],
            );
        }
        drop(mapped);
        buffer.unmap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        image::RgbaImage::from_raw(size[0], size[1], pixels)
            .unwrap()
            .save(path)
            .unwrap();
    }
}
