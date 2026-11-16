#[cfg(not(target_arch = "wasm32"))]
impl PaintFEApp {
    fn start_plugin_effect(
        &mut self,
        ctx: &egui::Context,
        pack: String,
        effect: crate::services::plugins::Effect,
        source: String,
    ) -> Result<(), String> {
        if self.document_modal_open() || self.paste_overlay.is_some() {
            return Err("Finish the current operation before running a plugin.".into());
        }
        self.finish_pending_line_edit();
        self.commit_pending_tool_history();
        let p = self
            .active_project()
            .ok_or("Open an image to run this plugin.")?;
        let state = &p.canvas_state;
        let layer = state
            .layers
            .get(state.active_layer_index)
            .ok_or("No active layer")?;
        if state.edit_layer_mask
            || !matches!(layer.content, crate::canvas::LayerContent::Raster)
            || layer.deep_pixels.is_some()
        {
            return Err("Plugins currently require an 8-bit raster layer. Choose or create a raster layer first.".into());
        }
        if u64::from(state.width) * u64::from(state.height) > 16 * 1024 * 1024 {
            return Err(
                "This image exceeds the plugin processing limit (16 million pixels).".into(),
            );
        }
        effect.validate_values(&effect.defaults())?;
        crate::ui::panels::plugin_effect::open(
            ctx,
            crate::ui::panels::plugin_effect::Target {
                document: p.id,
                layer: layer.runtime_id,
                revision: p.edit_revision,
                width: state.width,
                height: state.height,
                pixels: std::sync::Arc::new(layer.pixels.extract_region_rgba(
                    0,
                    0,
                    state.width,
                    state.height,
                )),
                mask: state.selection_mask.as_ref().map(|m| m.as_raw().clone()),
                pack,
                effect,
                source,
            },
        );
        Ok(())
    }
    fn preview_authored_plugin(&mut self, ctx: &egui::Context) {
        let r = crate::ui::panels::plugin_author::preview(ctx, &self.script_editor.code).and_then(
            |(id, e, source)| {
                self.plugin_modal_open = crate::ui::panels::plugin_effect::is_open(ctx)
                    || crate::ui::panels::plugin_browser::is_open(ctx);
                self.start_plugin_effect(ctx, id, e, source)
            },
        );
        if let Err(e) = r {
            self.filter_error = Some(e);
        }
    }
    fn apply_plugin_result(
        &mut self,
        applied: crate::ui::panels::plugin_effect::Applied,
    ) -> Result<(), String> {
        let target = &applied.target;
        if applied.pixels.len() as u64 != u64::from(target.width) * u64::from(target.height) * 4 {
            return Err("Invalid plugin output dimensions".into());
        }
        let project = self
            .projects
            .iter_mut()
            .find(|p| p.id == target.document && p.edit_revision == target.revision)
            .ok_or(
                "The document changed while the plugin was running. Its result was not applied.",
            )?;
        let state = &mut project.canvas_state;
        let index = state
            .layers
            .iter()
            .position(|l| l.runtime_id == target.layer)
            .ok_or("The plugin's original layer is no longer available.")?;
        if state.width != target.width
            || state.height != target.height
            || state.layers[index].deep_pixels.is_some()
            || !matches!(
                state.layers[index].content,
                crate::canvas::LayerContent::Raster
            )
        {
            return Err("The plugin target changed. Its result was not applied.".into());
        }
        if applied.pixels != *target.pixels {
            let mut cmd = SingleLayerSnapshotCommand::new_for_layer(
                HistoryAction::Script.label(target.effect.name.clone()),
                state,
                index,
            );
            state.layers[index].pixels = crate::canvas::TiledImage::from_raw_rgba(
                state.width,
                state.height,
                &applied.pixels,
            );
            cmd.set_after(state);
            project.history.push(Box::new(cmd));
            state.mark_dirty(None);
            project.mark_dirty();
        }
        let at = crate::services::plugins::root();
        let mut prefs = crate::services::plugins::Preferences::load(&at);
        prefs
            .packs
            .entry(target.pack.clone())
            .or_default()
            .values
            .insert(target.effect.id.clone(), applied.values);
        if let Err(e) = prefs.save(&at) {
            self.filter_error = Some(format!(
                "Effect applied, but its parameters could not be saved: {e}"
            ));
        }
        Ok(())
    }
    fn update_plugins(&mut self, ctx: &egui::Context) {
        let owned_input_at_start = crate::ui::panels::plugin_effect::is_open(ctx)
            || crate::ui::panels::plugin_browser::is_open(ctx)
            || crate::ui::panels::plugin_author::is_open(ctx)
            || ctx
                .data(|d| d.get_temp::<AppearancePreview>(egui::Id::new("appearance_preview")))
                .is_some();
        self.plugin_modal_open = crate::ui::panels::plugin_effect::is_open(ctx);
        use crate::ui::panels::{plugin_browser as browser, plugin_effect as effect};
        let action = browser::frame(ctx);
        self.plugin_modal_open = effect::is_open(ctx)
            || ctx
                .data(|d| d.get_temp::<AppearancePreview>(egui::Id::new("appearance_preview")))
                .is_some()
            || browser::is_open(ctx)
            || crate::ui::panels::plugin_author::is_open(ctx);
        if let Some(action) = action {
            match action {
                browser::Action::Run(pack, e) => {
                    let result =
                        crate::services::plugins::effect_source(&pack, &e).and_then(|source| {
                            self.start_plugin_effect(ctx, pack.manifest.id, e, source)
                        });
                    if let Err(message) = result {
                        self.filter_error = Some(message);
                    }
                }
                browser::Action::Create => {
                    crate::ui::panels::plugin_author::create(ctx, &self.script_editor.code);
                    self.window_visibility.script_editor = true;
                }
                browser::Action::Edit(pack) => {
                    if let Err(e) = crate::ui::panels::plugin_author::edit(
                        ctx,
                        &pack,
                        &mut self.script_editor.code,
                    ) {
                        self.filter_error = Some(e);
                    } else {
                        self.window_visibility.script_editor = true;
                    }
                }
                browser::Action::RemovedAppearance(pack) => {
                    if std::path::Path::new(&self.settings.icon_pack_path)
                        .starts_with(&pack.directory)
                    {
                        self.assets.clear_icon_pack();
                        self.settings.icon_pack_path.clear();
                        self.assets.reload_icons(
                            ctx,
                            matches!(self.theme.mode, crate::theme::ThemeMode::Dark),
                        );
                        self.settings.save();
                    }
                }
                browser::Action::Appearance(pack) => {
                    if let Err(e) = self.preview_appearance_pack(ctx, &pack) {
                        self.filter_error = Some(e);
                    }
                }
                browser::Action::Export(pack) => {
                    if let Err(e) = crate::ui::panels::plugin_author::export_installed(&pack) {
                        self.filter_error = Some(e);
                    }
                }
            }
        }
        self.appearance_preview_dialog(ctx);
        crate::ui::panels::plugin_author::frame(ctx, &mut self.script_editor.code);
        if let Some(applied) = effect::frame(ctx) {
            if let Err(e) = self.apply_plugin_result(applied) {
                self.filter_error = Some(e);
            }
            browser::reload(ctx);
        }
        self.plugin_modal_open = owned_input_at_start
            || effect::is_open(ctx)
            || ctx
                .data(|d| d.get_temp::<AppearancePreview>(egui::Id::new("appearance_preview")))
                .is_some()
            || browser::is_open(ctx)
            || crate::ui::panels::plugin_author::is_open(ctx);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct AppearancePreview {
    settings: AppSettings,
    icons: String,
}
#[cfg(not(target_arch = "wasm32"))]
impl PaintFEApp {
    fn rebuild_community_appearance(&mut self, ctx: &egui::Context) {
        let accent = if self.settings.theme_preset == crate::theme::ThemePreset::Custom {
            self.settings.custom_accent
        } else {
            self.settings.theme_preset.accent_colors()
        };
        self.theme = match self.settings.theme_mode {
            crate::theme::ThemeMode::Dark => {
                Theme::dark_with_accent(self.settings.theme_preset, accent)
            }
            crate::theme::ThemeMode::Light => {
                Theme::light_with_accent(self.settings.theme_preset, accent)
            }
        };
        self.theme
            .apply_overrides(&self.settings.build_theme_overrides());
        self.theme.polish = self.settings.ui_polish.clone();
        self.theme.density = self.settings.ui_density;
        self.theme.apply(ctx);
        crate::ui::glass::configure(ctx, &self.settings.glass, self.theme.mode == crate::theme::ThemeMode::Dark);
    }
    fn preview_appearance_pack(
        &mut self,
        ctx: &egui::Context,
        pack: &crate::services::plugins::Installed,
    ) -> Result<(), String> {
        use crate::services::plugins::{PackKind, hash, read_limited};
        let m = &pack.manifest;
        m.validate()?;
        let resource = m
            .files
            .iter()
            .find(|f| f.path == m.appearance)
            .ok_or("Missing appearance resource")?;
        let path = pack.directory.join(&m.appearance);
        let data = read_limited(&path, 4 * 1024 * 1024)?;
        if hash(&data) != resource.sha256 {
            return Err("Appearance resource verification failed".into());
        }
        let previous = AppearancePreview {
            settings: self.settings.clone(),
            icons: self.settings.icon_pack_path.clone(),
        };
        match m.kind {
            PackKind::Theme => {
                let text = std::str::from_utf8(&data).map_err(|_| "Invalid theme text")?;
                self.settings.import_theme_from_string(text);
                self.rebuild_community_appearance(ctx);
            }
            PackKind::Icons => {
                self.assets.load_icon_pack(&path)?;
                self.settings.icon_pack_path = path.to_string_lossy().into_owned();
                self.assets.reload_icons(
                    ctx,
                    matches!(self.theme.mode, crate::theme::ThemeMode::Dark),
                );
            }
            _ => return Err("Not an appearance pack".into()),
        }
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("appearance_preview"), previous));
        Ok(())
    }
    fn appearance_preview_dialog(&mut self, ctx: &egui::Context) {
        let id = egui::Id::new("appearance_preview");
        let Some(previous) = ctx.data(|d| d.get_temp::<AppearancePreview>(id)) else {
            return;
        };
        let mut keep = false;
        let mut revert = false;
        let response = egui::Modal::new(egui::Id::new("appearance_confirm")).show(ctx, |ui| {
            let colours = crate::ui::dialogs::core::DialogColors::from_ctx(ctx);
            if crate::ui::dialogs::core::paint_dialog_header(
                ui,
                &colours,
                "",
                &crate::t!("packs.preview_title"),
            ) {
                revert = true;
            }
            ui.label(crate::t!("packs.preview_notice"));
            ui.horizontal(|ui| {
                let keep_response = ui.button(crate::t!("packs.keep"));
                #[cfg(test)]
                ctx.data_mut(|d| {
                    d.insert_temp(egui::Id::new("appearance_keep_rect"), keep_response.rect)
                });
                keep = keep_response.clicked();
                revert |= ui.button(crate::t!("packs.revert")).clicked();
            });
        });
        revert |= response.should_close();
        if keep || revert {
            ctx.data_mut(|d| d.remove::<AppearancePreview>(id));
            if revert {
                self.settings = previous.settings;
                self.settings.icon_pack_path = previous.icons;
                self.rebuild_community_appearance(ctx);
                if self.settings.icon_pack_path.is_empty() {
                    self.assets.clear_icon_pack();
                } else {
                    let _ = self
                        .assets
                        .load_icon_pack(std::path::Path::new(&self.settings.icon_pack_path));
                }
                self.assets.reload_icons(
                    ctx,
                    matches!(self.theme.mode, crate::theme::ThemeMode::Dark),
                );
            }
            self.settings.save();
        }
    }
}
