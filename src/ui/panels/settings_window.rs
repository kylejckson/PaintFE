use crate::config::brushes::Assets;
use crate::config::icons::Icon;
use crate::config::keybindings::{BindableAction, KeyBindings, KeyCombo, key_name};
use crate::config::settings::{AppSettings, PixelGridMode, ZoomFilterMode};
use crate::theme::{AccentColors, ThemeMode, ThemePreset, UiDensity};
use eframe::egui;
use egui::{Color32, Sense, Vec2};

pub struct SettingsWindow {
    pub open: bool,
    /// Set when the user applies the "Pixel Art" preset; consumed by the app
    /// to also update the live tool state (hardness, stabilization).
    pub pending_pixel_art_preset: bool,
    /// Icon pack folder chosen by the user; consumed by the app (needs &mut Assets).
    pub pending_icon_pack_load: Option<std::path::PathBuf>,
    /// Set when the user removes the active icon pack.
    pub pending_icon_pack_clear: bool,
    /// Set when icon display options changed and textures must be re-uploaded.
    pub pending_icon_pack_reload: bool,
    active_tab: SettingsTab,
    search_query: String,
    section_generation: u64,
    section_settle_until: f64,
    #[cfg(not(target_arch = "wasm32"))]
    storage_error: Option<(bool, String)>,
    /// Editing values for Interface; saved through the common automatic apply path.
    staged_accent: AccentColors,
    staged_preset: ThemePreset,
    staged_mode: ThemeMode,
    /// Whether staged values need applying
    dirty: bool,
    /// Cached list of available GPU adapter names
    gpu_adapters: Vec<String>,
    /// Receiver for the background GPU enumeration task (None when not running)
    gpu_adapters_receiver: Option<std::sync::mpsc::Receiver<Vec<String>>>,
    /// Staging copy of ONNX Runtime DLL path (AI tab)
    staged_onnx_path: String,
    /// Staging copy of BiRefNet model path (AI tab)
    staged_model_path: String,
    /// Result of last ONNX Runtime probe (None = not tested yet)
    onnx_probe_result: Option<Result<String, String>>,
    onnx_test_receiver: Option<(String, std::sync::mpsc::Receiver<Result<String, String>>)>,
    /// Staging copy of keybindings for the Keybinds tab
    staged_keybindings: KeyBindings,
    /// Which action is currently being rebound (waiting for key press)
    pub rebinding_action: Option<BindableAction>,
    /// Brief status message after theme import/export (cleared on next frame)
    theme_status: Option<(String, f64)>,
    #[cfg(not(target_arch = "wasm32"))]
    plugin_manager: crate::paintdotnet_plugins::PluginManager,
    #[cfg(not(target_arch = "wasm32"))]
    plugin_status: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    pending_trust_plugin: Option<String>,
    /// Web only: last error from uploading a custom font file, if any.
    #[cfg(target_arch = "wasm32")]
    font_upload_error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SettingsTab {
    General,
    #[cfg(not(target_arch = "wasm32"))]
    OptionalFeatures,
    Editing,
    Interface,
    Hardware,
    Keybinds,
    AI,
    #[cfg(not(target_arch = "wasm32"))]
    Plugins,
    #[cfg(target_arch = "wasm32")]
    Fonts,
}

struct SearchEntry {
    tab: SettingsTab,
    key: &'static str,
    english: &'static str,
    section: &'static str,
}
#[derive(Clone)]
struct SearchTarget {
    key: String,
    section: String,
    frames: u8,
    until: f64,
}
include!("settings_search_catalog.rs");
#[cfg(test)]
thread_local! { static AI_PATH_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

trait SearchableResponse {
    fn search_target(self, ui: &egui::Ui, key: &str) -> Self;
}
impl SearchableResponse for egui::Response {
    fn search_target(self, ui: &egui::Ui, key: &str) -> Self {
        let target = ui
            .ctx()
            .data(|d| d.get_temp::<SearchTarget>(egui::Id::new("preferences_search_target")));
        if target.is_some_and(|target| target.key == key) {
            SettingsWindow::search_mark(ui, &crate::i18n::translate(key), &self);
        }
        self
    }
}

impl SearchEntry {
    fn matches(&self, query: &str) -> bool {
        let text = format!(
            "{} {} {}",
            self.english,
            crate::i18n::translate(self.key),
            self.key.replace(['.', '_'], " ")
        )
        .to_lowercase();
        query
            .split_whitespace()
            .all(|word| text.contains(&word.to_lowercase()))
    }
    fn rank(&self, query: &str) -> (u8, usize) {
        let query = query.trim().to_lowercase();
        let labels = [
            self.english.to_lowercase(),
            crate::i18n::translate(self.key).to_lowercase(),
        ];
        let rank = if labels.iter().any(|label| label == &query) {
            0
        } else if labels.iter().any(|label| label.starts_with(&query)) {
            1
        } else if labels.iter().any(|label| {
            query.split_whitespace().all(|word| {
                label
                    .split(|c: char| !c.is_alphanumeric())
                    .any(|part| part == word)
            })
        }) {
            2
        } else {
            3
        };
        (rank, self.english.len())
    }
}

impl Default for SettingsWindow {
    fn default() -> Self {
        let preset = ThemePreset::Signal;
        Self {
            open: false,
            pending_pixel_art_preset: false,
            pending_icon_pack_load: None,
            pending_icon_pack_clear: false,
            pending_icon_pack_reload: false,
            active_tab: SettingsTab::General,
            search_query: String::new(),
            section_generation: 0,
            section_settle_until: 0.0,
            #[cfg(not(target_arch = "wasm32"))]
            storage_error: None,
            staged_accent: preset.accent_colors(),
            staged_preset: preset,
            staged_mode: ThemeMode::Light,
            dirty: false,
            gpu_adapters: Vec::new(),
            gpu_adapters_receiver: None,
            staged_onnx_path: String::new(),
            staged_model_path: String::new(),
            onnx_probe_result: None,
            onnx_test_receiver: None,
            staged_keybindings: KeyBindings::default(),
            rebinding_action: None,
            theme_status: None,
            #[cfg(not(target_arch = "wasm32"))]
            plugin_manager: crate::paintdotnet_plugins::PluginManager::load(),
            #[cfg(not(target_arch = "wasm32"))]
            plugin_status: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_trust_plugin: None,
            #[cfg(target_arch = "wasm32")]
            font_upload_error: None,
        }
    }
}

impl SettingsWindow {
    /// Sync staged state from current settings (call when opening)
    fn sync_from_settings(&mut self, settings: &AppSettings) {
        let _scope = crate::ui::perf::Scope::new(17);
        self.staged_mode = settings.theme_mode;
        self.staged_preset = settings.theme_preset;
        self.staged_accent = if settings.theme_preset == ThemePreset::Custom {
            settings.custom_accent
        } else {
            settings.theme_preset.accent_colors()
        };
        self.staged_onnx_path = settings.onnx_runtime_path.clone();
        self.staged_model_path = settings.birefnet_model_path.clone();
        self.onnx_probe_result = None;
        self.onnx_test_receiver = None;
        self.staged_keybindings = settings.keybindings.clone();
        self.rebinding_action = None;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _scope = crate::ui::perf::Scope::new(18);
            self.plugin_manager = crate::paintdotnet_plugins::PluginManager::load();
            self.plugin_status = None;
            self.pending_trust_plugin = None;
        }
        self.dirty = false;
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        settings: &mut AppSettings,
        theme: &mut crate::theme::Theme,
        assets: &Assets,
    ) -> Option<egui::Rect> {
        if !self.open {
            return None;
        }

        if crate::ui::perf::requested_scenario() == Some("interface") {
            let first = ctx.data_mut(|d| {
                let id = egui::Id::new("profile_initial_navigation");
                let first = !d.get_temp::<bool>(id).unwrap_or(false);
                d.insert_temp(id, true);
                first
            });
            if first {
                self.active_tab = SettingsTab::Interface;
            }
        }

        crate::ui::perf::scenario(match self.active_tab {
            SettingsTab::Interface => "preferences_interface",
            SettingsTab::General => "preferences_general",
            _ => "preferences_other",
        });

        crate::config::persistence::begin();

        // Sync on first frame the window is shown
        let id = egui::Id::new("settings_sync_flag");
        let was_open_last_frame = ctx.data_mut(|d| {
            let prev: bool = d.get_temp(id).unwrap_or(false);
            d.insert_temp(id, true);
            prev
        });
        if !was_open_last_frame {
            self.sync_from_settings(settings);
        }

        let show = self.open;
        let was_rebinding = self.rebinding_action.is_some();
        let mut should_close = false;

        let position_id = egui::Id::new("preferences_heading_position");
        let position = ctx
            .data(|d| d.get_temp::<egui::Pos2>(position_id))
            .unwrap_or(ctx.content_rect().center() - egui::vec2(340.0, 270.0));
        let mut moved_position = None;

        let preferences_scope = crate::ui::perf::Scope::new(19);
        let window_response = crate::ui::polish::window(ctx, "settings_window_internal")
            .title_bar(false)
            .movable(false)
            .current_pos(position)
            .resizable(true)
            .collapsible(false)
            .default_width(680.0)
            .default_height(540.0)
            .default_pos(ctx.content_rect().center() - egui::vec2(340.0, 270.0))
            .min_width(600.0)
            .min_height(400.0)
            .max_width(ctx.content_rect().width() * 0.9)
            .max_height(ctx.content_rect().height() * 0.9)
            .show(ctx, |ui| {
                ui.memory_mut(|memory| memory.set_modal_layer(ui.layer_id()));
                let header_scope = crate::ui::perf::Scope::new(20);
                // ── Custom header strip ─────────────────────────────────────
                {
                    let available_width = ui.available_width();
                    let header_height = 32.0;
                    let v = ctx.global_style().visuals.clone();
                    let accent = v.selection.stroke.color;
                    let is_dark = v.dark_mode;
                    let accent_faint = if is_dark {
                        Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 35)
                    } else {
                        Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 25)
                    };

                    let (rect, _) = ui.allocate_exact_size(
                        Vec2::new(available_width, header_height),
                        Sense::hover(),
                    );
                    let heading_rect = egui::Rect::from_min_max(
                        rect.min,
                        egui::pos2(rect.right() - header_height, rect.bottom()),
                    );
                    let heading = ui.interact(
                        heading_rect,
                        ui.id().with("preferences_heading_drag"),
                        Sense::drag(),
                    );
                    if heading.dragged() {
                        moved_position = Some(position + ui.input(|i| i.pointer.delta()));
                        ctx.request_repaint();
                    }
                    let painter = ui.painter();
                    painter.rect_filled(rect, egui::CornerRadius::ZERO, accent_faint);
                    painter.rect_filled(
                        egui::Rect::from_min_size(rect.min, Vec2::new(3.0, header_height)),
                        egui::CornerRadius::ZERO,
                        accent,
                    );
                    painter.text(
                        egui::pos2(rect.min.x + 12.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        format!("\u{2699} {}", t!("settings.title")),
                        egui::FontId::proportional(14.0),
                        accent,
                    );

                    if let Some(error) = crate::config::persistence::error() {
                        ui.colored_label(
                            ui.visuals().error_fg_color,
                            crate::i18n::format_translation(
                                "settings.save_error",
                                &[("error", error)],
                            ),
                        );
                    }
                    // X close button on the right
                    let btn_size = Vec2::splat(header_height);
                    let btn_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.max.x - btn_size.x, rect.min.y),
                        btn_size,
                    );
                    let btn_response =
                        ui.interact(btn_rect, ui.id().with("hdr_close"), Sense::click());
                    btn_response.widget_info(|| {
                        egui::WidgetInfo::labeled(
                            egui::WidgetType::Button,
                            ui.is_enabled(),
                            crate::t!("ui.close_preferences"),
                        )
                    });
                    crate::ui::polish::control(
                        ui,
                        &btn_response,
                        false,
                        theme.widget_rounding.min(6.0),
                    );
                    crate::signal_widgets::paint_close_cross(ui, btn_rect, theme.text_muted);
                    if btn_response.clicked() {
                        should_close = true;
                    }
                }
                drop(header_scope);
                self.show_search(ui);
                // Force the window body to use the full allocated height so the
                // sidebar doesn't collapse the window to just ~165 px.
                let available_h = ui.available_height();
                ui.set_min_height(available_h);
                ui.horizontal(|ui| {
                    // -- Left Sidebar --
                    ui.vertical(|ui| {
                        let _scope = crate::ui::perf::Scope::new(21);
                        ui.set_min_width(132.0);
                        ui.set_max_width(132.0);
                        ui.set_min_height(available_h);
                        ui.add_space(4.0);

                        let tabs: Vec<(SettingsTab, Icon, String)> = vec![
                            (
                                SettingsTab::General,
                                Icon::SettingsGeneral,
                                t!("settings.tab.general"),
                            ),
                            (
                                SettingsTab::Editing,
                                Icon::MenuEditPaste,
                                t!("settings.tab.editing"),
                            ),
                            (
                                SettingsTab::Interface,
                                Icon::SettingsInterface,
                                t!("settings.tab.interface"),
                            ),
                            (
                                SettingsTab::Hardware,
                                Icon::SettingsHardware,
                                t!("settings.tab.hardware"),
                            ),
                            (
                                SettingsTab::Keybinds,
                                Icon::SettingsKeybinds,
                                t!("settings.tab.keybinds"),
                            ),
                            // AI features (ONNX Runtime, background removal) need
                            // dynamic library loading, which has no wasm32
                            // equivalent — hide the tab entirely on web.
                            #[cfg(not(target_arch = "wasm32"))]
                            (SettingsTab::AI, Icon::SettingsAI, t!("settings.tab.ai")),
                            #[cfg(not(target_arch = "wasm32"))]
                            (
                                SettingsTab::OptionalFeatures,
                                Icon::SettingsPlugins,
                                t!("packs.title"),
                            ),
                            #[cfg(not(target_arch = "wasm32"))]
                            (
                                SettingsTab::Plugins,
                                Icon::SettingsPlugins,
                                t!("settings.tab.plugins"),
                            ),
                            // Font upload / Google Fonts only exist on web —
                            // desktop already has full OS font-store access.
                            #[cfg(target_arch = "wasm32")]
                            (SettingsTab::Fonts, Icon::Text, crate::t!("ui.fonts")),
                        ];
                        for (tab, icon, label) in &tabs {
                            let selected = self.active_tab == *tab;
                            // Render icon + text as a selectable row
                            let icon_size = Vec2::splat(16.0);
                            let total_width = ui.available_width();
                            let (rect, response) = ui
                                .allocate_exact_size(Vec2::new(total_width, 30.0), Sense::click());
                            if response.clicked() {
                                self.active_tab = *tab;
                            }
                            response.widget_info(|| {
                                egui::WidgetInfo::selected(
                                    egui::WidgetType::Button,
                                    ui.is_enabled(),
                                    selected,
                                    label,
                                )
                            });
                            crate::ui::polish::control(
                                ui,
                                &response,
                                selected,
                                theme.widget_rounding.min(6.0),
                            );
                            // Draw icon
                            let icon_rect = egui::Rect::from_min_size(
                                egui::pos2(rect.left() + 6.0, rect.center().y - icon_size.y / 2.0),
                                icon_size,
                            );
                            let text_color = if selected {
                                ui.visuals().strong_text_color()
                            } else {
                                ui.visuals().text_color()
                            };
                            if let Some(texture) = assets.textures.get(icon) {
                                ui.painter().image(
                                    texture.id(),
                                    icon_rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    assets.icon_tint(*icon, text_color),
                                );
                            }
                            // Draw label
                            let text_pos = egui::pos2(
                                icon_rect.right() + theme.polish.icon_gap,
                                rect.center().y,
                            );
                            let mut job = egui::text::LayoutJob::simple(
                                label.clone(),
                                egui::FontId::proportional(12.0 * theme.polish.text_scale),
                                text_color,
                                (rect.right() - text_pos.x - 6.0).max(1.0),
                            );
                            job.wrap.max_rows = 1;
                            job.wrap.break_anywhere = true;
                            let galley = ui.painter().layout_job(job);
                            let origin = egui::pos2(text_pos.x, text_pos.y - galley.size().y * 0.5);
                            ui.painter().galley(origin, galley, text_color);
                            response.on_hover_text(label);
                        }
                    });

                    ui.separator();

                    // -- Right Content Area --
                    ui.vertical(|ui| {
                        ui.set_min_width(450.0);
                        ui.set_min_height(available_h);
                        // Wrap all tab content in a per-tab ScrollArea.
                        // `set_min_height` ensures short tabs don't shrink the window;
                        // tall tabs scroll instead of growing the window.
                        let scroll_id = format!("settings_tab_scroll_{:?}", self.active_tab);
                        let _scroll = egui::ScrollArea::vertical()
                            .id_salt(scroll_id)
                            .auto_shrink([false; 2])
                            .max_height(available_h.max(120.0))
                            .show(ui, |ui| {
                                let _scope = crate::ui::perf::Scope::new(22);
                                ui.ctx().data_mut(|d| {
                                    d.remove::<egui::Rect>(egui::Id::new(
                                        "preferences_search_exact_rect",
                                    ));
                                    d.remove::<egui::Rect>(egui::Id::new(
                                        "preferences_search_section_rect",
                                    ));
                                    d.remove::<String>(egui::Id::new(
                                        "preferences_search_current_section",
                                    ));
                                });
                                ui.add_space(4.0);
                                match self.active_tab {
                                    #[cfg(not(target_arch = "wasm32"))]
                                    SettingsTab::OptionalFeatures => {
                                        super::optional_features::show_settings(ui);
                                    }
                                    SettingsTab::General => {
                                        self.show_general_tab(ui, settings);
                                    }
                                    SettingsTab::Editing => {
                                        self.show_editing_tab(ui, settings);
                                    }
                                    SettingsTab::Interface => {
                                        crate::ui::controls::without_wheel_adjustment(ui, |ui| {
                                            self.show_interface_tab(
                                                ui, ctx, settings, theme, assets,
                                            );
                                        });
                                    }
                                    SettingsTab::Hardware => {
                                        self.show_hardware_tab(ui, settings);
                                    }
                                    SettingsTab::Keybinds => {
                                        self.show_keybinds_tab(ui, settings, assets);
                                    }
                                    SettingsTab::AI => {
                                        self.show_ai_tab(ui, settings);
                                    }
                                    #[cfg(not(target_arch = "wasm32"))]
                                    SettingsTab::Plugins => {
                                        self.show_plugins_tab(ui, settings);
                                    }
                                    #[cfg(target_arch = "wasm32")]
                                    SettingsTab::Fonts => {
                                        self.show_fonts_tab(ui, ctx);
                                    }
                                }
                                Self::finish_search_navigation(ui);
                            });
                        #[cfg(test)]
                        ui.ctx().data_mut(|d| {
                            d.insert_temp(
                                egui::Id::new("preferences_test_scroll"),
                                _scroll.state.offset,
                            );
                        });
                    });
                });
            });

        drop(preferences_scope);
        self.apply_pending_settings(settings, theme, ctx);
        #[cfg(not(target_arch = "wasm32"))]
        let trust_rect = self.show_plugin_trust_modal(ctx);
        #[cfg(target_arch = "wasm32")]
        let trust_rect: Option<egui::Rect> = None;

        let (enter, escape) = crate::ui::dialogs::core::dialog_footer_keys(ctx);
        should_close |= !was_rebinding && self.rebinding_action.is_none() && (enter || escape);
        self.open = show && !should_close;
        if !self.open {
            // Persist any staged interface/theme edits when closing settings.
            if self.dirty {
                self.apply_theme(settings, theme, ctx);
            }
            // Clear the sync flag when window closes
            ctx.data_mut(|d| d.insert_temp(id, false));
        }

        let typing = ctx.input(|i| {
            i.pointer.any_down()
                || i.events
                    .iter()
                    .any(|e| matches!(e, egui::Event::Text(_) | egui::Event::Paste(_)))
        });
        if crate::config::persistence::flush(
            !self.open || (!typing && ctx.input(|i| i.pointer.any_released())),
        ) {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        // The visible window rect(s) — the caller registers them as
        // input-blocking so canvas input (wheel zoom, clicks, strokes) does
        // not leak through the window to the canvas behind it.
        let window_rect = window_response.map(|r| r.response.rect);
        if let Some(rect) = window_rect {
            ctx.data_mut(|d| d.insert_temp(position_id, moved_position.unwrap_or(rect.min)));
        }
        match (window_rect, trust_rect) {
            (Some(a), Some(b)) => Some(a.union(b)),
            (a, b) => a.or(b),
        }
    }

    fn show_search(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Reserve a fixed slot for clear, so typing never changes the
            // search field's identity or horizontal position.
            let width = 210.0_f32.min((ui.available_width() * 0.48).max(80.0));
            let hint_width = (ui.available_width() - width - 34.0).max(0.0);
            ui.add_sized(
                [hint_width, ui.spacing().interact_size.y],
                egui::Label::new(
                    egui::RichText::new(t!("settings.auto_save_hint"))
                        .small()
                        .weak(),
                )
                .truncate(),
            );
            ui.add_sized(
                [width, ui.spacing().interact_size.y],
                egui::TextEdit::singleline(&mut self.search_query)
                    .id(egui::Id::new("preferences_search_field"))
                    .hint_text(crate::t!("ctx.text.search")),
            );
            if ui
                .add_enabled(!self.search_query.is_empty(), egui::Button::new("\u{00d7}"))
                .clicked()
            {
                self.search_query.clear();
                ui.ctx().data_mut(|d| {
                    d.remove::<SearchTarget>(egui::Id::new("preferences_search_target"))
                });
                ui.ctx()
                    .memory_mut(|m| m.request_focus(egui::Id::new("preferences_search_field")));
            }
        });
        if self.search_query.trim().is_empty() {
            return;
        }
        let mut matches: Vec<_> = SEARCH_ENTRIES
            .iter()
            .filter(|entry| entry.matches(&self.search_query))
            .collect();
        matches.sort_by_key(|entry| entry.rank(&self.search_query));
        if matches.is_empty() {
            ui.weak(crate::t!("settings.search.empty"))
                .search_target(ui, "settings.search.empty");
        }
        let mut picked = None;
        egui::ScrollArea::vertical()
            .id_salt("preferences_search_results")
            .auto_shrink([false, true])
            .max_height(150.0)
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                for entry in matches {
                    let label = format!(
                        "{}  \u{203a}  {}",
                        crate::i18n::translate(entry.section),
                        crate::i18n::translate(entry.key)
                    );
                    let selected = ui
                        .ctx()
                        .data(|d| {
                            d.get_temp::<SearchTarget>(egui::Id::new("preferences_search_target"))
                        })
                        .is_some_and(|target| {
                            target.key == entry.key && target.section == entry.section
                        });
                    let response = ui
                        .push_id((entry.section, entry.key), |ui| {
                            ui.selectable_label(selected, label)
                        })
                        .inner;
                    #[cfg(test)]
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(
                            egui::Id::new(("search_result_rect", entry.section, entry.key)),
                            response.rect,
                        )
                    });
                    if response.clicked() {
                        picked = Some(entry);
                    }
                }
            });
        if let Some(entry) = picked {
            self.active_tab = entry.tab;
            let until = ui.input(|i| i.time) + 3.0;
            ui.ctx().data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("preferences_search_target"),
                    SearchTarget {
                        key: entry.key.into(),
                        section: entry.section.into(),
                        frames: 3,
                        until,
                    },
                )
            });
            ui.ctx().request_repaint();
        }
    }

    // Settings register their rendered labels, not guessed scroll offsets.
    // Destination identities remain translation keys; translated text is only
    // used to associate the existing label with its owning destination.
    fn search_mark(ui: &egui::Ui, label: &str, response: &egui::Response) {
        let target = ui
            .ctx()
            .data(|d| d.get_temp::<SearchTarget>(egui::Id::new("preferences_search_target")));
        if let Some(target) = target {
            let section = ui.ctx().data(|d| {
                d.get_temp::<String>(egui::Id::new("preferences_search_current_section"))
            });
            if label == crate::i18n::translate(&target.key)
                && section.as_deref() == Some(crate::i18n::translate(&target.section).as_str())
            {
                ui.ctx().data_mut(|d| {
                    d.insert_temp(
                        egui::Id::new("preferences_search_exact_rect"),
                        response.rect,
                    );
                    #[cfg(test)]
                    d.insert_temp(egui::Id::new("search_target_clip"), ui.clip_rect());
                });
            }
        }
    }

    fn search_reveals(ctx: &egui::Context, keys: &[&str]) -> Option<bool> {
        ctx.data(|d| d.get_temp::<SearchTarget>(egui::Id::new("preferences_search_target")))
            .filter(|target| {
                target.frames > 0
                    && keys
                        .iter()
                        .any(|key| target.key == *key || target.section == *key)
            })
            .map(|_| true)
    }

    fn search_group(
        ui: &egui::Ui,
        id: egui::Id,
        keys: &[&str],
    ) -> egui::collapsing_header::CollapsingState {
        let mut state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
        if Self::search_reveals(ui.ctx(), keys).is_some() {
            state.set_open(true);
        }
        state
    }

    fn finish_search_navigation(ui: &mut egui::Ui) {
        let id = egui::Id::new("preferences_search_target");
        if let Some(mut target) = ui.ctx().data(|d| d.get_temp::<SearchTarget>(id)) {
            if ui.input(|i| i.time) >= target.until {
                ui.ctx().data_mut(|d| d.remove::<SearchTarget>(id));
                return;
            }
            let rect = ui.ctx().data(|d| {
                d.get_temp::<egui::Rect>(egui::Id::new("preferences_search_exact_rect"))
                    .or_else(|| {
                        d.get_temp::<egui::Rect>(egui::Id::new("preferences_search_section_rect"))
                    })
            });
            if let Some(rect) = rect {
                if target.frames > 0 {
                    ui.scroll_to_rect_animation(
                        rect,
                        Some(egui::Align::Center),
                        egui::style::ScrollAnimation::none(),
                    );
                    target.frames -= 1;
                    ui.ctx().request_repaint();
                }
                if ui.input(|i| i.time) < target.until {
                    ui.painter().rect_stroke(
                        rect.expand(2.0),
                        3.0,
                        egui::Stroke::new(1.5, ui.visuals().selection.stroke.color),
                        egui::StrokeKind::Outside,
                    );
                }
            }
            ui.ctx().data_mut(|d| d.insert_temp(id, target));
        }
    }

    /// Draws a bold section label followed by a thin separator rule.
    /// Used consistently across all settings tabs.
    pub(super) fn section_header(ui: &mut egui::Ui, label: &str) {
        ui.ctx().data_mut(|d| {
            d.insert_temp(
                egui::Id::new("preferences_search_current_section"),
                label.to_owned(),
            )
        });
        ui.add_space(10.0);
        let highlighted = ui
            .ctx()
            .data(|d| d.get_temp::<(String, f64)>(egui::Id::new("preferences_search_highlight")))
            .is_some_and(|(target, until)| target == label && ui.input(|i| i.time) < until);
        let text = egui::RichText::new(label).strong().size(13.0);
        let response = ui.label(if highlighted {
            text.background_color(ui.visuals().selection.bg_fill)
        } else {
            text
        });
        if ui
            .ctx()
            .data(|d| d.get_temp::<SearchTarget>(egui::Id::new("preferences_search_target")))
            .is_some_and(|target| crate::i18n::translate(&target.section) == label)
        {
            ui.ctx().data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("preferences_search_section_rect"),
                    response.rect,
                )
            });
        }
        if ui
            .ctx()
            .data(|d| d.get_temp::<String>(egui::Id::new("preferences_search_jump")))
            .as_deref()
            == Some(label)
        {
            response.scroll_to_me_animation(
                Some(egui::Align::Min),
                egui::style::ScrollAnimation::none(),
            );
            ui.ctx().request_repaint();
            ui.ctx().data_mut(|d| {
                let id = egui::Id::new("preferences_search_jump_frames");
                let frames = d.get_temp::<u32>(id).unwrap_or(1).saturating_sub(1);
                d.insert_temp(id, frames);
                if frames == 0 {
                    d.remove::<String>(egui::Id::new("preferences_search_jump"));
                }
            });
        }
        ui.separator();
        ui.add_space(3.0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn show_plugins_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        Self::section_header(ui, &t!("settings.plugins.title"));
        ui.label(
            egui::RichText::new(t!("settings.plugins.warning"))
                .color(egui::Color32::from_rgb(190, 120, 20)),
        )
        .search_target(ui, "settings.plugins.warning");
        ui.add_space(8.0);

        if ui
            .checkbox(
                &mut settings.paintdotnet_plugins_enabled,
                t!("settings.plugins.enable"),
            )
            .search_target(ui, "settings.plugins.enable")
            .changed()
        {
            settings.save();
        }

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    settings.paintdotnet_plugins_enabled,
                    egui::Button::new(t!("settings.plugins.import")),
                )
                .clicked()
                && let Some(paths) = rfd::FileDialog::new()
                    .add_filter(crate::t!("ui.paint_net_effect"), &["dll"])
                    .pick_files()
            {
                self.plugin_status = Some(
                    match crate::paintdotnet_plugins::PluginManager::import_files(&paths) {
                        Ok(plugin) => {
                            format!("{}: {}", t!("settings.plugins.imported"), plugin.name)
                        }
                        Err(error) => error,
                    },
                );
                self.plugin_manager = crate::paintdotnet_plugins::PluginManager::load();
            }
            if ui
                .button(t!("settings.plugins.rescan"))
                .search_target(ui, "settings.plugins.rescan")
                .clicked()
            {
                self.plugin_manager = crate::paintdotnet_plugins::PluginManager::load();
                self.plugin_manager.rescan();
                self.plugin_status = None;
            }
        });

        if let Some(status) = &self.plugin_status {
            ui.label(status);
        }
        if let Err(error) = crate::paintdotnet_plugins::host_path() {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }

        Self::section_header(ui, &t!("settings.plugins.installed"));
        if self.plugin_manager.plugins.is_empty() {
            ui.label(t!("settings.plugins.none"))
                .search_target(ui, "settings.plugins.none");
        }

        let mut trust_change: Option<String> = None;
        let mut remove: Option<String> = None;
        for plugin in &self.plugin_manager.plugins {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&plugin.name).strong());
                    ui.label(format!("{} · {}", plugin.category, plugin.profile));
                });
                ui.label(crate::i18n::format_translation(
                    "ui.sha_256_0",
                    &[(
                        "0",
                        plugin
                            .sha256
                            .get(..16)
                            .unwrap_or(&plugin.sha256)
                            .to_string(),
                    )],
                ));
                if let Some(error) = plugin.display_error() {
                    ui.colored_label(ui.visuals().error_fg_color, error);
                }
                ui.horizontal(|ui| {
                    let label = if plugin.trusted {
                        t!("settings.plugins.disable")
                    } else {
                        t!("settings.plugins.trust_enable")
                    };
                    if ui.button(label).clicked() {
                        if plugin.trusted {
                            trust_change = Some(plugin.sha256.clone());
                        } else {
                            self.pending_trust_plugin = Some(plugin.sha256.clone());
                        }
                    }
                    if ui
                        .button(t!("common.delete"))
                        .search_target(ui, "common.delete")
                        .clicked()
                    {
                        remove = Some(plugin.sha256.clone());
                    }
                });
            });
        }
        if let Some(hash) = trust_change
            && let Err(error) = self.plugin_manager.set_trusted_enabled(&hash, false)
        {
            self.plugin_status = Some(error);
        }
        if let Some(hash) = remove
            && let Err(error) = self.plugin_manager.remove(&hash)
        {
            self.plugin_status = Some(error);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn show_plugin_trust_modal(&mut self, ctx: &egui::Context) -> Option<egui::Rect> {
        let hash = self.pending_trust_plugin.clone()?;
        let Some(plugin) = self
            .plugin_manager
            .plugins
            .iter()
            .find(|plugin| plugin.sha256 == hash)
            .cloned()
        else {
            self.pending_trust_plugin = None;
            return None;
        };

        let window_response = crate::ui::polish::window(ctx, "paintdotnet_plugin_trust_confirm")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_width(460.0);
                ui.label(
                    egui::RichText::new(crate::t!("ui.trust_paint_net_plugin"))
                        .strong()
                        .size(16.0),
                )
                .search_target(ui, "ui.trust_paint_net_plugin");
                ui.add_space(8.0);
                ui.colored_label(
                    ui.visuals().error_fg_color,
                    crate::t!(
                        "ui.this_dll_is_executable_code_it_can_access_your_files_network_and_syste"
                    ),
                );
                ui.add_space(8.0);
                ui.label(crate::i18n::format_translation(
                    "ui.plugin_0",
                    &[("0", plugin.name.to_string())],
                ));
                ui.label(crate::i18n::format_translation(
                    "ui.path_0",
                    &[("0", plugin.source_file.to_string())],
                ));
                ui.label(crate::i18n::format_translation(
                    "ui.sha_256_0_2",
                    &[("0", plugin.sha256.to_string())],
                ));
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(crate::t!("common.cancel"))
                        .search_target(ui, "common.cancel")
                        .clicked()
                    {
                        self.pending_trust_plugin = None;
                    }
                    if ui
                        .button(crate::t!("ui.trust_this_exact_file"))
                        .search_target(ui, "ui.trust_this_exact_file")
                        .clicked()
                    {
                        if let Err(error) = self.plugin_manager.set_trusted_enabled(&hash, true) {
                            self.plugin_status = Some(error);
                        }
                        self.pending_trust_plugin = None;
                    }
                });
            });

        window_response.map(|r| r.response.rect)
    }

    // -- General Tab -------------------------------------------
    fn show_editing_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        Self::section_header(ui, &t!("settings.editing.clipboard"));
        ui.checkbox(
            &mut settings.paste_at_source_position,
            t!("settings.editing.paste_source"),
        )
        .search_target(ui, "settings.editing.paste_source");
        ui.weak(t!("settings.editing.paste_source_hint"))
            .search_target(ui, "settings.editing.paste_source_hint");
        ui.checkbox(
            &mut settings.paste_keep_visible,
            t!("settings.editing.paste_visible"),
        )
        .search_target(ui, "settings.editing.paste_visible");
        ui.weak(t!("settings.editing.paste_visible_hint"))
            .search_target(ui, "settings.editing.paste_visible_hint");
        ui.add_space(6.0);
        ui.checkbox(
            &mut settings.clipboard_copy_transparent_cutout,
            crate::t!("ui.transparent_pasted_pixels_overwrite_destination"),
        )
        .search_target(ui, "ui.transparent_pasted_pixels_overwrite_destination");
        ui.label(
            egui::RichText::new(crate::t!(
                "ui.when_enabled_pasted_selections_keep_their_original_silhouette_instead_"
            ))
            .small()
            .weak(),
        );

        ui.add_space(6.0);
        ui.checkbox(
            &mut settings.select_after_paste,
            crate::t!("ui.select_the_pasted_image_after_pasting"),
        )
        .search_target(ui, "ui.select_the_pasted_image_after_pasting");
        ui.label(
            egui::RichText::new(crate::t!(
                "ui.when_enabled_committing_a_paste_leaves_the_pasted_region_selected_read"
            ))
            .small()
            .weak(),
        );

        Self::section_header(ui, &t!("settings.editing.selection_keys"));
        ui.checkbox(
            &mut settings.escape_deselect,
            t!("settings.interface.escape_deselect"),
        )
        .search_target(ui, "settings.interface.escape_deselect");
        ui.checkbox(
            &mut settings.enter_deselect,
            t!("settings.editing.enter_deselect"),
        )
        .search_target(ui, "settings.editing.enter_deselect");
        ui.weak(t!("settings.editing.enter_deselect_hint"))
            .search_target(ui, "settings.editing.enter_deselect_hint");
        ui.checkbox(
            &mut settings.legacy_clipboard_shortcuts,
            t!("settings.editing.legacy_clipboard"),
        )
        .search_target(ui, "settings.editing.legacy_clipboard");
        ui.weak(t!("settings.editing.legacy_clipboard_hint"))
            .search_target(ui, "settings.editing.legacy_clipboard_hint");
        settings.save();
    }

    fn validated_ai_path(raw: &str, model: bool) -> Result<String, String> {
        #[cfg(test)]
        AI_PATH_PROBES.with(|count| count.set(count.get() + 1));
        let value = raw.trim().trim_matches('"');
        if value.is_empty() {
            return Ok(String::new());
        }
        let path = std::path::Path::new(value);
        if !path.is_file() {
            return Err(crate::i18n::format_translation(
                "settings.ai.file_not_found",
                &[("path", value.to_owned())],
            ));
        }
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_ascii_lowercase();
        let valid = if model {
            name.ends_with(".onnx")
        } else {
            name.ends_with(".dll")
                || name.ends_with(".dylib")
                || name.ends_with(".so")
                || name.contains(".so.")
        };
        if !valid {
            return Err(t!(if model {
                "settings.ai.expected_model"
            } else {
                "settings.ai.expected_runtime"
            }));
        }
        Ok(value.to_owned())
    }

    fn apply_pending_settings(
        &mut self,
        settings: &mut AppSettings,
        theme: &mut crate::theme::Theme,
        ctx: &egui::Context,
    ) {
        if self.dirty {
            self.apply_theme(settings, theme, ctx);
        }
        let mut changed = false;
        if settings.keybindings.bindings != self.staged_keybindings.bindings
            || settings.keybindings.unbound != self.staged_keybindings.unbound
        {
            settings.keybindings = self.staged_keybindings.clone();
            changed = true;
        }
        for (raw, model, saved) in [
            (
                &self.staged_onnx_path,
                false,
                &mut settings.onnx_runtime_path,
            ),
            (
                &self.staged_model_path,
                true,
                &mut settings.birefnet_model_path,
            ),
        ] {
            // Saved paths have already been validated. Rechecking the file
            // system on every scrolling frame can block on slow/removable disks.
            if raw.trim().trim_matches('"') == saved.as_str() {
                continue;
            }
            if let Ok(path) = Self::validated_ai_path(raw, model)
                && *saved != path
            {
                *saved = path;
                changed = true;
            }
        }
        if changed {
            settings.save();
        }
    }

    fn show_general_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let storage = crate::services::storage::initialize();
            Self::section_header(ui, &t!("settings.storage.heading"));
            let mut portable = crate::services::storage::requested_portable();
            let changed = ui
                .horizontal(|ui| {
                    let profile = ui
                        .radio_value(&mut portable, false, crate::t!("ui.user_profile"))
                        .changed();
                    let portable = ui
                        .radio_value(&mut portable, true, crate::t!("ui.portable"))
                        .changed();
                    profile || portable
                })
                .inner;
            if changed {
                settings.save();
                self.storage_error = crate::services::storage::request_mode(portable, false)
                    .err()
                    .map(|error| (portable, error));
            }
            ui.label(format!(
                "{}: {}",
                t!("settings.storage.location"),
                storage.settings_path().parent().unwrap().display()
            ))
            .search_target(ui, "settings.storage.location");
            if crate::services::storage::requested_portable() != storage.portable {
                ui.label(t!("settings.storage.restart"))
                    .search_target(ui, "settings.storage.restart");
            }
            if let Some(warning) = &storage.warning {
                ui.label(warning);
            }
            if let Some((target, error)) = self.storage_error.clone() {
                ui.colored_label(egui::Color32::RED, error);
                if ui
                    .button(t!("settings.storage.replace"))
                    .search_target(ui, "settings.storage.replace")
                    .clicked()
                {
                    self.storage_error = crate::services::storage::request_mode(target, true)
                        .err()
                        .map(|error| (target, error));
                }
            }
            ui.add_space(8.0);
        }
        // -- Language -------------------------------------------
        Self::section_header(ui, &crate::t!("ui.language"));
        ui.horizontal(|ui| {
            ui.label(crate::t!("ui.language_2"))
                .search_target(ui, "ui.language_2");
            let current_code = if settings.language.is_empty() {
                "auto".to_string()
            } else {
                settings.language.clone()
            };
            let display_text = if current_code == "auto" {
                crate::t!("ui.auto_system")
            } else {
                crate::i18n::LANGUAGES
                    .iter()
                    .find(|(c, _)| *c == current_code.as_str())
                    .map(|(_, name)| name.to_string())
                    .unwrap_or_else(|| current_code.clone())
            };
            egui::ComboBox::from_id_salt("language_select")
                .selected_text(&display_text)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(current_code == "auto", crate::t!("ui.auto_system"))
                        .clicked()
                    {
                        settings.language = String::new();
                        let detected = crate::i18n::detect_system_language();
                        #[cfg(not(target_arch = "wasm32"))]
                        if !super::optional_features::select_language(ui.ctx(), &detected, true) {
                            return;
                        }
                        crate::i18n::set_language(&detected);
                    }
                    for &(code, name) in crate::i18n::LANGUAGES {
                        if ui
                            .selectable_label(
                                current_code.as_str() == code,
                                format!("{} ({})", name, code),
                            )
                            .clicked()
                        {
                            #[cfg(not(target_arch = "wasm32"))]
                            if !super::optional_features::select_language(ui.ctx(), code, false) {
                                continue;
                            }
                            settings.language = code.to_string();
                            crate::i18n::set_language(code);
                        }
                    }
                });
        });
        ui.label(
            egui::RichText::new(crate::t!("ui.restart_recommended_after_changing_language"))
                .small()
                .weak(),
        )
        .search_target(ui, "ui.restart_recommended_after_changing_language");

        // -- History & Auto-save -------------------------------------------
        Self::section_header(ui, &t!("settings.general.history"));
        egui::Grid::new("general_history_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .min_col_width(160.0)
            .show(ui, |ui| {
                ui.label(t!("settings.general.max_undo_steps"))
                    .search_target(ui, "settings.general.max_undo_steps");
                Self::settings_drag_usize(ui, &mut settings.max_undo_steps, 10..=500, "", 50);
                ui.end_row();

                ui.label(crate::t!("ui.auto_save_interval"))
                    .search_target(ui, "ui.auto_save_interval");
                {
                    const OPTIONS: &[(u32, &str)] = &[
                        (0, "Disabled"),
                        (1, "Every 1 minute"),
                        (3, "Every 3 minutes"),
                        (5, "Every 5 minutes"),
                        (10, "Every 10 minutes"),
                        (15, "Every 15 minutes"),
                        (30, "Every 30 minutes"),
                    ];
                    let current_label = OPTIONS
                        .iter()
                        .find(|(v, _)| *v == settings.auto_save_minutes)
                        .map(|(_, l)| *l)
                        .unwrap_or("Custom");
                    egui::ComboBox::from_id_salt("auto_save_interval")
                        .selected_text(crate::i18n::translate_english(current_label))
                        .show_ui(ui, |ui| {
                            for &(val, label) in OPTIONS {
                                if ui
                                    .selectable_label(
                                        settings.auto_save_minutes == val,
                                        crate::i18n::translate_english(label),
                                    )
                                    .clicked()
                                {
                                    settings.auto_save_minutes = val;
                                }
                            }
                        });
                }
                ui.end_row();
            });
        ui.label(
            egui::RichText::new(t!("settings.general.max_undo_hint"))
                .small()
                .weak(),
        )
        .search_target(ui, "settings.general.max_undo_hint");
        if settings.auto_save_minutes > 0 {
            if let Some(dir) = crate::io::autosave_dir() {
                ui.label(
                    egui::RichText::new(crate::i18n::format_translation(
                        "ui.auto_saves_to_0",
                        &[("0", format!("{}", dir.display()))],
                    ))
                    .small()
                    .weak(),
                );
            }
        } else {
            ui.label(
                egui::RichText::new(crate::t!(
                    "ui.auto_save_is_disabled_enable_it_to_protect_against_crashes"
                ))
                .small()
                .weak(),
            );
        }

        // -- Canvas Display -------------------------------------------
        Self::section_header(ui, &t!("settings.general.display"));
        ui.horizontal(|ui| {
            ui.label(t!("settings.general.pixel_grid"))
                .search_target(ui, "settings.general.pixel_grid");
            egui::ComboBox::from_id_salt("pixel_grid_mode")
                .selected_text(settings.pixel_grid_mode.name())
                .show_ui(ui, |ui| {
                    for &mode in PixelGridMode::all() {
                        if ui
                            .selectable_label(mode == settings.pixel_grid_mode, mode.name())
                            .clicked()
                        {
                            settings.pixel_grid_mode = mode;
                        }
                    }
                });
        });

        // -- Startup Canvas -------------------------------------------
        Self::section_header(ui, &crate::t!("ui.startup_canvas"));
        ui.checkbox(
            &mut settings.create_canvas_on_startup,
            crate::t!("ui.create_a_blank_canvas_on_startup"),
        )
        .search_target(ui, "ui.create_a_blank_canvas_on_startup");
        ui.label(
            egui::RichText::new(crate::t!(
                "ui.when_disabled_the_app_starts_empty_use_file_new_or_the_tab_to_create_a"
            ))
            .small()
            .weak(),
        );
        if settings.create_canvas_on_startup {
            ui.add_space(4.0);
            egui::Grid::new("startup_canvas_grid")
                .num_columns(2)
                .spacing([16.0, 6.0])
                .min_col_width(160.0)
                .show(ui, |ui| {
                    ui.label(crate::t!("ui.default_width"))
                        .search_target(ui, "ui.default_width");
                    Self::settings_drag_u32(
                        ui,
                        &mut settings.default_canvas_width,
                        1..=65535u32,
                        " px",
                        800,
                    );
                    ui.end_row();

                    ui.label(crate::t!("ui.default_height"))
                        .search_target(ui, "ui.default_height");
                    Self::settings_drag_u32(
                        ui,
                        &mut settings.default_canvas_height,
                        1..=65535u32,
                        " px",
                        600,
                    );
                    ui.end_row();
                });
        }

        // -- Behaviour -------------------------------------------------
        Self::section_header(ui, &crate::t!("ui.behaviour"));
        ui.checkbox(
            &mut settings.confirm_on_exit,
            crate::t!("ui.confirm_before_exiting_with_unsaved_changes"),
        )
        .search_target(ui, "ui.confirm_before_exiting_with_unsaved_changes");
        ui.label(
            egui::RichText::new(crate::t!(
                "ui.shows_a_save_prompt_when_quitting_with_unsaved_projects"
            ))
            .small()
            .weak(),
        );

        // -- Debug Info ---------------------------------------------------
        Self::section_header(ui, &t!("settings.general.debug_info"));
        ui.checkbox(
            &mut settings.show_debug_panel,
            t!("settings.general.show_debug"),
        )
        .search_target(ui, "settings.general.show_debug");
        ui.label(
            egui::RichText::new(t!("settings.general.debug_hint"))
                .small()
                .weak(),
        )
        .search_target(ui, "settings.general.debug_hint");
        if settings.show_debug_panel {
            ui.indent("debug_options", |ui| {
                ui.checkbox(
                    &mut settings.debug_show_canvas_size,
                    t!("settings.general.debug_canvas_size"),
                )
                .search_target(ui, "settings.general.debug_canvas_size");
                ui.checkbox(
                    &mut settings.debug_show_zoom,
                    t!("settings.general.debug_zoom"),
                )
                .search_target(ui, "settings.general.debug_zoom");
                ui.checkbox(
                    &mut settings.debug_show_fps,
                    t!("settings.general.debug_fps"),
                )
                .search_target(ui, "settings.general.debug_fps");
                ui.checkbox(
                    &mut settings.debug_show_gpu,
                    t!("settings.general.debug_gpu"),
                )
                .search_target(ui, "settings.general.debug_gpu");
                ui.checkbox(
                    &mut settings.debug_show_operations,
                    t!("settings.general.debug_operations"),
                )
                .search_target(ui, "settings.general.debug_operations");
                ui.checkbox(
                    &mut settings.show_tool_info,
                    t!("settings.general.show_tool_info"),
                )
                .search_target(ui, "settings.general.show_tool_info");
            });
        }

        // -- About ------------------------------------------------------
        Self::section_header(ui, &crate::t!("ui.about"));
        egui::Grid::new("about_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new(crate::t!("ui.version")).strong())
                    .search_target(ui, "ui.version");
                #[cfg(target_arch = "wasm32")]
                ui.label("PaintFE - Web 1.0");
                #[cfg(not(target_arch = "wasm32"))]
                ui.label(env!("CARGO_PKG_VERSION"));
                ui.end_row();

                ui.label(egui::RichText::new(crate::t!("ui.author")).strong())
                    .search_target(ui, "ui.author");
                ui.label("Kyle Jackson");
                ui.end_row();

                ui.label(egui::RichText::new(crate::t!("ui.license")).strong())
                    .search_target(ui, "ui.license");
                ui.label("MIT");
                ui.end_row();

                ui.label(egui::RichText::new(crate::t!("ui.source")).strong())
                    .search_target(ui, "ui.source");
                ui.hyperlink_to("paintfe.com", "https://paintfe.com");
                ui.end_row();
            });
        ui.collapsing("JPEG XL · libjxl / jxl-rs", |ui| {
            egui::ScrollArea::vertical()
                .max_height(200.0)
                .show(ui, |ui| {
                    ui.label(include_str!("../../../assets/licenses/jpeg-xl.txt"));
                });
        });
        #[cfg(target_arch = "wasm32")]
        {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(crate::t!(
                    "ui.paintfe_web_is_an_experimental_beta_port_of_the_desktop_app_some_featu"
                ))
                .small()
                .weak(),
            );
        }

        // -- Reset ------------------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);
        if ui
            .button(t!("settings.general.reset_defaults"))
            .search_target(ui, "settings.general.reset_defaults")
            .clicked()
        {
            *settings = AppSettings::default();
            self.pending_icon_pack_clear = true;
        }
        settings.save();
    }

    // -- Hardware Tab -----------------------------------------------
    fn show_hardware_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        // -- Preferred Adapter --------------------------------------------
        Self::section_header(ui, &t!("settings.hardware.preferred_gpu"));

        // Poll background GPU enumeration result
        if let Some(rx) = &self.gpu_adapters_receiver
            && let Ok(adapters) = rx.try_recv()
        {
            self.gpu_adapters = adapters;
            self.gpu_adapters_receiver = None;
        }

        // GPU adapter enumeration needs a background thread (std::thread::spawn,
        // unavailable on wasm32) and blocks on an async wgpu call that never
        // resolves synchronously in the browser sandbox — not available on web.
        #[cfg(target_arch = "wasm32")]
        if self.gpu_adapters.is_empty() {
            self.gpu_adapters = vec!["Auto".to_string()];
        }

        // Kick off background enumeration on first visit (non-blocking)
        #[cfg(not(target_arch = "wasm32"))]
        if self.gpu_adapters.is_empty() && self.gpu_adapters_receiver.is_none() {
            let (tx, rx) = std::sync::mpsc::channel();
            self.gpu_adapters_receiver = Some(rx);
            std::thread::spawn(move || {
                // Use wgpu's own adapter enumeration — zero child processes spawned,
                // no console flash, and it returns the same GPU names the app actually uses.
                let mut adapters = vec!["Auto".to_string()];

                let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::all(),
                    ..wgpu::InstanceDescriptor::new_without_display_handle()
                });
                for adapter in
                    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
                {
                    let info = adapter.get_info();
                    // Skip software/fallback adapters (CPU-simulated or unknown)
                    if info.device_type != wgpu::DeviceType::Other
                        && info.device_type != wgpu::DeviceType::Cpu
                    {
                        let name = info.name.clone();
                        if !name.is_empty() && !adapters.contains(&name) {
                            adapters.push(name);
                        }
                    }
                }
                let _ = tx.send(adapters);
            });
        }

        egui::Grid::new("hardware_gpu_grid")
            .num_columns(2)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label(t!("settings.hardware.adapter"))
                    .search_target(ui, "settings.hardware.adapter");
                if self.gpu_adapters_receiver.is_some() {
                    // Still loading — show a spinner label
                    ui.weak(crate::t!("ui.detecting_gpus"))
                        .search_target(ui, "ui.detecting_gpus");
                    ui.ctx().request_repaint();
                } else {
                    egui::ComboBox::from_id_salt("gpu_adapter_sel")
                        .selected_text(&settings.preferred_gpu)
                        .show_ui(ui, |ui| {
                            for adapter in &self.gpu_adapters {
                                if ui
                                    .selectable_label(settings.preferred_gpu == *adapter, adapter)
                                    .clicked()
                                {
                                    settings.preferred_gpu = adapter.clone();
                                }
                            }
                        });
                }
                ui.end_row();
            });
        ui.label(
            egui::RichText::new(t!("settings.hardware.restart_warning"))
                .small()
                .weak(),
        )
        .search_target(ui, "settings.hardware.restart_warning");

        // -- Status -----------------------------------------------------
        Self::section_header(ui, &t!("settings.hardware.status"));
        let status_text =
            t!("settings.hardware.running_on").replace("{0}", &settings.preferred_gpu);
        ui.label(egui::RichText::new(status_text).strong());

        // -- Reset ------------------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);
        if ui
            .button(t!("settings.hardware.reset"))
            .search_target(ui, "settings.hardware.reset")
            .clicked()
        {
            settings.preferred_gpu = "Auto".to_string();
            self.gpu_adapters.clear();
            self.gpu_adapters_receiver = None;
        }
        settings.save();
    }

    /// Tools settings use the same standard slider as other preferences.
    #[track_caller]
    fn tool_shelf_slider(
        ui: &mut egui::Ui,
        value: &mut f32,
        max: f32,
        label: &str,
        suffix: &str,
        default: f32,
        inherit: Option<&mut bool>,
    ) -> bool {
        let row = crate::ui::controls::ValueRow::new(label)
            .label_width(180.0)
            .show(ui, |ui| {
                let mut slider = crate::ui::controls::Slider::new(value, 0.0..=max)
                    .step_by(1.0)
                    .max_decimals(1)
                    .suffix(suffix)
                    .steppers(true)
                    .double_click_reset_to(default);
                if let Some(inherit) = inherit {
                    slider = slider.on_double_click_reset(|| *inherit = true);
                }
                ui.add(slider).changed()
            });
        Self::search_mark(ui, label, &row.label);
        row.inner
    }

    pub(crate) fn open_tools_settings(&mut self, settings: &AppSettings) {
        self.sync_from_settings(settings);
        self.active_tab = SettingsTab::Interface;
        self.open = true;
    }

    fn show_tools_visibility(ui: &mut egui::Ui, settings: &mut AppSettings, assets: &Assets) {
        use crate::ui::panels::tools::palette;
        egui::CollapsingHeader::new(t!("settings.tools.visible"))
            .id_salt("tools_visibility")
            .open(
                Self::search_reveals(ui.ctx(), &["settings.tools.title"]).or_else(|| {
                    ui.ctx()
                        .data(|d| {
                            d.get_temp::<(String, f64)>(egui::Id::new(
                                "preferences_search_highlight",
                            ))
                        })
                        .filter(|(section, until)| {
                            section == &t!("settings.tools.title") && *until > ui.input(|i| i.time)
                        })
                        .map(|_| true)
                }),
            )
            .show(ui, |ui| {
                let mut changed = false;
                ui.weak(t!("settings.tools.visibility_hint"))
                    .search_target(ui, "settings.tools.visibility_hint");
                if ui
                    .button(t!("settings.tools.show_all"))
                    .search_target(ui, "settings.tools.show_all")
                    .clicked()
                {
                    settings.tools_palette.hidden.clear();
                    changed = true;
                }
                for (group, tools) in palette::GROUPS.iter().enumerate() {
                    if group > 0 {
                        ui.separator();
                    }
                    for (tool, id) in tools.iter().copied().chain(
                        (group == 3).then_some((crate::components::tools::Tool::Shapes, "shapes")),
                    ) {
                        ui.horizontal(|ui| {
                            let icon = if tool == crate::components::tools::Tool::Shapes {
                                Icon::ShapesSmall
                            } else {
                                tool.icon()
                            };
                            if let Some(texture) = assets.get_texture(icon) {
                                ui.add(
                                    egui::Image::from_texture(
                                        egui::load::SizedTexture::from_handle(texture),
                                    )
                                    .fit_to_exact_size(egui::Vec2::splat(20.)),
                                );
                            }
                            let mut visible = palette::visible(&settings.tools_palette, id);
                            if ui.checkbox(&mut visible, tool.icon().tooltip()).changed() {
                                settings.tools_palette.hidden.retain(|hidden| hidden != id);
                                if !visible {
                                    settings.tools_palette.hidden.push(id.into());
                                }
                                changed = true;
                            }
                        });
                    }
                }
                if changed {
                    settings.save();
                }
            });
    }

    // -- Interface Tab ----------------------------------------------
    fn show_interface_tab(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        settings: &mut AppSettings,
        theme: &mut crate::theme::Theme,
        assets: &crate::assets::Assets,
    ) {
        // Poll the browser file picker for an imported theme file.
        #[cfg(target_arch = "wasm32")]
        if let Some((_name, bytes)) = crate::web_bridge::drain_pending("theme_import")
            .into_iter()
            .next()
        {
            match String::from_utf8(bytes) {
                Ok(content) => {
                    settings.import_theme_from_string(&content);
                    self.sync_from_settings(settings);
                    self.dirty = true;
                    self.apply_theme(settings, theme, ctx);
                    self.theme_status = Some((
                        t!("settings.interface.theme_imported"),
                        ui.input(|i| i.time),
                    ));
                }
                Err(e) => {
                    self.theme_status = Some((
                        format!("{}: {e}", t!("settings.interface.theme_error")),
                        ui.input(|i| i.time),
                    ));
                }
            }
        }

        // Re-measure after edits, expansion, theme/font/locale/width changes.
        // Wheel and scrollbar motion alone reuse measured offscreen sections.
        if ui.input(|i| {
            i.events.iter().any(|event| {
                matches!(
                    event,
                    egui::Event::PointerButton { .. }
                        | egui::Event::Key { .. }
                        | egui::Event::Text(_)
                        | egui::Event::Paste(_)
                )
            })
        }) {
            self.section_generation = self.section_generation.wrapping_add(1);
            self.section_settle_until = ui.input(|i| i.time) + 0.4;
        }
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (
            self.section_generation,
            std::sync::Arc::as_ptr(&ctx.global_style()) as usize,
            crate::i18n::current_language(),
            ui.ctx().pixels_per_point().to_bits(),
            settings.advanced_customization,
        )
            .hash(&mut hasher);
        let revision = hasher.finish();
        let force = ui.input(|i| i.time) < self.section_settle_until
            || egui::Popup::is_any_open(ctx)
            || ctx.data(|d| {
                d.get_temp::<SearchTarget>(egui::Id::new("preferences_search_target"))
                    .is_some()
                    || d.get_temp::<u32>(egui::Id::new("preferences_search_jump_frames"))
                        .is_some_and(|frames| frames > 0)
            });
        super::settings_sections::show(ui, "toolbar", revision, force, |ui| {
            if ui
                .checkbox(
                    &mut settings.middle_click_close_tabs,
                    t!("settings.interface.middle_click_close_tabs"),
                )
                .search_target(ui, "settings.interface.middle_click_close_tabs")
                .changed()
            {
                settings.save();
            }
            ui.add_space(8.0);

            let mut layout_changed = false;
            Self::section_header(ui, &t!("settings.tools.title"));
            Self::show_tools_visibility(ui, settings, assets);
            layout_changed |= ui
                .checkbox(
                    &mut settings.tools_palette.animate_resize,
                    t!("settings.tools.animate_resize"),
                )
                .search_target(ui, "settings.tools.animate_resize")
                .changed();
            ui.horizontal(|ui| {
                for (mode, label) in [
                    ("auto", "settings.tools.auto"),
                    ("column", "settings.tools.column"),
                    ("row", "settings.tools.row"),
                    ("grid", "settings.tools.grid"),
                ] {
                    layout_changed |= ui
                        .selectable_value(&mut settings.tools_palette.mode, mode.into(), t!(label))
                        .changed();
                }
            });

            layout_changed |= ui
                .checkbox(
                    &mut settings.tools_show_colors,
                    t!("settings.interface.tools_show_colors"),
                )
                .search_target(ui, "settings.interface.tools_show_colors")
                .changed();
            if settings.tools_palette.mode == "grid" && layout_changed {
                settings.tools_palette.columns = 6;
            }
            Self::section_header(ui, &t!("settings.interface.tool_options"));
            ui.horizontal(|ui| {
                layout_changed |= ui
                    .selectable_value(
                        &mut settings.tool_shelf.docked,
                        false,
                        t!("settings.interface.floating"),
                    )
                    .changed();
                layout_changed |= ui
                    .selectable_value(
                        &mut settings.tool_shelf.docked,
                        true,
                        t!("settings.interface.docked"),
                    )
                    .changed();
                if ui.button("↺").clicked() {
                    settings.tool_shelf = Default::default();
                    settings.tool_shelf_rounding = None;
                    layout_changed = true;
                }
            });
            if settings.tool_shelf.docked {
                ui.label(t!("settings.separator.title"))
                    .search_target(ui, "settings.separator.title");
                let mut automatic = settings.tool_shelf.top_separator.is_none();
                if ui
                    .checkbox(&mut automatic, t!("settings.separator.theme"))
                    .search_target(ui, "settings.separator.theme")
                    .changed()
                {
                    settings.tool_shelf.top_separator = if automatic {
                        None
                    } else {
                        Some([
                            theme.tool_shelf_border.r(),
                            theme.tool_shelf_border.g(),
                            theme.tool_shelf_border.b(),
                        ])
                    };
                    layout_changed = true;
                }
                if let Some(rgb) = &mut settings.tool_shelf.top_separator {
                    let mut color = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                    if crate::ui::controls::color_edit_button_srgba(
                        ui,
                        &mut color,
                        egui::color_picker::Alpha::BlendOrAdditive,
                    )
                    .changed()
                    {
                        *rgb = [color.r(), color.g(), color.b()];
                        layout_changed = true;
                    }
                }
                let mut opacity = settings.tool_shelf.separator_opacity * 100.;
                if Self::tool_shelf_slider(
                    ui,
                    &mut opacity,
                    100.,
                    &t!("settings.separator.opacity"),
                    "%",
                    100.0,
                    None,
                ) {
                    settings.tool_shelf.separator_opacity = opacity / 100.;
                    layout_changed = true;
                }
                if ui
                    .button(t!("common.reset"))
                    .search_target(ui, "common.reset")
                    .clicked()
                {
                    settings.tool_shelf.top_separator = None;
                    settings.tool_shelf.separator_opacity = 1.;
                    layout_changed = true;
                }
            }
            let mut inherit_opacity = false;
            let mut opacity = settings.tool_shelf.effective_opacity() * 100.0;
            if Self::tool_shelf_slider(
                ui,
                &mut opacity,
                100.0,
                &t!("settings.interface.bar_opacity"),
                "%",
                if settings.tool_shelf.docked {
                    100.0
                } else {
                    90.0
                },
                Some(&mut inherit_opacity),
            ) {
                settings.tool_shelf.opacity = Some(opacity / 100.0);
                layout_changed = true;
            }
            if inherit_opacity {
                settings.tool_shelf.opacity = None;
            }
            let mut inherit_rounding = false;
            let mut rounding =
                settings
                    .tool_shelf_rounding
                    .unwrap_or(if settings.tool_shelf.docked {
                        0.0
                    } else {
                        theme.tool_shelf_rounding
                    });
            if Self::tool_shelf_slider(
                ui,
                &mut rounding,
                24.0,
                &t!("settings.interface.bar_radius"),
                " px",
                if settings.tool_shelf.docked {
                    0.0
                } else {
                    theme.tool_shelf_rounding
                },
                Some(&mut inherit_rounding),
            ) {
                settings.tool_shelf_rounding = Some(rounding);
                layout_changed = true;
            }
            if inherit_rounding {
                settings.tool_shelf_rounding = None;
            }
            egui::CollapsingHeader::new(t!("settings.interface.bar_spacing"))
                .id_salt("tool_shelf_spacing")
                .open(Self::search_reveals(
                    ui.ctx(),
                    &[
                        "settings.interface.bar_spacing",
                        "settings.interface.bar_top",
                        "settings.interface.bar_left",
                        "settings.interface.bar_right",
                        "settings.interface.bar_link_sides",
                        "settings.interface.bar_padding_x",
                        "settings.interface.bar_padding_y",
                    ],
                ))
                .show(ui, |ui| {
                    // Docked bars attach directly to the toolbar; their floating margins are retained.
                    ui.add_enabled_ui(!settings.tool_shelf.docked, |ui| {
                        layout_changed |= Self::tool_shelf_slider(
                            ui,
                            &mut settings.tool_shelf.top,
                            64.0,
                            &t!("settings.interface.bar_top"),
                            " px",
                            8.0,
                            None,
                        );
                        layout_changed |= ui
                            .checkbox(
                                &mut settings.tool_shelf.link_sides,
                                t!("settings.interface.bar_link_sides"),
                            )
                            .search_target(ui, "settings.interface.bar_link_sides")
                            .changed();
                        layout_changed |= Self::tool_shelf_slider(
                            ui,
                            &mut settings.tool_shelf.left,
                            64.0,
                            &t!("settings.interface.bar_left"),
                            " px",
                            12.0,
                            None,
                        );
                        ui.add_enabled_ui(!settings.tool_shelf.link_sides, |ui| {
                            layout_changed |= Self::tool_shelf_slider(
                                ui,
                                &mut settings.tool_shelf.right,
                                64.0,
                                &t!("settings.interface.bar_right"),
                                " px",
                                12.0,
                                None,
                            );
                        });
                    });
                    layout_changed |= Self::tool_shelf_slider(
                        ui,
                        &mut settings.tool_shelf.padding_x,
                        32.0,
                        &t!("settings.interface.bar_padding_x"),
                        " px",
                        10.0,
                        None,
                    );
                    layout_changed |= Self::tool_shelf_slider(
                        ui,
                        &mut settings.tool_shelf.padding_y,
                        32.0,
                        &t!("settings.interface.bar_padding_y"),
                        " px",
                        5.0,
                        None,
                    );
                });
            if layout_changed {
                settings.tool_shelf.normalize();
                self.dirty = true;
                settings.save();
                ctx.request_repaint();
            }
            ui.add_space(8.0);
        });
        super::settings_sections::show(ui, "grip", revision, force, |ui| {
            Self::section_header(ui, &t!("settings.interface.resize_grip"));
            let mut grip_changed = false;
            ui.horizontal(|ui| {
                let mut automatic = settings.resize_grip_color.is_none();
                if ui
                    .checkbox(&mut automatic, t!("settings.interface.resize_grip_auto"))
                    .search_target(ui, "settings.interface.resize_grip_auto")
                    .changed()
                {
                    settings.resize_grip_color = if automatic {
                        None
                    } else {
                        Some(ctx.global_style().visuals.weak_text_color().to_opaque())
                    };
                    grip_changed = true;
                }
                if let Some(color) = settings.resize_grip_color.as_mut() {
                    grip_changed |= crate::ui::controls::color_edit_button_srgba(
                        ui,
                        color,
                        egui::color_picker::Alpha::Opaque,
                    )
                    .changed();
                }
            });
            ui.horizontal(|ui| {
                ui.label(t!("settings.interface.resize_grip_opacity"))
                    .search_target(ui, "settings.interface.resize_grip_opacity");
                grip_changed |= ui
                    .add(
                        crate::ui::controls::Slider::new(
                            &mut settings.resize_grip_opacity,
                            0.0..=1.0,
                        )
                        .reset_to(0.5)
                        .custom_formatter(|value, _| format!("{:.0}%", value * 100.0)),
                    )
                    .changed();
                if ui.button("↺").clicked() {
                    settings.resize_grip_color = None;
                    settings.resize_grip_opacity = 0.5;
                    grip_changed = true;
                }
            });
            if grip_changed {
                settings.save();
            }
            ui.add_space(8.0);
        });
        super::settings_sections::show(ui, "theme", revision, force, |ui| {
            // -- Theme ----------------------------------------------------
            Self::section_header(ui, &t!("settings.interface.theme_mode"));
            egui::Grid::new("interface_theme_grid")
                .num_columns(2)
                .spacing([16.0, 6.0])
                .min_col_width(140.0)
                .show(ui, |ui| {
                    ui.label(t!("settings.interface.mode"))
                        .search_target(ui, "settings.interface.mode");
                    let current_name = match self.staged_mode {
                        ThemeMode::Light => t!("settings.interface.mode.light"),
                        ThemeMode::Dark => t!("settings.interface.mode.dark"),
                    };
                    egui::ComboBox::from_id_salt("theme_mode_sel")
                        .selected_text(&current_name)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    self.staged_mode == ThemeMode::Light,
                                    t!("settings.interface.mode.light"),
                                )
                                .clicked()
                            {
                                self.staged_mode = ThemeMode::Light;
                                self.dirty = true;
                            }
                            if ui
                                .selectable_label(
                                    self.staged_mode == ThemeMode::Dark,
                                    t!("settings.interface.mode.dark"),
                                )
                                .clicked()
                            {
                                self.staged_mode = ThemeMode::Dark;
                                self.dirty = true;
                            }
                        });
                    ui.end_row();

                    ui.label(t!("settings.interface.accent_preset"))
                        .search_target(ui, "settings.interface.accent_preset");
                    egui::ComboBox::from_id_salt("accent_preset")
                        .selected_text(crate::i18n::translate_english(&self.staged_preset.label()))
                        .show_ui(ui, |ui| {
                            for &preset in ThemePreset::all() {
                                if ui
                                    .selectable_label(
                                        self.staged_preset == preset,
                                        crate::i18n::translate_english(&preset.label()),
                                    )
                                    .clicked()
                                {
                                    self.staged_preset = preset;
                                    self.staged_accent = preset.accent_colors();
                                    self.dirty = true;
                                }
                            }
                            if self.staged_preset == ThemePreset::Custom {
                                let _ = ui.selectable_label(true, crate::t!("theme_preset.custom"));
                            }
                        });
                    ui.end_row();
                });

            // -- Accent Colors ---------------------------------------------
            Self::section_header(ui, &t!("settings.interface.accent_color"));
            let mode_colors_label = match self.staged_mode {
                ThemeMode::Light => t!("settings.interface.light_mode_colors"),
                ThemeMode::Dark => t!("settings.interface.dark_mode_colors"),
            };
            ui.label(egui::RichText::new(mode_colors_label).weak());
            ui.add_space(4.0);

            // Show the 3 color slots for the staged mode
            let changed = match self.staged_mode {
                ThemeMode::Light => {
                    let mut c = false;
                    c |= Self::color_row(
                        ui,
                        &t!("settings.interface.accent_normal"),
                        &mut self.staged_accent.light_normal,
                    );
                    c |= Self::color_row(
                        ui,
                        &t!("settings.interface.accent_faint"),
                        &mut self.staged_accent.light_faint,
                    );
                    c |= Self::color_row(
                        ui,
                        &t!("settings.interface.accent_strong"),
                        &mut self.staged_accent.light_strong,
                    );
                    c
                }
                ThemeMode::Dark => {
                    let mut c = false;
                    c |= Self::color_row(
                        ui,
                        &t!("settings.interface.accent_normal"),
                        &mut self.staged_accent.dark_normal,
                    );
                    c |= Self::color_row(
                        ui,
                        &t!("settings.interface.accent_faint"),
                        &mut self.staged_accent.dark_faint,
                    );
                    c |= Self::color_row(
                        ui,
                        &t!("settings.interface.accent_strong"),
                        &mut self.staged_accent.dark_strong,
                    );
                    c
                }
            };
            if changed {
                self.staged_preset = ThemePreset::Custom;
                self.dirty = true;
            }
        });
        super::settings_sections::show(ui, "icons", revision, force, |ui| {
            // -- Icon Pack -------------------------------------------------
            Self::section_header(ui, &t!("settings.interface.icon_pack"));
            ui.horizontal(|ui| {
                use crate::config::icon_packs::BundledIconStyle;
                ui.label(crate::t!("ui.icon_style"))
                    .search_target(ui, "ui.icon_style");
                let selected = assets
                    .icon_pack_name()
                    .map(|name| {
                        crate::i18n::format_translation("ui.custom_0", &[("0", name.to_string())])
                    })
                    .unwrap_or_else(|| {
                        crate::i18n::translate_english(settings.bundled_icon_style.label())
                            .to_string()
                    });
                egui::ComboBox::from_id_salt("bundled_icon_style")
                    .selected_text(selected)
                    .show_ui(ui, |ui| {
                        for style in [BundledIconStyle::Luminous, BundledIconStyle::Classic] {
                            let selected = assets.icon_pack_name().is_none()
                                && settings.bundled_icon_style == style;
                            if ui
                                .selectable_label(
                                    selected,
                                    crate::i18n::translate_english(style.label()),
                                )
                                .clicked()
                            {
                                settings.bundled_icon_style = style;
                                self.pending_icon_pack_clear = true;
                            }
                        }
                        if let Some(name) = assets.icon_pack_name() {
                            let _ = ui.selectable_label(
                                true,
                                crate::i18n::format_translation(
                                    "ui.custom_0",
                                    &[("0", name.to_string())],
                                ),
                            );
                        }
                    });
            });
            ui.horizontal(|ui| {
                #[cfg(not(target_arch = "wasm32"))]
                ui.menu_button(t!("settings.interface.icon_pack_browse"), |ui| {
                    if ui
                        .button(t!("settings.interface.icon_pack_folder"))
                        .search_target(ui, "settings.interface.icon_pack_folder")
                        .clicked()
                    {
                        ui.close();
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            self.pending_icon_pack_load = Some(dir);
                        }
                    }
                    if ui
                        .button(t!("settings.interface.icon_pack_archive"))
                        .search_target(ui, "settings.interface.icon_pack_archive")
                        .clicked()
                    {
                        ui.close();
                        if let Some(file) = rfd::FileDialog::new()
                            .add_filter(crate::t!("settings.interface.icon_pack_current"), &["zip"])
                            .pick_file()
                        {
                            self.pending_icon_pack_load = Some(file);
                        }
                    }
                })
                .response
                .on_hover_text(t!("settings.interface.icon_pack_browse_hint"));
                if ui
                    .add_enabled(
                        assets.icon_pack_name().is_some(),
                        egui::Button::new(crate::t!("ui.use_bundled_style")),
                    )
                    .clicked()
                {
                    self.pending_icon_pack_clear = true;
                }
                #[cfg(not(target_arch = "wasm32"))]
                if ui
                    .button(t!("settings.interface.icon_pack_export"))
                    .search_target(ui, "settings.interface.icon_pack_export")
                    .on_hover_text(t!("settings.interface.icon_pack_export_hint"))
                    .clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                    && let Err(e) = assets.export_icon_template(&dir)
                {
                    crate::log_info!("Icon pack export failed: {e}");
                }
            });
            if assets.icon_pack_name().is_some()
                && ui
                    .checkbox(
                        &mut settings.icon_pack_invert_mismatch,
                        t!("settings.interface.icon_pack_invert"),
                    )
                    .search_target(ui, "settings.interface.icon_pack_invert")
                    .on_hover_text(t!("settings.interface.icon_pack_invert_hint"))
                    .changed()
            {
                settings.save();
                self.pending_icon_pack_reload = true;
            }
        });
        super::settings_sections::show(ui, "canvas", revision, force, |ui| {
            // -- Canvas Rendering -----------------------------------------
            Self::section_header(ui, &t!("settings.interface.canvas_rendering"));

            // Quick-setup button (kept at the top of the section for visibility).
            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new(format!(
                            "⚡ {}",
                            t!("settings.interface.pixel_art_preset_apply")
                        ))
                        .strong(),
                    )
                    .min_size(egui::vec2(ui.available_width(), 28.0)),
                )
                .on_hover_text(t!("settings.interface.pixel_art_preset_hint"))
                .clicked()
            {
                settings.zoom_filter_mode = ZoomFilterMode::Nearest;
                settings.pixel_grid_mode = PixelGridMode::AlwaysOn;
                settings.pixel_grid_opacity = 0.45;
                settings.pixel_grid_dashed = true;
                settings.selection_outline_opacity = 0.6;
                settings.persisted_brush_hardness = 1.0;
                settings.persisted_stroke_stabilization = 0.0;
                settings.animated_selection_ants = false;
                settings.low_latency_present = true;
                settings.save();
                self.pending_pixel_art_preset = true;
            }
            ui.add_space(4.0);

            egui::Grid::new("interface_canvas_grid")
                .num_columns(2)
                .spacing([16.0, 6.0])
                .min_col_width(140.0)
                .show(ui, |ui| {
                    ui.label(t!("settings.interface.zoom_filter"))
                        .search_target(ui, "settings.interface.zoom_filter");
                    egui::ComboBox::from_id_salt("zoom_filter_sel")
                        .selected_text(settings.zoom_filter_mode.name())
                        .show_ui(ui, |ui| {
                            for &mode in ZoomFilterMode::all() {
                                if ui
                                    .selectable_label(
                                        settings.zoom_filter_mode == mode,
                                        mode.name(),
                                    )
                                    .clicked()
                                {
                                    settings.zoom_filter_mode = mode;
                                    settings.save();
                                }
                            }
                        });
                    ui.end_row();

                    ui.label(t!("settings.interface.brightness"))
                        .search_target(ui, "settings.interface.brightness");
                    if Self::settings_slider(
                        ui,
                        &mut settings.checkerboard_brightness,
                        0.5..=2.0,
                        0.1,
                        1.0,
                    ) {
                        settings.save();
                    }
                    ui.end_row();

                    ui.label(t!("settings.interface.low_latency_present"))
                        .search_target(ui, "settings.interface.low_latency_present");
                    if ui
                        .checkbox(&mut settings.low_latency_present, "")
                        .on_hover_text(t!("settings.interface.low_latency_present_hint"))
                        .changed()
                    {
                        settings.save();
                    }
                    ui.end_row();

                    ui.label(t!("settings.interface.stroke_stabilization"))
                        .search_target(ui, "settings.interface.stroke_stabilization");
                    if Self::settings_slider(
                        ui,
                        &mut settings.persisted_stroke_stabilization,
                        0.0..=0.9,
                        0.05,
                        0.0,
                    ) {
                        settings.save();
                    }
                    ui.end_row();

                    ui.label(t!("settings.interface.animated_ants"))
                        .search_target(ui, "settings.interface.animated_ants");
                    if ui
                        .checkbox(&mut settings.animated_selection_ants, "")
                        .on_hover_text(t!("settings.interface.animated_ants_hint"))
                        .changed()
                    {
                        settings.save();
                    }
                    ui.end_row();

                    // Pixel grid outline color
                    ui.label(crate::t!("ui.pixel_grid_outline"))
                        .search_target(ui, "ui.pixel_grid_outline");
                    ui.horizontal(|ui| {
                        if Self::color_row(ui, "", &mut settings.pixel_grid_outline_color) {
                            settings.save();
                        }
                        if ui
                            .button("↺")
                            .on_hover_text(crate::t!("ui.reset_to_default"))
                            .clicked()
                        {
                            settings.pixel_grid_outline_color = Color32::from_black_alpha(90);
                            settings.save();
                        }
                    });
                    ui.end_row();

                    ui.label(t!("settings.interface.pixel_grid_opacity"))
                        .search_target(ui, "settings.interface.pixel_grid_opacity");
                    if Self::settings_slider(
                        ui,
                        &mut settings.pixel_grid_opacity,
                        0.0..=1.0,
                        0.05,
                        0.5,
                    ) {
                        settings.save();
                    }
                    ui.end_row();

                    ui.label(t!("settings.interface.pixel_grid_dashed"))
                        .search_target(ui, "settings.interface.pixel_grid_dashed");
                    if ui
                        .checkbox(&mut settings.pixel_grid_dashed, "")
                        .on_hover_text(t!("settings.interface.pixel_grid_dashed_hint"))
                        .changed()
                    {
                        settings.save();
                    }
                    ui.end_row();

                    ui.label(t!("settings.interface.selection_outline_opacity"))
                        .search_target(ui, "settings.interface.selection_outline_opacity");
                    if Self::settings_slider(
                        ui,
                        &mut settings.selection_outline_opacity,
                        0.1..=1.0,
                        0.05,
                        0.7,
                    ) {
                        settings.save();
                    }
                    ui.end_row();

                    ui.label(crate::t!("ui.selection_stripes"))
                        .search_target(ui, "ui.selection_stripes");
                    ui.horizontal(|ui| {
                        if Self::color_row(ui, "", &mut settings.selection_stripe_color) {
                            settings.save();
                        }
                        let mut alpha = settings.selection_stripe_alpha as f32;
                        if ui
                            .add(
                                crate::ui::controls::Slider::new(&mut alpha, 0.0..=96.0)
                                    .show_value(true),
                            )
                            .changed()
                        {
                            settings.selection_stripe_alpha = alpha.round() as u8;
                            settings.save();
                        }
                        if ui
                            .button("↺")
                            .on_hover_text(crate::t!("ui.reset_to_default"))
                            .clicked()
                        {
                            settings.selection_stripe_color =
                                Color32::from_rgba_premultiplied(255, 255, 255, 255);
                            settings.selection_stripe_alpha = 22;
                            settings.save();
                        }
                    });
                    ui.end_row();
                });
            ui.label(
                egui::RichText::new(t!("settings.interface.zoom_filter_hint"))
                    .small()
                    .weak(),
            )
            .search_target(ui, "settings.interface.zoom_filter_hint");
            ui.label(
                egui::RichText::new(t!("settings.interface.brightness_hint"))
                    .small()
                    .weak(),
            )
            .search_target(ui, "settings.interface.brightness_hint");
            ui.label(
                egui::RichText::new(crate::t!(
                    "ui.pixel_grid_outline_center_colors_control_the_dual_stroke_grid_visibili"
                ))
                .small()
                .weak(),
            );
        });
        super::settings_sections::show(ui, "folders", revision, force, |ui| {
            // -- Layer Folders --------------------------------------------
            Self::section_header(ui, &crate::t!("ui.layer_folder_colors"));
            let mut folder_palette_changed = false;
            egui::Grid::new("interface_folder_color_grid")
                .num_columns(2)
                .spacing([16.0, 6.0])
                .min_col_width(140.0)
                .show(ui, |ui| {
                    for (idx, color) in settings.folder_color_palette.iter_mut().enumerate() {
                        ui.label(crate::i18n::format_translation(
                            "ui.folder_0",
                            &[("0", format!("{}", idx + 1))],
                        ));
                        if ui
                            .push_id(("folder_colour", idx), |ui| Self::color_row(ui, "", color))
                            .inner
                        {
                            folder_palette_changed = true;
                        }
                        ui.end_row();
                    }
                });
            if folder_palette_changed {
                settings.save();
            }
            if ui
                .button(crate::t!("ui.reset_folder_colors"))
                .search_target(ui, "ui.reset_folder_colors")
                .clicked()
            {
                settings.folder_color_palette = AppSettings::default_folder_color_palette();
                settings.save();
            }
        });
        super::settings_sections::show(ui, "interaction", revision, force, |ui| {
            Self::section_header(ui, &crate::t!("ui.motion_and_interaction"));
            ui.horizontal(|ui| {
                ui.label(crate::t!("ui.tooltip_delay_seconds"))
                    .search_target(ui, "ui.tooltip_delay_seconds")
                    .on_hover_text(crate::t!(
                        "ui.time_before_tooltips_appear_set_to_0_for_immediate_tooltips"
                    ));
                if ui
                    .add(
                        crate::ui::controls::Numeric::new(&mut settings.tooltip_delay)
                            .range(0.0..=3.0)
                            .speed(0.05),
                    )
                    .changed()
                {
                    settings.save();
                }
            });
            if crate::ui::polish::preferences(ui, &mut settings.ui_polish) {
                settings.history_animations = Some(settings.ui_polish.row_motion);
                theme.history_animations = settings.ui_polish.row_motion;
                theme.polish = settings.ui_polish.clone();
                theme.density = settings.ui_density;
                theme.apply(ctx);
                settings.save();
            }
            Self::section_header(ui, &crate::t!("ui.workspace"));
            if crate::ui::workspace::preferences(
                ui,
                &mut settings.workspace,
                &mut settings.workspace_profiles,
            ) {
                settings.save();
            }
        });
        super::settings_sections::show(ui, "advanced", revision, force, |ui| {
            // -- Advanced Customization -----------------------------------
            Self::section_header(ui, &crate::t!("ui.advanced_customization"));
            ui.checkbox(
                &mut settings.advanced_customization,
                crate::t!("ui.enable_advanced_theme_overrides"),
            )
            .search_target(ui, "ui.enable_advanced_theme_overrides");
            ui.label(
                egui::RichText::new(crate::t!(
                    "ui.fine_tune_individual_colors_geometry_and_effects_overrides_apply_on_to"
                ))
                .small()
                .weak(),
            );

            ui.group(|ui| {
                Self::section_header(ui, &crate::t!("info.appearance"));
                ui.horizontal(|ui| {
                    ui.label(crate::t!("info.font_size"))
                        .search_target(ui, "info.font_size");
                    if crate::ui::dialogs::core::numeric_field_with_buttons(
                        ui,
                        &mut settings.info_font_size,
                        0.2,
                        10.0..=24.0,
                        " pt",
                        1.0,
                    ) {
                        settings.save();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(crate::t!("settings.interface.resize_grip_opacity"))
                        .search_target(ui, "settings.interface.resize_grip_opacity");
                    if Self::settings_slider(ui, &mut settings.info_opacity, 0.0..=1.0, 0.05, 1.0) {
                        settings.save();
                    }
                });
                for (label, value, default) in [
                    (
                        crate::t!("ui.primary_text"),
                        &mut settings.info_text_color,
                        ui.visuals().strong_text_color(),
                    ),
                    (
                        crate::t!("ui.background"),
                        &mut settings.info_background_color,
                        ui.visuals().panel_fill,
                    ),
                ] {
                    ui.horizontal(|ui| {
                        let mut enabled = value.is_some();
                        let mut color = value.unwrap_or([default.r(), default.g(), default.b()]);
                        if ui.checkbox(&mut enabled, label).changed() {
                            *value = enabled.then_some(color);
                            self.dirty = true;
                        }
                        if enabled
                            && crate::ui::controls::color_edit_button_srgb(ui, &mut color).changed()
                        {
                            *value = Some(color);
                            self.dirty = true;
                        }
                        if !enabled {
                            ui.weak(crate::t!("ui.preset_default"))
                                .search_target(ui, "ui.preset_default");
                        }
                    });
                }
                if ui
                    .small_button("↺")
                    .on_hover_text(crate::t!("common.reset"))
                    .clicked()
                {
                    settings.info_font_size = 12.0;
                    settings.info_opacity = 1.0;
                    settings.info_text_color = None;
                    settings.info_background_color = None;
                    settings.save();
                }
            });

            ui.ctx().data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("preferences_search_current_section"),
                    crate::t!("ui.advanced_customization"),
                )
            });
            let reveal_advanced = ui
                .ctx()
                .data(|d| d.get_temp::<SearchTarget>(egui::Id::new("preferences_search_target")))
                .is_some_and(|target| target.section == "ui.advanced_customization");
            if settings.advanced_customization || reveal_advanced {
                ui.scope(|ui| {
                    if !settings.advanced_customization {
                        ui.disable();
                    }
                    ui.add_space(6.0);

                    // UI Density
                    egui::Grid::new("advanced_density_grid")
                        .num_columns(2)
                        .spacing([16.0, 6.0])
                        .min_col_width(140.0)
                        .show(ui, |ui| {
                            ui.label(crate::t!("ui.ui_density"))
                                .search_target(ui, "ui.ui_density");
                            egui::ComboBox::from_id_salt("ui_density_sel")
                                .selected_text(crate::i18n::translate_english(
                                    settings.ui_density.label(),
                                ))
                                .show_ui(ui, |ui| {
                                    for &density in UiDensity::all() {
                                        if ui
                                            .selectable_label(
                                                settings.ui_density == density,
                                                crate::i18n::translate_english(density.label()),
                                            )
                                            .clicked()
                                        {
                                            settings.ui_density = density;
                                            self.dirty = true;
                                        }
                                    }
                                });
                            ui.end_row();
                        });

                    // --- Surface Colors ---
                    let id = ui.make_persistent_id("adv_surface");
                    Self::search_group(
                        ui,
                        id,
                        &[
                            "ui.active_bg3",
                            "ui.background",
                            "ui.elevated_bg2",
                            "ui.panel",
                            "ui.surface_colors",
                            "ui.window",
                        ],
                    )
                    .show_header(ui, |ui| {
                        ui.strong(crate::t!("ui.surface_colors"))
                            .search_target(ui, "ui.surface_colors");
                    })
                    .body(|ui| {
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.background"),
                            &mut settings.ov_bg_color,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.panel"),
                            &mut settings.ov_panel_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.window"),
                            &mut settings.ov_window_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.elevated_bg2"),
                            &mut settings.ov_bg2,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.active_bg3"),
                            &mut settings.ov_bg3,
                            &mut self.dirty,
                        );
                    });

                    // --- Text & Labels ---
                    let id = ui.make_persistent_id("adv_text");
                    Self::search_group(
                        ui,
                        id,
                        &[
                            "ui.faint_text",
                            "ui.muted_text",
                            "ui.primary_text",
                            "ui.text_labels",
                        ],
                    )
                    .show_header(ui, |ui| {
                        ui.strong(crate::t!("ui.text_labels"))
                            .search_target(ui, "ui.text_labels");
                    })
                    .body(|ui| {
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.primary_text"),
                            &mut settings.ov_text_color,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.muted_text"),
                            &mut settings.ov_text_muted,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.faint_text"),
                            &mut settings.ov_text_faint,
                            &mut self.dirty,
                        );
                    });

                    // --- Borders & Separators ---
                    let id = ui.make_persistent_id("adv_borders");
                    Self::search_group(
                        ui,
                        id,
                        &[
                            "ui.border",
                            "ui.borders_separators",
                            "ui.hover_border",
                            "ui.separator",
                        ],
                    )
                    .show_header(ui, |ui| {
                        ui.strong(crate::t!("ui.borders_separators"))
                            .search_target(ui, "ui.borders_separators");
                    })
                    .body(|ui| {
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.border"),
                            &mut settings.ov_border_color,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.hover_border"),
                            &mut settings.ov_border_lit,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.separator"),
                            &mut settings.ov_separator_color,
                            &mut self.dirty,
                        );
                    });

                    // --- Buttons & Controls ---
                    let id = ui.make_persistent_id("adv_buttons");
                    Self::search_group(
                        ui,
                        id,
                        &[
                            "ui.active",
                            "ui.button_bg",
                            "ui.buttons_controls",
                            "ui.control_border",
                            "ui.hover",
                            "ui.icon_button_active",
                            "ui.icon_button_bg",
                            "ui.icon_button_disabled",
                            "ui.manual_input_background",
                            "ui.slider_control_track",
                            "ui.stepper_background",
                        ],
                    )
                    .show_header(ui, |ui| {
                        ui.strong(crate::t!("ui.buttons_controls"))
                            .search_target(ui, "ui.buttons_controls");
                    })
                    .body(|ui| {
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.button_bg"),
                            &mut settings.ov_button_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.hover"),
                            &mut settings.ov_button_hover,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.active"),
                            &mut settings.ov_button_active,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.icon_button_bg"),
                            &mut settings.ov_icon_button_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.icon_button_active"),
                            &mut settings.ov_icon_button_active,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.icon_button_disabled"),
                            &mut settings.ov_icon_button_disabled,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.slider_control_track"),
                            &mut settings.ov_bg3,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.manual_input_background"),
                            &mut settings.ov_text_input_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.stepper_background"),
                            &mut settings.ov_stepper_button_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.control_border"),
                            &mut settings.ov_border_lit,
                            &mut self.dirty,
                        );
                    });

                    // --- Panels & Windows ---
                    let id = ui.make_persistent_id("adv_panels");
                    Self::search_group(
                        ui,
                        id,
                        &[
                            "ui.floating_window",
                            "ui.menu",
                            "ui.panels_windows",
                            "ui.tool_shelf_bg",
                            "ui.tool_shelf_border",
                            "ui.toolbar",
                        ],
                    )
                    .show_header(ui, |ui| {
                        ui.strong(crate::t!("ui.panels_windows"))
                            .search_target(ui, "ui.panels_windows");
                    })
                    .body(|ui| {
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.floating_window"),
                            &mut settings.ov_floating_window_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.tool_shelf_bg"),
                            &mut settings.ov_tool_shelf_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.tool_shelf_border"),
                            &mut settings.ov_tool_shelf_border,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.toolbar"),
                            &mut settings.ov_toolbar_bg,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.menu"),
                            &mut settings.ov_menu_bg,
                            &mut self.dirty,
                        );
                    });

                    // --- Canvas Background ---
                    let id = ui.make_persistent_id("adv_canvas");
                    Self::search_group(
                        ui,
                        id,
                        &[
                            "ui.canvas_background",
                            "ui.gradient_bottom",
                            "ui.gradient_top",
                            "ui.grid_opacity",
                            "ui.grid_visible",
                        ],
                    )
                    .show_header(ui, |ui| {
                        ui.strong(crate::t!("ui.canvas_background"))
                            .search_target(ui, "ui.canvas_background");
                    })
                    .body(|ui| {
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.gradient_top"),
                            &mut settings.ov_canvas_bg_top,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.gradient_bottom"),
                            &mut settings.ov_canvas_bg_bottom,
                            &mut self.dirty,
                        );
                        ui.horizontal(|ui| {
                            ui.label(crate::t!("ui.grid_visible"))
                                .search_target(ui, "ui.grid_visible");
                            ui.checkbox(&mut settings.canvas_grid_visible, "");
                        });
                        ui.horizontal(|ui| {
                            ui.label(crate::t!("ui.grid_opacity"))
                                .search_target(ui, "ui.grid_opacity");
                            if Self::settings_slider(
                                ui,
                                &mut settings.canvas_grid_opacity,
                                0.0..=1.0,
                                0.05,
                                0.4,
                            ) {
                                self.dirty = true;
                            }
                        });
                    });

                    // --- Geometry & Shape ---
                    let id = ui.make_persistent_id("adv_geometry");
                    Self::search_group(
                        ui,
                        id,
                        &[
                            "ui.badge_cornerradius",
                            "ui.dialog_cornerradius",
                            "ui.geometry_shape",
                            "ui.history_card_cornerradius",
                            "ui.history_placeholder_opacity",
                            "ui.menu_cornerradius",
                            "ui.tab_cornerradius",
                            "ui.tool_button_cornerradius",
                            "ui.tool_shelf_cornerradius",
                            "ui.widget_cornerradius",
                            "ui.window_cornerradius",
                        ],
                    )
                    .show_header(ui, |ui| {
                        ui.strong(crate::t!("ui.geometry_shape"))
                            .search_target(ui, "ui.geometry_shape");
                    })
                    .body(|ui| {
                        ui.weak(crate::t!(
                            "ui.history_motion_is_configured_in_motion_and_interaction"
                        ));
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.history_card_cornerradius"),
                            &mut settings.history_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.history_placeholder_opacity"),
                            &mut settings.history_placeholder_opacity,
                            0.0,
                            1.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.widget_cornerradius"),
                            &mut settings.widget_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.window_cornerradius"),
                            &mut settings.window_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.tool_shelf_cornerradius"),
                            &mut settings.tool_shelf_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.tool_button_cornerradius"),
                            &mut settings.tool_button_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.menu_cornerradius"),
                            &mut settings.menu_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.badge_cornerradius"),
                            &mut settings.badge_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.tab_cornerradius"),
                            &mut settings.tab_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                        Self::opt_f32_row(
                            ui,
                            &crate::t!("ui.dialog_cornerradius"),
                            &mut settings.dialog_rounding,
                            0.0,
                            24.0,
                            &mut self.dirty,
                        );
                    });

                    // --- Additional Accents ---
                    let id = ui.make_persistent_id("adv_accents");
                    Self::search_group(
                        ui,
                        id,
                        &[
                            "ui.additional_accents",
                            "ui.reset_all_overrides",
                            "ui.secondary_green",
                            "ui.tertiary_amber",
                        ],
                    )
                    .show_header(ui, |ui| {
                        ui.strong(crate::t!("ui.additional_accents"))
                            .search_target(ui, "ui.additional_accents");
                    })
                    .body(|ui| {
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.secondary_green"),
                            &mut settings.ov_accent3,
                            &mut self.dirty,
                        );
                        Self::opt_color_row(
                            ui,
                            &crate::t!("ui.tertiary_amber"),
                            &mut settings.ov_accent4,
                            &mut self.dirty,
                        );
                    });

                    // Reset overrides button
                    ui.add_space(8.0);
                    if ui
                        .button(crate::t!("ui.reset_all_overrides"))
                        .search_target(ui, "ui.reset_all_overrides")
                        .clicked()
                    {
                        settings.ov_bg_color = None;
                        settings.ov_panel_bg = None;
                        settings.ov_window_bg = None;
                        settings.ov_bg2 = None;
                        settings.ov_bg3 = None;
                        settings.ov_text_color = None;
                        settings.ov_text_muted = None;
                        settings.ov_text_faint = None;
                        settings.ov_border_color = None;
                        settings.ov_border_lit = None;
                        settings.ov_separator_color = None;
                        settings.ov_button_bg = None;
                        settings.ov_button_hover = None;
                        settings.ov_button_active = None;
                        settings.ov_floating_window_bg = None;
                        settings.ov_tool_shelf_bg = None;
                        settings.ov_tool_shelf_border = None;
                        settings.ov_toolbar_bg = None;
                        settings.ov_menu_bg = None;
                        settings.ov_icon_button_bg = None;
                        settings.ov_icon_button_active = None;
                        settings.ov_icon_button_disabled = None;
                        settings.ov_text_input_bg = None;
                        settings.ov_stepper_button_bg = None;
                        settings.ov_canvas_bg_top = None;
                        settings.ov_canvas_bg_bottom = None;
                        settings.ov_glow_accent = None;
                        settings.ov_accent3 = None;
                        settings.ov_accent4 = None;
                        settings.history_rounding = None;
                        settings.history_placeholder_opacity = None;
                        settings.history_animations = None;
                        settings.widget_rounding = None;
                        settings.window_rounding = None;
                        settings.menu_rounding = None;
                        settings.tool_shelf_rounding = None;
                        settings.tool_button_rounding = None;
                        settings.dialog_rounding = None;
                        settings.glow_intensity = 1.0;
                        settings.shadow_strength = 1.0;
                        settings.canvas_grid_visible = true;
                        settings.canvas_grid_opacity = 0.4;
                        settings.ui_density = UiDensity::Normal;
                        self.dirty = true;
                    }
                });
            }
        });
        super::settings_sections::show(ui, "reset", revision, force, |ui| {
            // -- Reset ---------------------------------------------------
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui
                    .button(t!("settings.interface.reset_to_signal"))
                    .search_target(ui, "settings.interface.reset_to_signal")
                    .clicked()
                {
                    self.staged_preset = ThemePreset::Signal;
                    self.staged_accent = ThemePreset::Signal.accent_colors();
                    self.staged_mode = ThemeMode::Light;
                    self.dirty = true;
                    self.apply_theme(settings, theme, ctx);
                }

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(4.0);

                if ui
                    .button(t!("settings.interface.export_theme"))
                    .search_target(ui, "settings.interface.export_theme")
                    .clicked()
                {
                    let theme_str = settings.export_theme_to_string();
                    #[cfg(target_arch = "wasm32")]
                    {
                        crate::web_fs::trigger_download(
                            "my_theme.paintfe-theme",
                            theme_str.as_bytes(),
                        );
                        self.theme_status = Some((
                            t!("settings.interface.theme_exported"),
                            ui.input(|i| i.time),
                        ));
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(path) = crate::services::file_dialogs::dialog(
                        crate::services::file_dialogs::Purpose::Themes,
                        None,
                    )
                    .add_filter(crate::t!("ui.paintfe_theme"), &["paintfe-theme"])
                    .set_file_name("my_theme.paintfe-theme")
                    .save_file()
                    {
                        crate::services::file_dialogs::remember(
                            crate::services::file_dialogs::Purpose::Themes,
                            &path,
                        );
                        match std::fs::write(&path, &theme_str) {
                            Ok(()) => {
                                self.theme_status = Some((
                                    t!("settings.interface.theme_exported"),
                                    ui.input(|i| i.time),
                                ));
                            }
                            Err(e) => {
                                self.theme_status = Some((
                                    format!("{}: {e}", t!("settings.interface.theme_error")),
                                    ui.input(|i| i.time),
                                ));
                            }
                        }
                    }
                }
                #[cfg(target_arch = "wasm32")]
                if ui
                    .button(t!("settings.interface.import_theme"))
                    .search_target(ui, "settings.interface.import_theme")
                    .clicked()
                {
                    crate::web_bridge::open_picker(
                        "theme_import",
                        ".paintfe-theme,text/plain",
                        false,
                    );
                }
                #[cfg(not(target_arch = "wasm32"))]
                if ui
                    .button(t!("settings.interface.import_theme"))
                    .search_target(ui, "settings.interface.import_theme")
                    .clicked()
                    && let Some(path) = crate::services::file_dialogs::dialog(
                        crate::services::file_dialogs::Purpose::Themes,
                        None,
                    )
                    .add_filter(crate::t!("ui.paintfe_theme"), &["paintfe-theme"])
                    .pick_file()
                {
                    crate::services::file_dialogs::remember(
                        crate::services::file_dialogs::Purpose::Themes,
                        &path,
                    );
                    match std::fs::read_to_string(&path) {
                        Ok(content) => {
                            settings.import_theme_from_string(&content);
                            self.sync_from_settings(settings);
                            self.dirty = true;
                            self.apply_theme(settings, theme, ctx);
                            self.theme_status = Some((
                                t!("settings.interface.theme_imported"),
                                ui.input(|i| i.time),
                            ));
                        }
                        Err(e) => {
                            self.theme_status = Some((
                                format!("{}: {e}", t!("settings.interface.theme_error")),
                                ui.input(|i| i.time),
                            ));
                        }
                    }
                }
            });

            // Show status message for 3 seconds
            if let Some((msg, time)) = &self.theme_status {
                let now = ui.input(|i| i.time);
                if now - time < 3.0 {
                    ui.label(egui::RichText::new(msg).small().weak());
                } else {
                    self.theme_status = None;
                }
            }
        });
    }

    /// Render a single color picker row with alpha support, returns true if value changed
    #[track_caller]
    fn color_row(ui: &mut egui::Ui, label: &str, color: &mut egui::Color32) -> bool {
        let label = crate::i18n::translate_english(label);
        let mut changed = false;
        let row = crate::ui::controls::ValueRow::new(format!("{}:", label)).show(ui, |ui| {
            let mut egui_color = *color;
            if crate::ui::controls::color_edit_button_srgba(
                ui,
                &mut egui_color,
                egui::color_picker::Alpha::OnlyBlend,
            )
            .changed()
            {
                *color = egui_color;
                changed = true;
            }
            // Alpha DragValue
            let mut alpha = color.a() as u16;
            if ui
                .add(
                    crate::ui::controls::Numeric::new(&mut alpha)
                        .range(0..=255)
                        .speed(1),
                )
                .changed()
            {
                *color = {
                    let [r, g, b, _] = color.to_srgba_unmultiplied();
                    Color32::from_rgba_unmultiplied(r, g, b, alpha as u8)
                };
                changed = true;
            }
        });
        Self::search_mark(ui, &label, &row.label);
        changed
    }

    /// Render an optional color override row with an enable checkbox + color picker + alpha input + reset button.
    #[track_caller]
    fn opt_color_row(ui: &mut egui::Ui, label: &str, opt: &mut Option<Color32>, dirty: &mut bool) {
        let label = crate::i18n::translate_english(label);
        ui.horizontal(|ui| {
            let mut enabled = opt.is_some();
            if ui.checkbox(&mut enabled, "").changed() {
                if enabled {
                    // Initialise to a neutral grey so the user sees something
                    *opt = Some(opt.unwrap_or(Color32::from_rgb(128, 128, 128)));
                } else {
                    *opt = None;
                }
                *dirty = true;
            }
            let response = ui.label(format!("{label}:"));
            Self::search_mark(ui, &label, &response);
            if let Some(color) = opt {
                let mut egui_color = *color;
                if crate::ui::controls::color_edit_button_srgba(
                    ui,
                    &mut egui_color,
                    egui::color_picker::Alpha::OnlyBlend,
                )
                .changed()
                {
                    *color = egui_color;
                    *dirty = true;
                }
                // Alpha DragValue
                let mut alpha = color.a() as u16;
                if ui
                    .add(
                        crate::ui::controls::Numeric::new(&mut alpha)
                            .range(0..=255)
                            .speed(1),
                    )
                    .changed()
                {
                    let [r, g, b, _] = color.to_srgba_unmultiplied();
                    *color = Color32::from_rgba_unmultiplied(r, g, b, alpha as u8);
                    *dirty = true;
                }
                if ui
                    .small_button("\u{21BA}")
                    .on_hover_text(crate::t!("common.reset"))
                    .clicked()
                {
                    *opt = None;
                    *dirty = true;
                }
            } else {
                ui.weak(crate::t!("ui.preset_default"))
                    .search_target(ui, "ui.preset_default");
            }
        });
    }

    /// Render an optional float override row with an enable checkbox + slider + reset button.
    #[track_caller]
    fn opt_f32_row(
        ui: &mut egui::Ui,
        label: &str,
        opt: &mut Option<f32>,
        min: f32,
        max: f32,
        dirty: &mut bool,
    ) {
        let mut inherit = false;
        ui.horizontal(|ui| {
            let mut enabled = opt.is_some();
            if ui.checkbox(&mut enabled, "").changed() {
                if enabled {
                    *opt = Some((min + max) * 0.5);
                } else {
                    *opt = None;
                }
                *dirty = true;
            }
            let response = ui.label(format!("{label}:"));
            Self::search_mark(ui, label, &response);
            if let Some(val) = opt {
                if ui
                    .add(
                        crate::ui::controls::Slider::new(val, min..=max)
                            .step_by(if max - min <= 1.0 { 0.01 } else { 0.5 })
                            .on_double_click_reset(|| inherit = true),
                    )
                    .changed()
                {
                    *dirty = true;
                }
                if ui
                    .small_button("\u{21BA}")
                    .on_hover_text(crate::t!("common.reset"))
                    .clicked()
                {
                    *opt = None;
                    *dirty = true;
                }
            } else {
                ui.weak(crate::t!("ui.preset_default"))
                    .search_target(ui, "ui.preset_default");
            }
        });
        if inherit {
            *opt = None;
            *dirty = true;
        }
    }

    /// Render a settings slider with ◀/▶ step buttons and a ↺ reset button.
    /// Returns true if the value changed.
    #[track_caller]
    fn settings_slider(
        ui: &mut egui::Ui,
        value: &mut f32,
        range: std::ops::RangeInclusive<f32>,
        step: f64,
        default: f32,
    ) -> bool {
        ui.add(
            crate::ui::controls::Slider::new(value, range)
                .step_by(step)
                .max_decimals(3)
                .steppers(true)
                .reset_to(default),
        )
        .changed()
    }

    /// Render a settings DragValue (usize) with −/+ buttons and a ↺ reset button.
    /// Returns true if the value changed.
    #[track_caller]
    fn settings_drag_usize(
        ui: &mut egui::Ui,
        value: &mut usize,
        range: std::ops::RangeInclusive<usize>,
        suffix: &str,
        default: usize,
    ) -> bool {
        ui.add(
            crate::ui::controls::Numeric::new(value)
                .range(range)
                .suffix(suffix)
                .speed(1.0)
                .steppers(1.0)
                .reset_to(default),
        )
        .changed()
    }

    /// Render a settings DragValue (u32) with −/+ buttons and a ↺ reset button.
    /// Returns true if the value changed.
    #[track_caller]
    fn settings_drag_u32(
        ui: &mut egui::Ui,
        value: &mut u32,
        range: std::ops::RangeInclusive<u32>,
        suffix: &str,
        default: u32,
    ) -> bool {
        ui.add(
            crate::ui::controls::Numeric::new(value)
                .range(range)
                .suffix(suffix)
                .speed(1.0)
                .steppers(1.0)
                .reset_to(default),
        )
        .changed()
    }

    /// Commit staged theme changes to settings and rebuild the theme
    fn apply_theme(
        &mut self,
        settings: &mut AppSettings,
        theme: &mut crate::theme::Theme,
        ctx: &egui::Context,
    ) {
        let effective_accent = if self.staged_preset == ThemePreset::Custom {
            self.staged_accent
        } else {
            self.staged_preset.accent_colors()
        };

        settings.theme_mode = self.staged_mode;
        settings.theme_preset = self.staged_preset;
        settings.custom_accent = effective_accent;
        self.staged_accent = effective_accent;

        *theme = theme.with_accent(self.staged_preset, effective_accent);
        // If mode changed, rebuild with correct mode
        if settings.theme_mode != theme.mode {
            *theme = match settings.theme_mode {
                ThemeMode::Light => {
                    crate::theme::Theme::light_with_accent(self.staged_preset, effective_accent)
                }
                ThemeMode::Dark => {
                    crate::theme::Theme::dark_with_accent(self.staged_preset, effective_accent)
                }
            };
        }
        // Apply user overrides
        let ov = settings.build_theme_overrides();
        theme.apply_overrides(&ov);
        theme.polish = settings.ui_polish.clone();
        theme.density = settings.ui_density;
        theme.apply(ctx);
        self.dirty = false;
        settings.save();
    }

    // -- AI Tab -------------------------------------------------------
    fn show_ai_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        // -- ONNX Runtime ----------------------------------------------
        Self::section_header(ui, &t!("settings.ai.onnx_runtime"));

        ui.label(t!("settings.ai.library_path"))
            .search_target(ui, "settings.ai.library_path");
        ui.horizontal(|ui| {
            let field_w = (ui.available_width() - 38.0).max(120.0);
            if ui
                .add(
                    egui::TextEdit::singleline(&mut self.staged_onnx_path)
                        .desired_width(field_w)
                        .hint_text(t!("settings.ai.onnx_path_placeholder")),
                )
                .changed()
            {
                self.onnx_probe_result = None;
            }
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("\u{1F4C2}").clicked()
                && let Some(path) = crate::services::file_dialogs::dialog(
                    crate::services::file_dialogs::Purpose::Runtime,
                    Some(std::path::Path::new(&self.staged_onnx_path)),
                )
                .add_filter(crate::t!("ui.dynamic_library"), &["dll", "so", "dylib"])
                // Package-managed Linux libraries may end in `.so.1.23.2`.
                .add_filter(crate::t!("ui.all_files"), &["*"])
                .pick_file()
            {
                crate::services::file_dialogs::remember(
                    crate::services::file_dialogs::Purpose::Runtime,
                    &path,
                );
                self.staged_onnx_path = path.display().to_string();
                self.onnx_probe_result = None;
            }
        });
        if let Err(error) = Self::validated_ai_path(&self.staged_onnx_path, false) {
            ui.colored_label(ui.visuals().error_fg_color, error);
            ui.weak(t!("settings.ai.invalid_path_kept"))
                .search_target(ui, "settings.ai.invalid_path_kept");
        }
        ui.label(
            egui::RichText::new(t!("settings.ai.library_hint"))
                .small()
                .weak(),
        )
        .search_target(ui, "settings.ai.library_hint");

        // Security warning — remind users to only use the official runtime
        ui.add_space(4.0);
        let warning_color = egui::Color32::from_rgb(180, 120, 0);
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(t!("settings.ai.security_warning"))
                    .small()
                    .color(warning_color),
            )
            .search_target(ui, "settings.ai.security_warning");
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    self.onnx_test_receiver.is_none(),
                    egui::Button::new(t!("settings.ai.test")),
                )
                .search_target(ui, "settings.ai.test")
                .clicked()
            {
                self.onnx_probe_result = None;
                match Self::validated_ai_path(&self.staged_onnx_path, false) {
                    Ok(path) if path.is_empty() => {
                        self.onnx_probe_result = Some(Err(t!("settings.ai.path_empty")))
                    }
                    Ok(path) => {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let (sender, receiver) = std::sync::mpsc::channel();
                            let ctx = ui.ctx().clone();
                            let probe_path = path.clone();
                            match std::thread::Builder::new()
                                .name("paintfe-onnx-test".into())
                                .spawn(move || {
                                    let _ = sender.send(
                                        crate::ops::ai::probe_onnx_runtime(&probe_path)
                                            .map_err(|e| e.to_string()),
                                    );
                                    ctx.request_repaint();
                                }) {
                                Ok(_) => self.onnx_test_receiver = Some((path, receiver)),
                                Err(error) => self.onnx_probe_result = Some(Err(error.to_string())),
                            }
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            self.onnx_probe_result = Some(
                                crate::ops::ai::probe_onnx_runtime(&path)
                                    .map_err(|e| e.to_string()),
                            );
                        }
                    }
                    Err(error) => self.onnx_probe_result = Some(Err(error)),
                }
            }
            if let Some((path, receiver)) = &self.onnx_test_receiver {
                match receiver.try_recv() {
                    Ok(result) => {
                        if self.staged_onnx_path.trim().trim_matches('"') == path {
                            self.onnx_probe_result = Some(result);
                        }
                        self.onnx_test_receiver = None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.onnx_test_receiver = None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        ui.spinner();
                    }
                }
            }
            match &self.onnx_probe_result {
                Some(Ok(version)) => {
                    ui.label(
                        egui::RichText::new(t!("settings.ai.onnx_loaded").replace("{0}", version))
                            .color(egui::Color32::from_rgb(0, 180, 0)),
                    )
                    .search_target(ui, "settings.ai.onnx_loaded");
                }
                Some(Err(e)) => {
                    ui.label(
                        egui::RichText::new(format!("\u{274C} {}", e))
                            .color(egui::Color32::from_rgb(220, 0, 0)),
                    );
                }
                None => {
                    ui.weak(t!("settings.ai.not_tested"))
                        .search_target(ui, "settings.ai.not_tested");
                }
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(t!("settings.ai.download_from"))
                    .small()
                    .weak(),
            )
            .search_target(ui, "settings.ai.download_from");
            ui.hyperlink_to(
                egui::RichText::new(crate::t!("ui.github_com_microsoft_onnxruntime")).small(),
                "https://github.com/microsoft/onnxruntime/releases",
            );
        });

        // -- Segmentation Model ----------------------------------------
        Self::section_header(ui, &t!("settings.ai.segmentation_model"));

        ui.label(t!("settings.ai.model_path"))
            .search_target(ui, "settings.ai.model_path");
        ui.horizontal(|ui| {
            let field_w = (ui.available_width() - 38.0).max(120.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.staged_model_path)
                    .desired_width(field_w)
                    .hint_text(t!("settings.ai.model_path_placeholder")),
            );
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("\u{1F4C2}").clicked()
                && let Some(path) = crate::services::file_dialogs::dialog(
                    crate::services::file_dialogs::Purpose::Model,
                    Some(std::path::Path::new(&self.staged_model_path)),
                )
                .add_filter(crate::t!("ui.onnx_model"), &["onnx"])
                .pick_file()
            {
                crate::services::file_dialogs::remember(
                    crate::services::file_dialogs::Purpose::Model,
                    &path,
                );
                self.staged_model_path = path.display().to_string();
            }
        });
        if let Err(error) = Self::validated_ai_path(&self.staged_model_path, true) {
            ui.colored_label(ui.visuals().error_fg_color, error);
            ui.weak(t!("settings.ai.invalid_path_kept"))
                .search_target(ui, "settings.ai.invalid_path_kept");
        }
        ui.label(
            egui::RichText::new(t!("settings.ai.model_hint"))
                .small()
                .weak(),
        )
        .search_target(ui, "settings.ai.model_hint");

        ui.add_space(12.0);
        ui.label(egui::RichText::new(t!("settings.ai.supported_models")).strong())
            .search_target(ui, "settings.ai.supported_models");
        ui.add_space(2.0);
        egui::Grid::new("supported_models_grid")
            .num_columns(3)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(t!("settings.ai.model_header"))
                        .strong()
                        .small(),
                )
                .search_target(ui, "settings.ai.model_header");
                ui.label(
                    egui::RichText::new(t!("settings.ai.input_size_header"))
                        .strong()
                        .small(),
                )
                .search_target(ui, "settings.ai.input_size_header");
                ui.label(
                    egui::RichText::new(t!("settings.ai.best_for_header"))
                        .strong()
                        .small(),
                )
                .search_target(ui, "settings.ai.best_for_header");
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.birefnet")).small())
                    .search_target(ui, "settings.ai.birefnet");
                ui.label(egui::RichText::new("1024\u{00D7}1024").small().weak());
                ui.label(
                    egui::RichText::new(t!("settings.ai.birefnet_desc"))
                        .small()
                        .weak(),
                )
                .search_target(ui, "settings.ai.birefnet_desc");
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.u2net")).small())
                    .search_target(ui, "settings.ai.u2net");
                ui.label(egui::RichText::new("320\u{00D7}320").small().weak());
                ui.label(
                    egui::RichText::new(t!("settings.ai.u2net_desc"))
                        .small()
                        .weak(),
                )
                .search_target(ui, "settings.ai.u2net_desc");
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.isnet")).small())
                    .search_target(ui, "settings.ai.isnet");
                ui.label(egui::RichText::new("1024\u{00D7}1024").small().weak());
                ui.label(
                    egui::RichText::new(t!("settings.ai.isnet_desc"))
                        .small()
                        .weak(),
                )
                .search_target(ui, "settings.ai.isnet_desc");
                ui.end_row();
            });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(t!("settings.ai.download_models"))
                    .small()
                    .weak(),
            )
            .search_target(ui, "settings.ai.download_models");
            ui.hyperlink_to(
                egui::RichText::new("BiRefNet").small(),
                "https://github.com/ZhengPeng7/BiRefNet/releases",
            );
            ui.label(egui::RichText::new("\u{2022}").small().weak());
            ui.hyperlink_to(
                egui::RichText::new(crate::t!("settings.ai.u2net")).small(),
                "https://github.com/xuebinqin/U-2-Net",
            );
            ui.label(egui::RichText::new("\u{2022}").small().weak());
            ui.hyperlink_to(
                egui::RichText::new("IS-Net").small(),
                "https://github.com/xuebinqin/DIS",
            );
        });

        egui::CollapsingHeader::new(t!("settings.ai.advanced"))
            .open(Self::search_reveals(
                ui.ctx(),
                &[
                    "settings.ai.advanced",
                    "settings.ai.graph_optimization",
                    "settings.ai.graph_all",
                    "settings.ai.graph_basic",
                    "settings.ai.graph_disabled",
                    "settings.ai.cpu_arena",
                    "settings.ai.memory_hint",
                ],
            ))
            .show(ui, |ui| {
                let old = settings.onnx_graph_optimization;
                ui.label(t!("settings.ai.graph_optimization"))
                    .search_target(ui, "settings.ai.graph_optimization");
                egui::ComboBox::from_id_salt("onnx_graph_optimization")
                    .selected_text(format!("{:?}", old))
                    .show_ui(ui, |ui| {
                        for (value, label) in [
                            (
                                crate::ops::ai::GraphOptimization::All,
                                t!("settings.ai.graph_all"),
                            ),
                            (
                                crate::ops::ai::GraphOptimization::Basic,
                                t!("settings.ai.graph_basic"),
                            ),
                            (
                                crate::ops::ai::GraphOptimization::Disabled,
                                t!("settings.ai.graph_disabled"),
                            ),
                        ] {
                            ui.selectable_value(
                                &mut settings.onnx_graph_optimization,
                                value,
                                label,
                            );
                        }
                    });
                let arena_changed = ui
                    .checkbox(
                        &mut settings.onnx_cpu_memory_arena,
                        t!("settings.ai.cpu_arena"),
                    )
                    .search_target(ui, "settings.ai.cpu_arena")
                    .changed();
                if old != settings.onnx_graph_optimization || arena_changed {
                    settings.save();
                }
                ui.label(
                    egui::RichText::new(t!("settings.ai.memory_hint"))
                        .small()
                        .weak(),
                )
                .search_target(ui, "settings.ai.memory_hint");
                ui.add_space(8.0);
            });

        // -- Reset ---------------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui
                .button(t!("common.reset"))
                .search_target(ui, "common.reset")
                .clicked()
            {
                self.staged_onnx_path.clear();
                self.staged_model_path.clear();
                self.onnx_probe_result = None;
                settings.onnx_runtime_path.clear();
                settings.birefnet_model_path.clear();
                settings.save();
            }
        });

        ui.add_space(8.0);
        // Show current status
        if [&self.staged_onnx_path, &self.staged_model_path]
            .iter()
            .all(|p| !p.trim().is_empty())
            && Self::validated_ai_path(&self.staged_onnx_path, false).is_ok()
            && Self::validated_ai_path(&self.staged_model_path, true).is_ok()
        {
            ui.label(
                egui::RichText::new(t!("settings.ai.configured"))
                    .small()
                    .color(egui::Color32::from_rgb(0, 160, 0)),
            )
            .search_target(ui, "settings.ai.configured");
        } else {
            ui.label(
                egui::RichText::new(t!("settings.ai.not_configured"))
                    .small()
                    .weak(),
            )
            .search_target(ui, "settings.ai.not_configured");
        }
    }

    // -- Fonts Tab (web only) ---------------------------------------------
    #[cfg(target_arch = "wasm32")]
    fn show_fonts_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // Poll the browser file picker for an uploaded font.
        if let Some((name, bytes)) = crate::web_bridge::drain_pending("font_upload")
            .into_iter()
            .next()
        {
            let family = std::path::Path::new(&name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&name)
                .to_string();
            match crate::ops::text::custom_fonts::register_from_bytes(family, bytes) {
                Ok(()) => self.font_upload_error = None,
                Err(e) => self.font_upload_error = Some(e),
            }
        }

        Self::section_header(ui, &crate::t!("ui.upload_a_font"));
        ui.label(
            egui::RichText::new(crate::t!(
                "ui.browsers_can_t_read_your_system_s_installed_fonts_so_upload_a_ttf_otf_"
            ))
            .small()
            .weak(),
        );
        ui.add_space(4.0);
        if ui
            .button(crate::t!("ui.upload_font_file"))
            .search_target(ui, "ui.upload_font_file")
            .clicked()
        {
            crate::web_bridge::open_picker("font_upload", ".ttf,.otf,font/ttf,font/otf", false);
        }
        if let Some(err) = &self.font_upload_error {
            ui.add_space(4.0);
            ui.colored_label(
                egui::Color32::from_rgb(220, 60, 60),
                format!("\u{274C} {err}"),
            );
        }

        Self::section_header(ui, &crate::t!("ui.google_fonts"));
        ui.label(
            egui::RichText::new(crate::t!(
                "ui.fetches_the_font_directly_from_the_google_fonts_source_repository_no_a"
            ))
            .small()
            .weak(),
        );
        ui.add_space(4.0);
        egui::ScrollArea::vertical()
            .id_salt("google_fonts_list")
            .max_height(220.0)
            .show(ui, |ui| {
                for &(family, slug) in crate::ops::google_fonts::CURATED {
                    ui.horizontal(|ui| {
                        ui.label(family);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            match crate::ops::google_fonts::status_of(family) {
                                Some(crate::ops::google_fonts::FetchStatus::Loading) => {
                                    ui.add(egui::Spinner::new().size(14.0));
                                }
                                Some(crate::ops::google_fonts::FetchStatus::Done) => {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(0, 180, 0),
                                        crate::t!("ui.added"),
                                    );
                                }
                                Some(crate::ops::google_fonts::FetchStatus::Failed(e)) => {
                                    if ui
                                        .button(crate::t!("ui.retry"))
                                        .search_target(ui, "ui.retry")
                                        .clicked()
                                    {
                                        crate::ops::google_fonts::fetch(family, slug, ctx.clone());
                                    }
                                    ui.colored_label(
                                        egui::Color32::from_rgb(220, 60, 60),
                                        format!("\u{274C} {e}"),
                                    );
                                }
                                None => {
                                    if ui
                                        .button(crate::t!("selection_mode.add"))
                                        .search_target(ui, "selection_mode.add")
                                        .clicked()
                                    {
                                        crate::ops::google_fonts::fetch(family, slug, ctx.clone());
                                    }
                                }
                            }
                        });
                    });
                }
            });

        let installed = crate::ops::text::custom_fonts::names();
        if !installed.is_empty() {
            Self::section_header(ui, &crate::t!("ui.installed_custom_fonts"));
            let mut to_remove: Option<String> = None;
            for name in &installed {
                ui.horizontal(|ui| {
                    ui.label(name);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button(crate::t!("ui.remove"))
                            .search_target(ui, "ui.remove")
                            .clicked()
                        {
                            to_remove = Some(name.clone());
                        }
                    });
                });
            }
            if let Some(name) = to_remove {
                crate::ops::text::custom_fonts::remove(&name);
            }
        }
    }

    // -- Keybinds Tab ---------------------------------------------
    fn show_keybinds_tab(
        &mut self,
        ui: &mut egui::Ui,
        _settings: &mut AppSettings,
        _assets: &Assets,
    ) {
        Self::section_header(ui, &t!("settings.keybinds.title"));
        ui.weak(t!("settings.keybinds.hint"))
            .search_target(ui, "settings.keybinds.hint");
        ui.add_space(8.0);

        // Detect key press for rebinding
        //
        // On Windows, modifier keys (Ctrl/Shift/Alt) cannot be reliably detected
        // from keyboard events because WM_CHAR for Ctrl+letter is suppressed by
        // configure_event_loop, and winit may not include modifier state in
        // KeyboardInput events. As a workaround, the rebinding button only
        // captures the KEY itself — modifiers are set via toggle buttons in the
        // keybind row below.
        if let Some(rebind_action) = self.rebinding_action {
            let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
            if esc {
                self.rebinding_action = None;
            } else {
                // Capture the key from Event::Key events (modifier-free).
                // Symbol characters (e.g. [, ], /, =, -) come via Event::Text.
                let new_combo = ui.input(|i| {
                    let mut found = None;
                    for ev in &i.events {
                        match ev {
                            egui::Event::Text(t) if t.chars().count() == 1 => {
                                let ch = t.chars().next().unwrap();
                                let is_symbol = !ch.is_ascii_alphanumeric()
                                    && !ch.is_whitespace()
                                    && !ch.is_control();
                                if is_symbol {
                                    found = Some(KeyCombo {
                                        ctrl: false,
                                        shift: false,
                                        alt: false,
                                        key: None,
                                        text_char: Some(t.clone()),
                                    });
                                    break;
                                }
                            }
                            egui::Event::Key {
                                key, pressed: true, ..
                            } if *key != egui::Key::Escape && key_name(*key) != "?" => {
                                found = Some(KeyCombo {
                                    ctrl: false,
                                    shift: false,
                                    alt: false,
                                    key: Some(*key),
                                    text_char: None,
                                });
                                break;
                            }
                            _ => {}
                        }
                    }
                    found
                });

                if let Some(combo) = new_combo {
                    // Preserve any existing modifier flags from the staged binding
                    let existing = self.staged_keybindings.get(rebind_action).cloned();
                    let merged = KeyCombo {
                        ctrl: existing.as_ref().map(|c| c.ctrl).unwrap_or(false),
                        shift: existing.as_ref().map(|c| c.shift).unwrap_or(false),
                        alt: existing.as_ref().map(|c| c.alt).unwrap_or(false),
                        ..combo
                    };
                    self.staged_keybindings.set(rebind_action, merged);
                    self.rebinding_action = None;
                }
            }

            // Consume all keyboard events so they don't trigger canvas shortcuts
            ui.input_mut(|i| {
                i.events.retain(|e| {
                    matches!(
                        e,
                        egui::Event::Copy
                            | egui::Event::Cut
                            | egui::Event::Paste(_)
                            | egui::Event::PointerMoved(_)
                    )
                })
            });
        }

        // Keybinds table — rendered inline without a nested ScrollArea
        // (the outer ScrollArea in show() handles scrolling the tab content)
        let mut current_category = String::new();
        for action in BindableAction::all() {
            let cat = action.category();
            if cat != current_category {
                if !current_category.is_empty() {
                    ui.add_space(6.0);
                }
                ui.label(egui::RichText::new(&cat).strong().size(13.0));
                ui.separator();
                current_category = cat;
            }

            ui.horizontal(|ui| {
                let name = action.display_name();
                let response = ui.label(egui::RichText::new(&name).size(12.0));
                Self::search_mark(ui, &name, &response);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if matches!(
                        action,
                        BindableAction::BrushResizeDragModifier
                            | BindableAction::SelectionPreserveAspectModifier
                    ) {
                        let mut current = self
                            .staged_keybindings
                            .get(*action)
                            .cloned()
                            .unwrap_or_else(|| {
                                if *action == BindableAction::SelectionPreserveAspectModifier {
                                    KeyCombo::modifiers_only(false, false, true)
                                } else {
                                    KeyCombo::modifiers_only(false, true, false)
                                }
                            });
                        egui::ComboBox::from_id_salt(format!("modifier_binding_{action:?}"))
                            .selected_text(current.display())
                            .show_ui(ui, |ui| {
                                let options: Vec<(&str, KeyCombo)> =
                                    if *action == BindableAction::SelectionPreserveAspectModifier {
                                        vec![
                                            ("Alt", KeyCombo::modifiers_only(false, false, true)),
                                            ("Ctrl", KeyCombo::modifiers_only(true, false, false)),
                                        ]
                                    } else {
                                        vec![
                                            ("Shift", KeyCombo::modifiers_only(false, true, false)),
                                            ("Ctrl", KeyCombo::modifiers_only(true, false, false)),
                                            ("Alt", KeyCombo::modifiers_only(false, false, true)),
                                        ]
                                    };
                                for (label, combo) in options {
                                    ui.selectable_value(&mut current, combo, label);
                                }
                            });
                        self.staged_keybindings.set(*action, current);
                    } else {
                        let is_rebinding = self.rebinding_action == Some(*action);

                        // Clear button to unbind (if keybind is set)
                        if let Some(combo) = self.staged_keybindings.get(*action)
                            && (combo.key.is_some()
                                || combo.text_char.is_some()
                                || combo.ctrl
                                || combo.shift
                                || combo.alt)
                        {
                            let unbind_btn = ui.button("x");
                            if unbind_btn.clicked() {
                                self.staged_keybindings
                                    .set(*action, KeyCombo::modifiers_only(false, false, false));
                                self.rebinding_action = None;
                            }
                            unbind_btn.on_hover_text(t!("settings.keybinds.clear_keybind"));
                        }

                        // Modifier prefix toggle buttons (Ctrl / Shift / Alt)
                        // Manual toggle since automatic modifier detection from
                        // keyboard events is unreliable on Windows.
                        let mod_btn_size = egui::vec2(30.0, 20.0);
                        let ctrl_active = self
                            .staged_keybindings
                            .get(*action)
                            .map(|c| c.ctrl)
                            .unwrap_or(false);
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(crate::t!("ui.ctrl")).size(9.0).color(
                                        if ctrl_active {
                                            Color32::WHITE
                                        } else {
                                            Color32::GRAY
                                        },
                                    ),
                                )
                                .min_size(mod_btn_size)
                                .fill(if ctrl_active {
                                    ui.visuals().selection.bg_fill
                                } else {
                                    ui.visuals().widgets.inactive.bg_fill
                                }),
                            )
                            .clicked()
                        {
                            let mut c = self
                                .staged_keybindings
                                .get(*action)
                                .cloned()
                                .unwrap_or_else(|| KeyCombo::modifiers_only(false, false, false));
                            c.ctrl = !c.ctrl;
                            self.staged_keybindings.set(*action, c);
                        }
                        let shift_active = self
                            .staged_keybindings
                            .get(*action)
                            .map(|c| c.shift)
                            .unwrap_or(false);
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(crate::t!("ui.shft")).size(9.0).color(
                                        if shift_active {
                                            Color32::WHITE
                                        } else {
                                            Color32::GRAY
                                        },
                                    ),
                                )
                                .min_size(mod_btn_size)
                                .fill(if shift_active {
                                    ui.visuals().selection.bg_fill
                                } else {
                                    ui.visuals().widgets.inactive.bg_fill
                                }),
                            )
                            .clicked()
                        {
                            let mut c = self
                                .staged_keybindings
                                .get(*action)
                                .cloned()
                                .unwrap_or_else(|| KeyCombo::modifiers_only(false, false, false));
                            c.shift = !c.shift;
                            self.staged_keybindings.set(*action, c);
                        }
                        let alt_active = self
                            .staged_keybindings
                            .get(*action)
                            .map(|c| c.alt)
                            .unwrap_or(false);
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new(crate::t!("ui.alt")).size(9.0).color(
                                        if alt_active {
                                            Color32::WHITE
                                        } else {
                                            Color32::GRAY
                                        },
                                    ),
                                )
                                .min_size(mod_btn_size)
                                .fill(if alt_active {
                                    ui.visuals().selection.bg_fill
                                } else {
                                    ui.visuals().widgets.inactive.bg_fill
                                }),
                            )
                            .clicked()
                        {
                            let mut c = self
                                .staged_keybindings
                                .get(*action)
                                .cloned()
                                .unwrap_or_else(|| KeyCombo::modifiers_only(false, false, false));
                            c.alt = !c.alt;
                            self.staged_keybindings.set(*action, c);
                        }

                        let btn_text = if is_rebinding {
                            egui::RichText::new(t!("settings.keybinds.press_key"))
                                .italics()
                                .color(if ui.visuals().dark_mode {
                                    Color32::from_rgb(100, 200, 255)
                                } else {
                                    Color32::from_rgb(0, 100, 200)
                                })
                        } else {
                            let combo_text = self
                                .staged_keybindings
                                .get(*action)
                                .map(|c| c.display())
                                .unwrap_or_else(|| "—".to_string());
                            egui::RichText::new(combo_text).monospace()
                        };
                        let btn =
                            ui.add(egui::Button::new(btn_text).min_size(egui::vec2(100.0, 20.0)));
                        if btn.clicked() && !is_rebinding {
                            self.rebinding_action = Some(*action);
                        }
                    }
                });
            });
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui
                .button(t!("settings.keybinds.reset_defaults"))
                .search_target(ui, "settings.keybinds.reset_defaults")
                .clicked()
            {
                self.staged_keybindings = KeyBindings::default();
                self.rebinding_action = None;
            }
        });
    }
}

#[cfg(test)]
mod preference_consistency_tests {
    use super::*;

    #[test]
    fn unchanged_ai_paths_are_not_probed_on_scrolling_frames() {
        let ctx = context();
        let mut settings = AppSettings {
            birefnet_model_path: "missing-model.onnx".into(),
            onnx_runtime_path: "missing-runtime.dll".into(),
            ..Default::default()
        };
        let mut window = SettingsWindow {
            staged_model_path: settings.birefnet_model_path.clone(),
            staged_onnx_path: settings.onnx_runtime_path.clone(),
            ..Default::default()
        };
        let mut theme = crate::theme::Theme::light();
        let before = AI_PATH_PROBES.with(|count| count.get());
        for _ in 0..5 {
            window.apply_pending_settings(&mut settings, &mut theme, &ctx);
        }
        assert_eq!(AI_PATH_PROBES.with(|count| count.get()), before);
        window.staged_model_path = "different-missing-model.onnx".into();
        window.apply_pending_settings(&mut settings, &mut theme, &ctx);
        assert_eq!(AI_PATH_PROBES.with(|count| count.get()), before + 1);
        assert_eq!(settings.birefnet_model_path, "missing-model.onnx");
    }

    #[test]
    fn interface_wheel_scrolls_over_tracks_and_numbers_without_changing_values() {
        crate::i18n::init();
        for numeric in [false, true] {
            let ctx = context();
            let mut settings = AppSettings::default();
            let before = settings.tool_shelf.clone();
            let mut theme = crate::theme::Theme::light();
            let mut assets = Assets::new();
            assets.init(&ctx);
            let mut window = SettingsWindow {
                open: true,
                active_tab: SettingsTab::Interface,
                ..Default::default()
            };
            let mut frame = 0;
            let mut run = |events| {
                frame += 1;
                ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1200.0, 900.0),
                        )),
                        time: Some(frame as f64 / 60.0),
                        events,
                        ..Default::default()
                    },
                    |_| {
                        window.show(&ctx, &mut settings, &mut theme, &assets);
                    },
                )
            };
            run(vec![]);
            run(vec![]);
            let output = run(vec![]);
            fn find(shape: &egui::Shape) -> Option<egui::Rect> {
                match shape {
                    egui::Shape::Text(s) if s.galley.text().ends_with('%') => {
                        Some(egui::Rect::from_min_size(s.pos, s.galley.size()))
                    }
                    egui::Shape::Vec(v) => v.iter().find_map(find),
                    _ => None,
                }
            }
            let number = output
                .shapes
                .iter()
                .find_map(|s| find(&s.shape))
                .expect("opacity numeric field visible");
            let pos = if numeric {
                number.center()
            } else {
                number.center() - egui::vec2(70.0, 0.0)
            };
            for _ in 0..8 {
                run(vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::vec2(0.0, -25.0),
                        modifiers: egui::Modifiers::NONE,
                        phase: egui::TouchPhase::Move,
                    },
                ]);
            }
            assert_eq!(settings.tool_shelf, before, "scrolling edited tool options");
            assert_eq!(settings.resize_grip_opacity, 0.5);
            assert_eq!(
                settings.ui_polish,
                crate::ui::polish::PolishSettings::default()
            );
            assert!(!window.dirty);
            assert!(
                ctx.data(|d| d.get_temp::<egui::Vec2>(egui::Id::new("preferences_test_scroll")))
                    .unwrap()
                    .y
                    > 0.0,
                "wheel over a control did not scroll the page"
            );
        }
    }

    #[test]
    #[ignore = "manual Preferences CPU profile"]
    fn profile_interface_scrolling() {
        crate::i18n::init();
        for advanced in [false, true] {
            let ctx = context();
            let mut settings = AppSettings {
                advanced_customization: advanced,
                ..Default::default()
            };
            let mut theme = crate::theme::Theme::light();
            let mut assets = Assets::new();
            assets.init(&ctx);
            let mut window = SettingsWindow {
                open: true,
                active_tab: SettingsTab::Interface,
                ..Default::default()
            };
            let mut samples = Vec::new();
            for frame in 0..100 {
                let start = std::time::Instant::now();
                let _ = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1200.0, 900.0),
                        )),
                        time: Some(frame as f64 / 60.0),
                        events: vec![
                            egui::Event::PointerMoved(egui::pos2(750.0, 450.0)),
                            egui::Event::MouseWheel {
                                unit: egui::MouseWheelUnit::Point,
                                delta: egui::vec2(0.0, if frame < 55 { -25.0 } else { 25.0 }),
                                modifiers: egui::Modifiers::NONE,
                                phase: egui::TouchPhase::Move,
                            },
                        ],
                        ..Default::default()
                    },
                    |_| {
                        window.show(&ctx, &mut settings, &mut theme, &assets);
                    },
                );
                if frame >= 10 {
                    samples.push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "Interface advanced={advanced}: median={:.2}ms p95={:.2}ms",
                samples[samples.len() / 2],
                samples[samples.len() * 95 / 100]
            );
        }
    }

    #[test]
    fn preferences_numeric_drag_keeps_window_still_and_heading_moves_it() {
        crate::i18n::init();
        let ctx = context();
        let mut settings = AppSettings::default();
        let mut theme = crate::theme::Theme::light();
        let mut assets = Assets::new();
        assets.init(&ctx);
        let mut window = SettingsWindow {
            open: true,
            active_tab: SettingsTab::Interface,
            ..Default::default()
        };
        ctx.data_mut(|d| {
            d.insert_temp(
                egui::Id::new("preferences_search_jump"),
                crate::t!("info.appearance"),
            );
            d.insert_temp(egui::Id::new("preferences_search_jump_frames"), 3_u32);
        });
        let mut run = |events| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1200.0, 900.0),
                    )),
                    events,
                    ..Default::default()
                },
                |_| {
                    window.show(&ctx, &mut settings, &mut theme, &assets);
                },
            )
        };
        run(Vec::new());
        run(Vec::new());
        let output = run(Vec::new());
        fn text_rect(shape: &egui::Shape, text: &str) -> Option<egui::Rect> {
            match shape {
                egui::Shape::Text(s) if s.galley.text().contains(text) => {
                    Some(egui::Rect::from_min_size(s.pos, s.galley.size()))
                }
                egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| text_rect(s, text)),
                _ => None,
            }
        }
        let numeric = output
            .shapes
            .iter()
            .find_map(|s| text_rect(&s.shape, " pt"))
            .expect("font-size numeric field is visible");
        let position_id = egui::Id::new("preferences_heading_position");
        let before = ctx.data(|d| d.get_temp::<egui::Pos2>(position_id)).unwrap();
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            pressed,
            button: egui::PointerButton::Primary,
            modifiers: egui::Modifiers::NONE,
        };
        let start = numeric.center();
        run(vec![egui::Event::PointerMoved(start)]);
        run(vec![button(start, true)]);
        for offset in [8.0, 16.0, 24.0, 32.0] {
            run(vec![egui::Event::PointerMoved(
                start + egui::vec2(offset, 0.0),
            )]);
        }
        run(vec![button(start + egui::vec2(32.0, 0.0), false)]);
        assert_eq!(
            ctx.data(|d| d.get_temp::<egui::Pos2>(position_id)),
            Some(before)
        );
        let heading = before + egui::vec2(100.0, 20.0);
        run(vec![egui::Event::PointerMoved(heading)]);
        run(vec![button(heading, true)]);
        run(vec![egui::Event::PointerMoved(
            heading + egui::vec2(30.0, 10.0),
        )]);
        run(vec![button(heading + egui::vec2(30.0, 10.0), false)]);
        assert!(
            settings.info_font_size > 12.0,
            "drag changed the numeric value"
        );
        let after = ctx.data(|d| d.get_temp::<egui::Pos2>(position_id)).unwrap();
        assert!((after - before - egui::vec2(30.0, 10.0)).length() < 2.0);
        AppSettings::default().save();
    }

    #[test]
    fn preferences_search_stays_in_a_compact_row_in_both_themes() {
        crate::i18n::init();
        for dark in [false, true] {
            for width in [420.0, 680.0, 1000.0] {
                let ctx = context();
                ctx.set_visuals(if dark {
                    egui::Visuals::dark()
                } else {
                    egui::Visuals::light()
                });
                let mut settings = SettingsWindow::default();
                for query in ["", "theme", "no such purple elephant", ""] {
                    settings.search_query = query.into();
                    for _ in 0..3 {
                        let _ = ctx.run_ui(egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 800.0))),
                            ..Default::default()
                        }, |ui| {
                                let top = ui.cursor().top();
                                settings.show_search(ui);
                                let content = ui.label("Settings content remains visible");
                                let maximum = if query.is_empty() { 50.0 } else { 300.0 };
                                assert!(content.rect.top() - top < maximum,
                                    "search consumed the content area: dark={dark}, width={width}, query={query}");
                                assert!(content.rect.bottom() < 800.0);
                        });
                    }
                }
            }
        }
    }

    #[test]
    fn issue_186_search_matches_english_and_selected_language() {
        crate::i18n::init();
        let previous = crate::i18n::current_language();
        crate::i18n::set_language("ru");
        let entry = SEARCH_ENTRIES
            .iter()
            .find(|entry| entry.key == "settings.editing.paste_source")
            .unwrap();
        assert!(entry.matches("original position"));
        assert!(entry.matches(&crate::i18n::translate(entry.key)));
        assert!(!entry.matches("no such purple elephant"));
        assert_eq!(entry.tab, SettingsTab::Editing);
        assert_eq!(entry.section, "settings.editing.clipboard");
        crate::i18n::set_language(&previous);
    }

    #[test]
    fn issue_186_typing_keeps_focus_across_the_first_character() {
        crate::i18n::init();
        let ctx = context();
        let mut window = SettingsWindow::default();
        let id = egui::Id::new("preferences_search_field");
        for text in ["", "p", "a", "s", "t", "e"] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(680.0, 600.0),
                    )),
                    events: if text.is_empty() {
                        vec![]
                    } else {
                        vec![egui::Event::Text(text.into())]
                    },
                    ..Default::default()
                },
                |ui| {
                    if text.is_empty() {
                        ui.ctx().memory_mut(|m| m.request_focus(id));
                    }
                    window.show_search(ui);
                },
            );
            assert!(ctx.memory(|m| m.has_focus(id)), "focus lost after {text}");
        }
        assert_eq!(window.search_query, "paste");
    }

    #[test]
    fn issue_186_picking_a_result_preserves_query_and_results() {
        crate::i18n::init();
        let ctx = context();
        let mut window = SettingsWindow {
            search_query: "Paste".into(),
            ..Default::default()
        };
        let frame = |events, window: &mut SettingsWindow| {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(680.0, 600.0),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| window.show_search(ui),
            );
        };
        frame(vec![], &mut window);
        let rect = ctx
            .data(|d| {
                d.get_temp::<egui::Rect>(egui::Id::new((
                    "search_result_rect",
                    "settings.keybinds.title",
                    "keybind.paste",
                )))
            })
            .unwrap();
        frame(vec![egui::Event::PointerMoved(rect.center())], &mut window);
        for pressed in [true, false] {
            frame(
                vec![egui::Event::PointerButton {
                    pos: rect.center(),
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                }],
                &mut window,
            );
        }
        assert_eq!(window.search_query, "Paste");
        assert_eq!(window.active_tab, SettingsTab::Keybinds);
        assert_eq!(
            ctx.data(|d| d.get_temp::<SearchTarget>(egui::Id::new("preferences_search_target")))
                .unwrap()
                .key,
            "keybind.paste"
        );
    }

    #[test]
    fn issue_186_exact_destinations_reveal_and_scroll_in_both_languages() {
        crate::i18n::init();
        let previous = crate::i18n::current_language();
        for language in ["en", "ru"] {
            crate::i18n::set_language(language);
            for key in [
                "settings.interface.bar_padding_x",
                "keybind.paste",
                "ui.panel",
                "info.font_size",
                "settings.ai.cpu_arena",
            ] {
                let entry = SEARCH_ENTRIES
                    .iter()
                    .find(|entry| entry.key == key)
                    .unwrap();
                let ctx = context();
                let mut settings = AppSettings::default();
                let mut theme = crate::theme::Theme::light();
                let mut assets = Assets::new();
                assets.init(&ctx);
                let mut window = SettingsWindow {
                    open: true,
                    active_tab: entry.tab,
                    search_query: entry.english.into(),
                    ..Default::default()
                };
                ctx.data_mut(|d| {
                    d.insert_temp(
                        egui::Id::new("preferences_search_target"),
                        SearchTarget {
                            key: key.into(),
                            section: entry.section.into(),
                            frames: 5,
                            until: 3.0,
                        },
                    )
                });
                for frame in 0..20 {
                    let _ = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(1200.0, 900.0),
                            )),
                            time: Some(frame as f64 / 30.0),
                            ..Default::default()
                        },
                        |_| {
                            window.show(&ctx, &mut settings, &mut theme, &assets);
                        },
                    );
                }
                let rect = ctx
                    .data(|d| {
                        d.get_temp::<egui::Rect>(egui::Id::new("preferences_search_exact_rect"))
                    })
                    .unwrap_or_else(|| panic!("no exact destination: {language} {key}"));
                let clip = ctx
                    .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("search_target_clip")))
                    .unwrap();
                assert!(
                    clip.contains(rect.center()),
                    "destination offscreen: {language} {key}: {rect:?}, clip {clip:?}"
                );
                assert!(
                    !settings.advanced_customization,
                    "search must not enable theme overrides"
                );
            }
        }
        crate::i18n::set_language(&previous);
    }

    #[test]
    fn issue_186_whole_words_rank_before_incidental_substrings() {
        let add = SearchEntry {
            tab: SettingsTab::Interface,
            key: "selection_mode.add",
            english: "Add",
            section: "settings.tools.title",
        };
        let padding = SEARCH_ENTRIES
            .iter()
            .find(|entry| entry.key == "settings.interface.bar_padding_x")
            .unwrap();
        assert!(add.rank("add") < padding.rank("add"));
    }

    fn context() -> egui::Context {
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
    fn ai_paths_validate_file_type_and_preserve_saved_paths_when_invalid() {
        crate::i18n::init();
        let dir = std::env::temp_dir().join(format!(
            "paintfe-ai-path-validation-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for name in [
            "model.onnx",
            "runtime.dll",
            "libonnxruntime.so.1.30.0",
            "runtime.dylib",
            "wrong.txt",
        ] {
            std::fs::write(dir.join(name), b"path validation fixture").unwrap();
        }
        for name in ["runtime.dll", "libonnxruntime.so.1.30.0", "runtime.dylib"] {
            assert!(
                SettingsWindow::validated_ai_path(&dir.join(name).to_string_lossy(), false).is_ok()
            );
        }
        assert!(
            SettingsWindow::validated_ai_path(&dir.join("model.onnx").to_string_lossy(), true)
                .is_ok()
        );
        assert!(
            SettingsWindow::validated_ai_path(&dir.join("wrong.txt").to_string_lossy(), true)
                .is_err()
        );
        assert!(SettingsWindow::validated_ai_path(&dir.to_string_lossy(), false).is_err());
        let model = dir.join("model.onnx").to_string_lossy().into_owned();
        assert_eq!(
            SettingsWindow::validated_ai_path(&format!("  \"{model}\"  "), true).unwrap(),
            model
        );
        assert_eq!(SettingsWindow::validated_ai_path("", true).unwrap(), "");
        let mut settings = AppSettings {
            onnx_runtime_path: "previous-runtime".into(),
            birefnet_model_path: model.clone(),
            ..Default::default()
        };
        let mut window = SettingsWindow::default();
        window.sync_from_settings(&settings);
        window.staged_onnx_path = dir.join("missing.dll").to_string_lossy().into_owned();
        window.staged_model_path = dir.join("wrong.txt").to_string_lossy().into_owned();
        let mut theme = crate::theme::Theme::light();
        window.apply_pending_settings(&mut settings, &mut theme, &context());
        assert_eq!(settings.onnx_runtime_path, "previous-runtime");
        assert_eq!(settings.birefnet_model_path, model);
        assert!(dir.starts_with(std::env::temp_dir()));
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn closing_preferences_saves_pending_theme_and_keybindings() {
        crate::i18n::init();
        let ctx = context();
        let mut settings = AppSettings::default();
        let mut theme = crate::theme::Theme::light();
        let mut window = SettingsWindow {
            open: true,
            ..Default::default()
        };
        window.sync_from_settings(&settings);
        window.staged_mode = ThemeMode::Dark;
        window.dirty = true;
        window
            .staged_keybindings
            .set(BindableAction::Copy, KeyCombo::ctrl_key(egui::Key::F8));
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("settings_sync_flag"), true));
        let mut assets = Assets::new();
        assets.init(&ctx);
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |_ui| {
                window.show(&ctx, &mut settings, &mut theme, &assets);
            },
        );
        assert!(!window.open);
        assert_eq!(settings.theme_mode, ThemeMode::Dark);
        assert_eq!(
            settings.keybindings.get(BindableAction::Copy),
            Some(&KeyCombo::ctrl_key(egui::Key::F8))
        );
        AppSettings::flush_pending_saves().unwrap();
        let restored = AppSettings::load();
        assert_eq!(restored.theme_mode, ThemeMode::Dark);
        assert_eq!(
            restored.keybindings.get(BindableAction::Copy),
            settings.keybindings.get(BindableAction::Copy)
        );
        AppSettings::default().save();
        AppSettings::flush_pending_saves().unwrap();
    }
}

#[cfg(test)]
mod shared_colour_row_tests {
    use super::*;
    #[test]
    fn alpha_capsules_preserve_straight_rgb_in_both_override_variants() {
        for optional in [false, true] {
            let ctx = egui::Context::default();
            let mut color = Some(Color32::from_rgba_unmultiplied(200, 100, 50, 128));
            fn frame(
                ctx: &egui::Context,
                color: &mut Option<Color32>,
                optional: bool,
                events: Vec<egui::Event>,
            ) {
                let _ = ctx.run_ui(
                    egui::RawInput {
                        events,
                        ..Default::default()
                    },
                    |ui| {
                        if optional {
                            SettingsWindow::opt_color_row(ui, "Colour", color, &mut false);
                        } else {
                            SettingsWindow::color_row(ui, "Colour", color.as_mut().unwrap());
                        }
                    },
                );
            }
            frame(&ctx, &mut color, optional, vec![]);
            let field = ctx
                .data(|d| d.get_temp::<egui::Response>(egui::Id::new("control_test_editor")))
                .unwrap();
            ctx.memory_mut(|m| m.request_focus(field.id));
            frame(&ctx, &mut color, optional, vec![]);
            let key = |key, modifiers| egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            };
            frame(
                &ctx,
                &mut color,
                optional,
                vec![
                    key(
                        egui::Key::A,
                        egui::Modifiers {
                            ctrl: true,
                            command: true,
                            ..Default::default()
                        },
                    ),
                    egui::Event::Text("64".into()),
                ],
            );
            frame(
                &ctx,
                &mut color,
                optional,
                vec![key(egui::Key::Tab, egui::Modifiers::NONE)],
            );
            let [r, g, b, a] = color.unwrap().to_srgba_unmultiplied();
            assert_eq!(a, 64);
            for (actual, expected) in [(r, 200_u8), (g, 100), (b, 50)] {
                assert!(actual.abs_diff(expected) <= 3);
            }
        }
    }
}
