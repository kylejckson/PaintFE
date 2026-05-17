use crate::config::brushes::Assets;
use crate::config::icons::Icon;
use crate::config::keybindings::{BindableAction, KeyBindings, KeyCombo, key_name};
use crate::config::settings::{AppSettings, PixelGridMode, ZoomFilterMode};
use crate::theme::{AccentColors, ThemeMode, ThemePreset, UiDensity};
use eframe::egui;
use egui::{Color32, Sense, Vec2};

pub struct SettingsWindow {
    pub open: bool,
    active_tab: SettingsTab,
    /// Staging copy of accent colors for the "Interface" tab (applied on "Apply")
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
    /// Staging copy of keybindings for the Keybinds tab
    staged_keybindings: KeyBindings,
    /// Which action is currently being rebound (waiting for key press)
    pub rebinding_action: Option<BindableAction>,
    /// Brief status message after theme import/export (cleared on next frame)
    theme_status: Option<(String, f64)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SettingsTab {
    General,
    Interface,
    Hardware,
    Keybinds,
    AI,
}

impl Default for SettingsWindow {
    fn default() -> Self {
        let preset = ThemePreset::Signal;
        Self {
            open: false,
            active_tab: SettingsTab::General,
            staged_accent: preset.accent_colors(),
            staged_preset: preset,
            staged_mode: ThemeMode::Light,
            dirty: false,
            gpu_adapters: Vec::new(),
            gpu_adapters_receiver: None,
            staged_onnx_path: String::new(),
            staged_model_path: String::new(),
            onnx_probe_result: None,
            staged_keybindings: KeyBindings::default(),
            rebinding_action: None,
            theme_status: None,
        }
    }
}

impl SettingsWindow {
    /// Sync staged state from current settings (call when opening)
    fn sync_from_settings(&mut self, settings: &AppSettings) {
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
        self.staged_keybindings = settings.keybindings.clone();
        self.rebinding_action = None;
        self.dirty = false;
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        settings: &mut AppSettings,
        theme: &mut crate::theme::Theme,
        assets: &Assets,
    ) {
        if !self.open {
            return;
        }

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
        let mut should_close = false;

        egui::Window::new("settings_window_internal")
            .title_bar(false)
            .resizable(true)
            .collapsible(false)
            .default_width(680.0)
            .default_height(540.0)
            .min_width(600.0)
            .min_height(400.0)
            .max_width(ctx.content_rect().width() * 0.9)
            .max_height(ctx.content_rect().height() * 0.9)
            .show(ctx, |ui| {
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

                    // X close button on the right
                    let btn_size = Vec2::splat(header_height);
                    let btn_rect = egui::Rect::from_min_size(
                        egui::pos2(rect.max.x - btn_size.x, rect.min.y),
                        btn_size,
                    );
                    let btn_response =
                        ui.interact(btn_rect, ui.id().with("hdr_close"), Sense::click());
                    if btn_response.hovered() {
                        painter.rect_filled(
                            btn_rect,
                            egui::CornerRadius::ZERO,
                            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 55),
                        );
                    }
                    painter.text(
                        btn_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "×",
                        egui::FontId::proportional(14.0),
                        accent,
                    );
                    if btn_response.clicked() {
                        should_close = true;
                    }
                }
                // Force the window body to use the full allocated height so the
                // sidebar doesn't collapse the window to just ~165 px.
                let available_h = ui.available_height();
                ui.set_min_height(available_h);
                ui.horizontal(|ui| {
                    // -- Left Sidebar --
                    ui.vertical(|ui| {
                        ui.set_min_width(132.0);
                        ui.set_max_width(132.0);
                        ui.set_min_height(available_h);
                        ui.add_space(4.0);

                        let tabs: [(SettingsTab, Icon, String); 5] = [
                            (
                                SettingsTab::General,
                                Icon::SettingsGeneral,
                                t!("settings.tab.general"),
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
                            (SettingsTab::AI, Icon::SettingsAI, t!("settings.tab.ai")),
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
                            // Draw selection background
                            if selected {
                                ui.painter()
                                    .rect_filled(rect, 2.0, ui.visuals().selection.bg_fill);
                            } else if response.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    2.0,
                                    ui.visuals().widgets.hovered.bg_fill,
                                );
                            }
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
                                    text_color,
                                );
                            }
                            // Draw label
                            let text_pos =
                                egui::pos2(icon_rect.right() + 6.0, rect.center().y - 6.0);
                            ui.painter().text(
                                text_pos,
                                egui::Align2::LEFT_TOP,
                                label,
                                egui::FontId::proportional(12.0),
                                text_color,
                            );
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
                        egui::ScrollArea::vertical()
                            .id_salt(scroll_id)
                            .auto_shrink([false; 2])
                            .max_height(1000.0)
                            .show(ui, |ui| {
                                ui.set_min_height(1000.0);
                                ui.add_space(4.0);
                                match self.active_tab {
                                    SettingsTab::General => {
                                        self.show_general_tab(ui, settings);
                                    }
                                    SettingsTab::Interface => {
                                        self.show_interface_tab(ui, ctx, settings, theme);
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
                                }
                            });
                    });
                });
            });

        self.open = show && !should_close;
        if !self.open {
            // Persist any staged interface/theme edits when closing settings.
            if self.dirty {
                self.apply_theme(settings, theme, ctx);
            }
            // Clear the sync flag when window closes
            ctx.data_mut(|d| d.insert_temp(id, false));
        }
    }

    /// Draws a bold section label followed by a thin separator rule.
    /// Used consistently across all settings tabs.
    fn section_header(ui: &mut egui::Ui, label: &str) {
        ui.add_space(10.0);
        ui.label(egui::RichText::new(label).strong().size(13.0));
        ui.separator();
        ui.add_space(3.0);
    }

    // -- General Tab -------------------------------------------
    fn show_general_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        // -- Language -------------------------------------------
        Self::section_header(ui, "Language");
        ui.horizontal(|ui| {
            ui.label("Language:");
            let current_code = if settings.language.is_empty() {
                "auto".to_string()
            } else {
                settings.language.clone()
            };
            let display_text = if current_code == "auto" {
                "Auto (System)".to_string()
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
                        .selectable_label(current_code == "auto", "Auto (System)")
                        .clicked()
                    {
                        settings.language = String::new();
                        let detected = crate::i18n::detect_system_language();
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
                            settings.language = code.to_string();
                            crate::i18n::set_language(code);
                        }
                    }
                });
        });
        ui.label(
            egui::RichText::new("Restart recommended after changing language.")
                .small()
                .weak(),
        );

        // -- History & Auto-save -------------------------------------------
        Self::section_header(ui, &t!("settings.general.history"));
        egui::Grid::new("general_history_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .min_col_width(160.0)
            .show(ui, |ui| {
                ui.label(t!("settings.general.max_undo_steps"));
                Self::settings_drag_usize(ui, &mut settings.max_undo_steps, 10..=500, " steps", 50);
                ui.end_row();

                ui.label("Auto-save interval:");
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
                        .selected_text(current_label)
                        .show_ui(ui, |ui| {
                            for &(val, label) in OPTIONS {
                                if ui
                                    .selectable_label(settings.auto_save_minutes == val, label)
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
        );
        if settings.auto_save_minutes > 0 {
            if let Some(dir) = crate::io::autosave_dir() {
                ui.label(
                    egui::RichText::new(format!("Auto-saves to: {}", dir.display()))
                        .small()
                        .weak(),
                );
            }
        } else {
            ui.label(
                egui::RichText::new(
                    "Auto-save is disabled — enable it to protect against crashes.",
                )
                .small()
                .weak(),
            );
        }

        // -- Canvas Display -------------------------------------------
        Self::section_header(ui, &t!("settings.general.display"));
        ui.horizontal(|ui| {
            ui.label(t!("settings.general.pixel_grid"));
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
        Self::section_header(ui, "Startup Canvas");
        ui.checkbox(
            &mut settings.create_canvas_on_startup,
            "Create a blank canvas on startup",
        );
        ui.label(
            egui::RichText::new(
                "When disabled, the app starts empty — use File > New or the + tab to create a canvas.",
            )
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
                    ui.label("Default width:");
                    Self::settings_drag_u32(
                        ui,
                        &mut settings.default_canvas_width,
                        1..=65535u32,
                        " px",
                        800,
                    );
                    ui.end_row();

                    ui.label("Default height:");
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
        Self::section_header(ui, "Behaviour");
        ui.checkbox(
            &mut settings.confirm_on_exit,
            "Confirm before exiting with unsaved changes",
        );
        ui.label(
            egui::RichText::new("Shows a save prompt when quitting with unsaved projects.")
                .small()
                .weak(),
        );

        ui.add_space(6.0);
        ui.checkbox(
            &mut settings.clipboard_copy_transparent_cutout,
            "Transparent pasted pixels overwrite destination",
        );
        ui.label(
            egui::RichText::new(
                "When enabled, pasted selections keep their original silhouette instead of filling their bounding box.",
            )
            .small()
            .weak(),
        );

        // -- Debug Info ---------------------------------------------------
        Self::section_header(ui, &t!("settings.general.debug_info"));
        ui.checkbox(
            &mut settings.show_debug_panel,
            t!("settings.general.show_debug"),
        );
        ui.label(
            egui::RichText::new(t!("settings.general.debug_hint"))
                .small()
                .weak(),
        );
        if settings.show_debug_panel {
            ui.add_space(4.0);
            ui.indent("debug_options", |ui| {
                ui.checkbox(
                    &mut settings.debug_show_canvas_size,
                    t!("settings.general.debug_canvas_size"),
                );
                ui.checkbox(
                    &mut settings.debug_show_zoom,
                    t!("settings.general.debug_zoom"),
                );
                ui.checkbox(
                    &mut settings.debug_show_fps,
                    t!("settings.general.debug_fps"),
                );
                ui.checkbox(
                    &mut settings.debug_show_gpu,
                    t!("settings.general.debug_gpu"),
                );
                ui.checkbox(
                    &mut settings.debug_show_operations,
                    t!("settings.general.debug_operations"),
                );
                ui.checkbox(
                    &mut settings.show_tool_info,
                    t!("settings.general.show_tool_info"),
                );
            });
        }

        // -- About ------------------------------------------------------
        Self::section_header(ui, "About");
        egui::Grid::new("about_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Version").strong());
                ui.label(env!("CARGO_PKG_VERSION"));
                ui.end_row();

                ui.label(egui::RichText::new("Author").strong());
                ui.label("Kyle Jackson");
                ui.end_row();

                ui.label(egui::RichText::new("License").strong());
                ui.label("MIT");
                ui.end_row();

                ui.label(egui::RichText::new("Source").strong());
                ui.hyperlink_to("paintfe.com", "https://paintfe.com");
                ui.end_row();
            });

        // -- Reset ------------------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);
        if ui.button(t!("settings.general.reset_defaults")).clicked() {
            *settings = AppSettings::default();
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

        // Kick off background enumeration on first visit (non-blocking)
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
                ui.label(t!("settings.hardware.adapter"));
                if self.gpu_adapters_receiver.is_some() {
                    // Still loading — show a spinner label
                    ui.weak("Detecting GPUs...");
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
        );

        // -- Status -----------------------------------------------------
        Self::section_header(ui, &t!("settings.hardware.status"));
        let status_text =
            t!("settings.hardware.running_on").replace("{0}", &settings.preferred_gpu);
        ui.label(egui::RichText::new(status_text).strong());

        // -- Reset ------------------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);
        if ui.button(t!("settings.hardware.reset")).clicked() {
            settings.preferred_gpu = "Auto".to_string();
            self.gpu_adapters.clear();
            self.gpu_adapters_receiver = None;
        }
        settings.save();
    }

    // -- Interface Tab ----------------------------------------------
    fn show_interface_tab(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        settings: &mut AppSettings,
        theme: &mut crate::theme::Theme,
    ) {
        // -- Theme ----------------------------------------------------
        Self::section_header(ui, &t!("settings.interface.theme_mode"));
        egui::Grid::new("interface_theme_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .min_col_width(140.0)
            .show(ui, |ui| {
                ui.label(t!("settings.interface.mode"));
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

                ui.label(t!("settings.interface.accent_preset"));
                egui::ComboBox::from_id_salt("accent_preset")
                    .selected_text(self.staged_preset.label())
                    .show_ui(ui, |ui| {
                        for &preset in ThemePreset::all() {
                            if ui
                                .selectable_label(self.staged_preset == preset, preset.label())
                                .clicked()
                            {
                                self.staged_preset = preset;
                                self.staged_accent = preset.accent_colors();
                                self.dirty = true;
                            }
                        }
                        if self.staged_preset == ThemePreset::Custom {
                            let _ = ui.selectable_label(true, "Custom");
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

        // -- Canvas Rendering -----------------------------------------
        Self::section_header(ui, &t!("settings.interface.canvas_rendering"));
        egui::Grid::new("interface_canvas_grid")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .min_col_width(140.0)
            .show(ui, |ui| {
                ui.label(t!("settings.interface.zoom_filter"));
                egui::ComboBox::from_id_salt("zoom_filter_sel")
                    .selected_text(settings.zoom_filter_mode.name())
                    .show_ui(ui, |ui| {
                        for &mode in ZoomFilterMode::all() {
                            if ui
                                .selectable_label(settings.zoom_filter_mode == mode, mode.name())
                                .clicked()
                            {
                                settings.zoom_filter_mode = mode;
                                settings.save();
                            }
                        }
                    });
                ui.end_row();

                ui.label(t!("settings.interface.brightness"));
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

                // Pixel grid outline color
                ui.label("Pixel Grid Outline");
                ui.horizontal(|ui| {
                    if Self::color_row(ui, "", &mut settings.pixel_grid_outline_color) {
                        settings.save();
                    }
                    if ui.button("↺").on_hover_text("Reset to default").clicked() {
                        settings.pixel_grid_outline_color = Color32::from_black_alpha(90);
                        settings.save();
                    }
                });
                ui.end_row();

                ui.label("Selection Stripes");
                ui.horizontal(|ui| {
                    if Self::color_row(ui, "", &mut settings.selection_stripe_color) {
                        settings.save();
                    }
                    let mut alpha = settings.selection_stripe_alpha as f32;
                    if ui
                        .add(egui::Slider::new(&mut alpha, 0.0..=96.0).show_value(true))
                        .changed()
                    {
                        settings.selection_stripe_alpha = alpha.round() as u8;
                        settings.save();
                    }
                    if ui.button("↺").on_hover_text("Reset to default").clicked() {
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
        );
        ui.label(
            egui::RichText::new(t!("settings.interface.brightness_hint"))
                .small()
                .weak(),
        );
        ui.label(
            egui::RichText::new("Pixel grid outline/center colors control the dual-stroke grid visibility on any background.")
                .small()
                .weak(),
        );

        // -- Advanced Customization -----------------------------------
        Self::section_header(ui, "Advanced Customization");
        ui.checkbox(
            &mut settings.advanced_customization,
            "Enable advanced theme overrides",
        );
        ui.label(
            egui::RichText::new(
                "Fine-tune individual colors, geometry, and effects. Overrides apply on top of the selected preset.",
            )
            .small()
            .weak(),
        );

        if settings.advanced_customization {
            ui.add_space(6.0);

            // UI Density
            egui::Grid::new("advanced_density_grid")
                .num_columns(2)
                .spacing([16.0, 6.0])
                .min_col_width(140.0)
                .show(ui, |ui| {
                    ui.label("UI Density");
                    egui::ComboBox::from_id_salt("ui_density_sel")
                        .selected_text(settings.ui_density.label())
                        .show_ui(ui, |ui| {
                            for &density in UiDensity::all() {
                                if ui
                                    .selectable_label(
                                        settings.ui_density == density,
                                        density.label(),
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
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.strong("Surface Colors");
                })
                .body(|ui| {
                    Self::opt_color_row(
                        ui,
                        "Background",
                        &mut settings.ov_bg_color,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(ui, "Panel", &mut settings.ov_panel_bg, &mut self.dirty);
                    Self::opt_color_row(ui, "Window", &mut settings.ov_window_bg, &mut self.dirty);
                    Self::opt_color_row(
                        ui,
                        "Elevated (bg2)",
                        &mut settings.ov_bg2,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(ui, "Active (bg3)", &mut settings.ov_bg3, &mut self.dirty);
                });

            // --- Text & Labels ---
            let id = ui.make_persistent_id("adv_text");
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.strong("Text & Labels");
                })
                .body(|ui| {
                    Self::opt_color_row(
                        ui,
                        "Primary Text",
                        &mut settings.ov_text_color,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Muted Text",
                        &mut settings.ov_text_muted,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Faint Text",
                        &mut settings.ov_text_faint,
                        &mut self.dirty,
                    );
                });

            // --- Borders & Separators ---
            let id = ui.make_persistent_id("adv_borders");
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.strong("Borders & Separators");
                })
                .body(|ui| {
                    Self::opt_color_row(
                        ui,
                        "Border",
                        &mut settings.ov_border_color,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Hover Border",
                        &mut settings.ov_border_lit,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Separator",
                        &mut settings.ov_separator_color,
                        &mut self.dirty,
                    );
                });

            // --- Buttons & Controls ---
            let id = ui.make_persistent_id("adv_buttons");
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.strong("Buttons & Controls");
                })
                .body(|ui| {
                    Self::opt_color_row(
                        ui,
                        "Button BG",
                        &mut settings.ov_button_bg,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Hover",
                        &mut settings.ov_button_hover,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Active",
                        &mut settings.ov_button_active,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Icon Button BG",
                        &mut settings.ov_icon_button_bg,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Icon Button Active",
                        &mut settings.ov_icon_button_active,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Icon Button Disabled",
                        &mut settings.ov_icon_button_disabled,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Slider/Control Track",
                        &mut settings.ov_bg3,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Manual Input Background",
                        &mut settings.ov_text_input_bg,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Stepper +/- Background",
                        &mut settings.ov_stepper_button_bg,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Control Border",
                        &mut settings.ov_border_lit,
                        &mut self.dirty,
                    );
                });

            // --- Panels & Windows ---
            let id = ui.make_persistent_id("adv_panels");
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.strong("Panels & Windows");
                })
                .body(|ui| {
                    Self::opt_color_row(
                        ui,
                        "Floating Window",
                        &mut settings.ov_floating_window_bg,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Tool Shelf BG",
                        &mut settings.ov_tool_shelf_bg,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Tool Shelf Border",
                        &mut settings.ov_tool_shelf_border,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Toolbar",
                        &mut settings.ov_toolbar_bg,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(ui, "Menu", &mut settings.ov_menu_bg, &mut self.dirty);
                });

            // --- Canvas Background ---
            let id = ui.make_persistent_id("adv_canvas");
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.strong("Canvas Background");
                })
                .body(|ui| {
                    Self::opt_color_row(
                        ui,
                        "Gradient Top",
                        &mut settings.ov_canvas_bg_top,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Gradient Bottom",
                        &mut settings.ov_canvas_bg_bottom,
                        &mut self.dirty,
                    );
                    ui.horizontal(|ui| {
                        ui.label("Grid Visible:");
                        ui.checkbox(&mut settings.canvas_grid_visible, "");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Grid Opacity:");
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
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.strong("Geometry & Shape");
                })
                .body(|ui| {
                    Self::opt_f32_row(
                        ui,
                        "Widget CornerRadius",
                        &mut settings.widget_rounding,
                        0.0,
                        24.0,
                        &mut self.dirty,
                    );
                    Self::opt_f32_row(
                        ui,
                        "Window CornerRadius",
                        &mut settings.window_rounding,
                        0.0,
                        24.0,
                        &mut self.dirty,
                    );
                    Self::opt_f32_row(
                        ui,
                        "Tool Shelf CornerRadius",
                        &mut settings.tool_shelf_rounding,
                        0.0,
                        24.0,
                        &mut self.dirty,
                    );
                    Self::opt_f32_row(
                        ui,
                        "Tool Button CornerRadius",
                        &mut settings.tool_button_rounding,
                        0.0,
                        24.0,
                        &mut self.dirty,
                    );
                    Self::opt_f32_row(
                        ui,
                        "Menu CornerRadius",
                        &mut settings.menu_rounding,
                        0.0,
                        24.0,
                        &mut self.dirty,
                    );
                    Self::opt_f32_row(
                        ui,
                        "Badge CornerRadius",
                        &mut settings.badge_rounding,
                        0.0,
                        24.0,
                        &mut self.dirty,
                    );
                    Self::opt_f32_row(
                        ui,
                        "Tab CornerRadius",
                        &mut settings.tab_rounding,
                        0.0,
                        24.0,
                        &mut self.dirty,
                    );
                });

            // --- Additional Accents ---
            let id = ui.make_persistent_id("adv_accents");
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                .show_header(ui, |ui| {
                    ui.strong("Additional Accents");
                })
                .body(|ui| {
                    Self::opt_color_row(
                        ui,
                        "Secondary (Green)",
                        &mut settings.ov_accent3,
                        &mut self.dirty,
                    );
                    Self::opt_color_row(
                        ui,
                        "Tertiary (Amber)",
                        &mut settings.ov_accent4,
                        &mut self.dirty,
                    );
                });

            // Reset overrides button
            ui.add_space(8.0);
            if ui.button("Reset All Overrides").clicked() {
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
                settings.widget_rounding = None;
                settings.window_rounding = None;
                settings.menu_rounding = None;
                settings.tool_shelf_rounding = None;
                settings.tool_button_rounding = None;
                settings.glow_intensity = 1.0;
                settings.shadow_strength = 1.0;
                settings.canvas_grid_visible = true;
                settings.canvas_grid_opacity = 0.4;
                settings.ui_density = UiDensity::Normal;
                self.dirty = true;
            }
        }

        // -- Apply / Reset -------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let apply_btn = ui.add_enabled(
                self.dirty,
                egui::Button::new(format!("  {}  ", t!("settings.interface.apply_changes"))),
            );
            if apply_btn.clicked() {
                self.apply_theme(settings, theme, ctx);
            }
            if ui
                .button(t!("settings.interface.reset_to_signal"))
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

            if ui.button(t!("settings.interface.export_theme")).clicked() {
                let theme_str = settings.export_theme_to_string();
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PaintFE Theme", &["paintfe-theme"])
                    .set_file_name("my_theme.paintfe-theme")
                    .save_file()
                {
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
            if ui.button(t!("settings.interface.import_theme")).clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("PaintFE Theme", &["paintfe-theme"])
                    .pick_file()
            {
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
    }

    /// Render a single color picker row with alpha support, returns true if value changed
    fn color_row(ui: &mut egui::Ui, label: &str, color: &mut egui::Color32) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(format!("{}:", label));
            let mut egui_color = *color;
            if egui::color_picker::color_edit_button_srgba(
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
                .add(egui::DragValue::new(&mut alpha).range(0..=255).speed(1))
                .changed()
            {
                *color =
                    Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), alpha as u8);
                changed = true;
            }
        });
        changed
    }

    /// Render an optional color override row with an enable checkbox + color picker + alpha input + reset button.
    fn opt_color_row(ui: &mut egui::Ui, label: &str, opt: &mut Option<Color32>, dirty: &mut bool) {
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
            ui.label(format!("{label}:"));
            if let Some(color) = opt {
                let mut egui_color = *color;
                if egui::color_picker::color_edit_button_srgba(
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
                    .add(egui::DragValue::new(&mut alpha).range(0..=255).speed(1))
                    .changed()
                {
                    *color = Color32::from_rgba_premultiplied(
                        color.r(),
                        color.g(),
                        color.b(),
                        alpha as u8,
                    );
                    *dirty = true;
                }
                if ui.small_button("\u{21BA}").on_hover_text("Reset").clicked() {
                    *opt = None;
                    *dirty = true;
                }
            } else {
                ui.weak("(preset default)");
            }
        });
    }

    /// Render an optional float override row with an enable checkbox + slider + reset button.
    fn opt_f32_row(
        ui: &mut egui::Ui,
        label: &str,
        opt: &mut Option<f32>,
        min: f32,
        max: f32,
        dirty: &mut bool,
    ) {
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
            ui.label(format!("{label}:"));
            if let Some(val) = opt {
                if ui
                    .add(egui::Slider::new(val, min..=max).step_by(0.5_f64))
                    .changed()
                {
                    *dirty = true;
                }
                if ui.small_button("\u{21BA}").on_hover_text("Reset").clicked() {
                    *opt = None;
                    *dirty = true;
                }
            } else {
                ui.weak("(preset default)");
            }
        });
    }

    /// Render a settings slider with ◀/▶ step buttons and a ↺ reset button.
    /// Returns true if the value changed.
    fn settings_slider(
        ui: &mut egui::Ui,
        value: &mut f32,
        range: std::ops::RangeInclusive<f32>,
        step: f64,
        default: f32,
    ) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            let min = *range.start();
            let max = *range.end();
            let step_f = step as f32;

            if ui
                .add(egui::Slider::new(value, range).step_by(step))
                .changed()
            {
                changed = true;
            }

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                if Self::themed_stepper_button(ui, "-", "Decrease").clicked() {
                    *value = (*value - step_f).max(min);
                    changed = true;
                }

                if ui
                    .add_sized(
                        [58.0, 16.0],
                        egui::DragValue::new(value)
                            .range(min..=max)
                            .speed(step_f.max(0.001) as f64)
                            .max_decimals(3),
                    )
                    .changed()
                {
                    changed = true;
                }

                if Self::themed_stepper_button(ui, "+", "Increase").clicked() {
                    *value = (*value + step_f).min(max);
                    changed = true;
                }
            });

            let at_default = (*value - default).abs() < step_f * 0.01;
            let reset_btn = ui.add_enabled(!at_default, egui::Button::new("\u{21BA}").small());
            if reset_btn.on_hover_text("Reset to default").clicked() {
                *value = default;
                changed = true;
            }
        });
        changed
    }

    /// Render a settings DragValue (usize) with −/+ buttons and a ↺ reset button.
    /// Returns true if the value changed.
    fn settings_drag_usize(
        ui: &mut egui::Ui,
        value: &mut usize,
        range: std::ops::RangeInclusive<usize>,
        suffix: &str,
        default: usize,
    ) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            let min = *range.start();
            let max = *range.end();
            if Self::themed_stepper_button(ui, "-", "Decrease").clicked() && *value > min {
                *value -= 1;
                changed = true;
            }
            if ui
                .add(
                    egui::DragValue::new(value)
                        .range(range)
                        .speed(1.0)
                        .suffix(suffix),
                )
                .changed()
            {
                changed = true;
            }
            if Self::themed_stepper_button(ui, "+", "Increase").clicked() && *value < max {
                *value += 1;
                changed = true;
            }
            let reset_btn =
                ui.add_enabled(*value != default, egui::Button::new("\u{21BA}").small());
            if reset_btn.on_hover_text("Reset to default").clicked() {
                *value = default;
                changed = true;
            }
        });
        changed
    }

    /// Render a settings DragValue (u32) with −/+ buttons and a ↺ reset button.
    /// Returns true if the value changed.
    fn settings_drag_u32(
        ui: &mut egui::Ui,
        value: &mut u32,
        range: std::ops::RangeInclusive<u32>,
        suffix: &str,
        default: u32,
    ) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            let min = *range.start();
            let max = *range.end();
            if Self::themed_stepper_button(ui, "-", "Decrease").clicked() && *value > min {
                *value -= 1;
                changed = true;
            }
            if ui
                .add(
                    egui::DragValue::new(value)
                        .range(range)
                        .speed(1.0)
                        .suffix(suffix),
                )
                .changed()
            {
                changed = true;
            }
            if Self::themed_stepper_button(ui, "+", "Increase").clicked() && *value < max {
                *value += 1;
                changed = true;
            }
            let reset_btn =
                ui.add_enabled(*value != default, egui::Button::new("\u{21BA}").small());
            if reset_btn.on_hover_text("Reset to default").clicked() {
                *value = default;
                changed = true;
            }
        });
        changed
    }

    fn themed_stepper_button(ui: &mut egui::Ui, label: &str, tooltip: &str) -> egui::Response {
        let stepper_bg = crate::theme::Theme::stepper_button_bg_for(ui);
        ui.scope(|ui| {
            ui.visuals_mut().widgets.inactive.bg_fill = stepper_bg;
            ui.visuals_mut().widgets.inactive.weak_bg_fill = stepper_bg;
            ui.small_button(label).on_hover_text(tooltip)
        })
        .inner
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
        theme.apply(ctx);
        self.dirty = false;
        settings.save();
    }

    // -- AI Tab -------------------------------------------------------
    fn show_ai_tab(&mut self, ui: &mut egui::Ui, settings: &mut AppSettings) {
        // -- ONNX Runtime ----------------------------------------------
        Self::section_header(ui, &t!("settings.ai.onnx_runtime"));

        ui.label(t!("settings.ai.library_path"));
        ui.horizontal(|ui| {
            let field_w = (ui.available_width() - 38.0).max(120.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.staged_onnx_path)
                    .desired_width(field_w)
                    .hint_text(t!("settings.ai.onnx_path_placeholder")),
            );
            if ui.button("\u{1F4C2}").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("Dynamic Library", &["dll", "so", "dylib"])
                    .pick_file()
            {
                self.staged_onnx_path = path.display().to_string();
                self.onnx_probe_result = None;
            }
        });
        ui.label(
            egui::RichText::new(t!("settings.ai.library_hint"))
                .small()
                .weak(),
        );

        // Security warning — remind users to only use the official runtime
        ui.add_space(4.0);
        let warning_color = egui::Color32::from_rgb(180, 120, 0);
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(t!("settings.ai.security_warning"))
                    .small()
                    .color(warning_color),
            );
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button(t!("settings.ai.test")).clicked() {
                if self.staged_onnx_path.is_empty() {
                    self.onnx_probe_result = Some(Err(t!("settings.ai.path_empty")));
                } else {
                    match crate::ops::ai::probe_onnx_runtime(&self.staged_onnx_path) {
                        Ok(version) => {
                            self.onnx_probe_result = Some(Ok(version));
                        }
                        Err(e) => {
                            self.onnx_probe_result = Some(Err(format!("{}", e)));
                        }
                    }
                }
            }
            match &self.onnx_probe_result {
                Some(Ok(version)) => {
                    ui.label(
                        egui::RichText::new(t!("settings.ai.onnx_loaded").replace("{0}", version))
                            .color(egui::Color32::from_rgb(0, 180, 0)),
                    );
                }
                Some(Err(e)) => {
                    ui.label(
                        egui::RichText::new(format!("\u{274C} {}", e))
                            .color(egui::Color32::from_rgb(220, 0, 0)),
                    );
                }
                None => {
                    ui.weak(t!("settings.ai.not_tested"));
                }
            }
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(t!("settings.ai.download_from"))
                    .small()
                    .weak(),
            );
            ui.hyperlink_to(
                egui::RichText::new("github.com/microsoft/onnxruntime").small(),
                "https://github.com/microsoft/onnxruntime/releases",
            );
        });

        // -- Segmentation Model ----------------------------------------
        Self::section_header(ui, &t!("settings.ai.segmentation_model"));

        ui.label(t!("settings.ai.model_path"));
        ui.horizontal(|ui| {
            let field_w = (ui.available_width() - 38.0).max(120.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.staged_model_path)
                    .desired_width(field_w)
                    .hint_text(t!("settings.ai.model_path_placeholder")),
            );
            if ui.button("\u{1F4C2}").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("ONNX Model", &["onnx"])
                    .pick_file()
            {
                self.staged_model_path = path.display().to_string();
            }
        });
        ui.label(
            egui::RichText::new(t!("settings.ai.model_hint"))
                .small()
                .weak(),
        );

        ui.add_space(12.0);
        ui.label(egui::RichText::new(t!("settings.ai.supported_models")).strong());
        ui.add_space(2.0);
        egui::Grid::new("supported_models_grid")
            .num_columns(3)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(t!("settings.ai.model_header"))
                        .strong()
                        .small(),
                );
                ui.label(
                    egui::RichText::new(t!("settings.ai.input_size_header"))
                        .strong()
                        .small(),
                );
                ui.label(
                    egui::RichText::new(t!("settings.ai.best_for_header"))
                        .strong()
                        .small(),
                );
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.birefnet")).small());
                ui.label(egui::RichText::new("1024\u{00D7}1024").small().weak());
                ui.label(
                    egui::RichText::new(t!("settings.ai.birefnet_desc"))
                        .small()
                        .weak(),
                );
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.u2net")).small());
                ui.label(egui::RichText::new("320\u{00D7}320").small().weak());
                ui.label(
                    egui::RichText::new(t!("settings.ai.u2net_desc"))
                        .small()
                        .weak(),
                );
                ui.end_row();

                ui.label(egui::RichText::new(t!("settings.ai.isnet")).small());
                ui.label(egui::RichText::new("1024\u{00D7}1024").small().weak());
                ui.label(
                    egui::RichText::new(t!("settings.ai.isnet_desc"))
                        .small()
                        .weak(),
                );
                ui.end_row();
            });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(t!("settings.ai.download_models"))
                    .small()
                    .weak(),
            );
            ui.hyperlink_to(
                egui::RichText::new("BiRefNet").small(),
                "https://github.com/ZhengPeng7/BiRefNet/releases",
            );
            ui.label(egui::RichText::new("\u{2022}").small().weak());
            ui.hyperlink_to(
                egui::RichText::new("U\u{00B2}-Net").small(),
                "https://github.com/xuebinqin/U-2-Net",
            );
            ui.label(egui::RichText::new("\u{2022}").small().weak());
            ui.hyperlink_to(
                egui::RichText::new("IS-Net").small(),
                "https://github.com/xuebinqin/DIS",
            );
        });

        // -- Apply / Reset -------------------------------------------
        ui.add_space(16.0);
        ui.separator();
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui.button(t!("common.apply")).clicked() {
                settings.onnx_runtime_path = self.staged_onnx_path.clone();
                settings.birefnet_model_path = self.staged_model_path.clone();
                settings.save();
            }
            if ui.button(t!("common.reset")).clicked() {
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
        if !settings.onnx_runtime_path.is_empty() && !settings.birefnet_model_path.is_empty() {
            ui.label(
                egui::RichText::new(t!("settings.ai.configured"))
                    .small()
                    .color(egui::Color32::from_rgb(0, 160, 0)),
            );
        } else {
            ui.label(
                egui::RichText::new(t!("settings.ai.not_configured"))
                    .small()
                    .weak(),
            );
        }
    }

    // -- Keybinds Tab ---------------------------------------------
    fn show_keybinds_tab(
        &mut self,
        ui: &mut egui::Ui,
        settings: &mut AppSettings,
        _assets: &Assets,
    ) {
        Self::section_header(ui, &t!("settings.keybinds.title"));
        ui.weak(t!("settings.keybinds.hint"));
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
                ui.label(egui::RichText::new(name).size(12.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if *action == BindableAction::BrushResizeDragModifier {
                        let mut current = self
                            .staged_keybindings
                            .get(*action)
                            .cloned()
                            .unwrap_or_else(|| KeyCombo::modifiers_only(false, true, false));
                        egui::ComboBox::from_id_salt("brush_resize_drag_modifier")
                            .selected_text(current.display())
                            .show_ui(ui, |ui| {
                                let options = [
                                    ("Shift", KeyCombo::modifiers_only(false, true, false)),
                                    ("Ctrl", KeyCombo::modifiers_only(true, false, false)),
                                    ("Alt", KeyCombo::modifiers_only(false, false, true)),
                                ];
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
                                egui::Button::new(egui::RichText::new("Ctrl").size(9.0).color(
                                    if ctrl_active {
                                        Color32::WHITE
                                    } else {
                                        Color32::GRAY
                                    },
                                ))
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
                                egui::Button::new(egui::RichText::new("Shft").size(9.0).color(
                                    if shift_active {
                                        Color32::WHITE
                                    } else {
                                        Color32::GRAY
                                    },
                                ))
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
                                egui::Button::new(egui::RichText::new("Alt").size(9.0).color(
                                    if alt_active {
                                        Color32::WHITE
                                    } else {
                                        Color32::GRAY
                                    },
                                ))
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
            if ui.button(t!("common.apply")).clicked() {
                settings.keybindings = self.staged_keybindings.clone();
                settings.save();
            }
            if ui.button(t!("settings.keybinds.reset_defaults")).clicked() {
                self.staged_keybindings = KeyBindings::default();
                self.rebinding_action = None;
            }
        });
    }
}
