// ============================================================================
// PaintFE Script Editor Panel — Floating code editor with syntax highlighting
// ============================================================================

use std::sync::{Arc, atomic::AtomicBool};
use std::collections::HashSet;
use egui::{self, Color32, Rounding, Stroke, Margin, RichText, Layout, Align, Vec2, TextFormat, FontId};
use egui::text::LayoutJob;

/// Maximum panel height to prevent infinite growth. The actual height is
/// clamped to this or the available remaining height in the parent window.
const MAX_PANEL_HEIGHT: f32 = 900.0;

// ============================================================================
// Data structs
// ============================================================================

#[derive(Clone, Debug)]
pub struct ConsoleLine {
    pub text: String,
    pub kind: ConsoleLineKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConsoleLineKind {
    Output,   // print() output
    Error,    // error messages
    Info,     // timing info, pixel counts
}

#[derive(Clone, Debug)]
pub struct SavedScript {
    pub name: String,
    pub code: String,
    pub pinned: bool,
}

/// A script effect registered in the Filter > Custom menu.
/// Stored in the custom_effects subdirectory, separate from saved scripts.
#[derive(Clone, Debug)]
pub struct CustomScriptEffect {
    pub name: String,
    pub code: String,
}

/// Load custom script effects from disk
pub fn load_custom_effects() -> Vec<CustomScriptEffect> {
    let dir = custom_effects_directory();
    if !dir.exists() { return Vec::new(); }

    let mut effects = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "rhai").unwrap_or(false) {
                if let Ok(code) = std::fs::read_to_string(&path) {
                    let name = path.file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    effects.push(CustomScriptEffect { name, code });
                }
            }
        }
    }
    effects.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    effects
}

/// Save a custom script effect to disk
pub fn save_custom_effect(effect: &CustomScriptEffect) {
    let dir = custom_effects_directory();
    let _ = std::fs::create_dir_all(&dir);
    let filename = sanitize_filename(&effect.name);
    let path = dir.join(format!("{}.rhai", filename));
    let _ = std::fs::write(&path, &effect.code);
}

/// Delete a custom script effect from disk
pub fn delete_custom_effect(name: &str) {
    let dir = custom_effects_directory();
    let filename = sanitize_filename(name);
    let path = dir.join(format!("{}.rhai", filename));
    let _ = std::fs::remove_file(&path);
}

fn custom_effects_directory() -> std::path::PathBuf {
    let base = app_data_directory();
    base.join("custom_effects")
}

pub struct ScriptEditorPanel {
    pub code: String,
    pub console_output: Vec<ConsoleLine>,
    pub is_running: bool,
    pub cancel_flag: Arc<AtomicBool>,
    pub progress: Option<f32>,
    pub saved_scripts: Vec<SavedScript>,
    pub selected_script_idx: Option<usize>,
    pub console_expanded: bool,
    highlighter: SyntaxHighlighter,
    pub run_requested: bool,
    pub stop_requested: bool,
    pub close_requested: bool,
    /// Updates canvas on sleep() calls during execution.
    pub live_preview: bool,
    /// Error line (1-based) for red gutter highlight.
    pub error_line: Option<usize>,
    /// When Some, the name-effect popup is open.
    pub naming_effect: Option<String>,
    /// Pending add-effect request (name, code).
    pub pending_add_effect: Option<(String, String)>,
}

impl Default for ScriptEditorPanel {
    fn default() -> Self {
        Self {
            code: String::from("// Write your script here\n// Example: Invert all pixels\nmap_channels(|r, g, b, a| {\n    [255 - r, 255 - g, 255 - b, a]\n});\n"),
            console_output: Vec::new(),
            is_running: false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            progress: None,
            saved_scripts: Vec::new(),
            selected_script_idx: None,
            console_expanded: false,
            highlighter: SyntaxHighlighter::new(),
            run_requested: false,
            stop_requested: false,
            close_requested: false,
            live_preview: true,
            error_line: None,
            naming_effect: None,
            pending_add_effect: None,
        }
    }
}

impl ScriptEditorPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, theme: &crate::theme::Theme) {
        self.run_requested = false;
        self.stop_requested = false;
        self.close_requested = false;

        let is_dark = matches!(theme.mode, crate::theme::ThemeMode::Dark);
        let accent = theme.accent;
        let text_color = theme.text_color;
        let muted = theme.text_muted;
        let time = ui.input(|i| i.time);

        // -- Custom title bar --
        ui.horizontal(|ui| {
            // Neon icon
            ui.label(RichText::new("◆").color(accent).size(14.0));
            ui.label(RichText::new(crate::t!("script.title")).size(14.0).color(text_color).strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.small_button("×").clicked() {
                    self.close_requested = true;
                }
            });
        });

        ui.add_space(2.0);

        // -- Toolbar row --
        ui.horizontal(|ui| {
            if self.is_running {
                // -- Animated "Running…" indicator --
                // Pulsing dot animation
                let dot_count = ((time * 3.0) as usize) % 4;
                let dots: String = ".".repeat(dot_count);
                let pulse = ((time * 4.0).sin() * 0.5 + 0.5) as f32;
                let glow_alpha = (120.0 + pulse * 135.0) as u8;
                let running_color = if is_dark {
                    Color32::from_rgba_unmultiplied(255, 170, 50, glow_alpha)
                } else {
                    Color32::from_rgba_unmultiplied(200, 100, 0, glow_alpha)
                };
                let spinner_text = format!("⟳ {}{}", crate::t!("script.progress.running"), dots);
                let stop_btn = ui.button(
                    RichText::new(spinner_text).color(running_color).strong()
                );
                if stop_btn.clicked() {
                    self.stop_requested = true;
                    self.cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                // Keep repainting for animation
                ui.ctx().request_repaint();
            } else {
                // -- Run button — theme-aware green --
                let run_color = if is_dark {
                    Color32::from_rgb(80, 220, 120)   // soft neon green for dark
                } else {
                    Color32::from_rgb(20, 140, 60)    // readable dark green for light
                };
                let run_btn = ui.button(
                    RichText::new(format!("▶ {}", crate::t!("script.run"))).color(run_color).strong()
                );
                if run_btn.clicked() {
                    self.run_requested = true;
                }
            };

            ui.separator();

            // Live preview toggle
            let preview_icon = if self.live_preview { "👁" } else { "○" };
            let preview_color = if self.live_preview { accent } else { muted };
            if ui.button(RichText::new(format!("{} Live", preview_icon)).color(preview_color).size(11.0)).on_hover_text("Preview changes on canvas during sleep() calls").clicked() {
                self.live_preview = !self.live_preview;
            }

            ui.separator();

            // Save button
            if ui.button(RichText::new(crate::t!("script.save")).color(text_color)).clicked() {
                self.save_current_script();
            }

            // Load button
            if ui.button(RichText::new(crate::t!("script.load")).color(text_color)).clicked() {
                self.load_script_dialog();
            }

            // Add to Filters button
            let filter_color = if is_dark {
                Color32::from_rgb(150, 120, 255) // soft purple
            } else {
                Color32::from_rgb(100, 60, 200)
            };
            if ui.button(RichText::new("+ Filter").color(filter_color).size(11.0))
                .on_hover_text("Add this script as a custom filter effect (Filter > Custom)")
                .clicked()
            {
                // Open the naming popup
                self.naming_effect = Some(String::from("My Effect"));
            }

            ui.separator();

            // Saved scripts dropdown
            if !self.saved_scripts.is_empty() {
                let selected_name = self.selected_script_idx
                    .and_then(|i| self.saved_scripts.get(i))
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| crate::t!("script.scripts_dropdown"));

                egui::ComboBox::from_id_source("script_dropdown")
                    .selected_text(RichText::new(selected_name).color(text_color))
                    .show_ui(ui, |ui: &mut egui::Ui| {
                        for (i, script) in self.saved_scripts.iter().enumerate() {
                            let label = if script.pinned {
                                format!("★ {}", script.name)
                            } else {
                                script.name.clone()
                            };
                            if ui.selectable_label(self.selected_script_idx == Some(i), label).clicked() {
                                self.selected_script_idx = Some(i);
                                self.code = script.code.clone();
                            }
                        }
                    });
            }
        });

        // -- "Name your effect" inline popup --
        if self.naming_effect.is_some() {
            ui.add_space(2.0);
            let frame_bg = if is_dark {
                Color32::from_rgb(35, 37, 48)
            } else {
                Color32::from_rgb(238, 236, 248)
            };
            egui::Frame::none()
                .fill(frame_bg)
                .rounding(Rounding::same(4.0))
                .stroke(Stroke::new(1.0, accent.linear_multiply(0.5)))
                .inner_margin(Margin::same(6.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Effect name:").color(text_color).size(12.0));
                        let name = self.naming_effect.as_mut().unwrap();
                        let resp = ui.add(
                            egui::TextEdit::singleline(name)
                                .desired_width(160.0)
                                .font(FontId::proportional(12.0))
                        );
                        // Auto-focus
                        if resp.gained_focus() || ui.memory(|m| m.focus().is_none()) {
                            resp.request_focus();
                        }
                        let enter_pressed = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        let add_color = if is_dark {
                            Color32::from_rgb(80, 220, 120)
                        } else {
                            Color32::from_rgb(20, 140, 60)
                        };
                        if ui.button(RichText::new("Add").color(add_color).strong()).clicked() || enter_pressed {
                            let effect_name = self.naming_effect.take().unwrap_or_default();
                            if !effect_name.trim().is_empty() {
                                self.pending_add_effect = Some((effect_name.trim().to_string(), self.code.clone()));
                            }
                        }
                        if ui.button(RichText::new("Cancel").color(muted).size(11.0)).clicked() {
                            self.naming_effect = None;
                        }
                    });
                });
        }

        ui.add_space(4.0);

        // -- Neon accent line --
        let line_rect = ui.available_rect_before_wrap();
        let line_y = ui.cursor().min.y;
        let line_start = egui::pos2(line_rect.min.x, line_y);
        let line_end = egui::pos2(line_rect.max.x, line_y);
        ui.painter().line_segment(
            [line_start, line_end],
            Stroke::new(2.0, accent.linear_multiply(0.8)),
        );
        // Glow effect — wider, more transparent line behind
        ui.painter().line_segment(
            [line_start, line_end],
            Stroke::new(6.0, Color32::from_rgba_premultiplied(
                accent.r(), accent.g(), accent.b(), 40
            )),
        );
        ui.add_space(4.0);

        // -- Calculate layout: code area and console --
        // Clamp available height to prevent infinite vertical growth feedback loop.
        // available_height() returns remaining space in the parent; cap it to a
        // reasonable maximum so set_max_height below doesn't expand the window.
        let total_h = ui.available_height().min(MAX_PANEL_HEIGHT);
        let console_h = if self.console_expanded { (total_h * 0.25).max(80.0).min(200.0) } else { 24.0 };
        let progress_h = if self.progress.is_some() || self.is_running { 24.0 } else { 0.0 };
        let code_h = (total_h - console_h - progress_h - 8.0).max(100.0);

        // -- Code editor area (recessed well, theme-aware) --
        let (code_bg, code_border, gutter_bg, line_num_color) = if is_dark {
            (
                Color32::from_rgb(24, 26, 32),
                Color32::from_rgb(16, 17, 22),
                Color32::from_rgb(30, 32, 40),
                Color32::from_rgb(80, 85, 100),
            )
        } else {
            (
                Color32::from_rgb(252, 252, 254),
                Color32::from_rgb(210, 212, 220),
                Color32::from_rgb(240, 241, 245),
                Color32::from_rgb(150, 155, 170),
            )
        };
        let code_frame = egui::Frame::none()
            .fill(code_bg)
            .rounding(Rounding::same(4.0))
            .stroke(Stroke::new(1.0, code_border))
            .inner_margin(Margin::same(0.0));

        code_frame.show(ui, |ui| {
            ui.set_max_height(code_h);

            egui::ScrollArea::vertical()
                .id_source("script_code_scroll")
                .max_height(code_h)
                .show(ui, |ui: &mut egui::Ui| {
                    ui.horizontal_top(|ui: &mut egui::Ui| {
                        // -- Line number gutter --
                        let line_count = self.code.lines().count().max(1);
                        let gutter_width = 28.0;
                        let error_line = self.error_line;

                        let (gutter_rect, _) = ui.allocate_exact_size(
                            Vec2::new(gutter_width, line_count as f32 * 17.0),
                            egui::Sense::hover(),
                        );

                        ui.painter().rect_filled(gutter_rect, 0.0, gutter_bg);

                        for i in 0..line_count {
                            let y = gutter_rect.min.y + i as f32 * 17.0;
                            let is_error = error_line == Some(i + 1);
                            if is_error {
                                // Red background behind error line number
                                let line_rect = egui::Rect::from_min_size(
                                    egui::pos2(gutter_rect.min.x, y),
                                    Vec2::new(gutter_width, 17.0),
                                );
                                let err_gutter_bg = if is_dark {
                                    Color32::from_rgb(80, 20, 20)
                                } else {
                                    Color32::from_rgb(255, 220, 220)
                                };
                                ui.painter().rect_filled(line_rect, 0.0, err_gutter_bg);
                            }
                            let num_color = if is_error {
                                Color32::from_rgb(255, 100, 100)
                            } else {
                                line_num_color
                            };
                            ui.painter().text(
                                egui::pos2(gutter_rect.max.x - 6.0, y),
                                egui::Align2::RIGHT_TOP,
                                format!("{}", i + 1),
                                FontId::monospace(12.0),
                                num_color,
                            );
                        }

                        // -- Code text editor with syntax highlighting --
                        let highlighter = &self.highlighter;
                        let dark_mode = is_dark;
                        let mut layouter = |ui: &egui::Ui, text: &str, wrap_width: f32| {
                            let layout_job = highlighter.highlight(text, wrap_width, error_line, dark_mode);
                            ui.fonts(|f| f.layout_job(layout_job))
                        };

                        let _output = egui::TextEdit::multiline(&mut self.code)
                            .font(FontId::monospace(13.0))
                            .desired_width(ui.available_width())
                            .desired_rows(10)
                            .lock_focus(true)
                            .code_editor()
                            .layouter(&mut layouter)
                            .show(ui);
                    });
                });
        });

        ui.add_space(2.0);

        // -- Console area (theme-aware) --
        let (console_bg, console_border) = if is_dark {
            (Color32::from_rgb(18, 20, 26), Color32::from_rgb(30, 32, 40))
        } else {
            (Color32::from_rgb(248, 248, 252), Color32::from_rgb(210, 212, 220))
        };

        // Console header
        ui.horizontal(|ui| {
            let arrow = if self.console_expanded { "▼" } else { "▶" };
            if ui.button(RichText::new(format!("{} {}", arrow, crate::t!("script.console"))).color(muted).size(11.0)).clicked() {
                self.console_expanded = !self.console_expanded;
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.small_button(RichText::new(crate::t!("script.console.clear")).color(muted).size(10.0)).clicked() {
                    self.console_output.clear();
                }
            });
        });

        if self.console_expanded {
            egui::Frame::none()
                .fill(console_bg)
                .rounding(Rounding { nw: 0.0, ne: 0.0, sw: 4.0, se: 4.0 })
                .stroke(Stroke::new(1.0, console_border))
                .inner_margin(Margin::same(6.0))
                .show(ui, |ui| {
                    ui.set_max_height(console_h - 24.0);

                    // Build console text as a single selectable/copyable block
                    let mut console_text = String::new();
                    for line in &self.console_output {
                        let prefix = match line.kind {
                            ConsoleLineKind::Output => ">",
                            ConsoleLineKind::Error => "✗",
                            ConsoleLineKind::Info => "ℹ",
                        };
                        if !console_text.is_empty() {
                            console_text.push('\n');
                        }
                        console_text.push_str(&format!("{} {}", prefix, line.text));
                    }
                    if console_text.is_empty() {
                        console_text = String::from("(no output)");
                    }

                    // Build a colored layout job for the console
                    let console_lines_ref = &self.console_output;
                    let dark_mode = is_dark;
                    let mut layouter = |_ui: &egui::Ui, _text: &str, wrap_width: f32| {
                        let mut job = LayoutJob::default();
                        job.wrap.max_width = wrap_width;
                        let font = FontId::monospace(11.0);

                        let muted_color = if dark_mode {
                            Color32::from_rgb(80, 85, 100)
                        } else {
                            Color32::from_rgb(150, 155, 170)
                        };

                        if console_lines_ref.is_empty() {
                            job.append("(no output)", 0.0, TextFormat {
                                font_id: font.clone(),
                                color: muted_color,
                                ..Default::default()
                            });
                        } else {
                            let mut first = true;
                            for line in console_lines_ref {
                                if !first {
                                    job.append("\n", 0.0, TextFormat {
                                        font_id: font.clone(),
                                        color: Color32::TRANSPARENT,
                                        ..Default::default()
                                    });
                                }
                                first = false;

                                let (prefix, prefix_color, text_color, bg_color) = if dark_mode {
                                    match line.kind {
                                        ConsoleLineKind::Output => (
                                            "> ",
                                            Color32::from_rgb(120, 130, 145),
                                            Color32::from_rgb(171, 178, 191),
                                            Color32::TRANSPARENT,
                                        ),
                                        ConsoleLineKind::Error => (
                                            "✗ ",
                                            Color32::from_rgb(255, 80, 80),
                                            Color32::from_rgb(255, 120, 120),
                                            Color32::from_rgba_premultiplied(100, 20, 20, 50),
                                        ),
                                        ConsoleLineKind::Info => (
                                            "ℹ ",
                                            Color32::from_rgb(100, 180, 255),
                                            Color32::from_rgb(140, 160, 180),
                                            Color32::TRANSPARENT,
                                        ),
                                    }
                                } else {
                                    match line.kind {
                                        ConsoleLineKind::Output => (
                                            "> ",
                                            Color32::from_rgb(80, 90, 110),
                                            Color32::from_rgb(50, 55, 65),
                                            Color32::TRANSPARENT,
                                        ),
                                        ConsoleLineKind::Error => (
                                            "✗ ",
                                            Color32::from_rgb(200, 40, 40),
                                            Color32::from_rgb(170, 30, 30),
                                            Color32::from_rgba_premultiplied(255, 220, 220, 60),
                                        ),
                                        ConsoleLineKind::Info => (
                                            "ℹ ",
                                            Color32::from_rgb(30, 110, 200),
                                            Color32::from_rgb(60, 70, 90),
                                            Color32::TRANSPARENT,
                                        ),
                                    }
                                };

                                // Prefix symbol
                                job.append(prefix, 0.0, TextFormat {
                                    font_id: font.clone(),
                                    color: prefix_color,
                                    background: bg_color,
                                    ..Default::default()
                                });
                                // Message text
                                job.append(&line.text, 0.0, TextFormat {
                                    font_id: font.clone(),
                                    color: text_color,
                                    background: bg_color,
                                    ..Default::default()
                                });
                            }
                        }
                        _ui.fonts(|f| f.layout_job(job))
                    };

                    egui::ScrollArea::vertical()
                        .id_source("script_console_scroll")
                        .stick_to_bottom(true)
                        .max_height(console_h - 24.0)
                        .show(ui, |ui: &mut egui::Ui| {
                            // Use a read-only TextEdit with colored layouter
                            egui::TextEdit::multiline(&mut console_text.as_str())
                                .font(FontId::monospace(11.0))
                                .desired_width(ui.available_width())
                                .frame(false)
                                .layouter(&mut layouter)
                                .show(ui);
                        });
                });
        }

        // -- Progress bar --
        if self.is_running {
            ui.add_space(2.0);
            let avail_w = ui.available_width();
            if let Some(p) = self.progress {
                let bar = egui::ProgressBar::new(p)
                    .text(format!("{:.0}%", p * 100.0))
                    .desired_width(avail_w);
                ui.add(bar);
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new(crate::t!("script.progress.running")).color(muted).size(11.0));
                });
            }
        }
    }

    pub fn add_console_line(&mut self, text: String, kind: ConsoleLineKind) {
        self.console_output.push(ConsoleLine { text, kind });
    }

    pub fn clear_console(&mut self) {
        self.console_output.clear();
    }

    // Saves to in-memory list; persistence is handled by the app layer.
    fn save_current_script(&mut self) {
        // If a script is selected, update it
        if let Some(idx) = self.selected_script_idx {
            if let Some(script) = self.saved_scripts.get_mut(idx) {
                script.code = self.code.clone();
                return;
            }
        }
        // Otherwise add new
        let name = format!("Script {}", self.saved_scripts.len() + 1);
        self.saved_scripts.push(SavedScript {
            name,
            code: self.code.clone(),
            pinned: false,
        });
        self.selected_script_idx = Some(self.saved_scripts.len() - 1);
    }

    fn load_script_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Rhai Script", &["rhai"])
            .add_filter("All Files", &["*"])
            .pick_file()
        {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                self.code = contents;
                self.selected_script_idx = None;
            }
        }
    }

    /// Pinned scripts, for Filter > Scripts menu.
    pub fn pinned_scripts(&self) -> Vec<(usize, String)> {
        self.saved_scripts.iter().enumerate()
            .filter(|(_, s)| s.pinned)
            .map(|(i, s)| (i, s.name.clone()))
            .collect()
    }

    pub fn load_saved_scripts(&mut self) {
        let dir = scripts_directory();
        if !dir.exists() { return; }

        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "rhai").unwrap_or(false) {
                    if let Ok(code) = std::fs::read_to_string(&path) {
                        let name = path.file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        // Avoid duplicates
                        if !self.saved_scripts.iter().any(|s| s.name == name) {
                            self.saved_scripts.push(SavedScript {
                                name,
                                code,
                                pinned: false,
                            });
                        }
                    }
                }
            }
        }
    }

    pub fn save_scripts_to_disk(&self) {
        let dir = scripts_directory();
        let _ = std::fs::create_dir_all(&dir);

        for script in &self.saved_scripts {
            let filename = format!("{}.rhai", script.name.replace(|c: char| !c.is_alphanumeric() && c != '_' && c != '-', "_"));
            let path = dir.join(filename);
            let _ = std::fs::write(&path, &script.code);
        }
    }
}

fn scripts_directory() -> std::path::PathBuf {
    app_data_directory().join("scripts")
}

fn app_data_directory() -> std::path::PathBuf {
    // Windows: %APPDATA%/PaintFE
    if let Some(appdata) = std::env::var_os("APPDATA") {
        return std::path::PathBuf::from(appdata).join("PaintFE");
    }
    // Linux/macOS: ~/.local/share/PaintFE or ~/.config/PaintFE
    if let Some(home) = std::env::var_os("HOME") {
        let local_share = std::path::PathBuf::from(&home).join(".local/share/PaintFE");
        return local_share;
    }
    // Absolute fallback
    std::path::PathBuf::from("PaintFE_data")
}

fn sanitize_filename(name: &str) -> String {
    name.replace(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != ' ', "_")
}

// ============================================================================
// Syntax Highlighter
// ============================================================================

pub struct SyntaxHighlighter {
    keywords: HashSet<&'static str>,
    builtins: HashSet<&'static str>,
}

impl SyntaxHighlighter {
    pub fn new() -> Self {
        let keywords: HashSet<&str> = [
            "let", "const", "if", "else", "for", "while", "loop", "in",
            "fn", "return", "true", "false", "break", "continue",
            "switch", "import", "export", "as", "is", "type_of",
            "throw", "try", "catch",
        ].into_iter().collect();

        let builtins: HashSet<&str> = [
            // Canvas
            "width", "height", "is_selected",
            // Pixel access
            "get_pixel", "set_pixel", "get_r", "get_g", "get_b", "get_a",
            "set_r", "set_g", "set_b", "set_a",
            // Bulk iteration
            "for_each_pixel", "for_region", "map_channels",
            // Effects
            "apply_blur", "apply_box_blur", "apply_motion_blur",
            "apply_sharpen", "apply_reduce_noise", "apply_median",
            "apply_invert", "apply_desaturate", "apply_sepia",
            "apply_brightness_contrast", "apply_hsl", "apply_exposure", "apply_levels",
            "apply_noise", "apply_pixelate", "apply_crystallize",
            "apply_bulge", "apply_twist",
            "apply_glow", "apply_vignette", "apply_halftone",
            "apply_ink", "apply_oil_painting",
            // Utility
            "print", "print_line", "sleep", "progress",
            "rand_int", "rand_float",
            "clamp", "clamp_f", "lerp", "distance",
            "abs_i", "min_i", "max_i", "min_f", "max_f",
            "abs", "min", "max",
            "floor", "ceil", "round", "sqrt", "pow",
            "sin", "cos", "tan", "atan2", "PI",
            "rgb_to_hsl", "hsl_to_rgb",
        ].into_iter().collect();

        Self { keywords, builtins }
    }

    /// Produce a LayoutJob with syntax-highlighted spans for the given code.
    /// If `error_line` is Some(n), line n (1-based) gets a red background tint.
    pub fn highlight(&self, code: &str, wrap_width: f32, error_line: Option<usize>, is_dark: bool) -> LayoutJob {
        let mut job = LayoutJob::default();
        job.wrap.max_width = wrap_width;

        // Pre-compute which byte offsets belong to the error line (if any)
        let error_byte_range = error_line.and_then(|target| {
            let mut current_line = 1usize;
            let mut line_start = 0usize;
            for (i, ch) in code.char_indices() {
                if ch == '\n' {
                    if current_line == target {
                        return Some((line_start, i));
                    }
                    current_line += 1;
                    line_start = i + 1;
                }
            }
            if current_line == target {
                Some((line_start, code.len()))
            } else {
                None
            }
        });

        let bytes = code.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            let ch = bytes[i] as char;

            // Single-line comment
            if ch == '/' && i + 1 < len && bytes[i + 1] == b'/' {
                let start = i;
                while i < len && bytes[i] != b'\n' {
                    i += 1;
                }
                let on_error = is_in_error_range(error_byte_range, start, i);
                self.push_span(&mut job, &code[start..i], TokenKind::Comment, on_error, is_dark);
                continue;
            }

            // Block comment
            if ch == '/' && i + 1 < len && bytes[i + 1] == b'*' {
                let start = i;
                i += 2;
                while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                if i + 1 < len { i += 2; } else { i = len; }
                let on_error = is_in_error_range(error_byte_range, start, i);
                self.push_span(&mut job, &code[start..i], TokenKind::Comment, on_error, is_dark);
                continue;
            }

            // String literal (double quote)
            if ch == '"' {
                let start = i;
                i += 1;
                while i < len && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < len { i += 1; }
                    i += 1;
                }
                if i < len { i += 1; }
                let on_error = is_in_error_range(error_byte_range, start, i);
                self.push_span(&mut job, &code[start..i], TokenKind::StringLit, on_error, is_dark);
                continue;
            }

            // String literal (single quote)
            if ch == '\'' {
                let start = i;
                i += 1;
                while i < len && bytes[i] != b'\'' {
                    if bytes[i] == b'\\' && i + 1 < len { i += 1; }
                    i += 1;
                }
                if i < len { i += 1; }
                let on_error = is_in_error_range(error_byte_range, start, i);
                self.push_span(&mut job, &code[start..i], TokenKind::StringLit, on_error, is_dark);
                continue;
            }

            // Number
            if ch.is_ascii_digit() || (ch == '.' && i + 1 < len && (bytes[i + 1] as char).is_ascii_digit()) {
                let start = i;
                // Hex prefix
                if ch == '0' && i + 1 < len && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
                    i += 2;
                    while i < len && (bytes[i] as char).is_ascii_hexdigit() { i += 1; }
                } else {
                    while i < len && ((bytes[i] as char).is_ascii_digit() || bytes[i] == b'.') { i += 1; }
                }
                let on_error = is_in_error_range(error_byte_range, start, i);
                self.push_span(&mut job, &code[start..i], TokenKind::Number, on_error, is_dark);
                continue;
            }

            // Identifier or keyword
            if ch.is_ascii_alphabetic() || ch == '_' {
                let start = i;
                while i < len && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &code[start..i];
                let kind = if self.keywords.contains(word) {
                    TokenKind::Keyword
                } else if self.builtins.contains(word) {
                    TokenKind::Builtin
                } else {
                    TokenKind::Identifier
                };
                let on_error = is_in_error_range(error_byte_range, start, i);
                self.push_span(&mut job, word, kind, on_error, is_dark);
                continue;
            }

            // Operators
            if "+-*/%=!<>&|^~?:".contains(ch) {
                let start = i;
                i += 1;
                // Consume multi-char operators
                while i < len && "+-*/%=!<>&|^~?:".contains(bytes[i] as char) && (i - start) < 3 {
                    i += 1;
                }
                let on_error = is_in_error_range(error_byte_range, start, i);
                self.push_span(&mut job, &code[start..i], TokenKind::Operator, on_error, is_dark);
                continue;
            }

            // Punctuation
            if "(){}[],;.".contains(ch) {
                let on_error = is_in_error_range(error_byte_range, i, i + 1);
                self.push_span(&mut job, &code[i..i+1], TokenKind::Punctuation, on_error, is_dark);
                i += 1;
                continue;
            }

            // Whitespace and other
            let on_error = is_in_error_range(error_byte_range, i, i + 1);
            self.push_span(&mut job, &code[i..i+1], TokenKind::Default, on_error, is_dark);
            i += 1;
        }

        job
    }

    fn push_span(&self, job: &mut LayoutJob, text: &str, kind: TokenKind, is_error_line: bool, is_dark: bool) {
        let color = if is_dark {
            match kind {
                TokenKind::Keyword    => Color32::from_rgb(198, 120, 221), // purple
                TokenKind::Builtin    => Color32::from_rgb(97, 175, 239),  // blue
                TokenKind::Number     => Color32::from_rgb(209, 154, 102), // orange
                TokenKind::StringLit  => Color32::from_rgb(152, 195, 121), // green
                TokenKind::Comment    => Color32::from_rgb(92, 99, 112),   // grey
                TokenKind::Operator   => Color32::from_rgb(86, 182, 194),  // cyan
                TokenKind::Punctuation => Color32::from_rgb(171, 178, 191), // light grey
                TokenKind::Identifier => Color32::from_rgb(229, 192, 123), // yellow
                TokenKind::Default    => Color32::from_rgb(171, 178, 191), // light grey
            }
        } else {
            // Light theme — One Light inspired
            match kind {
                TokenKind::Keyword    => Color32::from_rgb(166, 38, 164),  // magenta
                TokenKind::Builtin    => Color32::from_rgb(64, 120, 242),  // blue
                TokenKind::Number     => Color32::from_rgb(152, 104, 1),   // amber
                TokenKind::StringLit  => Color32::from_rgb(80, 141, 38),   // green
                TokenKind::Comment    => Color32::from_rgb(140, 148, 160), // grey
                TokenKind::Operator   => Color32::from_rgb(1, 132, 188),   // teal
                TokenKind::Punctuation => Color32::from_rgb(56, 58, 66),   // dark grey
                TokenKind::Identifier => Color32::from_rgb(180, 110, 10),  // golden
                TokenKind::Default    => Color32::from_rgb(56, 58, 66),    // dark grey
            }
        };

        let background = if is_error_line {
            if is_dark {
                Color32::from_rgba_premultiplied(120, 30, 30, 80) // translucent red overlay
            } else {
                Color32::from_rgba_premultiplied(255, 200, 200, 100)
            }
        } else {
            Color32::TRANSPARENT
        };

        job.append(text, 0.0, TextFormat {
            font_id: FontId::monospace(13.0),
            color,
            background,
            ..Default::default()
        });
    }
}

/// Check if a byte range [span_start..span_end) overlaps with the error line range.
fn is_in_error_range(error_range: Option<(usize, usize)>, span_start: usize, span_end: usize) -> bool {
    match error_range {
        Some((err_start, err_end)) => span_start < err_end && span_end > err_start,
        None => false,
    }
}

#[derive(Debug, Clone, Copy)]
enum TokenKind {
    Keyword,
    Builtin,
    Number,
    StringLit,
    Comment,
    Operator,
    Punctuation,
    Identifier,
    Default,
}
