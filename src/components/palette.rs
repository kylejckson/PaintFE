use super::color_widgets as cw;
use crate::assets::Assets;
use eframe::egui::{self, Color32};

const MAX_RECENT: usize = 30;
const MAX_COLORS: usize = 4096;

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredPalette {
    version: u8,
    swatches: Vec<[u8; 4]>,
    favorites: Vec<[u8; 4]>,
    recent: Vec<[u8; 4]>,
    tab: usize,
}

pub struct PalettePanel {
    recent: Vec<Color32>,
    palette: Vec<Color32>,
    favorites: Vec<Color32>,
    tab: usize,
}

impl Default for PalettePanel {
    fn default() -> Self {
        Self {
            recent: Vec::new(),
            palette: default_palette(),
            favorites: Vec::new(),
            tab: 0,
        }
    }
}

impl PalettePanel {
    // This existing settings slot now contains versioned state for all collections.
    pub fn serialize_recent_colors(&self) -> String {
        serde_json::to_string(&StoredPalette {
            version: 2,
            swatches: self.palette.iter().copied().map(cw::straight).collect(),
            favorites: self.favorites.iter().copied().map(cw::straight).collect(),
            recent: self.recent.iter().copied().map(cw::straight).collect(),
            tab: self.tab,
        })
        .unwrap_or_default()
    }
    pub fn load_recent_colors_from_serialized(&mut self, text: &str) {
        if let Ok(state) = serde_json::from_str::<StoredPalette>(text) {
            if state.version != 2 {
                return;
            }
            let convert = |colors: Vec<[u8; 4]>| {
                colors
                    .into_iter()
                    .take(MAX_COLORS)
                    .map(|[r, g, b, a]| cw::rgba(r, g, b, a))
                    .collect::<Vec<_>>()
            };
            self.palette = convert(state.swatches);
            self.favorites = convert(state.favorites);
            self.recent = convert(state.recent);
            self.recent.truncate(MAX_RECENT);
            self.tab = state.tab.min(2);
        } else if !text.trim().is_empty() {
            // Legacy channels were written premultiplied; do not multiply a second time.
            self.recent = text
                .split(',')
                .filter_map(parse_legacy)
                .take(MAX_RECENT)
                .collect();
        }
    }
    pub fn observe_color(&mut self, color: Color32) {
        if self.recent.first() == Some(&color) {
            return;
        }
        self.recent.retain(|c| *c != color);
        self.recent.insert(0, color);
        self.recent.truncate(MAX_RECENT);
    }
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        _assets: &Assets,
        primary: Color32,
        secondary: Color32,
        editing_secondary: bool,
    ) -> Option<(Color32, bool)> {
        let active = if editing_secondary {
            secondary
        } else {
            primary
        };
        ui.spacing_mut().item_spacing = egui::vec2(5.0, 5.0);
        ui.horizontal(|ui| {
            ui.allocate_ui(egui::vec2(ui.available_width() - 29.0, 24.0), |ui| {
                cw::tabs(
                    ui,
                    "palette_tabs",
                    &["Swatches", "Recently Used", "Favorites"],
                    &mut self.tab,
                );
            });
            let response = ui.add_enabled(
                self.tab != 1,
                egui::Button::new("+").min_size(egui::vec2(24.0, 24.0)),
            );
            if response
                .on_hover_text("Add selected color to this collection")
                .clicked()
            {
                let list = if self.tab == 2 {
                    &mut self.favorites
                } else {
                    &mut self.palette
                };
                if !list.contains(&active) && list.len() < MAX_COLORS {
                    list.push(active);
                }
            }
        });
        ui.add_space(2.0);
        let mut action = None;
        let mut favorite = None;
        let mut remove = None;
        let list = match self.tab {
            1 => &mut self.recent,
            2 => &mut self.favorites,
            _ => &mut self.palette,
        };
        egui::ScrollArea::vertical()
            .id_salt("palette_collection")
            .auto_shrink([false, false])
            .max_height(ui.available_height().max(23.0))
            .show_viewport(ui, |ui, viewport| {
                let column_count = columns(ui.available_width());
                let row_count = list.len().div_ceil(column_count);
                let row_height = 23.0 + ui.spacing().item_spacing.y;
                ui.set_min_height(
                    (row_count as f32 * row_height - ui.spacing().item_spacing.y).max(0.0),
                );
                let first = (viewport.min.y / row_height).floor().max(0.0) as usize;
                let last = ((viewport.max.y / row_height).ceil() as usize).min(row_count);
                if list.is_empty() {
                    ui.add_space(20.0);
                    ui.label(
                        egui::RichText::new(if self.tab == 1 {
                            "Colors appear here after painting."
                        } else {
                            "Add a color with + to start."
                        })
                        .size(11.0)
                        .color(ui.visuals().weak_text_color()),
                    );
                }
                ui.add_space(first.min(row_count) as f32 * row_height);
                for row in first.min(row_count)..last {
                    let start = row * column_count;
                    let end = (start + column_count).min(list.len());
                    let chunk = &mut list[start..end];
                    ui.horizontal(|ui| {
                        for (column, color) in chunk.iter_mut().enumerate() {
                            let index = row * column_count + column;
                            let response = cw::swatch(
                                ui,
                                ("collection", self.tab, index),
                                *color,
                                23.0,
                                *color == active,
                            );
                            if response.clicked() {
                                action = Some((*color, editing_secondary));
                            }
                            if response.secondary_clicked() {
                                action = Some((*color, true));
                            }
                            response.context_menu(|ui| {
                                if self.tab != 1 {
                                    if ui.button("Replace with Primary").clicked() {
                                        *color = primary;
                                        ui.close();
                                    }
                                    if ui.button("Replace with Secondary").clicked() {
                                        *color = secondary;
                                        ui.close();
                                    }
                                }
                                if self.tab != 2 && ui.button("Add to Favorites").clicked() {
                                    favorite = Some(*color);
                                    ui.close();
                                }
                                if ui.button("Remove color").clicked() {
                                    remove = Some(index);
                                    ui.close();
                                }
                            });
                        }
                    });
                }
            });
        if let Some(index) = remove {
            list.remove(index);
        }
        if let Some(color) = favorite
            && !self.favorites.contains(&color)
            && self.favorites.len() < MAX_COLORS
        {
            self.favorites.push(color);
        }
        action
    }
    pub fn reset_palette_default(&mut self) {
        self.palette = default_palette();
    }
    pub fn reset_recent_default(&mut self) {
        self.recent.clear();
    }
    pub fn save_palette_dialog(&self) {
        let mut text = String::from("# PaintFE palette v2: straight RGBA\n");
        for color in &self.palette {
            let [r, g, b, a] = cw::straight(*color);
            text.push_str(&format!("{r:02X}{g:02X}{b:02X}{a:02X}\n"));
        }
        #[cfg(target_arch = "wasm32")]
        crate::web_fs::trigger_download("paintfe.pfepalette", text.as_bytes());
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Palette", &["pfepalette"])
            .set_file_name("paintfe.pfepalette")
            .save_file()
        {
            let _ = std::fs::write(path, text);
        }
    }
    pub fn load_palette_dialog(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("Palette", &["pfepalette"])
                .pick_file()
            else {
                return;
            };
            let Ok(text) = std::fs::read_to_string(path) else {
                return;
            };
            let colors = parse_palette(&text);
            if !colors.is_empty() {
                self.palette = colors;
                self.tab = 0;
            }
        }
    }
}

fn parse_legacy(text: &str) -> Option<Color32> {
    let text = text.trim();
    if text.len() != 8 || !text.is_ascii() {
        return None;
    }
    let [r, g, b] = cw::parse_hex(&text[..6])?;
    let a = u8::from_str_radix(&text[6..], 16).ok()?;
    Some(Color32::from_rgba_premultiplied(
        r.min(a),
        g.min(a),
        b.min(a),
        a,
    ))
}
fn parse_palette(text: &str) -> Vec<Color32> {
    let modern = text
        .lines()
        .next()
        .is_some_and(|line| line.starts_with("# PaintFE palette v2"));
    text.lines()
        .filter_map(|line| {
            let c = parse_legacy(line)?;
            if modern {
                let t = line.trim();
                let [r, g, b] = cw::parse_hex(&t[..6])?;
                Some(cw::rgba(r, g, b, c.a()))
            } else {
                Some(c)
            }
        })
        .take(MAX_COLORS)
        .collect()
}
fn columns(width: f32) -> usize {
    ((width + 5.0) / 28.0).floor().max(1.0) as usize
}

fn default_palette() -> Vec<Color32> {
    const RGB: [[u8; 3]; 20] = [
        [30, 32, 31],
        [90, 93, 94],
        [161, 164, 166],
        [255, 255, 255],
        [208, 43, 37],
        [255, 119, 13],
        [255, 211, 0],
        [99, 188, 30],
        [0, 145, 72],
        [0, 158, 172],
        [0, 172, 240],
        [35, 99, 235],
        [73, 37, 216],
        [136, 25, 237],
        [212, 40, 160],
        [242, 43, 139],
        [243, 69, 77],
        [255, 143, 154],
        [255, 203, 166],
        [255, 226, 198],
    ];
    RGB.into_iter()
        .map(|[r, g, b]| cw::rgba(r, g, b, 255))
        .collect()
}

#[cfg(test)]
mod redesign_tests {
    use super::*;
    #[test]
    fn responsive_columns_keep_all_saved_colors() {
        assert_eq!(default_palette().len(), 20);
        assert_eq!(columns(280.0), 10);
        assert_eq!(columns(560.0), 20);
        assert_eq!(columns(0.0), 1);
        let panel = PalettePanel {
            palette: parse_palette(&"FF0000FF\n".repeat(80)),
            ..Default::default()
        };
        let mut restored = PalettePanel::default();
        restored.load_recent_colors_from_serialized(&panel.serialize_recent_colors());
        assert_eq!(restored.palette.len(), 80);
    }
    #[test]
    fn collections_roundtrip_empty_and_legacy_alpha() {
        let mut panel = PalettePanel::default();
        panel.palette.clear();
        panel.favorites.push(cw::rgba(240, 80, 20, 128));
        panel.tab = 2;
        let text = panel.serialize_recent_colors();
        let mut restored = PalettePanel::default();
        restored.load_recent_colors_from_serialized(&text);
        assert!(restored.palette.is_empty());
        assert_eq!(restored.favorites, panel.favorites);
        assert_eq!(restored.tab, 2);
        restored.load_recent_colors_from_serialized("80402080,nothex");
        assert_eq!(
            restored.recent[0],
            Color32::from_rgba_premultiplied(128, 64, 32, 128)
        );
    }
    #[test]
    fn import_varied_sizes_and_recency_deduplication() {
        assert_eq!(
            parse_palette("# PaintFE palette v2: straight RGBA\nFF804080\n")[0],
            cw::rgba(255, 128, 64, 128)
        );
        assert_eq!(parse_palette(&"FF0000FF\n".repeat(40)).len(), 40);
        let mut panel = PalettePanel::default();
        for i in 0..40 {
            panel.observe_color(cw::rgba(i, 0, 0, 255));
        }
        panel.observe_color(cw::rgba(10, 0, 0, 255));
        assert_eq!(panel.recent.len(), MAX_RECENT);
        assert_eq!(
            panel
                .recent
                .iter()
                .filter(|c| **c == cw::rgba(10, 0, 0, 255))
                .count(),
            1
        );
    }
}
