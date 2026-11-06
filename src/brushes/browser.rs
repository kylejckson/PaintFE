//! Persistent brush browser preferences and per-preset parameter reuse.
use super::BrushEngineSettings;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BrowserSettings {
    pub compact: bool,
    pub spacing: f32,
    pub preview_size: f32,
    pub rounding: Option<f32>,
    #[serde(with = "color")]
    pub background: Option<egui::Color32>,
    #[serde(with = "color")]
    pub selection: Option<egui::Color32>,
    #[serde(with = "color")]
    pub hover: Option<egui::Color32>,
    pub favourites: Vec<String>,
    pub recent: Vec<String>,
    pub category: String,
}
impl Default for BrowserSettings {
    fn default() -> Self {
        Self {
            compact: true,
            spacing: 4.,
            preview_size: 24.,
            rounding: None,
            background: None,
            selection: None,
            hover: None,
            favourites: Vec::new(),
            recent: Vec::new(),
            category: "All".into(),
        }
    }
}
impl BrowserSettings {
    pub fn copy_appearance(&mut self, other: &Self) {
        self.compact = other.compact;
        self.spacing = if other.spacing.is_finite() {
            other.spacing.clamp(0., 12.)
        } else {
            4.
        };
        self.preview_size = if other.preview_size.is_finite() {
            other.preview_size.clamp(18., 48.)
        } else {
            24.
        };
        self.rounding = other
            .rounding
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0., 24.));
        self.background = other.background;
        self.selection = other.selection;
        self.hover = other.hover;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SavedPreset {
    pub engine: BrushEngineSettings,
    pub flow: f32,
    pub opacity: f32,
    pub spacing: f32,
    pub hardness: f32,
    pub rotation: f32,
    pub random_rotation: bool,
    pub mode: u8,
    pub pressure_size: bool,
    pub pressure_flow: bool,
    pub minimum_size: f32,
    pub minimum_flow: f32,
    pub anti_aliased: bool,
    pub scatter: f32,
    pub hue_jitter: f32,
    pub brightness_jitter: f32,
    pub rotation_range: (f32, f32),
}

impl Default for SavedPreset {
    fn default() -> Self {
        Self {
            engine: Default::default(),
            flow: 1.,
            opacity: 1.,
            spacing: 0.025,
            hardness: 1.,
            rotation: 0.,
            random_rotation: false,
            mode: 0,
            pressure_size: false,
            pressure_flow: false,
            minimum_size: 0.1,
            minimum_flow: 0.1,
            anti_aliased: true,
            scatter: 0.,
            hue_jitter: 0.,
            brightness_jitter: 0.,
            rotation_range: (0., 360.),
        }
    }
}
impl SavedPreset {
    pub fn for_tip(tip: super::PaintTip) -> Self {
        Self {
            engine: tip.settings(),
            flow: tip.flow(),
            spacing: if tip.has_bristles() { 0.025 } else { 0.05 },
            rotation: if tip.has_bristles() { 0. } else { 30. },
            ..Default::default()
        }
    }
}

mod color {
    use serde::{Deserialize, Serialize};
    pub fn serialize<S: serde::Serializer>(
        value: &Option<egui::Color32>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.map(|color| color.to_array()).serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<egui::Color32>, D::Error> {
        Ok(Option::<[u8; 4]>::deserialize(deserializer)?
            .map(|[r, g, b, a]| egui::Color32::from_rgba_premultiplied(r, g, b, a)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn browser_colours_roundtrip_without_double_premultiplication() {
        let browser = BrowserSettings {
            background: Some(egui::Color32::from_rgba_unmultiplied(100, 160, 200, 128)),
            selection: Some(egui::Color32::from_rgba_premultiplied(30, 40, 50, 0)),
            favourites: vec!["Round Paint".into()],
            ..Default::default()
        };
        let restored: BrowserSettings =
            serde_json::from_str(&serde_json::to_string(&browser).unwrap()).unwrap();
        assert_eq!(restored, browser);
    }
    #[test]
    fn appearance_import_keeps_personal_navigation_and_bounds_invalid_sizes() {
        let mut browser = BrowserSettings {
            favourites: vec!["Rake".into()],
            ..Default::default()
        };
        browser.copy_appearance(&BrowserSettings {
            preview_size: f32::NAN,
            spacing: 999.,
            ..Default::default()
        });
        assert_eq!(browser.preview_size, 24.);
        assert_eq!(browser.spacing, 12.);
        assert_eq!(browser.favourites, ["Rake"]);
    }
}
