use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub type Values = BTreeMap<String, serde_json::Value>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameter {
    #[serde(skip, default = "uuid::Uuid::new_v4")]
    pub(crate) editor_id: uuid::Uuid,
    pub id: String,
    pub label: String,
    pub kind: String,
    #[serde(default)]
    pub group: String,
    pub default: serde_json::Value,
    #[serde(default)]
    pub help: String,
    #[serde(default)]
    pub min: f64,
    #[serde(default = "one")]
    pub max: f64,
    #[serde(default = "one")]
    pub step: f64,
    #[serde(default)]
    pub choices: Vec<String>,
}
fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Effect {
    pub id: String,
    pub name: String,
    pub category: String,
    pub script: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    #[serde(default)]
    pub histogram: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackKind {
    All,
    #[default]
    Effects,
    Theme,
    Icons,
}
impl PackKind {
    pub fn label(self) -> String {
        crate::i18n::translate(match self {
            Self::All => "packs.all",
            Self::Effects => "packs.effects",
            Self::Theme => "packs.themes",
            Self::Icons => "packs.icons",
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    #[serde(default, skip_serializing_if = "is_effects")]
    pub kind: PackKind,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub appearance: String,
    pub schema: u32,
    pub api: u32,
    pub id: String,
    pub version: String,
    pub name: String,
    pub author: String,
    pub description: String,
    pub license: String,
    pub tags: Vec<String>,
    pub effects: Vec<Effect>,
    pub files: Vec<PackFile>,
    #[serde(default)]
    pub icon: String,
    #[serde(default)]
    pub menu_light: String,
    #[serde(default)]
    pub menu_dark: String,
    #[serde(default)]
    pub screenshot: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Listing {
    pub manifest: Manifest,
    pub url: String,
    pub sha256: String,
    pub size: u64,
    pub published: u64,
    pub updated: u64,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub recent_downloads: u64,
    #[serde(default)]
    pub window_days: u32,
    #[serde(default)]
    pub screenshot_url: String,
    #[serde(default)]
    pub icon_url: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Catalogue {
    pub schema: u32,
    pub sequence: u64,
    pub updated: u64,
    pub packs: Vec<Listing>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub payload: String,
    pub signature: String,
}

pub fn identifier(s: &str) -> bool {
    !s.is_empty()
        && !s.ends_with('.')
        && s.len() <= 80
        && s.split('.').all(portable_component)
        && s.bytes().enumerate().all(|(i, c)| {
            c.is_ascii_lowercase()
                || (i > 0 && (c.is_ascii_digit() || c == b'_' || c == b'-' || c == b'.'))
        })
}
pub fn resource_path(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 160
        && !s.contains('\\')
        && !s.contains(':')
        && s.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && portable_component(part)
                && part
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        })
}
fn portable_component(s: &str) -> bool {
    let stem = s.split('.').next().unwrap_or_default().to_ascii_uppercase();
    !s.is_empty()
        && !s.ends_with('.')
        && !matches!(
            stem.as_str(),
            "CON"
                | "PRN"
                | "AUX"
                | "NUL"
                | "COM1"
                | "COM2"
                | "COM3"
                | "COM4"
                | "COM5"
                | "COM6"
                | "COM7"
                | "COM8"
                | "COM9"
                | "LPT1"
                | "LPT2"
                | "LPT3"
                | "LPT4"
                | "LPT5"
                | "LPT6"
                | "LPT7"
                | "LPT8"
                | "LPT9"
        )
}
fn text(s: &str, max: usize) -> bool {
    !s.trim().is_empty() && s.chars().count() <= max && !s.chars().any(|c| c.is_control())
}
pub fn digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())
}
pub fn version(s: &str) -> bool {
    let p: Vec<_> = s.split('.').collect();
    p.len() == 3
        && p.iter()
            .all(|p| !p.is_empty() && p.len() <= 8 && p.bytes().all(|c| c.is_ascii_digit()))
}
pub fn version_key(s: &str) -> Vec<u32> {
    s.split('.').filter_map(|p| p.parse().ok()).collect()
}
pub fn allowed_url(url: &str) -> bool {
    [
        "https://github.com/kylejckson/PaintFE_plugins/releases/download/",
        "https://kylejckson.github.io/PaintFE_plugins/",
    ]
    .iter()
    .any(|root| url.starts_with(root))
        && url
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b":/-_.".contains(&c))
        && !url.contains("/../")
}
impl Parameter {
    pub fn validate_value(&self, value: &serde_json::Value) -> bool {
        match self.kind.as_str() {
            "number" | "integer" => value.as_f64().is_some_and(|n| {
                n.is_finite()
                    && n >= self.min
                    && n <= self.max
                    && (self.kind != "integer" || n.fract() == 0.)
            }),
            "seed" => value.as_u64().is_some_and(|seed| seed <= u32::MAX as u64),
            "point" => value.as_array().is_some_and(|p| {
                p.len() == 2 && p.iter().all(|v| v.as_f64().is_some_and(|n| (0.0..=1.0).contains(&n)))
            }),
            "boolean" => value.is_boolean(),
            "text" => value.as_str().is_some_and(|s| s.chars().count() <= 2048),
            "choice" => value
                .as_str()
                .is_some_and(|s| self.choices.iter().any(|c| c == s)),
            "colour" => value.as_array().is_some_and(|a| {
                a.len() == 4 && a.iter().all(|v| v.as_u64().is_some_and(|n| n <= 255))
            }),
            _ => false,
        }
    }
}
impl Effect {
    pub fn defaults(&self) -> Values {
        self.parameters
            .iter()
            .map(|p| (p.id.clone(), p.default.clone()))
            .collect()
    }
    pub fn validate_values(&self, values: &Values) -> Result<(), String> {
        if values.len() != self.parameters.len()
            || !self
                .parameters
                .iter()
                .all(|p| values.get(&p.id).is_some_and(|v| p.validate_value(v)))
        {
            return Err("Invalid plugin parameters".into());
        }
        Ok(())
    }
}
fn is_effects(kind: &PackKind) -> bool {
    *kind == PackKind::Effects
}
impl Manifest {
    pub fn supports_script_api(&self) -> bool { matches!(self.api, 1 | 2) }
    pub fn validate(&self) -> Result<(), String> {
        self.validate_listing()?;
        if !self.supports_script_api() {
            return Err("This plugin requires another Script API version.".into());
        }
        Ok(())
    }
    pub fn validate_listing(&self) -> Result<(), String> {
        let valid = self.schema == 1
            && self.api > 0
            && identifier(&self.id)
            && version(&self.version)
            && text(&self.name, 80)
            && text(&self.author, 80)
            && !self.description.trim().is_empty()
            && self.description.chars().count() <= 4096
            && !self
                .description
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t')
            && text(&self.license, 120)
            && self.tags.len() <= 12
            && self.tags.iter().all(|s| text(s, 40))
            && ((self.kind == PackKind::Effects
                && !self.effects.is_empty()
                && self.appearance.is_empty())
                || (matches!(self.kind, PackKind::Theme | PackKind::Icons)
                    && self.effects.is_empty()))
            && self.effects.len() <= 16
            && self.files.len() <= 64;
        if !valid {
            return Err("Unsupported or invalid plugin manifest".into());
        }
        let mut paths = BTreeSet::new();
        let mut total = 0u64;
        for f in &self.files {
            if !resource_path(&f.path)
                || f.path == "manifest.json"
                || !digest(&f.sha256)
                || f.size == 0
                || f.size > 4 * 1024 * 1024
                || !paths.insert(&f.path)
            {
                return Err("Invalid plugin file list".into());
            }
            total = total.checked_add(f.size).ok_or("Plugin is too large")?;
            if total > super::MAX_PACKAGE as u64 {
                return Err("Plugin is too large".into());
            }
        }
        for art in [
            &self.icon,
            &self.menu_light,
            &self.menu_dark,
            &self.screenshot,
        ] {
            if !art.is_empty() && (!art.ends_with(".png") || !paths.contains(art)) {
                return Err("Missing plugin artwork".into());
            }
        }
        if self.kind == PackKind::Theme
            && self
                .files
                .iter()
                .any(|f| f.path == self.appearance && f.size > 512 * 1024)
        {
            return Err("Theme file exceeds its 512 KiB limit".into());
        }
        if self.kind != PackKind::Effects {
            let suffix = if self.kind == PackKind::Theme {
                ".txt"
            } else {
                ".zip"
            };
            if !self.appearance.ends_with(suffix)
                || !paths.contains(&self.appearance)
                || self.files.iter().any(|f| f.path.ends_with(".rhai"))
            {
                return Err("Invalid appearance pack resources".into());
            }
        }
        let mut ids = BTreeSet::new();
        for e in &self.effects {
            if !identifier(&e.id)
                || !ids.insert(&e.id)
                || !text(&e.name, 80)
                || !text(&e.category, 40)
                || !e.script.ends_with(".rhai")
                || !paths.contains(&e.script)
                || e.parameters.len() > 24
            {
                return Err("Invalid plugin effect".into());
            }
            if self.api <= 2 && e.parameters.iter().filter(|p| p.kind == "seed").count() > 1 {
                return Err("An effect can have only one random seed parameter".into());
            }
            let mut params = BTreeSet::new();
            for p in &e.parameters {
                if !identifier(&p.id)
                    || p.id.contains(['.', '-'])
                    || !params.insert(&p.id)
                    || !text(&p.label, 80)
                    || p.help.chars().count() > 1000
                    || p.group.chars().count() > 80
                    || !p.min.is_finite()
                    || !p.max.is_finite()
                    || p.min > p.max
                    || !p.step.is_finite()
                    || p.step <= 0.
                    || (p.kind == "integer"
                        && (p.min.fract() != 0.
                            || p.max.fract() != 0.
                            || p.step.fract() != 0.
                            || p.step < 1.
                            || p.min < i64::MIN as f64
                            || p.max >= i64::MAX as f64))
                    || p.choices.len() > 64
                    || p.choices.iter().any(|c| !text(c, 80))
                    || (self.api <= 2 && !p.validate_value(&p.default))
                    || (self.api == 1 && matches!(p.kind.as_str(), "seed" | "point"))
                {
                    return Err("Invalid plugin parameter".into());
                }
            }
        }
        Ok(())
    }
}
impl Catalogue {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != 1 || self.packs.len() > 1000 {
            return Err("Unsupported catalogue".into());
        }
        let mut ids = BTreeSet::new();
        for p in &self.packs {
            p.manifest.validate_listing()?;
            if !ids.insert(&p.manifest.id)
                || !allowed_url(&p.url)
                || !digest(&p.sha256)
                || p.size == 0
                || p.size > super::MAX_PACKAGE as u64
                || p.window_days > 7
                || p.recent_downloads > p.downloads
                || [&p.screenshot_url, &p.icon_url]
                    .iter()
                    .any(|s| !s.is_empty() && !allowed_url(s))
            {
                return Err("Invalid catalogue listing".into());
            }
        }
        Ok(())
    }
}
