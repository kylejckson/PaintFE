// ============================================================================
// ICON PACKS — user-supplied PNG icon overrides
// ============================================================================
//
// A pack is a folder of PNGs named after icon ids (snake_case of the `Icon`
// variant: `brush.png`, `menu_file_open.png`, ...), with optional per-theme
// variants `…_dark.png` / `…_light.png`. Anything not provided falls back to
// the built-in icon (which keeps its normal invert-in-dark behaviour), so
// partial packs work fine and mixed packs resolve per icon:
//
//   1. `id_<theme>.png`            (theme-specific override)
//   2. `id.png`                    (generic override; inverted in dark mode
//                                   when `invert_mismatch` is set)
//   3. built-in icon               (inverted in dark mode, as always)
//
// `pack.ini` may contain `name=…` for the pack's display name.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::config::icons::Icon;

/// Which source produced a displayed icon (used by the Preferences preview).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IconSource {
    /// Pack provided a theme-specific variant.
    PackTheme,
    /// Pack generic variant, used as-is for this theme.
    PackGeneric,
    /// Pack generic variant, inverted for the current theme.
    PackGenericInverted,
    /// Built-in icon, used as-is.
    Builtin,
    /// Built-in icon, inverted for dark mode.
    BuiltinInverted,
}

impl IconSource {
    pub fn label(&self) -> &'static str {
        match self {
            IconSource::PackTheme => "pack (theme)",
            IconSource::PackGeneric => "pack",
            IconSource::PackGenericInverted => "pack (inverted)",
            IconSource::Builtin => "default",
            IconSource::BuiltinInverted => "default (inverted)",
        }
    }

    pub fn from_pack(&self) -> bool {
        matches!(
            self,
            IconSource::PackTheme | IconSource::PackGeneric | IconSource::PackGenericInverted
        )
    }
}

#[derive(Default)]
struct PackIconVariants {
    dark: Option<image::RgbaImage>,
    light: Option<image::RgbaImage>,
    generic: Option<image::RgbaImage>,
}

/// A loaded icon pack: PNG overrides keyed by canonical icon id.
pub struct IconPack {
    pub path: PathBuf,
    pub name: String,
    icons: HashMap<String, PackIconVariants>,
}

impl IconPack {
    /// Load every `*.png` in `dir` as icon overrides.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let mut icons: HashMap<String, PackIconVariants> = HashMap::new();
        let entries =
            std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("png") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let (base, variant) = if let Some(b) = stem.strip_suffix("_dark") {
                (b, 0)
            } else if let Some(b) = stem.strip_suffix("_light") {
                (b, 1)
            } else {
                (stem, 2)
            };
            let Ok(img) = image::open(&path) else {
                continue; // skip unreadable files, keep the rest of the pack
            };
            let img = img.to_rgba8();
            let slot = icons.entry(base.to_string()).or_default();
            match variant {
                0 => slot.dark = Some(img),
                1 => slot.light = Some(img),
                _ => slot.generic = Some(img),
            }
        }
        if icons.is_empty() {
            return Err(format!("no PNG icons found in {}", dir.display()));
        }
        let name = std::fs::read_to_string(dir.join("pack.ini"))
            .ok()
            .and_then(|ini| {
                ini.lines()
                    .find_map(|l| l.strip_prefix("name="))
                    .map(|s| s.trim().to_string())
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                dir.file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "Icon Pack".to_string())
            });
        Ok(Self {
            path: dir.to_path_buf(),
            name,
            icons,
        })
    }

    /// Resolve display pixels for an icon. Returns `None` when the pack has no
    /// override for it (caller falls back to the built-in icon).
    pub fn resolve(
        &self,
        icon: Icon,
        dark: bool,
        invert_mismatch: bool,
    ) -> Option<(image::RgbaImage, IconSource)> {
        let v = self.icons.get(&icon_id_name(icon))?;
        if dark {
            if let Some(img) = &v.dark {
                return Some((img.clone(), IconSource::PackTheme));
            }
        } else if let Some(img) = &v.light {
            return Some((img.clone(), IconSource::PackTheme));
        }
        let img = v.generic.as_ref()?;
        if dark && invert_mismatch {
            Some((invert_image(img), IconSource::PackGenericInverted))
        } else {
            Some((img.clone(), IconSource::PackGeneric))
        }
    }

    /// True when the pack provides anything for this icon id.
    pub fn has_icon(&self, icon: Icon) -> bool {
        self.icons.contains_key(&icon_id_name(icon))
    }

    pub fn icon_count(&self) -> usize {
        self.icons.len()
    }
}

/// Canonical icon id used for pack filenames: snake_case of the variant name
/// (`Icon::MenuFileOpen` -> `menu_file_open`, `Icon::Brush` -> `brush`).
pub fn icon_id_name(icon: Icon) -> String {
    let raw = format!("{icon:?}");
    let mut out = String::with_capacity(raw.len() + 4);
    for (i, ch) in raw.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// Invert the RGB channels of an image (alpha preserved), matching the
/// built-in dark-mode icon treatment.
pub fn invert_image(img: &image::RgbaImage) -> image::RgbaImage {
    let mut out = img.clone();
    for px in out.pixels_mut() {
        px[0] = 255 - px[0];
        px[1] = 255 - px[1];
        px[2] = 255 - px[2];
    }
    out
}

/// Write an icon-pack template into `dir`: `pack.ini`, `icons.txt` (every
/// icon id) and the current icons as `icons/<id>.png` so artists can edit
/// them in place.
pub fn export_template(
    dir: &Path,
    icons: &[(String, image::RgbaImage)],
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("pack.ini"),
        "name=My Icon Pack\n# Optional: displayed in Preferences\n",
    )
    .map_err(|e| e.to_string())?;
    let mut list = String::from(
        "# PaintFE icon pack template\n# Drop PNGs named <id>.png here to override icons.\n# Add <id>_dark.png / <id>_light.png for theme-specific variants.\n\n",
    );
    for (id, _) in icons {
        list.push_str(id);
        list.push('\n');
    }
    std::fs::write(dir.join("icons.txt"), list).map_err(|e| e.to_string())?;

    let icons_dir = dir.join("icons");
    std::fs::create_dir_all(&icons_dir).map_err(|e| e.to_string())?;
    for (id, img) in icons {
        let path = icons_dir.join(format!("{id}.png"));
        img.save(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_png(path: &Path, rgb: [u8; 3]) {
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([rgb[0], rgb[1], rgb[2], 255]));
        img.save(path).unwrap();
    }

    #[test]
    fn icon_id_name_is_snake_case() {
        assert_eq!(icon_id_name(Icon::Brush), "brush");
        assert_eq!(icon_id_name(Icon::MenuFileOpen), "menu_file_open");
    }

    #[test]
    fn pack_resolution_prefers_theme_variants_then_generic_then_builtin() {
        let dir = std::env::temp_dir().join(format!(
            "paintfe_icon_pack_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_png(&dir.join("pencil_dark.png"), [255, 0, 0]);
        write_png(&dir.join("brush.png"), [0, 255, 0]);

        let pack = IconPack::load(&dir).unwrap();

        // Theme-specific variant wins in dark mode.
        let (_, src) = pack.resolve(Icon::Pencil, true, true).unwrap();
        assert_eq!(src, IconSource::PackTheme);
        // Light mode has no variant for pencil -> falls back to generic? none
        // provided, so the caller uses the built-in.
        assert!(pack.resolve(Icon::Pencil, false, true).is_none());

        // Generic variant: inverted in dark only when invert_mismatch is set.
        let (img, src) = pack.resolve(Icon::Brush, true, true).unwrap();
        assert_eq!(src, IconSource::PackGenericInverted);
        assert_eq!(
            img.get_pixel(0, 0).0,
            [255, 0, 255, 255],
            "generic inverted in dark (green -> magenta)"
        );
        let (_, src) = pack.resolve(Icon::Brush, true, false).unwrap();
        assert_eq!(src, IconSource::PackGeneric);
        let (_, src) = pack.resolve(Icon::Brush, false, true).unwrap();
        assert_eq!(src, IconSource::PackGeneric);

        // Missing icons report no override (built-in fallback).
        assert!(pack.resolve(Icon::Lasso, true, true).is_none());
        assert!(pack.has_icon(Icon::Brush));
        assert!(!pack.has_icon(Icon::Lasso));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
