// ============================================================================
// ICON PACKS — user-supplied PNG icon overrides
// ============================================================================
//
// A pack is a folder (or a .zip) of PNGs named after canonical icon ids
// (`tool_pencil.png`, `toolbar_new.png`, `menu_file_new.png`,
// `shape_heart.png`, ...), with optional per-theme variants `…_dark.png` /
// `…_light.png`. Anything not provided falls back to the built-in icon
// (which keeps its normal invert-in-dark behaviour), so partial packs work
// fine and mixed packs resolve per icon:
//
//   1. `id_<theme>.png`            (theme-specific override)
//   2. `id.png`                    (generic override; inverted in dark mode
//                                   when `invert_mismatch` is set)
//   3. built-in icon               (inverted in dark mode, as always)
//
// Packs made for the original naming may use the legacy ids (snake case of
// the enum variant: `pencil`, `new`, `menu_file_new`, …) — those still
// resolve through `legacy_id_name`.
//
// `pack.ini` may contain `name=…` for the pack's display name.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::config::icons::Icon;
use crate::ops::shapes::ShapeKind;

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

impl PackIconVariants {
    fn set(&mut self, variant: u8, img: image::RgbaImage) {
        match variant {
            0 => self.dark = Some(img),
            1 => self.light = Some(img),
            _ => self.generic = Some(img),
        }
    }
}

/// A loaded icon pack: PNG overrides keyed by canonical icon id.
pub struct IconPack {
    pub path: PathBuf,
    pub name: String,
    icons: HashMap<String, PackIconVariants>,
}

/// Safety caps for archive imports.
const ZIP_MAX_ENTRIES: usize = 4096;
const ZIP_MAX_ENTRY_BYTES: u64 = 32 * 1024 * 1024;

impl IconPack {
    /// Load a pack from a folder of PNGs, or a .zip archive of PNGs.
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut icons: HashMap<String, PackIconVariants> = HashMap::new();
        let mut ini = String::new();

        if path.is_dir() {
            // Scan the root plus one level of subfolders (Export Template puts
            // PNGs under `icons/`, and packs may be organized in folders).
            let mut files: Vec<PathBuf> = Vec::new();
            let entries =
                std::fs::read_dir(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            for entry in entries.flatten() {
                let file = entry.path();
                if file.is_dir() {
                    if let Ok(sub) = std::fs::read_dir(&file) {
                        files.extend(sub.flatten().map(|e| e.path()));
                    }
                } else {
                    files.push(file);
                }
            }
            for file in files {
                if file.extension().and_then(|e| e.to_str()) != Some("png") {
                    continue;
                }
                let Some(stem) = file.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                if let Ok(bytes) = std::fs::read(&file)
                    && let Ok(img) = image::load_from_memory(&bytes)
                {
                    ingest_png(&mut icons, stem, img.to_rgba8());
                }
            }
            ini = std::fs::read_to_string(path.join("pack.ini")).unwrap_or_default();
        } else if path.extension().and_then(|e| e.to_str()) == Some("zip") {
            let file = std::fs::File::open(path)
                .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
            let mut zip =
                zip::ZipArchive::new(file).map_err(|e| format!("bad zip {}: {e}", path.display()))?;
            if zip.len() > ZIP_MAX_ENTRIES {
                return Err(format!("too many entries in {}", path.display()));
            }
            for i in 0..zip.len() {
                let entry = zip
                    .by_index(i)
                    .map_err(|e| format!("bad zip entry: {e}"))?;
                if !entry.is_file() {
                    continue;
                }
                let Some(name) = entry.enclosed_name().map(|p| p.to_path_buf()) else {
                    continue; // path traversal guard
                };
                if name.components().count() > 2 {
                    continue; // root files + one folder at most
                }
                let stem = match name.extension().and_then(|e| e.to_str()) {
                    Some("png") => name
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default()
                        .to_string(),
                    Some("ini") if name.file_name().and_then(|s| s.to_str()) == Some("pack.ini") => {
                        let mut buf = String::new();
                        let mut limited = entry.take(ZIP_MAX_ENTRY_BYTES);
                        let _ = limited.read_to_string(&mut buf);
                        ini = buf;
                        continue;
                    }
                    _ => continue,
                };
                if stem.is_empty() || entry.size() > ZIP_MAX_ENTRY_BYTES {
                    continue;
                }
                let mut buf = Vec::new();
                let mut limited = entry.take(ZIP_MAX_ENTRY_BYTES);
                if limited.read_to_end(&mut buf).is_err() {
                    continue;
                }
                if let Ok(img) = image::load_from_memory(&buf) {
                    ingest_png(&mut icons, &stem, img.to_rgba8());
                }
            }
        } else {
            return Err(format!(
                "icon packs must be a folder or a .zip file: {}",
                path.display()
            ));
        }

        if icons.is_empty() {
            return Err(format!("no PNG icons found in {}", path.display()));
        }
        let name = ini
            .lines()
            .find_map(|l| l.strip_prefix("name="))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "Icon Pack".to_string())
            });
        Ok(Self {
            path: path.to_path_buf(),
            name,
            icons,
        })
    }

    /// Resolve display pixels for an icon. Returns `None` when the pack has no
    /// override for it (caller falls back to the built-in icon). Both the
    /// canonical id and the legacy id are accepted.
    pub fn resolve(
        &self,
        icon: Icon,
        dark: bool,
        invert_mismatch: bool,
    ) -> Option<(image::RgbaImage, IconSource)> {
        let variants = self
            .icons
            .get(&icon.pack_id())
            .or_else(|| self.icons.get(&legacy_id_name(icon)))?;
        resolve_variants(variants, dark, invert_mismatch)
    }

    /// Resolve display pixels for a shape-kind icon (see `resolve`).
    pub fn resolve_shape(
        &self,
        kind: ShapeKind,
        dark: bool,
        invert_mismatch: bool,
    ) -> Option<(image::RgbaImage, IconSource)> {
        let variants = self
            .icons
            .get(&shape_pack_id(kind))
            .or_else(|| self.icons.get(kind.icon_name()))?;
        resolve_variants(variants, dark, invert_mismatch)
    }

    /// True when the pack provides anything for this icon id.
    pub fn has_icon(&self, icon: Icon) -> bool {
        self.icons.contains_key(&icon.pack_id()) || self.icons.contains_key(&legacy_id_name(icon))
    }

    pub fn icon_count(&self) -> usize {
        self.icons.len()
    }
}

fn resolve_variants(
    v: &PackIconVariants,
    dark: bool,
    invert_mismatch: bool,
) -> Option<(image::RgbaImage, IconSource)> {
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

/// Parse `name.png` / `name_dark.png` / `name_light.png` into the pack map.
/// Dead placeholder icons are ignored.
fn ingest_png(icons: &mut HashMap<String, PackIconVariants>, stem: &str, img: image::RgbaImage) {
    let (base, variant) = if let Some(b) = stem.strip_suffix("_dark") {
        (b, 0)
    } else if let Some(b) = stem.strip_suffix("_light") {
        (b, 1)
    } else {
        (stem, 2)
    };
    if is_excluded_id(base) {
        return;
    }
    icons
        .entry(base.to_string())
        .or_default()
        .set(variant, img);
}

/// Legacy icon id used by packs made for the original naming: snake case of
/// the enum variant (`Icon::MenuFileOpen` -> `menu_file_open`,
/// `Icon::Brush` -> `brush`). Still accepted on pack import.
pub fn legacy_id_name(icon: Icon) -> String {
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

/// Canonical pack id for a shape-kind icon (`shape_heart`, `shape_star5`, …).
/// Legacy packs may use the bare stem (`heart`) — see `ShapeKind::icon_name`.
pub fn shape_pack_id(kind: ShapeKind) -> String {
    let raw = format!("{kind:?}");
    let mut base = String::with_capacity(raw.len() + 4);
    for (i, ch) in raw.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                base.push('_');
            }
            base.extend(ch.to_lowercase());
        } else {
            base.push(ch);
        }
    }
    format!("shape_{base}")
}

/// Icon ids that are dead placeholders: never shown in the UI, so they are
/// excluded from Export Template, hidden from the preview, and ignored when
/// loading a pack.
const EXCLUDED_IDS: &[&str] = &["menu_filter_sharpen"];

/// True when an icon id is a dead placeholder (see `EXCLUDED_IDS`).
pub fn is_excluded_id(id: &str) -> bool {
    EXCLUDED_IDS.contains(&id)
}

/// True when an icon is a dead placeholder (see `EXCLUDED_IDS`).
pub fn is_excluded_icon(icon: Icon) -> bool {
    is_excluded_id(&icon.pack_id())
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
/// canonical id plus a legacy-name migration list) and the current icons as
/// `icons/<id>.png` so artists can edit them in place.
pub fn export_template(
    dir: &Path,
    icons: &[(String, image::RgbaImage)],
    legacy_names: &[(String, String)],
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
    list.push_str(
        "\n# Legacy names (older packs) still resolve to these ids:\n#   legacy_name -> canonical_id\n",
    );
    for (legacy, canonical) in legacy_names {
        list.push_str(&format!("#   {legacy} -> {canonical}\n"));
    }
    std::fs::write(dir.join("icons.txt"), list).map_err(|e| e.to_string())?;

    let icons_dir = dir.join("icons");
    std::fs::create_dir_all(&icons_dir).map_err(|e| e.to_string())?;
    for (id, img) in icons {
        let path = icons_dir.join(format!("{id}.png"));
        img.save(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
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

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "paintfe_icon_pack_{tag}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn pack_ids_are_category_prefixed() {
        assert_eq!(Icon::Pencil.pack_id(), "tool_pencil");
        assert_eq!(Icon::New.pack_id(), "toolbar_new");
        assert_eq!(Icon::MenuFileOpen.pack_id(), "menu_file_open");
        assert_eq!(Icon::LayerAdd.pack_id(), "layer_add");
        assert_eq!(Icon::Settings.pack_id(), "settings_main");
        assert_eq!(Icon::SwapColors.pack_id(), "color_swap");
        assert_eq!(Icon::ShapeFilled.pack_id(), "shape_filled");
        assert_eq!(Icon::Delete.pack_id(), "ui_delete");
        assert_eq!(shape_pack_id(ShapeKind::Heart), "shape_heart");
    }

    #[test]
    fn legacy_names_still_resolve() {
        let dir = tmp_dir("legacy");
        // Legacy names from the original naming (snake of the enum variant).
        write_png(&dir.join("pencil.png"), [10, 20, 30]);
        write_png(&dir.join("menu_file_new.png"), [1, 2, 3]);
        write_png(&dir.join("heart.png"), [9, 8, 7]);

        let pack = IconPack::load(&dir).unwrap();
        let (img, src) = pack.resolve(Icon::Pencil, false, true).unwrap();
        assert_eq!(src, IconSource::PackGeneric);
        assert_eq!(img.get_pixel(0, 0).0[0], 10);
        assert!(pack.resolve(Icon::MenuFileNew, false, true).is_some());
        assert!(
            pack.resolve_shape(ShapeKind::Heart, false, true)
                .is_some(),
            "legacy shape stem resolves"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pack_resolution_prefers_theme_variants_then_generic_then_builtin() {
        let dir = tmp_dir("resolve");
        write_png(&dir.join("tool_pencil_dark.png"), [255, 0, 0]);
        write_png(&dir.join("tool_brush.png"), [0, 255, 0]);

        let pack = IconPack::load(&dir).unwrap();

        let (_, src) = pack.resolve(Icon::Pencil, true, true).unwrap();
        assert_eq!(src, IconSource::PackTheme);
        assert!(pack.resolve(Icon::Pencil, false, true).is_none());

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

        assert!(pack.resolve(Icon::Lasso, true, true).is_none());
        assert!(pack.has_icon(Icon::Brush));
        assert!(!pack.has_icon(Icon::Lasso));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zip_packs_load_like_folders() {
        let dir = tmp_dir("zip");
        let png = image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]));
        let mut buf = Vec::new();
        png.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Png,
        )
        .unwrap();
        let zip_path = dir.join("pack.zip");
        {
            let file = std::fs::File::create(&zip_path).unwrap();
            let mut zip = zip::ZipWriter::new(file);
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zip.start_file("tool_pencil.png", opts).unwrap();
            std::io::Write::write_all(&mut zip, &buf).unwrap();
            zip.start_file("pack.ini", opts).unwrap();
            std::io::Write::write_all(&mut zip, b"name=Zip Test Pack\n").unwrap();
            // Path traversal entries must be ignored.
            zip.start_file("../evil.png", opts).unwrap();
            std::io::Write::write_all(&mut zip, &buf).unwrap();
        }

        let pack = IconPack::load(&zip_path).unwrap();
        assert_eq!(pack.name, "Zip Test Pack");
        assert_eq!(pack.icon_count(), 1, "traversal entry ignored");
        assert!(pack.resolve(Icon::Pencil, false, false).is_some());
        assert!(pack.resolve(Icon::Brush, false, false).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn excluded_placeholder_icons_are_ignored() {
        let dir = tmp_dir("excluded");
        write_png(&dir.join("menu_filter_sharpen.png"), [1, 1, 1]);
        write_png(&dir.join("tool_pencil.png"), [2, 2, 2]);
        let pack = IconPack::load(&dir).unwrap();
        assert_eq!(pack.icon_count(), 1, "placeholder skipped");
        assert!(pack.resolve(Icon::MenuFilterSharpen, false, false).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_template_round_trips() {
        let src = tmp_dir("export_src");
        write_png(&src.join("tool_pencil.png"), [11, 22, 33]);
        let pack = IconPack::load(&src).unwrap();
        let (img, _) = pack.resolve(Icon::Pencil, false, false).unwrap();

        let out = tmp_dir("export_out");
        export_template(
            &out,
            &[("tool_pencil".to_string(), img.clone())],
            &[("pencil".to_string(), "tool_pencil".to_string())],
        )
        .unwrap();
        assert!(out.join("pack.ini").exists());
        assert!(out.join("icons/tool_pencil.png").exists());
        let listing = std::fs::read_to_string(out.join("icons.txt")).unwrap();
        assert!(listing.contains("tool_pencil"));
        assert!(listing.contains("pencil -> tool_pencil"), "legacy list");

        // Exported template loads and overrides the icon.
        let reloaded = IconPack::load(&out).unwrap();
        assert!(reloaded.resolve(Icon::Pencil, false, false).is_some());
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn legacy_id_matches_original_naming() {
        assert_eq!(legacy_id_name(Icon::Brush), "brush");
        assert_eq!(legacy_id_name(Icon::MenuFileOpen), "menu_file_open");
    }
}
