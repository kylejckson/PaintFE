//! Internationalization (i18n) module for PaintFE.
//!
//! Uses a simple key→string HashMap loaded at runtime from embedded translation data.
//! The `t!("key")` macro looks up the current language, falling back to English.
//! Language can be switched at runtime via `set_language()`.

use std::collections::{BTreeMap, HashMap};
mod overrides;
pub use overrides::*;
use std::sync::Mutex;

/// Global translation state.
static I18N: Mutex<Option<I18nState>> = Mutex::new(None);

struct I18nState {
    current_lang: String,
    /// lang_code → (key → translated_string)
    translations: HashMap<String, HashMap<String, String>>,
    /// Built-in English display labels used by enums and shared widgets.
    english_keys: HashMap<String, String>,
    overrides: HashMap<String, BTreeMap<String, String>>,
}

/// Supported languages: (code, native_name)
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("ar", "العربية"),
    ("es", "Español"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("pt-BR", "Português (Brasil)"),
    ("pt-PT", "Português (Portugal)"),
    ("it", "Italiano"),
    ("ja", "日本語"),
    ("zh-CN", "中文(简体)"),
    ("zh-TW", "中文(繁體)"),
    ("ru", "Русский"),
    ("uk", "Українська"),
    ("cs", "Čeština"),
    ("ro", "Română"),
    ("be", "Беларуская"),
    ("nl", "Nederlands"),
    ("pl", "Polski"),
    ("tr", "Türkçe"),
    ("ko", "한국어"),
    ("id", "Bahasa Indonesia"),
    ("vi", "Tiếng Việt"),
    ("th", "ไทย"),
    ("hu", "Magyar"),
    ("en-AU-x-bogan", "English (Bogan)"),
    ("fe", "Fancy English"),
];

const EMBEDDED_TRANSLATIONS: &[(&str, &str)] = &[
    ("en", include_str!("../locales/en.txt")),
    ("ar", include_str!("../locales/ar.txt")),
    ("es", include_str!("../locales/es.txt")),
    ("fr", include_str!("../locales/fr.txt")),
    ("de", include_str!("../locales/de.txt")),
    ("pt-BR", include_str!("../locales/pt-BR.txt")),
    ("pt-PT", include_str!("../locales/pt-PT.txt")),
    ("it", include_str!("../locales/it.txt")),
    ("ja", include_str!("../locales/ja.txt")),
    ("zh-CN", include_str!("../locales/zh-CN.txt")),
    ("zh-TW", include_str!("../locales/zh-TW.txt")),
    ("ru", include_str!("../locales/ru.txt")),
    ("uk", include_str!("../locales/uk.txt")),
    ("cs", include_str!("../locales/cs.txt")),
    ("ro", include_str!("../locales/ro.txt")),
    ("be", include_str!("../locales/be.txt")),
    ("nl", include_str!("../locales/nl.txt")),
    ("pl", include_str!("../locales/pl.txt")),
    ("tr", include_str!("../locales/tr.txt")),
    ("ko", include_str!("../locales/ko.txt")),
    ("id", include_str!("../locales/id.txt")),
    ("vi", include_str!("../locales/vi.txt")),
    ("th", include_str!("../locales/th.txt")),
    ("hu", include_str!("../locales/hu.txt")),
    (
        "en-AU-x-bogan",
        include_str!("../locales/en-AU-x-bogan.txt"),
    ),
    ("fe", include_str!("../locales/fe.txt")),
];

/// Initialize the i18n system with embedded translations.
/// Call once at startup.
pub fn init() {
    let translations: HashMap<String, HashMap<String, String>> = EMBEDDED_TRANSLATIONS
        .iter()
        .map(|(code, data)| ((*code).to_string(), parse_translations(data)))
        .collect();

    let mut english_entries: Vec<_> = translations["en"].iter().collect();
    english_entries.sort_by_key(|(key, _)| *key);
    let mut english_keys = HashMap::new();
    for (key, value) in english_entries {
        english_keys
            .entry(value.clone())
            .or_insert_with(|| key.clone());
        english_keys
            .entry(value.to_lowercase())
            .or_insert_with(|| key.clone());
    }
    // Retain lookup aliases for stable English display values after spelling changes.
    english_keys.insert("Color".into(), "gradient_mode.color".into());
    english_keys.insert("color".into(), "gradient_mode.color".into());
    english_keys.insert("Invert Colors".into(), "menu.color.invert_colors".into());
    english_keys.insert("invert colors".into(), "menu.color.invert_colors".into());
    english_keys.insert(
        "Color to Alpha...".into(),
        "menu.color.color_to_alpha".into(),
    );
    english_keys.insert(
        "color to alpha...".into(),
        "menu.color.color_to_alpha".into(),
    );
    english_keys.insert("Color Balance...".into(), "menu.color.color_balance".into());
    english_keys.insert("color balance...".into(), "menu.color.color_balance".into());
    english_keys.insert(
        "Color Filter...".into(),
        "menu.filter.stylize.color_filter".into(),
    );
    english_keys.insert(
        "color filter...".into(),
        "menu.filter.stylize.color_filter".into(),
    );
    english_keys.insert("Colors Panel".into(), "menu.view.colors_panel".into());
    english_keys.insert("colors panel".into(), "menu.view.colors_panel".into());
    english_keys.insert("Color Picker".into(), "keybind.tool_color_picker".into());
    english_keys.insert("color picker".into(), "keybind.tool_color_picker".into());
    english_keys.insert("Color Remover".into(), "keybind.tool_color_remover".into());
    english_keys.insert("color remover".into(), "keybind.tool_color_remover".into());
    english_keys.insert(
        "Color matching tolerance (0% = exact color, 100% = all colors)".into(),
        "ctx.tolerance_tooltip".into(),
    );
    english_keys.insert(
        "color matching tolerance (0% = exact color, 100% = all colors)".into(),
        "ctx.tolerance_tooltip".into(),
    );
    english_keys.insert(
        "Eraser strength — controlled by the primary color's alpha value".into(),
        "ctx.eraser_opacity_tooltip".into(),
    );
    english_keys.insert(
        "eraser strength — controlled by the primary color's alpha value".into(),
        "ctx.eraser_opacity_tooltip".into(),
    );
    english_keys.insert(
        "Pick color in Color panel → Apply Primary".into(),
        "ctx.gradient.color_hint".into(),
    );
    english_keys.insert(
        "pick color in color panel → apply primary".into(),
        "ctx.gradient.color_hint".into(),
    );
    english_keys.insert(
        "Color:".into(),
        "ctx.text.effects.inner_shadow.color".into(),
    );
    english_keys.insert(
        "color:".into(),
        "ctx.text.effects.inner_shadow.color".into(),
    );
    english_keys.insert("Center".into(), "ctx.text.effects.outline.center".into());
    english_keys.insert("center".into(), "ctx.text.effects.outline.center".into());
    english_keys.insert("Color Balance".into(), "dialog.color_balance".into());
    english_keys.insert("color balance".into(), "dialog.color_balance".into());
    english_keys.insert("Color Filter".into(), "dialog.color_filter".into());
    english_keys.insert("color filter".into(), "dialog.color_filter".into());
    english_keys.insert("Color to Alpha".into(), "dialog.color_to_alpha".into());
    english_keys.insert("color to alpha".into(), "dialog.color_to_alpha".into());
    english_keys.insert(
        "(Uses secondary color)".into(),
        "dialog.resize_canvas.uses_secondary_color".into(),
    );
    english_keys.insert(
        "(uses secondary color)".into(),
        "dialog.resize_canvas.uses_secondary_color".into(),
    );
    english_keys.insert(
        "Accent-colored shadows in dark theme".into(),
        "settings.general.neon_hint".into(),
    );
    english_keys.insert(
        "accent-colored shadows in dark theme".into(),
        "settings.general.neon_hint".into(),
    );
    english_keys.insert(
        "Follow theme color".into(),
        "settings.interface.resize_grip_auto".into(),
    );
    english_keys.insert(
        "follow theme color".into(),
        "settings.interface.resize_grip_auto".into(),
    );
    english_keys.insert(
        "Accent Color".into(),
        "settings.interface.accent_color".into(),
    );
    english_keys.insert(
        "accent color".into(),
        "settings.interface.accent_color".into(),
    );
    english_keys.insert(
        "Light Mode Colors".into(),
        "settings.interface.light_mode_colors".into(),
    );
    english_keys.insert(
        "light mode colors".into(),
        "settings.interface.light_mode_colors".into(),
    );
    english_keys.insert(
        "Dark Mode Colors".into(),
        "settings.interface.dark_mode_colors".into(),
    );
    english_keys.insert(
        "dark mode colors".into(),
        "settings.interface.dark_mode_colors".into(),
    );
    english_keys.insert("Color Burn".into(), "blend.color_burn".into());
    english_keys.insert("color burn".into(), "blend.color_burn".into());
    english_keys.insert("Color Dodge".into(), "blend.color_dodge".into());
    english_keys.insert("color dodge".into(), "blend.color_dodge".into());
    english_keys.insert("Nearest Neighbor".into(), "interpolation.nearest".into());
    english_keys.insert("nearest neighbor".into(), "interpolation.nearest".into());
    english_keys.insert(
        "Graph optimization".into(),
        "settings.ai.graph_optimization".into(),
    );
    english_keys.insert(
        "graph optimization".into(),
        "settings.ai.graph_optimization".into(),
    );
    english_keys.insert("Colors".into(), "ui.colors".into());
    english_keys.insert("colors".into(), "ui.colors".into());
    english_keys.insert("Hex color".into(), "ui.hex_color".into());
    english_keys.insert("hex color".into(), "ui.hex_color".into());
    english_keys.insert(
        "Edit or copy hex color".into(),
        "ui.edit_or_copy_hex_color".into(),
    );
    english_keys.insert(
        "edit or copy hex color".into(),
        "ui.edit_or_copy_hex_color".into(),
    );
    english_keys.insert(
        "Add selected color to this collection".into(),
        "ui.add_selected_color_to_this_collection".into(),
    );
    english_keys.insert(
        "add selected color to this collection".into(),
        "ui.add_selected_color_to_this_collection".into(),
    );
    english_keys.insert(
        "Colors appear here after painting.".into(),
        "ui.colors_appear_here_after_painting".into(),
    );
    english_keys.insert(
        "colors appear here after painting.".into(),
        "ui.colors_appear_here_after_painting".into(),
    );
    english_keys.insert(
        "Add a color with + to start.".into(),
        "ui.add_a_color_with_to_start".into(),
    );
    english_keys.insert(
        "add a color with + to start.".into(),
        "ui.add_a_color_with_to_start".into(),
    );
    english_keys.insert("Add to Favorites".into(), "ui.add_to_favorites".into());
    english_keys.insert("add to favorites".into(), "ui.add_to_favorites".into());
    english_keys.insert("Remove color".into(), "ui.remove_color".into());
    english_keys.insert("remove color".into(), "ui.remove_color".into());
    english_keys.insert("Center Anchor".into(), "ui.center_anchor".into());
    english_keys.insert("center anchor".into(), "ui.center_anchor".into());
    english_keys.insert("Pixel grid outline/center colors control the dual-stroke grid visibility on any background.".into(), "ui.pixel_grid_outline_center_colors_control_the_dual_stroke_grid_visibili".into());
    english_keys.insert("pixel grid outline/center colors control the dual-stroke grid visibility on any background.".into(), "ui.pixel_grid_outline_center_colors_control_the_dual_stroke_grid_visibili".into());
    english_keys.insert(
        "Layer Folder Colors".into(),
        "ui.layer_folder_colors".into(),
    );
    english_keys.insert(
        "layer folder colors".into(),
        "ui.layer_folder_colors".into(),
    );
    english_keys.insert(
        "Reset Folder Colors".into(),
        "ui.reset_folder_colors".into(),
    );
    english_keys.insert(
        "reset folder colors".into(),
        "ui.reset_folder_colors".into(),
    );
    english_keys.insert(
        "Advanced Customization".into(),
        "ui.advanced_customization".into(),
    );
    english_keys.insert(
        "advanced customization".into(),
        "ui.advanced_customization".into(),
    );
    english_keys.insert("Fine-tune individual colors, geometry, and effects. Overrides apply on top of the selected preset.".into(), "ui.fine_tune_individual_colors_geometry_and_effects_overrides_apply_on_to".into());
    english_keys.insert("fine-tune individual colors, geometry, and effects. overrides apply on top of the selected preset.".into(), "ui.fine_tune_individual_colors_geometry_and_effects_overrides_apply_on_to".into());
    english_keys.insert(
        "Click to open Colors".into(),
        "ui.click_to_open_colors".into(),
    );
    english_keys.insert(
        "click to open colors".into(),
        "ui.click_to_open_colors".into(),
    );
    english_keys.insert(
        "Preserve source colors".into(),
        "ui.preserve_source_colors".into(),
    );
    english_keys.insert(
        "preserve source colors".into(),
        "ui.preserve_source_colors".into(),
    );
    english_keys.insert(
        "Optimize automatically".into(),
        "ui.optimize_automatically".into(),
    );
    english_keys.insert(
        "optimize automatically".into(),
        "ui.optimize_automatically".into(),
    );
    english_keys.insert("Select Color Range".into(), "ui.select_color_range".into());
    english_keys.insert("select color range".into(), "ui.select_color_range".into());
    english_keys.insert("Hue Center".into(), "ui.hue_center".into());
    english_keys.insert("hue center".into(), "ui.hue_center".into());
    english_keys.insert("TARGET COLOR".into(), "ui.target_color".into());
    english_keys.insert("target color".into(), "ui.target_color".into());
    english_keys.insert(
        "Recover alpha and edge colors from artwork flattened over a noisy solid background."
            .into(),
        "ui.recover_alpha_and_edge_colors_from_artwork_flattened_over_a_noisy_soli".into(),
    );
    english_keys.insert(
        "recover alpha and edge colors from artwork flattened over a noisy solid background."
            .into(),
        "ui.recover_alpha_and_edge_colors_from_artwork_flattened_over_a_noisy_soli".into(),
    );
    english_keys.insert("Color Preservation".into(), "ui.color_preservation".into());
    english_keys.insert("color preservation".into(), "ui.color_preservation".into());
    english_keys.insert("Line Color".into(), "ui.line_color".into());
    english_keys.insert("line color".into(), "ui.line_color".into());
    english_keys.insert("Center X".into(), "ui.center_x".into());
    english_keys.insert("center x".into(), "ui.center_x".into());
    english_keys.insert("Center Y".into(), "ui.center_y".into());
    english_keys.insert("center y".into(), "ui.center_y".into());
    english_keys.insert("COLOR TINT".into(), "ui.color_tint".into());
    english_keys.insert("color tint".into(), "ui.color_tint".into());
    english_keys.insert(
        "Radial color tint at origin".into(),
        "ui.radial_color_tint_at_origin".into(),
    );
    english_keys.insert(
        "radial color tint at origin".into(),
        "ui.radial_color_tint_at_origin".into(),
    );
    english_keys.insert("Tint Color".into(), "ui.tint_color".into());
    english_keys.insert("tint color".into(), "ui.tint_color".into());
    english_keys.insert("Uses Primary Color".into(), "ui.uses_primary_color".into());
    english_keys.insert("uses primary color".into(), "ui.uses_primary_color".into());
    english_keys.insert("Folder color".into(), "ui.folder_color".into());
    english_keys.insert("folder color".into(), "ui.folder_color".into());
    english_keys.insert("4-neighbor".into(), "ui.4_neighbor".into());
    english_keys.insert("8-neighbor".into(), "ui.8_neighbor".into());
    english_keys.insert("Color {0}".into(), "ui.color_0".into());
    english_keys.insert("color {0}".into(), "ui.color_0".into());
    english_keys.insert("COLORS".into(), "ui.colors_2".into());
    english_keys.insert("colors".into(), "ui.colors".into());
    english_keys.insert("Favorites".into(), "ui.favorites".into());
    english_keys.insert("favorites".into(), "ui.favorites".into());
    english_keys.insert("Swap Colors".into(), "icon.swap_colors".into());
    english_keys.insert("swap colors".into(), "icon.swap_colors".into());
    english_keys.insert(
        "Apply Primary Color".into(),
        "icon.apply_primary_color".into(),
    );
    english_keys.insert(
        "apply primary color".into(),
        "icon.apply_primary_color".into(),
    );
    english_keys.insert(
        "Brush Dynamics (Pressure, Scatter, Color Jitter)".into(),
        "icon.brush_dynamics_flow_scatter_color_jitter".into(),
    );
    english_keys.insert(
        "brush dynamics (pressure, scatter, color jitter)".into(),
        "icon.brush_dynamics_flow_scatter_color_jitter".into(),
    );
    english_keys.insert("Surface Colors".into(), "ui.surface_colors".into());
    english_keys.insert("surface colors".into(), "ui.surface_colors".into());
    english_keys.insert(
        "Retarget is designed for reduction; enlargement uses nearest-neighbor pixels.".into(),
        "ui.retarget_is_designed_for_reduction_enlargement_uses_nearest_neighbor_p".into(),
    );
    english_keys.insert(
        "retarget is designed for reduction; enlargement uses nearest-neighbor pixels.".into(),
        "ui.retarget_is_designed_for_reduction_enlargement_uses_nearest_neighbor_p".into(),
    );
    english_keys.insert("Customize interface labels...".into(), "labels.edit".into());
    english_keys.insert("customize interface labels...".into(), "labels.edit".into());
    english_keys.insert("Customize interface labels".into(), "labels.title".into());
    english_keys.insert("customize interface labels".into(), "labels.title".into());
    english_keys.insert(
        "Select a label to customize its wording.".into(),
        "labels.select_label".into(),
    );
    english_keys.insert(
        "select a label to customize its wording.".into(),
        "labels.select_label".into(),
    );
    english_keys.insert("Text color".into(), "ruler.text_color".into());
    english_keys.insert("text color".into(), "ruler.text_color".into());
    english_keys.insert("Tick color".into(), "ruler.tick_color".into());
    english_keys.insert("tick color".into(), "ruler.tick_color".into());
    english_keys.insert("Background color".into(), "ruler.background_color".into());
    english_keys.insert("background color".into(), "ruler.background_color".into());
    english_keys.insert(
        "Grayscale without transparency".into(),
        "export.gray".into(),
    );
    english_keys.insert(
        "grayscale without transparency".into(),
        "export.gray".into(),
    );
    english_keys.insert(
        "Grayscale with transparency".into(),
        "export.gray_alpha".into(),
    );
    english_keys.insert(
        "grayscale with transparency".into(),
        "export.gray_alpha".into(),
    );
    english_keys.insert(
        "Saving as grayscale removes colour information.".into(),
        "export.loss_color".into(),
    );
    english_keys.insert(
        "saving as grayscale removes colour information.".into(),
        "export.loss_color".into(),
    );
    english_keys.insert("Grayscale".into(), "export.advanced.gray".into());
    english_keys.insert("grayscale".into(), "export.advanced.gray".into());
    english_keys.insert(
        "Uses Color Burn for dark brush colours and Color Dodge for light colours.".into(),
        "blend_help.21".into(),
    );
    english_keys.insert(
        "uses color burn for dark brush colours and color dodge for light colours.".into(),
        "blend_help.21".into(),
    );
    english_keys.insert(
        "Adds or subtracts light depending on whether the brush colour is above mid-gray.".into(),
        "blend_help.22".into(),
    );
    english_keys.insert(
        "adds or subtracts light depending on whether the brush colour is above mid-gray.".into(),
        "blend_help.22".into(),
    );
    english_keys.insert("Choose a tool, then adjust its options above the image. Pick colours in the Color panel. Zoom in to inspect detail and pan to move the view. Hold Space to pan temporarily.".into(), "help.start.navigate_body".into());
    english_keys.insert("choose a tool, then adjust its options above the image. pick colours in the color panel. zoom in to inspect detail and pan to move the view. hold space to pan temporarily.".into(), "help.start.navigate_body".into());
    let state = I18nState {
        current_lang: "en".to_string(),
        translations,
        english_keys,
        overrides: HashMap::new(),
    };
    *I18N.lock().unwrap() = Some(state);
}

/// Set the active language. If `code` is not a known language, falls back to "en".
pub fn set_language(code: &str) {
    if let Ok(mut guard) = I18N.lock()
        && let Some(ref mut state) = *guard
    {
        if state.translations.contains_key(code) {
            state.current_lang = code.to_string();
        } else {
            state.current_lang = "en".to_string();
        }
    }
}

/// Get the current language code.
pub fn current_language() -> String {
    if let Ok(guard) = I18N.lock()
        && let Some(ref state) = *guard
    {
        return state.current_lang.clone();
    }
    "en".to_string()
}

/// Translate a built-in display label while keeping its stable English value
/// available to persistence and identifiers. Do not use for user-supplied names.
pub fn translate_english(text: &str) -> String {
    if let Ok(guard) = I18N.lock()
        && let Some(state) = guard.as_ref()
        && let Some(key) = state
            .english_keys
            .get(text)
            .or_else(|| state.english_keys.get(&text.to_lowercase()))
        && let Some(value) = state
            .overrides
            .get(&state.current_lang)
            .and_then(|m| m.get(key))
            .or_else(|| {
                state
                    .translations
                    .get(&state.current_lang)
                    .and_then(|m| m.get(key))
            })
    {
        return if text.chars().any(|c| c.is_ascii_uppercase())
            && !text.chars().any(|c| c.is_ascii_lowercase())
        {
            value.to_uppercase()
        } else {
            value.clone()
        };
    }
    text.to_string()
}

/// Strength presets use "Light" to mean weak, rather than the light theme.
pub fn translate_strength_preset(text: &str) -> String {
    if text == "Light" {
        translate("preset.strength.light")
    } else {
        translate_english(text)
    }
}

/// Interpolate formatted values once; inserted text is never reinterpreted.
pub fn format_translation(key: &str, values: &[(&str, String)]) -> String {
    let template = translate(key);
    let mut result = String::with_capacity(template.len());
    let mut rest = template.as_str();
    while let Some(start) = rest.find('{') {
        result.push_str(&rest[..start]);
        rest = &rest[start..];
        let Some(end) = rest.find('}') else { break };
        let name = &rest[1..end];
        if let Some((_, value)) = values.iter().find(|(key, _)| *key == name) {
            result.push_str(value);
        } else {
            result.push_str(&rest[..=end]);
        }
        rest = &rest[end + 1..];
    }
    result.push_str(rest);
    result
}

/// Look up a translation key. Returns the translated string if found,
/// or falls back to English, or returns the key itself as last resort.
pub fn translate(key: &str) -> String {
    if let Ok(guard) = I18N.lock()
        && let Some(ref state) = *guard
    {
        if let Some(value) = state
            .overrides
            .get(&state.current_lang)
            .and_then(|m| m.get(key))
        {
            return value.clone();
        }
        // Try current language
        if let Some(map) = state.translations.get(&state.current_lang)
            && let Some(val) = map.get(key)
        {
            return val.clone();
        }
        // Fallback to English
        if state.current_lang != "en"
            && let Some(map) = state.translations.get("en")
            && let Some(val) = map.get(key)
        {
            return val.clone();
        }
    }
    // Last resort: return the key itself
    key.to_string()
}

/// Detect the system language and return the best matching language code.
/// Returns "en" if no match is found.
pub fn detect_system_language() -> String {
    // Try Windows API first
    #[cfg(target_os = "windows")]
    {
        if let Some(lang) = detect_windows_language() {
            return lang;
        }
    }

    // Try LANG / LC_ALL environment variables (Linux/macOS, sometimes set on Windows)
    for var in &["LANG", "LC_ALL", "LC_MESSAGES", "LANGUAGE"] {
        if let Ok(val) = std::env::var(var)
            && let Some(lang) = match_system_locale(&val)
        {
            return lang;
        }
    }

    "en".to_string()
}

#[cfg(target_os = "windows")]
fn detect_windows_language() -> Option<String> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    // Use GetUserDefaultLocaleName
    unsafe extern "system" {
        fn GetUserDefaultLocaleName(lp_locale_name: *mut u16, cch_locale_name: i32) -> i32;
    }

    let mut buf = [0u16; 85]; // LOCALE_NAME_MAX_LENGTH
    let len = unsafe { GetUserDefaultLocaleName(buf.as_mut_ptr(), buf.len() as i32) };
    if len > 0 {
        let os_str = OsString::from_wide(&buf[..((len - 1) as usize)]);
        if let Some(locale_str) = os_str.to_str() {
            return match_system_locale(locale_str);
        }
    }
    None
}

/// Match a system locale string (e.g. "en_US.UTF-8", "fr-FR", "ja_JP") to our supported languages.
fn match_system_locale(locale: &str) -> Option<String> {
    // Normalize: lowercase, replace _ with -
    let normalized = locale.to_lowercase().replace('_', "-");

    // Extract language part (before any '.' or '@')
    let lang_part = normalized.split('.').next().unwrap_or(&normalized);
    let lang_part = lang_part.split('@').next().unwrap_or(lang_part);

    // Try exact match first (e.g., "zh-cn" → "zh-CN")
    for &(code, _) in LANGUAGES {
        // Novelty translations are always an explicit user choice.
        if code.contains("-x-") || code == "fe" {
            continue;
        }
        if code.to_lowercase() == lang_part {
            return Some(code.to_string());
        }
    }

    // Try prefix match (e.g., "fr-fr" → "fr"). Generic Portuguese uses Brazil.
    let primary = lang_part.split('-').next().unwrap_or(lang_part);
    if primary == "pt" {
        return Some(
            if lang_part.split('-').skip(1).any(|part| part == "pt") {
                "pt-PT"
            } else {
                "pt-BR"
            }
            .to_string(),
        );
    }
    // For Chinese, check region to distinguish zh-CN vs zh-TW
    if primary == "zh" {
        let region = lang_part.split('-').nth(1).unwrap_or("");
        // Traditional Chinese regions / script tags
        if matches!(region, "tw" | "hk" | "mo" | "hant") {
            return Some("zh-TW".to_string());
        }
        return Some("zh-CN".to_string());
    }
    for &(code, _) in LANGUAGES {
        if code.contains("-x-") || code == "fe" {
            continue;
        }
        let code_primary = code.split('-').next().unwrap_or(code);
        if code_primary.to_lowercase() == primary {
            return Some(code.to_string());
        }
    }

    None
}

/// Before tag version 2, `be` was used for Bogan English, not Belarusian.
/// Only migrate saved preferences; a Belarusian system locale remains Belarusian.
pub fn migrate_saved_language(code: &str, tag_version: u32) -> String {
    if code == "be" && tag_version < 2 {
        "en-AU-x-bogan".to_string()
    } else if code == "pt" {
        // The original catalog uses Brazilian Portuguese (Salvar, Arquivo, Tela).
        "pt-BR".to_string()
    } else {
        code.to_string()
    }
}

/// Parse a simple key=value translation file.
/// Format: one `key=value` per line. Lines starting with `#` are comments. Empty lines ignored.
fn parse_translations(data: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, val)) = line.split_once('=') {
            map.insert(key.trim().to_string(), val.trim().to_string());
        }
    }
    map
}

/// Translation macro. Usage: `t!("menu.file")` or `t!("dialog.layer_name", name = "Layer 1")`
#[macro_export]
macro_rules! t {
    ($key:expr) => {
        $crate::i18n::translate($key)
    };
    ($key:expr, $($name:ident = $val:expr),+ $(,)?) => {{
        let mut s = $crate::i18n::translate($key);
        $(
            s = s.replace(concat!("{", stringify!($name), "}"), &format!("{}", $val));
        )+
        s
    }};
}

#[cfg(test)]
mod locale_tests {
    use super::*;

    #[test]
    fn real_system_locales_select_real_languages() {
        for (locale, expected) in [
            ("ru_RU.UTF-8", "ru"),
            ("ar_SA.UTF-8", "ar"),
            ("ar-EG", "ar"),
            ("uk-UA", "uk"),
            ("cs_CZ.UTF-8", "cs"),
            ("ro-RO", "ro"),
            ("be-BY", "be"),
            ("en-AU", "en"),
            ("en-AU-x-bogan", "en"),
            ("pt", "pt-BR"),
            ("pt_BR.UTF-8", "pt-BR"),
            ("pt-PT", "pt-PT"),
            ("pt-Latn-PT", "pt-PT"),
            ("ko-KR", "ko"),
            ("id-ID", "id"),
            ("vi-VN", "vi"),
            ("th-TH", "th"),
            ("hu-HU", "hu"),
        ] {
            assert_eq!(match_system_locale(locale).as_deref(), Some(expected));
        }
    }

    #[test]
    fn bogan_migration_only_applies_to_old_saved_preferences() {
        assert_eq!(migrate_saved_language("be", 1), "en-AU-x-bogan");
        assert_eq!(migrate_saved_language("be", 2), "be");
        assert_eq!(migrate_saved_language("en-AU-x-bogan", 2), "en-AU-x-bogan");
        assert_eq!(migrate_saved_language("", 1), "");
        assert_eq!(migrate_saved_language("ru", 1), "ru");
        assert_eq!(migrate_saved_language("pt", 1), "pt-BR");
        assert_eq!(migrate_saved_language("pt", 2), "pt-BR");
        assert_eq!(migrate_saved_language("pt-PT", 2), "pt-PT");
    }

    #[test]
    fn maintained_catalogs_cover_english_without_duplicate_keys() {
        let english = parse_translations(include_str!("../locales/en.txt"));
        let registered: Vec<_> = LANGUAGES.iter().map(|(code, _)| *code).collect();
        let embedded: Vec<_> = EMBEDDED_TRANSLATIONS
            .iter()
            .map(|(code, _)| *code)
            .collect();
        assert_eq!(
            registered, embedded,
            "language picker and embedded catalogs differ"
        );
        for (code, data) in EMBEDDED_TRANSLATIONS {
            let translated = parse_translations(data);
            let entries = data
                .lines()
                .filter(|line| !line.trim().starts_with('#') && line.contains('='))
                .count();
            assert_eq!(entries, translated.len(), "duplicate translation keys");
            assert_eq!(
                translated.len(),
                english.len(),
                "{code}: incomplete catalog"
            );
            for key in english.keys() {
                assert!(
                    translated.get(key).is_some_and(|value| !value.is_empty()),
                    "{key}"
                );
            }
        }
    }
}

#[cfg(test)]
mod spelling_alias_tests {
    #[test]
    fn australian_spelling_preserves_legacy_label_identities() {
        super::init();
        let guard = super::I18N.lock().unwrap();
        let state = guard.as_ref().unwrap();
        for (label, key) in [
            ("Color", "gradient_mode.color"),
            ("color", "gradient_mode.color"),
            ("Center", "ctx.text.effects.outline.center"),
            ("center", "ctx.text.effects.outline.center"),
            ("colors", "ui.colors"),
        ] {
            assert_eq!(state.english_keys.get(label).map(String::as_str), Some(key));
        }
        assert_eq!(state.translations["en"]["menu.color"], "Colour");
        assert_eq!(
            state.translations["en"]["tool.color_picker"],
            "Colour Picker"
        );
    }
}
