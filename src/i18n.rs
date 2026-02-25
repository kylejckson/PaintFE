//! Internationalization (i18n) module for PaintFE.
//!
//! Uses a simple key→string HashMap loaded at runtime from embedded translation data.
//! The `t!("key")` macro looks up the current language, falling back to English.
//! Language can be switched at runtime via `set_language()`.

use std::collections::HashMap;
use std::sync::Mutex;

/// Global translation state.
static I18N: Mutex<Option<I18nState>> = Mutex::new(None);

struct I18nState {
    current_lang: String,
    /// lang_code → (key → translated_string)
    translations: HashMap<String, HashMap<String, String>>,
}

/// Supported languages: (code, native_name)
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("es", "Español"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("pt", "Português"),
    ("it", "Italiano"),
    ("ja", "日本語"),
    ("zh-CN", "中文(简体)"),
    ("zh-TW", "中文(繁體)"),
    ("ru", "Русский"),
    ("nl", "Nederlands"),
    ("pl", "Polski"),
    ("tr", "Türkçe"),
    ("be", "Bogan English"),
    ("fe", "Fancy English"),
];

/// Initialize the i18n system with embedded translations.
/// Call once at startup.
pub fn init() {
    let mut translations: HashMap<String, HashMap<String, String>> = HashMap::new();

    // Parse all embedded translation files
    translations.insert(
        "en".to_string(),
        parse_translations(include_str!("../locales/en.txt")),
    );
    translations.insert(
        "es".to_string(),
        parse_translations(include_str!("../locales/es.txt")),
    );
    translations.insert(
        "fr".to_string(),
        parse_translations(include_str!("../locales/fr.txt")),
    );
    translations.insert(
        "de".to_string(),
        parse_translations(include_str!("../locales/de.txt")),
    );
    translations.insert(
        "pt".to_string(),
        parse_translations(include_str!("../locales/pt.txt")),
    );
    translations.insert(
        "it".to_string(),
        parse_translations(include_str!("../locales/it.txt")),
    );
    translations.insert(
        "ja".to_string(),
        parse_translations(include_str!("../locales/ja.txt")),
    );
    translations.insert(
        "zh-CN".to_string(),
        parse_translations(include_str!("../locales/zh-CN.txt")),
    );
    translations.insert(
        "zh-TW".to_string(),
        parse_translations(include_str!("../locales/zh-TW.txt")),
    );
    translations.insert(
        "ru".to_string(),
        parse_translations(include_str!("../locales/ru.txt")),
    );
    translations.insert(
        "nl".to_string(),
        parse_translations(include_str!("../locales/nl.txt")),
    );
    translations.insert(
        "pl".to_string(),
        parse_translations(include_str!("../locales/pl.txt")),
    );
    translations.insert(
        "tr".to_string(),
        parse_translations(include_str!("../locales/tr.txt")),
    );
    translations.insert(
        "be".to_string(),
        parse_translations(include_str!("../locales/be.txt")),
    );
    translations.insert(
        "fe".to_string(),
        parse_translations(include_str!("../locales/fe.txt")),
    );

    let state = I18nState {
        current_lang: "en".to_string(),
        translations,
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

/// Look up a translation key. Returns the translated string if found,
/// or falls back to English, or returns the key itself as last resort.
pub fn translate(key: &str) -> String {
    if let Ok(guard) = I18N.lock()
        && let Some(ref state) = *guard
    {
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
        if code.to_lowercase() == lang_part {
            return Some(code.to_string());
        }
    }

    // Try prefix match (e.g., "fr-fr" → "fr", "pt-br" → "pt")
    let primary = lang_part.split('-').next().unwrap_or(lang_part);
    for &(code, _) in LANGUAGES {
        let code_primary = code.split('-').next().unwrap_or(code);
        if code_primary.to_lowercase() == primary {
            return Some(code.to_string());
        }
    }

    None
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
