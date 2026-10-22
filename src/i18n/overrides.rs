//! Personal labels overlay bundled translations; identifiers remain unchanged.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

static REVISION: AtomicU64 = AtomicU64::new(0);
const MAX_FILE: usize = 1024 * 1024;

#[derive(Clone, Debug)]
pub struct LabelEntry {
    pub key: String,
    pub original: String,
    pub custom: Option<String>,
}

pub fn label_revision() -> u64 {
    REVISION.load(Ordering::Relaxed)
}

pub fn label_entries(language: &str) -> Vec<LabelEntry> {
    let guard = I18N.lock().unwrap_or_else(|e| e.into_inner());
    let Some(state) = guard.as_ref() else {
        return Vec::new();
    };
    let mut entries: Vec<_> = state.translations["en"]
        .iter()
        .map(|(key, english)| LabelEntry {
            key: key.clone(),
            original: state
                .translations
                .get(language)
                .and_then(|m| m.get(key))
                .unwrap_or(english)
                .clone(),
            custom: state
                .overrides
                .get(language)
                .and_then(|m| m.get(key))
                .cloned(),
        })
        .collect();
    entries.sort_by(|a, b| a.key.cmp(&b.key));
    entries
}

fn tokens(text: &str) -> Result<Vec<&str>, String> {
    let mut result = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('{') {
        rest = &rest[start..];
        let end = rest
            .find('}')
            .ok_or_else(|| "Unclosed placeholder".to_string())?;
        result.push(&rest[..=end]);
        rest = &rest[end + 1..];
    }
    result.sort_unstable();
    Ok(result)
}

pub fn validate_label(original: &str, replacement: &str) -> Result<(), String> {
    if replacement.trim().is_empty() || replacement.len() > 4096 {
        return Err(crate::t!("labels.invalid_length"));
    }
    if replacement
        .chars()
        .any(|c| c.is_control() && !matches!(c, '\n' | '\t'))
    {
        return Err(crate::t!("labels.invalid_text"));
    }
    if tokens(original)? != tokens(replacement)? {
        return Err(crate::t!("labels.placeholders"));
    }
    Ok(())
}

pub fn parse_label_overrides(
    language: &str,
    json: &str,
) -> Result<BTreeMap<String, String>, String> {
    if json.len() > MAX_FILE {
        return Err(crate::t!("labels.too_large"));
    }
    let mut values: BTreeMap<String, String> =
        serde_json::from_str(json).map_err(|e| e.to_string())?;
    let entries = label_entries(language);
    for (key, value) in &values {
        let original = entries
            .iter()
            .find(|entry| &entry.key == key)
            .ok_or_else(|| format!("{}: {key}", crate::t!("labels.unknown_key")))?;
        validate_label(&original.original, value)?;
    }
    values.retain(|key, value| {
        entries
            .iter()
            .any(|entry| &entry.key == key && &entry.original != value)
    });
    Ok(values)
}

fn known_language(language: &str) -> Result<(), String> {
    if LANGUAGES.iter().any(|(code, _)| *code == language) {
        Ok(())
    } else {
        Err("Unknown language".into())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn path(language: &str) -> std::path::PathBuf {
    crate::services::storage::data_dir()
        .join("labels")
        .join(format!("{language}.json"))
}

pub fn save_label_overrides(
    language: &str,
    values: BTreeMap<String, String>,
) -> Result<(), String> {
    known_language(language)?;
    let json = serde_json::to_string_pretty(&values).map_err(|e| e.to_string())?;
    let values = parse_label_overrides(language, &json)?;
    let json = serde_json::to_string_pretty(&values).map_err(|e| e.to_string())?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::io::Write;
        let target = path(language);
        let directory = target.parent().unwrap();
        std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        let temp = directory.join(format!(".labels-{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> std::io::Result<()> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            file.write_all(json.as_bytes())?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp, &target)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result.map_err(|e| e.to_string())?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let storage = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .ok_or_else(|| "Browser storage unavailable".to_string())?;
        storage
            .set_item(&format!("paintfe-labels-{language}"), &json)
            .map_err(|e| format!("{e:?}"))?;
    }
    apply(language, values);
    Ok(())
}

fn apply(language: &str, values: BTreeMap<String, String>) {
    if let Some(state) = I18N.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        state.overrides.insert(language.to_owned(), values);
    }
    REVISION.fetch_add(1, Ordering::Relaxed);
}

/// Load once after storage initialization, never during translation lookup.
pub fn load_label_overrides() -> Vec<String> {
    let mut errors = Vec::new();
    for &(language, _) in LANGUAGES {
        #[cfg(not(target_arch = "wasm32"))]
        let content = (|| -> Result<Option<String>, String> {
            let path = path(language);
            match std::fs::File::open(path) {
                Ok(file) => {
                    use std::io::Read;
                    let mut text = String::new();
                    file.take(MAX_FILE as u64 + 1)
                        .read_to_string(&mut text)
                        .map_err(|e| e.to_string())?;
                    Ok(Some(text))
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e.to_string()),
            }
        })();
        #[cfg(target_arch = "wasm32")]
        let content: Result<Option<String>, String> = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .map(|s| {
                s.get_item(&format!("paintfe-labels-{language}"))
                    .map_err(|e| format!("{e:?}"))
            })
            .unwrap_or(Ok(None));
        match content {
            Ok(Some(json)) => {
                // Removed keys and changed placeholders must not invalidate other customizations after updates.
                let parsed = (|| -> Result<BTreeMap<String, String>, String> {
                    if json.len() > MAX_FILE {
                        return Err("Label file exceeds 1 MiB".into());
                    }
                    let mut values: BTreeMap<String, String> =
                        serde_json::from_str(&json).map_err(|e| e.to_string())?;
                    let entries = label_entries(language);
                    values.retain(|key, value| {
                        entries
                            .iter()
                            .any(|e| &e.key == key && validate_label(&e.original, value).is_ok())
                    });
                    Ok(values)
                })();
                match parsed {
                    Ok(values) => apply(language, values),
                    Err(e) => errors.push(format!("{language}: {e}")),
                }
            }
            Ok(None) => {}
            Err(e) => errors.push(format!("{language}: {e}")),
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn placeholders_preserve_names_and_multiplicity() {
        assert!(validate_label("{name}: {count}", "{count} — {name}").is_ok());
        assert!(validate_label("{name}: {count}", "{name}").is_err());
        assert!(validate_label("{name}", "{name} {name}").is_err());
        assert!(validate_label("Save", "\0bad").is_err());
    }
    #[test]
    fn personal_labels_do_not_change_stable_english_lookup() {
        // The other i18n tests share global language state: this target runs serially.
        init();
        apply(
            "en",
            BTreeMap::from([
                ("menu.view".into(), "Panels".into()),
                ("keybind_category.view".into(), "Navigation".into()),
            ]),
        );
        assert_eq!(translate("menu.view"), "Panels");
        assert_eq!(translate_english("View"), "Navigation");
        assert_eq!(
            label_entries("en")
                .iter()
                .find(|e| e.key == "menu.view")
                .unwrap()
                .original,
            "View"
        );
        apply("en", BTreeMap::new());
        assert_eq!(translate("menu.view"), "View");
    }
}
