// Google Fonts picker (web only).
//
// ab_glyph (via ttf-parser) only understands raw TTF/OTF (sfnt) data, not
// the WOFF2 that fonts.googleapis.com's CSS API serves to browsers, and
// there's no pure-Rust WOFF2 decoder wired into this build. Instead of
// fighting that, this fetches the actual TTF straight from the
// github.com/google/fonts source repo, which hosts the original files the
// CSS API is built from — no API key needed, unlike the official Google
// Fonts Developer API.
//
// This is best-effort: the repo's layout could change, so failures are
// surfaced to the user rather than silently swallowed.

use std::cell::RefCell;
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

/// Curated list of well-known Google Fonts: (display name, repo folder slug).
/// Slugs are Google's own naming convention for these folders (lowercase,
/// no spaces/punctuation) and have stayed stable for years.
pub const CURATED: &[(&str, &str)] = &[
    ("Roboto", "roboto"),
    ("Open Sans", "opensans"),
    ("Lato", "lato"),
    ("Montserrat", "montserrat"),
    ("Oswald", "oswald"),
    ("Raleway", "raleway"),
    ("Poppins", "poppins"),
    ("Merriweather", "merriweather"),
    ("Playfair Display", "playfairdisplay"),
    ("Nunito", "nunito"),
    ("Nunito Sans", "nunitosans"),
    ("Ubuntu", "ubuntu"),
    ("Rubik", "rubik"),
    ("Karla", "karla"),
    ("Mulish", "mulish"),
    ("Work Sans", "worksans"),
    ("Inter", "inter"),
    ("Quicksand", "quicksand"),
    ("Source Sans 3", "sourcesans3"),
    ("PT Sans", "ptsans"),
    ("PT Serif", "ptserif"),
    ("Libre Baskerville", "librebaskerville"),
    ("Crimson Text", "crimsontext"),
    ("Bitter", "bitter"),
    ("Archivo", "archivo"),
    ("Barlow", "barlow"),
    ("Cabin", "cabin"),
    ("Josefin Sans", "josefinsans"),
    ("Comfortaa", "comfortaa"),
    ("Dancing Script", "dancingscript"),
    ("Pacifico", "pacifico"),
    ("Inconsolata", "inconsolata"),
    ("Roboto Mono", "robotomono"),
    ("Roboto Condensed", "robotocondensed"),
    ("Fira Sans", "firasans"),
    ("Exo 2", "exo2"),
];

#[derive(Clone, Debug)]
pub enum FetchStatus {
    Loading,
    Done,
    Failed(String),
}

thread_local! {
    static STATUS: RefCell<HashMap<String, FetchStatus>> = RefCell::new(HashMap::new());
}

pub fn status_of(family: &str) -> Option<FetchStatus> {
    STATUS.with(|s| s.borrow().get(family).cloned())
}

fn set_status(family: &str, status: FetchStatus) {
    STATUS.with(|s| s.borrow_mut().insert(family.to_string(), status));
}

/// Kick off fetching `family` (a display name from [`CURATED`]) in the
/// background. Poll [`status_of`] to show progress; once the font registers
/// with `ops::text::custom_fonts`, the Text tool's font list picks it up on
/// its own (see the generation-counter check in `text_style_controls.rs`).
pub fn fetch(family: &'static str, slug: &'static str, ctx: egui::Context) {
    set_status(family, FetchStatus::Loading);
    wasm_bindgen_futures::spawn_local(async move {
        let result = fetch_ttf_bytes(slug).await;
        match result {
            Ok(bytes) => {
                match crate::ops::text::custom_fonts::register_from_bytes(family.to_string(), bytes)
                {
                    Ok(()) => set_status(family, FetchStatus::Done),
                    Err(e) => set_status(family, FetchStatus::Failed(e)),
                }
            }
            Err(e) => set_status(family, FetchStatus::Failed(e)),
        }
        ctx.request_repaint();
    });
}

async fn fetch_json(url: &str) -> Result<serde_json::Value, String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|_| format!("network error fetching {url}"))?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| "unexpected fetch response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let text_promise = resp.text().map_err(|_| "no response body".to_string())?;
    let text_value = JsFuture::from(text_promise)
        .await
        .map_err(|_| "failed to read response body".to_string())?;
    let text = text_value.as_string().ok_or("response body not text")?;
    serde_json::from_str(&text).map_err(|e| format!("bad JSON from GitHub API: {e}"))
}

async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|_| format!("network error fetching {url}"))?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| "unexpected fetch response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let buf_promise = resp
        .array_buffer()
        .map_err(|_| "no response body".to_string())?;
    let buf = JsFuture::from(buf_promise)
        .await
        .map_err(|_| "failed to read response body".to_string())?;
    let array = js_sys::Uint8Array::new(&buf);
    Ok(array.to_vec())
}

/// Find a Regular-weight TTF/OTF in a GitHub contents-API directory listing.
fn find_regular_font(entries: &[serde_json::Value]) -> Option<String> {
    entries.iter().find_map(|entry| {
        let name = entry.get("name")?.as_str()?;
        let lower = name.to_lowercase();
        if (lower.ends_with(".ttf") || lower.ends_with(".otf")) && lower.contains("regular") {
            entry.get("download_url")?.as_str().map(|s| s.to_string())
        } else {
            None
        }
    })
}

async fn fetch_ttf_bytes(slug: &str) -> Result<Vec<u8>, String> {
    // Try each license directory the repo organizes fonts under — the slug
    // alone doesn't say which one a given family lives in.
    const LICENSE_DIRS: &[&str] = &["ofl", "apache", "ufl"];
    for license in LICENSE_DIRS {
        let listing_url =
            format!("https://api.github.com/repos/google/fonts/contents/{license}/{slug}");
        let Ok(entries) = fetch_json(&listing_url).await else {
            continue;
        };
        let Some(array) = entries.as_array() else {
            continue;
        };

        // Most modern families are variable fonts at the top level, with
        // flat static Regular instances kept in a `static/` subfolder.
        if let Some(url) = find_regular_font(array) {
            return fetch_bytes(&url).await;
        }
        let static_url =
            format!("https://api.github.com/repos/google/fonts/contents/{license}/{slug}/static");
        if let Ok(static_entries) = fetch_json(&static_url).await
            && let Some(static_array) = static_entries.as_array()
            && let Some(url) = find_regular_font(static_array)
        {
            return fetch_bytes(&url).await;
        }
    }
    Err("Could not find this font in the Google Fonts repository".to_string())
}
