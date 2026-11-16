use super::*;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
static DOWNLOAD_STOP: AtomicBool = AtomicBool::new(false);
static DOWNLOAD_ACTIVE: AtomicUsize = AtomicUsize::new(0);
struct DownloadGuard;
impl Drop for DownloadGuard {
    fn drop(&mut self) {
        DOWNLOAD_ACTIVE.fetch_sub(1, Ordering::SeqCst);
    }
}
/// Cancel outstanding explicit downloads before the application exits.
pub fn shutdown_downloads() {
    DOWNLOAD_STOP.store(true, Ordering::SeqCst);
    let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while DOWNLOAD_ACTIVE.load(Ordering::SeqCst) != 0 && std::time::Instant::now() < until {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
static INSTALL_LOCK: Mutex<()> = Mutex::new(());

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn hash(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
fn unhex<const N: usize>(s: &str) -> Result<[u8; N], String> {
    if s.len() != N * 2 || !s.is_ascii() {
        return Err("Invalid signature encoding".into());
    }
    let mut b = [0; N];
    for (i, v) in b.iter_mut().enumerate() {
        *v = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|_| "Invalid signature encoding")?;
    }
    Ok(b)
}
pub fn verified_payload(bytes: &[u8], allow_local: bool) -> Result<(String, bool), String> {
    if bytes.len() > MAX_CATALOGUE {
        return Err("Manifest is too large".into());
    }
    let envelope: Envelope =
        serde_json::from_slice(bytes).map_err(|_| "Invalid plugin metadata")?;
    if envelope.signature.is_empty() && allow_local {
        return Ok((envelope.payload, false));
    }
    let key = VerifyingKey::from_bytes(&unhex::<32>(include_str!("catalogue-key.hex").trim())?)
        .map_err(|_| "Invalid catalogue key")?;
    key.verify_strict(
        envelope.payload.as_bytes(),
        &Signature::from_bytes(&unhex::<64>(&envelope.signature)?),
    )
    .map_err(|_| "Plugin signature verification failed")?;
    Ok((envelope.payload, true))
}
pub fn read_limited(path: &Path, max: usize) -> Result<Vec<u8>, String> {
    let f = fs::File::open(path).map_err(|e| e.to_string())?;
    if f.metadata().map_err(|e| e.to_string())?.len() > max as u64 {
        return Err("File exceeds the plugin size limit".into());
    }
    let mut b = Vec::new();
    f.take(max as u64 + 1)
        .read_to_end(&mut b)
        .map_err(|e| e.to_string())?;
    if b.len() > max {
        return Err("File exceeds the plugin size limit".into());
    }
    Ok(b)
}
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("Invalid plugin path")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let tmp = parent.join(format!(".write-{}", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut f = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)
            .map_err(|e| e.to_string())?;
        use std::io::Write;
        f.write_all(bytes)
            .and_then(|_| f.sync_all())
            .map_err(|e| e.to_string())?;
        fs::rename(&tmp, path).map_err(|e| e.to_string())
    })();
    let _ = fs::remove_file(tmp);
    result
}
pub fn root() -> PathBuf {
    crate::services::storage::data_dir()
        .join("plugins")
        .join("scripts")
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PackSettings {
    pub disabled: bool,
    pub disabled_effects: BTreeSet<String>,
    pub favourite: bool,
    pub alias: String,
    pub menus: BTreeMap<String, String>,
    pub names: BTreeMap<String, String>,
    pub values: BTreeMap<String, Values>,
    pub current: String,
    pub previous: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Preferences {
    pub packs: BTreeMap<String, PackSettings>,
}
impl Preferences {
    pub fn load(at: &Path) -> Self {
        read_limited(&at.join("preferences.json"), 1024 * 1024)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }
    pub fn save(&self, at: &Path) -> Result<(), String> {
        let b = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        if b.len() > 1024 * 1024 {
            return Err("Plugin preferences exceed the storage limit".into());
        }
        atomic_write(&at.join("preferences.json"), &b)
    }
}
#[derive(Clone, Debug)]
pub struct Installed {
    pub manifest: Manifest,
    pub directory: PathBuf,
    pub signed: bool,
}
pub fn installed(at: &Path, prefs: &Preferences) -> Vec<Installed> {
    prefs
        .packs
        .iter()
        .filter_map(|(id, p)| {
            if !identifier(id) || !version(&p.current) {
                return None;
            }
            let directory = at.join("installed").join(id).join(&p.current);
            let bytes = read_limited(&directory.join("manifest.json"), MAX_CATALOGUE).ok()?;
            let (payload, signed) = verified_payload(&bytes, true).ok()?;
            let manifest: Manifest = serde_json::from_str(&payload).ok()?;
            manifest.validate().ok()?;
            if manifest.id != *id || manifest.version != p.current {
                return None;
            }
            Some(Installed {
                manifest,
                directory,
                signed,
            })
        })
        .collect()
}
pub fn effect_source(pack: &Installed, effect: &Effect) -> Result<String, String> {
    pack.manifest.validate()?;
    let file = pack
        .manifest
        .files
        .iter()
        .find(|f| f.path == effect.script)
        .ok_or("Effect source is missing")?;
    let path = pack.directory.join(&file.path);
    if fs::symlink_metadata(&path)
        .map_err(|e| e.to_string())?
        .file_type()
        .is_symlink()
    {
        return Err("Plugin resources cannot be links".into());
    }
    let bytes = read_limited(&path, 256 * 1024)?;
    if bytes.len() as u64 != file.size || hash(&bytes) != file.sha256 {
        return Err("Plugin source has changed. Import a new local copy before running it.".into());
    }
    String::from_utf8(bytes).map_err(|_| "Plugin source must be UTF-8".into())
}
pub fn install(
    bytes: &[u8],
    at: &Path,
    allow_local: bool,
    expected: Option<&Listing>,
) -> Result<Manifest, String> {
    let _lock = INSTALL_LOCK
        .lock()
        .map_err(|_| "Plugin installation is busy")?;
    if bytes.len() > MAX_PACKAGE {
        return Err("Plugin package is too large".into());
    }
    if let Some(e) = expected
        && (bytes.len() as u64 != e.size || hash(bytes) != e.sha256)
    {
        return Err("Downloaded package does not match the catalogue".into());
    }
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|_| "Invalid plugin archive")?;
    if archive.len() > 65 {
        return Err("Too many plugin files".into());
    }
    let envelope = {
        let mut f = archive
            .by_name("manifest.json")
            .map_err(|_| "Plugin manifest is missing")?;
        let mut b = Vec::new();
        f.by_ref()
            .take(MAX_CATALOGUE as u64 + 1)
            .read_to_end(&mut b)
            .map_err(|e| e.to_string())?;
        b
    };
    let (payload, signed) = verified_payload(&envelope, allow_local)?;
    let manifest: Manifest =
        serde_json::from_str(&payload).map_err(|_| "Invalid plugin manifest")?;
    manifest.validate()?;
    let old_preferences = Preferences::load(at);
    if !signed
        && installed(at, &old_preferences)
            .iter()
            .any(|p| p.manifest.id == manifest.id && p.signed)
    {
        return Err("An unsigned plugin cannot replace a community-signed pack. Save it as a separate local copy.".into());
    }
    if let Some(e) = expected
        && (e.manifest.id != manifest.id
            || e.manifest.version != manifest.version
            || serde_json::to_value(&e.manifest).ok() != serde_json::to_value(&manifest).ok())
    {
        return Err("Package identity does not match its listing".into());
    }
    if archive.len() != manifest.files.len() + 1 {
        return Err("Unexpected files in plugin package".into());
    }
    let mut seen = BTreeSet::new();
    for i in 0..archive.len() {
        let f = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = f.name().map_err(|e| e.to_string())?.into_owned();
        if !seen.insert(name.clone())
            || !resource_path(&name)
            || f.is_dir()
            || f.unix_mode().is_some_and(|m| m & 0o170000 == 0o120000)
        {
            return Err("Unsafe plugin archive entry".into());
        }
    }
    fs::create_dir_all(at).map_err(|e| e.to_string())?;
    let stage = at.join(format!(".install-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&stage).map_err(|e| e.to_string())?;
    let result = (|| {
        for file in &manifest.files {
            let mut f = archive
                .by_name(&file.path)
                .map_err(|_| "Plugin resource is missing")?;
            if f.size() != file.size {
                return Err("Plugin resource size mismatch".into());
            }
            let mut b = Vec::new();
            f.by_ref()
                .take(file.size + 1)
                .read_to_end(&mut b)
                .map_err(|e| e.to_string())?;
            if b.len() as u64 != file.size || hash(&b) != file.sha256 {
                return Err("Plugin resource verification failed".into());
            }
            if file.path.ends_with(".rhai") {
                if b.len() > 256 * 1024 || std::str::from_utf8(&b).is_err() {
                    return Err("Invalid script source".into());
                }
            } else if file.path.ends_with(".png") {
                validate_image(&b)?;
            } else if file.path.ends_with(".zip")
                && manifest.kind == PackKind::Icons
                && file.path == manifest.appearance
            {
                // The icon importer applies its own bounded archive validation.
            } else if !(file.path.ends_with(".md")
                || file.path.ends_with(".txt")
                || file.path == "LICENSE")
            {
                return Err("Unsupported plugin resource type".into());
            }
            let dest = stage.join(&file.path);
            fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
            fs::write(dest, b).map_err(|e| e.to_string())?;
        }
        if manifest.kind == PackKind::Icons {
            validate_icon_archive(&stage.join(&manifest.appearance))?;
            crate::config::icon_packs::IconPack::load(&stage.join(&manifest.appearance))?;
        }
        if manifest.kind == PackKind::Theme {
            let text = read_limited(&stage.join(&manifest.appearance), 512 * 1024)?;
            std::str::from_utf8(&text).map_err(|_| "Invalid theme text")?;
        }
        fs::write(stage.join("manifest.json"), &envelope).map_err(|e| e.to_string())?;
        let dest = at
            .join("installed")
            .join(&manifest.id)
            .join(&manifest.version);
        fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;
        let root = at.canonicalize().map_err(|e| e.to_string())?;
        if !dest
            .parent()
            .unwrap()
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(&root)
        {
            return Err("Plugin directory is outside its storage root".into());
        }
        let backup = if dest.exists() {
            if fs::symlink_metadata(&dest)
                .map_err(|e| e.to_string())?
                .file_type()
                .is_symlink()
            {
                return Err("Plugin directories cannot be links".into());
            }
            let previous = read_limited(&dest.join("manifest.json"), MAX_CATALOGUE)
                .and_then(|b| verified_payload(&b, true))
                .and_then(|(payload, _)| {
                    serde_json::from_str::<Manifest>(&payload).map_err(|e| e.to_string())
                });
            match previous {
                Ok(previous)
                    if serde_json::to_value(&previous).ok()
                        != serde_json::to_value(&manifest).ok() =>
                {
                    return Err("A changed plugin must have a new version.".into());
                }
                Err(_) if !signed => {
                    return Err("Use the signed community pack to repair damaged metadata.".into());
                }
                _ => {}
            }
            let backup = at.join(format!(".repair-{}", uuid::Uuid::new_v4()));
            fs::rename(&dest, &backup).map_err(|e| e.to_string())?;
            Some(backup)
        } else {
            None
        };
        if let Err(e) = fs::rename(&stage, &dest) {
            if let Some(old) = &backup {
                let _ = fs::rename(old, &dest);
            }
            return Err(e.to_string());
        }
        let mut prefs = Preferences::load(at);
        let p = prefs.packs.entry(manifest.id.clone()).or_default();
        if p.current != manifest.version {
            p.previous = p.current.clone();
        }
        p.current = manifest.version.clone();
        if let Err(e) = prefs.save(at) {
            let _ = fs::remove_dir_all(&dest);
            if let Some(old) = &backup
                && fs::rename(old, &dest).is_err()
            {
                return Err(format!(
                    "{e}. The previous pack remains in {}",
                    old.display()
                ));
            }
            return Err(e);
        }
        if let Some(old) = backup {
            let _ = fs::remove_dir_all(old);
        }
        Ok(manifest.clone())
    })();
    let _ = fs::remove_dir_all(stage);
    result
}
pub fn validate_image(bytes: &[u8]) -> Result<(), String> {
    let mut reader =
        image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(1600);
    limits.max_image_height = Some(1200);
    limits.max_alloc = Some(16 * 1024 * 1024);
    reader.limits(limits);
    reader
        .decode()
        .map(|_| ())
        .map_err(|_| "Plugin artwork is invalid or too large".into())
}
pub fn catalogue(bytes: &[u8]) -> Result<Catalogue, String> {
    let (payload, _) = verified_payload(bytes, false)?;
    let c: Catalogue = serde_json::from_str(&payload).map_err(|_| "Invalid catalogue")?;
    c.validate()?;
    Ok(c)
}
pub fn cached(at: &Path) -> Option<Catalogue> {
    read_limited(&at.join("catalogue.json"), MAX_CATALOGUE)
        .ok()
        .and_then(|b| catalogue(&b).ok())
}

/// No calls originate from startup or script code. Callers require an explicit user action.
pub fn fetch(url: &str, max: usize, cancel: &AtomicBool) -> Result<Vec<u8>, String> {
    DOWNLOAD_ACTIVE.fetch_add(1, Ordering::SeqCst);
    let _download = DownloadGuard;
    if DOWNLOAD_STOP.load(Ordering::SeqCst) {
        return Err("Download cancelled".into());
    }
    if url != CATALOGUE_URL && !allowed_url(url) {
        return Err("This download is outside the community repository".into());
    }
    let temp =
        std::env::temp_dir().join(format!("paintfe-plugin-download-{}", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut cmd = super::super::optional_packs::process::hidden_command(
            &super::super::optional_packs::process::curl_path(),
        );
        cmd.args([
            "--disable",
            "--fail",
            "--silent",
            "--location",
            "--max-redirs",
            "3",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            "15",
            "--max-time",
            "90",
            "--max-filesize",
            &max.to_string(),
            "--output",
        ])
        .arg(&temp)
        .arg(url)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
        let mut child = cmd.spawn().map_err(
            |_| "Could not start the download. You can import a pack downloaded in your browser.",
        )?;
        loop {
            if cancel.load(Ordering::Relaxed) || DOWNLOAD_STOP.load(Ordering::SeqCst) {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Download cancelled".into());
            }
            match child.try_wait() {
                Ok(Some(s)) => {
                    if !s.success() {
                        return Err("Download failed. Check your connection and try again.".into());
                    }
                    break;
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(25)),
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(e.to_string());
                }
            }
        }
        read_limited(&temp, max)
    })();
    let _ = fs::remove_file(temp);
    result
}

/// Removes only this validated pack's managed resources. Local copies have other IDs.
pub fn uninstall(id: &str, at: &Path) -> Result<(), String> {
    if !identifier(id) {
        return Err("Invalid plugin identity".into());
    }
    let _guard = INSTALL_LOCK
        .lock()
        .map_err(|_| "Plugin installation is busy")?;
    let root = at.canonicalize().map_err(|e| e.to_string())?;
    let dir = at.join("installed").join(id);
    let backup = if dir.exists() {
        if fs::symlink_metadata(&dir)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
            || !dir
                .canonicalize()
                .map_err(|e| e.to_string())?
                .starts_with(&root)
        {
            return Err("Unsafe plugin storage directory".into());
        }
        let backup = at.join(format!(".remove-{}", uuid::Uuid::new_v4()));
        fs::rename(&dir, &backup).map_err(|e| e.to_string())?;
        Some(backup)
    } else {
        None
    };
    let mut prefs = Preferences::load(at);
    let settings = prefs.packs.entry(id.to_string()).or_default();
    settings.current.clear();
    settings.previous.clear();
    if let Err(e) = prefs.save(at) {
        if let Some(old) = &backup {
            let _ = fs::rename(old, &dir);
        }
        return Err(e);
    }
    if let Some(old) = backup {
        fs::remove_dir_all(old).map_err(|e| {
            format!("Plugin removed, but its downloaded files could not be cleaned up: {e}")
        })?;
    }
    Ok(())
}

/// Fit preview pixels into a bounded texture without changing their aspect ratio.
pub fn thumbnail(image: &image::RgbaImage, max_width: u32, max_height: u32) -> image::RgbaImage {
    let scale = (max_width as f64 / image.width().max(1) as f64)
        .min(max_height as f64 / image.height().max(1) as f64)
        .min(1.);
    let width = (image.width() as f64 * scale)
        .round()
        .clamp(1., max_width.max(1) as f64) as u32;
    let height = (image.height() as f64 * scale)
        .round()
        .clamp(1., max_height.max(1) as f64) as u32;
    image::imageops::thumbnail(image, width, height)
}

/// Community downloads have tighter decoded budgets than manual icon imports.
pub fn validate_icon_archive(path: &Path) -> Result<(), String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|_| "Invalid icon archive")?;
    if archive.len() > 1024 {
        return Err("Too many icon archive entries".into());
    }
    let mut encoded = 0u64;
    let mut decoded = 0u64;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|_| "Invalid icon archive entry")?;
        if entry.enclosed_name().is_none()
            || entry.unix_mode().is_some_and(|m| m & 0o170000 == 0o120000)
        {
            return Err("Unsafe icon archive path".into());
        }
        if entry.is_dir() {
            continue;
        }
        encoded = encoded
            .checked_add(entry.size())
            .ok_or("Icon archive too large")?;
        if entry.size() > 2 * 1024 * 1024 || encoded > 16 * 1024 * 1024 {
            return Err("Icon archive exceeds its expanded-size limit".into());
        }
        if !entry
            .name()
            .map_err(|_| "Invalid icon archive name")?
            .ends_with(".png")
        {
            continue;
        }
        let expected = entry.size();
        let mut bytes = Vec::new();
        entry
            .by_ref()
            .take(2 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| "Invalid icon image")?;
        if bytes.len() as u64 != expected || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Err("Invalid PNG icon".into());
        }
        let (w, h) =
            image::ImageReader::with_format(std::io::Cursor::new(&bytes), image::ImageFormat::Png)
                .into_dimensions()
                .map_err(|_| "Invalid PNG dimensions")?;
        decoded = decoded
            .checked_add(u64::from(w) * u64::from(h) * 4)
            .ok_or("Decoded icon budget exceeded")?;
        if w == 0 || h == 0 || w > 1024 || h > 1024 || decoded > 64 * 1024 * 1024 {
            return Err("Decoded icon budget exceeded".into());
        }
    }
    Ok(())
}
