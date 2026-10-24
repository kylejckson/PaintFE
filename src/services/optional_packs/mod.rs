//! User-requested, release-pinned optional packs. No background network activity.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

pub mod avif;
mod process;
pub use process::encode_jxl;
#[cfg(test)]
mod tests;

pub const MAX_PACK: u64 = 96 * 1024 * 1024;
const MAX_MANIFEST: u64 = 32 * 1024;
static LOCK: Mutex<()> = Mutex::new(());
static REQUEST_JXL: AtomicBool = AtomicBool::new(false);
static REQUEST_AVIF: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pack {
    Cjk,
    Jxl,
    Avif,
    Heic,
}
impl Pack {
    // Retain the old enum value only to reject obsolete HEIC packs explicitly.
    pub const ALL: [Self; 3] = [Self::Cjk, Self::Jxl, Self::Avif];
    pub fn id(self) -> &'static str {
        match self {
            Self::Cjk => "cjk",
            Self::Jxl => "jxl",
            Self::Avif => "avif",
            Self::Heic => "heic",
        }
    }
    pub fn title(self) -> String {
        crate::t!(match self {
            Self::Cjk => "packs.cjk",
            Self::Jxl => "packs.jxl",
            Self::Avif => "packs.avif",
            Self::Heic => "packs.heic",
        })
    }
    pub fn payload(self) -> &'static str {
        match self {
            Self::Cjk => "NotoSansCJK-Regular.otf",
            Self::Heic => {
                if cfg!(windows) {
                    "paintfe-heic-encoder.exe"
                } else {
                    "paintfe-heic-encoder"
                }
            }
            Self::Avif => {
                if cfg!(windows) {
                    "paintfe-avif-codec.exe"
                } else {
                    "paintfe-avif-codec"
                }
            }
            Self::Jxl => {
                if cfg!(windows) {
                    "paintfe-jxl-encoder.exe"
                } else {
                    "paintfe-jxl-encoder"
                }
            }
        }
    }
    fn files(self) -> Vec<&'static str> {
        let mut files = vec![self.payload(), "LICENSE.txt"];
        if self == Self::Heic {
            files.extend(["SOURCES.tar.gz", "BUILDING.txt"]);
        }
        files
    }
    pub fn asset(self) -> String {
        format!(
            "PaintFE-pack-{}-{}-v{}.pfepack",
            self.id(),
            platform(),
            env!("CARGO_PKG_VERSION")
        )
    }
    pub fn url(self) -> String {
        format!(
            "https://github.com/kylejckson/PaintFE/releases/download/v{}/{}",
            env!("CARGO_PKG_VERSION"),
            self.asset()
        )
    }
    fn directory(self, root: &Path) -> PathBuf {
        root.join("protocol-1").join(self.id())
    }
}
pub fn platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}
pub fn root() -> PathBuf {
    super::storage::data_dir().join("packs")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackFile {
    pub name: String,
    pub sha256: String,
    pub size: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    pub pack: Pack,
    pub app_version: String,
    pub platform: String,
    pub protocol: u32,
    pub files: Vec<PackFile>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    manifest: String,
    signature: String,
}

fn from_hex<const N: usize>(hex: &str) -> Result<[u8; N], String> {
    if hex.len() != N * 2 {
        return Err("Invalid signing key or signature".into());
    }
    let mut result = [0; N];
    for (out, pair) in result.iter_mut().zip(hex.as_bytes().chunks_exact(2)) {
        let text = std::str::from_utf8(pair).map_err(|e| e.to_string())?;
        *out = u8::from_str_radix(text, 16).map_err(|e| e.to_string())?;
    }
    Ok(result)
}
fn trusted_key() -> Result<[u8; 32], String> {
    from_hex(option_env!("PAINTFE_PACK_PUBLIC_KEY").unwrap_or(""))
        .map_err(|_| "This build has no optional-pack signing key. Use an official release or the local pack build tool.".into())
}
fn verify_manifest(bytes: &[u8], pack: Pack, key: &[u8; 32]) -> Result<Manifest, String> {
    if bytes.len() as u64 > MAX_MANIFEST {
        return Err("Pack manifest is too large".into());
    }
    let envelope: Envelope = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let signature = ed25519_dalek::Signature::from_bytes(&from_hex::<64>(&envelope.signature)?);
    ed25519_dalek::VerifyingKey::from_bytes(key)
        .map_err(|e| e.to_string())?
        .verify_strict(envelope.manifest.as_bytes(), &signature)
        .map_err(|_| "Pack signature verification failed")?;
    let manifest: Manifest = serde_json::from_str(&envelope.manifest).map_err(|e| e.to_string())?;
    if manifest.schema != 1
        || manifest.protocol != 1
        || manifest.pack != pack
        || manifest.platform != platform()
    {
        return Err("This pack is for another feature, protocol or platform".into());
    }
    let expected = pack.files();
    if manifest.files.len() != expected.len()
        || !expected.iter().all(|name| {
            manifest
                .files
                .iter()
                .filter(|file| file.name == *name)
                .count()
                == 1
        })
        || manifest
            .files
            .iter()
            .any(|file| file.size == 0 || file.size > MAX_PACK)
    {
        return Err("Invalid pack file list".into());
    }
    Ok(manifest)
}
fn verify_file(bytes: &[u8], file: &PackFile) -> Result<(), String> {
    if bytes.len() as u64 != file.size || format!("{:x}", Sha256::digest(bytes)) != file.sha256 {
        return Err(format!("Pack integrity check failed: {}", file.name));
    }
    Ok(())
}
fn read_limited(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > max {
        return Err("Pack file is too large".into());
    }
    let mut data = Vec::new();
    file.take(max + 1)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() as u64 > max {
        return Err("Pack file is too large".into());
    }
    Ok(data)
}
fn installed_at(root: &Path, pack: Pack, key: &[u8; 32]) -> Result<PathBuf, String> {
    let directory = pack.directory(root);
    let manifest = verify_manifest(
        &read_limited(&directory.join("manifest.json"), MAX_MANIFEST)?,
        pack,
        key,
    )?;
    for file in &manifest.files {
        let path = directory.join(&file.name);
        if fs::symlink_metadata(&path)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
        {
            return Err("Pack files must not be symbolic links".into());
        }
        verify_file(&read_limited(&path, MAX_PACK)?, file)?;
    }
    Ok(directory.join(pack.payload()))
}
pub fn installed(pack: Pack) -> Result<PathBuf, String> {
    if pack == Pack::Heic {
        return Err(crate::io::heic::setup_message());
    }
    installed_at(&root(), pack, &trusted_key()?)
}
pub fn is_present(pack: Pack) -> bool {
    if pack == Pack::Heic {
        return false;
    }
    pack.directory(&root()).exists()
}

fn install_at(path: &Path, pack: Pack, root: &Path, key: &[u8; 32]) -> Result<(), String> {
    let bytes = read_limited(path, MAX_PACK)?;
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    if archive.len() != pack.files().len() + 1 {
        return Err("Unexpected pack contents".into());
    }
    let mut envelope = Vec::new();
    archive
        .by_name("manifest.json")
        .map_err(|e| e.to_string())?
        .take(MAX_MANIFEST + 1)
        .read_to_end(&mut envelope)
        .map_err(|e| e.to_string())?;
    let manifest = verify_manifest(&envelope, pack, key)?;
    let mut payloads = Vec::new();
    for file in &manifest.files {
        let mut entry = archive.by_name(&file.name).map_err(|e| e.to_string())?;
        if entry.size() != file.size
            || entry
                .unix_mode()
                .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            return Err("Invalid pack entry".into());
        }
        let mut bytes = Vec::new();
        entry
            .by_ref()
            .take(file.size + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        verify_file(&bytes, file)?;
        if pack == Pack::Cjk && file.name == pack.payload() {
            use ab_glyph::{Font, FontRef};
            let font = FontRef::try_from_slice(&bytes).map_err(|_| "Invalid CJK font")?;
            if "日本語中文简體한국어"
                .chars()
                .any(|c| font.glyph_id(c).0 == 0)
            {
                return Err("Font does not cover the required CJK scripts".into());
            }
        }
        payloads.push((&file.name, bytes));
    }
    fs::create_dir_all(root.join("protocol-1")).map_err(|e| e.to_string())?;
    let destination = pack.directory(root);
    let stage = destination.with_file_name(format!(".{}-{}", pack.id(), uuid::Uuid::new_v4()));
    fs::create_dir(&stage).map_err(|e| e.to_string())?;
    let result = (|| {
        for (name, bytes) in payloads {
            fs::write(stage.join(name), bytes).map_err(|e| e.to_string())?;
        }
        fs::write(stage.join("manifest.json"), envelope).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        if pack != Pack::Cjk {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(
                stage.join(pack.payload()),
                fs::Permissions::from_mode(0o700),
            )
            .map_err(|e| e.to_string())?;
        }
        let backup =
            destination.with_file_name(format!(".old-{}-{}", pack.id(), uuid::Uuid::new_v4()));
        let had_old = destination.exists();
        if had_old {
            fs::rename(&destination, &backup).map_err(|e| e.to_string())?;
        }
        if let Err(e) = fs::rename(&stage, &destination) {
            if had_old {
                let _ = fs::rename(&backup, &destination);
            }
            return Err(e.to_string());
        }
        if had_old {
            let _ = fs::remove_dir_all(backup);
        }
        Ok(())
    })();
    let _ = fs::remove_dir_all(stage);
    result
}
pub fn install(path: &Path, pack: Pack) -> Result<(), String> {
    if pack == Pack::Heic {
        return Err(crate::io::heic::setup_message());
    }
    let _guard = LOCK.lock().map_err(|e| e.to_string())?;
    install_at(path, pack, &root(), &trusted_key()?)
}
pub fn remove(pack: Pack) -> Result<(), String> {
    let _guard = LOCK.lock().map_err(|e| e.to_string())?;
    let path = pack.directory(&root());
    if path.exists() {
        fs::remove_dir_all(path).map_err(|e| e.to_string())?;
    }
    Ok(())
}
/// Staged fetch shared by network and offline transport tests. No destination
/// changes until the complete downloaded archive has passed verification.
fn download_with(
    pack: Pack,
    root: &Path,
    key: &[u8; 32],
    fetch: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    fs::create_dir_all(root).map_err(|e| e.to_string())?;
    let path = root.join(format!(".download-{}.pfepack", uuid::Uuid::new_v4()));
    let result = (|| {
        fetch(&path)?;
        let _guard = LOCK.lock().map_err(|e| e.to_string())?;
        install_at(&path, pack, root, key)
    })();
    let _ = fs::remove_file(path);
    result
}
/// Only called in a worker after the user explicitly chooses Download.
pub fn download(pack: Pack) -> Result<(), String> {
    if pack == Pack::Heic {
        return Err(crate::io::heic::setup_message());
    }
    let key = trusted_key()?;
    download_with(pack, &root(), &key, |path| {
        let mut command = process::hidden_command(&process::curl_path());
        command
            .args([
                "--disable",
                "--fail",
                "--silent",
                "--show-error",
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
                "300",
                "--max-filesize",
                &MAX_PACK.to_string(),
                "--output",
            ])
            .arg(path)
            .arg(pack.url())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());
        let output=command.output().map_err(|_|"Could not start the system download tool. Import a pack downloaded in your browser instead.".to_string())?;
        if !output.status.success() {
            return Err(format!(
                "Download failed. The pack may not be published for this version yet. {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(())
    })
}
pub fn request_avif() {
    REQUEST_AVIF.store(true, Ordering::Relaxed);
}
pub fn take_avif_request() -> bool {
    REQUEST_AVIF.swap(false, Ordering::Relaxed)
}
pub fn request_jxl() {
    REQUEST_JXL.store(true, Ordering::Relaxed);
}
pub fn take_jxl_request() -> bool {
    REQUEST_JXL.swap(false, Ordering::Relaxed)
}
pub fn needs_cjk(language: &str) -> bool {
    matches!(language, "ja" | "ko" | "zh-CN" | "zh-TW")
}
pub fn font_covers(bytes: &[u8], language: &str) -> bool {
    use ab_glyph::{Font, FontRef};
    let sample = match language {
        "ja" => "日本語あア",
        "ko" => "한국어",
        "zh-CN" => "中文简体",
        "zh-TW" => "中文繁體",
        _ => return true,
    };
    FontRef::try_from_slice(bytes).is_ok_and(|font| sample.chars().all(|c| font.glyph_id(c).0 != 0))
}

#[cfg(test)]
pub(crate) fn test_encoder_available() -> bool {
    static SETUP: std::sync::Once = std::sync::Once::new();
    SETUP.call_once(|| {
        if let Some(path) = std::env::var_os("PAINTFE_TEST_JXL_PACK") {
            install(Path::new(&path), Pack::Jxl).unwrap();
        }
    });
    installed(Pack::Jxl).is_ok()
}
