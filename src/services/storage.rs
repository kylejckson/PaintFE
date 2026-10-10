//! One storage decision shared by settings, logging and application-owned data.
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const SETTINGS: &str = "paintfe_settings.cfg";
const SELECTOR: &str = "paintfe_storage.cfg";

#[derive(Clone, Debug)]
pub struct Storage {
    pub executable_dir: PathBuf,
    pub profile_config: PathBuf,
    pub profile_data: PathBuf,
    pub portable: bool,
    pub warning: Option<String>,
}

impl Storage {
    pub fn settings_path(&self) -> PathBuf {
        self.config_dir().join(SETTINGS)
    }

    fn config_dir(&self) -> PathBuf {
        if self.portable {
            self.executable_dir.clone()
        } else {
            self.profile_config.clone()
        }
    }

    pub fn data_dir(&self) -> PathBuf {
        if self.portable {
            self.executable_dir.join("paintfe_data")
        } else {
            self.profile_data.clone()
        }
    }

    pub fn log_path(&self) -> PathBuf {
        self.config_dir().join("paintfe.log")
    }

    pub fn encode_path(&self, value: &str) -> String {
        if self.portable
            && let Ok(relative) = Path::new(value).strip_prefix(&self.executable_dir)
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_) | Component::CurDir))
        {
            let relative = relative
                .components()
                .filter_map(|part| match part {
                    Component::Normal(name) => Some(name.to_string_lossy()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("/");
            return format!("portable:{relative}");
        }
        value.to_owned()
    }

    pub fn decode_path(&self, value: &str) -> String {
        if let Some(relative) = value.strip_prefix("portable:") {
            let path = Path::new(relative);
            if path
                .components()
                .all(|part| matches!(part, Component::Normal(_) | Component::CurDir))
            {
                return self
                    .executable_dir
                    .join(path)
                    .to_string_lossy()
                    .into_owned();
            }
            return String::new();
        }
        value.to_owned()
    }

    fn target(&self, portable: bool) -> Self {
        Self {
            portable,
            ..self.clone()
        }
    }

    /// Copy only app-owned files. Sources are retained and conflicting content
    /// requires an explicit UI choice. Replaced files receive a backup.
    pub fn migrate(&self, portable: bool, replace: bool) -> Result<(), String> {
        self.migrate_with_origin(portable, replace, &self.executable_dir)
    }

    fn migrate_with_origin(
        &self,
        portable: bool,
        replace: bool,
        origin: &Path,
    ) -> Result<(), String> {
        let target = self.target(portable);
        let mut files = Vec::new();
        collect_files(&self.settings_path(), &target.settings_path(), &mut files)?;
        collect_files(&self.log_path(), &target.log_path(), &mut files)?;
        for name in ["autosave", "scripts", "effects", "plugins"] {
            collect_files(
                &self.data_dir().join(name),
                &target.data_dir().join(name),
                &mut files,
            )?;
        }
        // Older Linux builds kept autosaves and plugins alongside settings.
        // Preserve the current data-directory copy when both layouts exist.
        if !self.portable && self.config_dir() != self.data_dir() {
            let mut legacy = Vec::new();
            for name in ["autosave", "plugins"] {
                collect_files(
                    &self.config_dir().join(name),
                    &target.data_dir().join(name),
                    &mut legacy,
                )?;
            }
            let destinations = files
                .iter()
                .map(|(_, destination)| destination.clone())
                .collect::<std::collections::HashSet<_>>();
            files.extend(
                legacy
                    .into_iter()
                    .filter(|(_, destination)| !destinations.contains(destination)),
            );
        }
        // Complete conflict checks before publishing any destination files.
        for (source, destination) in &files {
            if destination.exists()
                && source != destination
                && std::fs::read(source).map_err(|e| e.to_string())?
                    != std::fs::read(destination).map_err(|e| e.to_string())?
                && !replace
            {
                return Err(format!(
                    "Destination contains different data: {}",
                    destination.display()
                ));
            }
        }
        for (source, destination) in files {
            if source == destination {
                continue;
            }
            let mut bytes = std::fs::read(&source).map_err(|e| e.to_string())?;
            if source == self.settings_path() {
                let content = String::from_utf8(bytes).map_err(|e| e.to_string())?;
                bytes = content
                    .lines()
                    .map(|line| {
                        if let Some((key, value)) = line.split_once('=')
                            && matches!(
                                key,
                                "onnx_runtime_path" | "birefnet_model_path" | "icon_pack_path"
                            )
                        {
                            let decoded = self.decode_path(value);
                            let rebased = Path::new(&decoded)
                                .strip_prefix(origin)
                                .ok()
                                .map(|relative| {
                                    target
                                        .executable_dir
                                        .join(relative)
                                        .to_string_lossy()
                                        .into_owned()
                                })
                                .unwrap_or(decoded);
                            return format!("{key}={}", target.encode_path(&rebased));
                        }
                        line.to_owned()
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
                    .into_bytes();
                bytes.push(b'\n');
            }
            if std::fs::read(&destination).ok().as_deref() == Some(bytes.as_slice()) {
                continue;
            }
            write_with_backup(&destination, &bytes)?;
        }
        Ok(())
    }
}

fn collect_files(
    source: &Path,
    destination: &Path,
    files: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<(), String> {
    let metadata = match std::fs::symlink_metadata(source) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.to_string()),
    };
    if metadata.file_type().is_symlink() {
        return Err(format!("Cannot migrate linked data: {}", source.display()));
    }
    if metadata.is_dir() {
        for entry in std::fs::read_dir(source).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            collect_files(&entry.path(), &destination.join(entry.file_name()), files)?;
        }
    } else if metadata.is_file() {
        files.push((source.to_owned(), destination.to_owned()));
    }
    Ok(())
}

fn write_with_backup(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("Storage path has no parent")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let temporary = parent.join(format!(".paintfe-{stamp}.tmp"));
    std::fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
    let backup = path.with_extension(format!("backup-{stamp}"));
    if path.exists()
        && let Err(error) = std::fs::rename(path, &backup)
    {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.to_string());
    }
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::rename(&backup, path);
        let _ = std::fs::remove_file(&temporary);
        return Err(error.to_string());
    }
    Ok(())
}

static STORAGE: OnceLock<Storage> = OnceLock::new();
static REQUESTED: Mutex<Option<bool>> = Mutex::new(None);

pub fn initialize() -> &'static Storage {
    STORAGE.get_or_init(resolve)
}
pub fn settings_path() -> PathBuf {
    initialize().settings_path()
}
pub fn data_dir() -> PathBuf {
    initialize().data_dir()
}
pub fn log_path() -> PathBuf {
    initialize().log_path()
}
pub fn encode_resource_path(value: &str) -> String {
    initialize().encode_path(value)
}
pub fn decode_resource_path(value: &str) -> String {
    initialize().decode_path(value)
}
pub fn requested_portable() -> bool {
    REQUESTED.lock().unwrap().unwrap_or(initialize().portable)
}

pub fn request_mode(portable: bool, replace: bool) -> Result<(), String> {
    let storage = initialize();
    if portable && !writable(&storage.executable_dir) {
        return Err("Executable directory is not writable".into());
    }
    storage.migrate(portable, replace)?;
    let selector = if writable(&storage.executable_dir) {
        storage.executable_dir.join(SELECTOR)
    } else {
        profile_selector(storage)
    };
    let mode = if portable { "portable" } else { "profile" };
    let previous = if storage.portable {
        "portable"
    } else {
        "profile"
    };
    // Recopy on the next startup so edits made after changing this preference
    // are carried across too. The current process keeps its original resolver.
    let selection = if portable == storage.portable {
        mode.to_owned()
    } else {
        format!(
            "{mode}\nmigrate_from={previous}\norigin_exe={}",
            serde_json::to_string(&storage.executable_dir.to_string_lossy())
                .map_err(|error| error.to_string())?
        )
    };
    write_with_backup(&selector, selection.as_bytes())?;
    *REQUESTED.lock().unwrap() = Some(portable);
    Ok(())
}

fn writable(directory: &Path) -> bool {
    if std::fs::create_dir_all(directory).is_err() {
        return false;
    }
    let path = directory.join(format!(".paintfe-writecheck-{}", std::process::id()));
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(file) => {
            drop(file);
            let _ = std::fs::remove_file(path);
            true
        }
        Err(_) => false,
    }
}

fn profile_selector(storage: &Storage) -> PathBuf {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    storage.executable_dir.hash(&mut hash);
    storage
        .profile_config
        .join(format!("paintfe_storage_{:x}.cfg", hash.finish()))
}

fn appimage_storage_directory(path: &Path) -> PathBuf {
    let config = PathBuf::from(format!("{}.config", path.display()));
    let home = PathBuf::from(format!("{}.home", path.display()));
    if config.is_dir() {
        config.join("paintfe")
    } else if home.is_dir() {
        home.join(".config/paintfe")
    } else {
        path.parent().unwrap_or(Path::new(".")).to_owned()
    }
}

fn resolve() -> Storage {
    // APPIMAGE identifies the original file; current_exe is inside a temporary,
    // read-only mount whose name changes on each launch.
    let executable = if cfg!(target_os = "linux") {
        std::env::var_os("APPIMAGE")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
    } else {
        None
    };
    let appimage_directory = executable.as_deref().map(appimage_storage_directory);
    let executable_dir = appimage_directory
        .or_else(|| {
            executable
                .or_else(|| std::env::current_exe().ok())
                .and_then(|p| p.parent().map(Path::to_owned))
        })
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| executable_dir.clone());
    let (profile_config, profile_data) = if cfg!(target_os = "windows") {
        let root = std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .unwrap_or(home)
            .join("PaintFE");
        (root.clone(), root)
    } else if cfg!(target_os = "macos") {
        let root = home.join("Library/Application Support/PaintFE");
        (root.clone(), root)
    } else {
        (
            std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".config"))
                .join("paintfe"),
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".local/share"))
                .join("PaintFE"),
        )
    };
    let mut storage = Storage {
        executable_dir,
        profile_config,
        profile_data,
        portable: cfg!(any(target_os = "windows", target_os = "linux")),
        warning: None,
    };
    #[cfg(test)]
    {
        let directory =
            std::env::temp_dir().join(format!("paintfe-unit-profile-{}", std::process::id()));
        storage.executable_dir = directory.join("portable");
        storage.profile_config = directory.join("profile");
        storage.profile_data = directory.join("profile");
    }
    #[cfg(debug_assertions)]
    if let Some(directory) = std::env::var_os("PAINTFE_TEST_STORAGE_DIR") {
        let directory = PathBuf::from(directory);
        storage.executable_dir = directory.join("portable");
        storage.profile_config = directory.join("profile");
        storage.profile_data = directory.join("profile");
    }
    let selector_path = if storage.executable_dir.join(SELECTOR).exists() {
        storage.executable_dir.join(SELECTOR)
    } else {
        profile_selector(&storage)
    };
    let selector = std::fs::read_to_string(storage.executable_dir.join(SELECTOR))
        .or_else(|_| std::fs::read_to_string(profile_selector(&storage)));
    if let Ok(mode) = &selector {
        storage.portable = mode.lines().next().unwrap_or("portable").trim() != "profile";
    }
    if storage.portable && !writable(&storage.executable_dir) {
        storage.portable = false;
        // Persist a safe fallback so protected installations do not prompt on
        // every launch. Settings still shows the effective storage location.
        if let Err(error) = write_with_backup(&profile_selector(&storage), b"profile") {
            storage.warning = Some(format!("Cannot remember profile storage: {error}"));
        }
    }
    if let Ok(selection) = &selector
        && let Some(previous) = selection
            .lines()
            .find_map(|line| line.strip_prefix("migrate_from="))
    {
        let source = storage.target(previous == "portable");
        if source.portable != storage.portable {
            let origin = selection
                .lines()
                .find_map(|line| line.strip_prefix("origin_exe="))
                .and_then(|value| serde_json::from_str::<String>(value).ok())
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .unwrap_or_else(|| storage.executable_dir.clone());
            let outcome = source
                .migrate_with_origin(storage.portable, true, &origin)
                .and_then(|_| {
                    write_with_backup(
                        &selector_path,
                        if storage.portable {
                            b"portable"
                        } else {
                            b"profile"
                        },
                    )
                });
            if let Err(error) = outcome {
                storage.portable = source.portable;
                storage.warning = Some(format!(
                    "Storage migration failed; keeping the previous location: {error}"
                ));
            }
        }
    }
    if storage.portable
        && !storage.settings_path().exists()
        && let Err(error) = storage.target(false).migrate(true, false)
    {
        // Preserve the established profile instead of silently resetting preferences.
        storage.portable = false;
        let _ = write_with_backup(&storage.executable_dir.join(SELECTOR), b"profile");
        storage.warning = Some(format!(
            "Portable migration failed; using the user profile: {error}"
        ));
    }
    storage
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn appimage_storage_uses_original_path_and_standard_portable_directories() {
        let fixture = Fixture::new();
        let image = fixture.root.join("PaintFE.AppImage");
        assert_eq!(appimage_storage_directory(&image), fixture.root);
        let home = fixture.root.join("PaintFE.AppImage.home");
        std::fs::create_dir_all(&home).unwrap();
        assert_eq!(
            appimage_storage_directory(&image),
            home.join(".config/paintfe")
        );
        let config = fixture.root.join("PaintFE.AppImage.config");
        std::fs::create_dir_all(&config).unwrap();
        assert_eq!(appimage_storage_directory(&image), config.join("paintfe"));
    }
    struct Fixture {
        root: PathBuf,
        storage: Storage,
    }
    impl Fixture {
        fn new() -> Self {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target/storage-tests")
                .join(format!("paintfe-storage-{stamp}"));
            let storage = Storage {
                executable_dir: root.join("portable"),
                profile_config: root.join("config"),
                profile_data: root.join("data"),
                portable: false,
                warning: None,
            };
            Self { root, storage }
        }
        fn write(&self, path: &Path, bytes: &[u8]) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn migration_preserves_sources_and_copies_all_owned_data() {
        let f = Fixture::new();
        f.write(
            &f.storage.settings_path(),
            b"middle_click_close_tabs=false\n",
        );
        for name in ["autosave", "scripts", "effects", "plugins"] {
            f.write(&f.storage.data_dir().join(name).join("owned"), b"owned");
        }
        f.write(
            &f.storage.profile_config.join("autosave/legacy.pfe"),
            b"legacy",
        );
        f.storage.migrate(true, false).unwrap();
        assert!(f.storage.settings_path().exists());
        let target = f.storage.target(true);
        for name in ["autosave", "scripts", "effects", "plugins"] {
            assert_eq!(
                std::fs::read(target.data_dir().join(name).join("owned")).unwrap(),
                b"owned"
            );
        }
        assert_eq!(
            std::fs::read(target.data_dir().join("autosave/legacy.pfe")).unwrap(),
            b"legacy"
        );
    }

    #[test]
    fn conflicts_abort_before_copy_and_explicit_replacement_keeps_backup() {
        let f = Fixture::new();
        let target = f.storage.target(true);
        f.write(&f.storage.settings_path(), b"new\n");
        f.write(&target.settings_path(), b"old\n");
        f.write(&f.storage.data_dir().join("scripts/extra"), b"extra");
        assert!(f.storage.migrate(true, false).is_err());
        assert!(!target.data_dir().join("scripts/extra").exists());
        assert_eq!(std::fs::read(target.settings_path()).unwrap(), b"old\n");
        f.storage.migrate(true, true).unwrap();
        assert_eq!(std::fs::read(target.settings_path()).unwrap(), b"new\n");
        assert!(
            std::fs::read_dir(&target.executable_dir)
                .unwrap()
                .any(|entry| entry
                    .unwrap()
                    .path()
                    .extension()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .starts_with("backup-"))
        );
        assert_eq!(std::fs::read(f.storage.settings_path()).unwrap(), b"new\n");
    }

    #[test]
    fn portable_paths_survive_relocation_and_reject_traversal() {
        let f = Fixture::new();
        let storage = f.storage.target(true);
        let resource = storage.executable_dir.join("models/model.onnx");
        let encoded = storage.encode_path(resource.to_str().unwrap());
        assert!(encoded.starts_with("portable:"));
        assert!(
            !encoded.contains('\\'),
            "portable resource paths must use platform-neutral separators"
        );
        let moved = Storage {
            executable_dir: f.root.join("moved"),
            ..storage.clone()
        };
        assert_eq!(
            moved.decode_path(&encoded),
            moved
                .executable_dir
                .join("models/model.onnx")
                .to_string_lossy()
        );
        assert_eq!(moved.decode_path("portable:../unsafe.dll"), "");
        let external = f.root.join("external.onnx").to_string_lossy().into_owned();
        assert_eq!(storage.encode_path(&external), external);
    }

    #[test]
    fn migration_converts_internal_resource_paths_for_the_target_mode() {
        let f = Fixture::new();
        let portable = f.storage.target(true);
        let path = portable.executable_dir.join("model.onnx");
        let content = format!("birefnet_model_path={}\n", path.display());
        f.write(&f.storage.settings_path(), content.as_bytes());
        f.storage.migrate(true, false).unwrap();
        assert!(
            std::fs::read_to_string(portable.settings_path())
                .unwrap()
                .contains("portable:model.onnx")
        );
        portable.migrate(false, true).unwrap();
        assert_eq!(
            std::fs::read_to_string(f.storage.settings_path()).unwrap(),
            content
        );
    }

    #[test]
    fn pending_profile_migration_rebases_resources_after_executable_relocation() {
        let f = Fixture::new();
        let previous = f.root.join("previous-installation");
        let content = format!(
            "birefnet_model_path={}\n",
            previous.join("models/tiny.onnx").display()
        );
        f.write(&f.storage.settings_path(), content.as_bytes());
        f.storage
            .migrate_with_origin(true, false, &previous)
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(f.storage.target(true).settings_path()).unwrap(),
            "birefnet_model_path=portable:models/tiny.onnx\n"
        );
    }

    #[test]
    fn unwritable_directory_is_detected_without_touching_existing_data() {
        let f = Fixture::new();
        let file = f.root.join("not-a-directory");
        f.write(&file, b"preserve");
        assert!(!writable(&file));
        assert_eq!(std::fs::read(file).unwrap(), b"preserve");
    }
}
