//! Out-of-process compatibility support for legacy Paint.NET 3.5 CPU effects.
//!
//! Plugin DLLs are untrusted code. They are never loaded into PaintFE itself;
//! metadata and rendering are delegated to the separately packaged .NET host.

use image::RgbaImage;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use crate::canvas::{CanvasState, TiledImage};
use crate::ops::dialogs::{
    DialogColors, DialogResult, accent_separator, dialog_footer, paint_dialog_header,
    preview_controls, section_label,
};

pub const PROFILE: &str = "legacy-3.5-cpu-v1";
const PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginProperty {
    pub name: String,
    pub kind: String,
    pub default: Value,
    pub min: Option<f64>,
    pub max: Option<f64>,
    #[serde(default)]
    pub choices: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PluginManifest {
    pub profile: String,
    pub source_file: String,
    pub sha256: String,
    pub trusted: bool,
    pub enabled: bool,
    pub name: String,
    pub category: String,
    pub effect_type: String,
    #[serde(default)]
    pub properties: Vec<PluginProperty>,
    pub error: Option<String>,
}

impl PluginManifest {
    pub fn display_error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

#[derive(Default)]
pub struct PluginManager {
    pub plugins: Vec<PluginManifest>,
    pub last_error: Option<String>,
}

pub struct PaintDotNetPluginDialog {
    pub plugin: PluginManifest,
    pub original_pixels: TiledImage,
    pub original_flat: RgbaImage,
    pub selection: Option<image::GrayImage>,
    pub layer_idx: usize,
    pub values: Map<String, Value>,
    pub live_preview: bool,
    pub render_error: Arc<Mutex<Option<String>>>,
}

impl PaintDotNetPluginDialog {
    pub fn new(state: &CanvasState, plugin: PluginManifest) -> Option<Self> {
        let layer_idx = state.active_layer_index;
        let original_pixels = state.layers.get(layer_idx)?.pixels.clone();
        let original_flat = original_pixels.to_rgba_image();
        let values = plugin
            .properties
            .iter()
            .map(|property| (property.name.clone(), property.default.clone()))
            .collect();
        Some(Self {
            plugin,
            original_pixels,
            original_flat,
            selection: state.selection_mask.clone(),
            layer_idx,
            values,
            live_preview: true,
            render_error: Arc::new(Mutex::new(None)),
        })
    }

    pub fn show(&mut self, ctx: &egui::Context) -> DialogResult<()> {
        let mut result = DialogResult::Open;
        let colors = DialogColors::from_ctx(ctx);
        egui::Window::new("dialog_paintdotnet_plugin")
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .default_pos(egui::pos2(ctx.content_rect().center().x - 190.0, 60.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                if paint_dialog_header(ui, &colors, "🧩", &self.plugin.name) {
                    result = DialogResult::Cancel;
                }
                ui.add_space(4.0);
                section_label(
                    ui,
                    &colors,
                    &format!("PAINT.NET · {}", self.plugin.category),
                );

                let mut changed = false;
                egui::Grid::new("paintdotnet_plugin_properties")
                    .num_columns(2)
                    .spacing([8.0, 6.0])
                    .show(ui, |ui| {
                        for property in &self.plugin.properties {
                            ui.label(&property.name);
                            let value = self
                                .values
                                .entry(property.name.clone())
                                .or_insert_with(|| property.default.clone());
                            match property.kind.as_str() {
                                "double" => {
                                    let mut number = value.as_f64().unwrap_or_default();
                                    let response = ui.add(egui::Slider::new(
                                        &mut number,
                                        property.min.unwrap_or(0.0)..=property.max.unwrap_or(1.0),
                                    ));
                                    if response.changed() {
                                        *value = Value::from(number);
                                        changed = true;
                                    }
                                }
                                "int32" => {
                                    let mut number = value.as_i64().unwrap_or_default() as i32;
                                    let response = ui.add(egui::Slider::new(
                                        &mut number,
                                        property.min.unwrap_or(0.0) as i32
                                            ..=property.max.unwrap_or(100.0) as i32,
                                    ));
                                    if response.changed() {
                                        *value = Value::from(number);
                                        changed = true;
                                    }
                                }
                                "boolean" => {
                                    let mut checked = value.as_bool().unwrap_or(false);
                                    if ui.checkbox(&mut checked, "").changed() {
                                        *value = Value::from(checked);
                                        changed = true;
                                    }
                                }
                                "choice" => {
                                    let mut index = value.as_i64().unwrap_or_default() as usize;
                                    egui::ComboBox::from_id_salt(("pdn_choice", &property.name))
                                        .selected_text(
                                            property
                                                .choices
                                                .get(index)
                                                .cloned()
                                                .unwrap_or_default(),
                                        )
                                        .show_ui(ui, |ui| {
                                            for (choice_index, choice) in
                                                property.choices.iter().enumerate()
                                            {
                                                if ui
                                                    .selectable_value(
                                                        &mut index,
                                                        choice_index,
                                                        choice,
                                                    )
                                                    .changed()
                                                {
                                                    changed = true;
                                                }
                                            }
                                        });
                                    *value = Value::from(index);
                                }
                                _ => {
                                    ui.label(format!("Unsupported: {}", property.kind));
                                }
                            }
                            ui.end_row();
                        }
                    });

                if let Ok(error) = self.render_error.lock()
                    && let Some(error) = error.as_deref()
                {
                    ui.colored_label(ui.visuals().error_fg_color, error);
                }
                accent_separator(ui, &colors);
                let manual = preview_controls(ui, &colors, &mut self.live_preview);
                if (changed && self.live_preview) || manual {
                    result = DialogResult::Changed;
                }
                let (ok, cancel) = dialog_footer(ui, &colors);
                if ok {
                    result = DialogResult::Ok(());
                }
                if cancel {
                    result = DialogResult::Cancel;
                }
            });
        result
    }
}

impl PluginManager {
    pub fn load() -> Self {
        let mut manager = Self::default();
        let Ok(entries) = fs::read_dir(plugin_root()) else {
            return manager;
        };
        for entry in entries.flatten() {
            let path = entry.path().join("manifest.json");
            let Ok(data) = fs::read(&path) else { continue };
            match serde_json::from_slice::<PluginManifest>(&data) {
                Ok(plugin) => manager.plugins.push(plugin),
                Err(error) => manager.last_error = Some(format!("{}: {error}", path.display())),
            }
        }
        manager.plugins.sort_by(|a, b| a.name.cmp(&b.name));
        manager
    }

    pub fn enabled_plugins(&self) -> impl Iterator<Item = &PluginManifest> {
        self.plugins
            .iter()
            .filter(|plugin| plugin.enabled && plugin.trusted && plugin.error.is_none())
    }

    pub fn rescan(&mut self) {
        for plugin in &mut self.plugins {
            match describe(Path::new(&plugin.source_file)) {
                Ok(response) => {
                    plugin.name = response.name.unwrap_or_else(|| plugin.name.clone());
                    plugin.category = response.category.unwrap_or_else(|| "Plugins".to_string());
                    plugin.effect_type = response.effect_type.unwrap_or_default();
                    plugin.properties = response.properties;
                    plugin.error = None;
                }
                Err(error) => {
                    plugin.enabled = false;
                    plugin.error = Some(error);
                }
            }
            if let Some(package) = Path::new(&plugin.source_file).parent() {
                let _ = save_manifest(package, plugin);
            }
        }
    }

    pub fn import_files(paths: &[PathBuf]) -> Result<PluginManifest, String> {
        let primary = paths.first().ok_or_else(|| "No DLL selected".to_string())?;
        let bytes = fs::read(primary).map_err(|error| error.to_string())?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let stem = primary
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("plugin");
        let safe_stem: String = stem
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let package_dir = plugin_root().join(format!("{}-{}", safe_stem, &hash[..12]));
        fs::create_dir_all(&package_dir).map_err(|error| error.to_string())?;
        for path in paths {
            let name = path
                .file_name()
                .ok_or_else(|| "Invalid DLL path".to_string())?;
            fs::copy(path, package_dir.join(name)).map_err(|error| error.to_string())?;
        }

        let installed = package_dir.join(
            primary
                .file_name()
                .ok_or_else(|| "Invalid DLL filename".to_string())?,
        );
        let description = describe(&installed);
        let mut manifest = match description {
            Ok(response) => PluginManifest {
                profile: PROFILE.to_string(),
                source_file: installed.to_string_lossy().into_owned(),
                sha256: hash,
                trusted: false,
                enabled: false,
                name: response.name.unwrap_or_else(|| safe_stem.clone()),
                category: response.category.unwrap_or_else(|| "Plugins".to_string()),
                effect_type: response.effect_type.unwrap_or_default(),
                properties: response.properties,
                error: None,
            },
            Err(error) => PluginManifest {
                profile: PROFILE.to_string(),
                source_file: installed.to_string_lossy().into_owned(),
                sha256: hash,
                trusted: false,
                enabled: false,
                name: safe_stem,
                category: "Plugins".to_string(),
                effect_type: String::new(),
                properties: Vec::new(),
                error: Some(error),
            },
        };
        save_manifest(&package_dir, &manifest)?;
        // Imported code must be explicitly trusted after its metadata is shown.
        manifest.trusted = false;
        Ok(manifest)
    }

    pub fn set_trusted_enabled(&mut self, sha256: &str, value: bool) -> Result<(), String> {
        let plugin = self
            .plugins
            .iter_mut()
            .find(|plugin| plugin.sha256 == sha256)
            .ok_or_else(|| "Plugin not found".to_string())?;
        plugin.trusted = value;
        plugin.enabled = value && plugin.error.is_none();
        let package = Path::new(&plugin.source_file)
            .parent()
            .ok_or_else(|| "Invalid plugin path".to_string())?;
        save_manifest(package, plugin)
    }

    pub fn remove(&mut self, sha256: &str) -> Result<(), String> {
        let index = self
            .plugins
            .iter()
            .position(|plugin| plugin.sha256 == sha256)
            .ok_or_else(|| "Plugin not found".to_string())?;
        let plugin = self.plugins.remove(index);
        let package = Path::new(&plugin.source_file)
            .parent()
            .ok_or_else(|| "Invalid plugin path".to_string())?;
        fs::remove_dir_all(package).map_err(|error| error.to_string())
    }
}

pub fn plugin_root() -> PathBuf {
    crate::assets::AppSettings::settings_path()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("plugins")
        .join("paintdotnet")
        .join("Effects")
}

pub fn host_path() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("PAINTFE_PDN_HOST").map(PathBuf::from)
        && path.is_file()
    {
        return Ok(path);
    }
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let directory = executable
        .parent()
        .ok_or_else(|| "Executable has no directory".to_string())?;
    let filename = if cfg!(windows) {
        "PaintFE.PaintDotNetHost.exe"
    } else {
        "PaintFE.PaintDotNetHost"
    };
    let development_host = directory
        .ancestors()
        .find(|path| path.file_name().and_then(|name| name.to_str()) == Some("target"))
        .map(|target| {
            let runtime_id = if cfg!(target_os = "windows") {
                "win-x64"
            } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "osx-arm64"
            } else if cfg!(target_os = "macos") {
                "osx-x64"
            } else {
                "linux-x64"
            };
            target.join("pdn-host").join(runtime_id).join(filename)
        });
    let candidates = [
        directory.join(filename),
        directory.join("paintdotnet-host").join(filename),
        development_host.unwrap_or_default(),
    ];
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| "Paint.NET compatibility host is not installed".to_string())
}

pub fn record_plugin_error(plugin: &PluginManifest, error: &str) {
    let mut failed = plugin.clone();
    failed.enabled = false;
    failed.error = Some(error.to_string());
    if let Some(package) = Path::new(&failed.source_file).parent() {
        let _ = save_manifest(package, &failed);
    }
}

pub fn describe(plugin_path: &Path) -> Result<HostResponse, String> {
    call_host(
        json_header("describe", plugin_path, None, 0, 0, Map::new(), 0, 0),
        &[],
        Duration::from_secs(10),
    )
}

pub fn render(
    plugin: &PluginManifest,
    image: &RgbaImage,
    parameters: &Map<String, Value>,
    selection: Option<&image::GrayImage>,
) -> Result<RgbaImage, String> {
    let pixels = image.as_raw();
    let mask = selection
        .filter(|mask| mask.dimensions() == image.dimensions())
        .map(|mask| mask.as_raw().as_slice())
        .unwrap_or(&[]);
    let mut payload = Vec::with_capacity(pixels.len() + mask.len());
    payload.extend_from_slice(pixels);
    payload.extend_from_slice(mask);
    let response = call_host(
        json_header(
            "render",
            Path::new(&plugin.source_file),
            Some(&plugin.effect_type),
            image.width(),
            image.height(),
            parameters.clone(),
            pixels.len(),
            mask.len(),
        ),
        &payload,
        Duration::from_secs(120),
    )?;
    let mut output = RgbaImage::from_raw(image.width(), image.height(), response.pixels)
        .ok_or_else(|| "Host returned an invalid RGBA payload".to_string())?;
    if let Some(mask) = selection {
        for y in 0..image.height() {
            for x in 0..image.width() {
                if x >= mask.width() || y >= mask.height() || mask.get_pixel(x, y).0[0] == 0 {
                    output.put_pixel(x, y, *image.get_pixel(x, y));
                }
            }
        }
    }
    Ok(output)
}

fn json_header(
    command: &str,
    plugin_path: &Path,
    effect_type: Option<&str>,
    width: u32,
    height: u32,
    parameters: Map<String, Value>,
    pixel_length: usize,
    mask_length: usize,
) -> Value {
    serde_json::json!({
        "protocolVersion": PROTOCOL_VERSION,
        "command": command,
        "pluginPath": plugin_path,
        "effectType": effect_type,
        "width": width,
        "height": height,
        "parameters": parameters,
        "pixelLength": pixel_length,
        "maskLength": mask_length,
    })
}

fn call_host(header: Value, pixels: &[u8], timeout: Duration) -> Result<HostResponse, String> {
    let mut child = Command::new(host_path()?)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("Failed to start plugin host: {error}"))?;
    let header = serde_json::to_vec(&header).map_err(|error| error.to_string())?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Host stdin unavailable".to_string())?;
    stdin
        .write_all(&(header.len() as u32).to_le_bytes())
        .and_then(|_| stdin.write_all(&header))
        .and_then(|_| stdin.write_all(pixels))
        .map_err(|error| error.to_string())?;
    drop(stdin);

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Host stdout unavailable".to_string())?;
    let child = Arc::new(Mutex::new(child));
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let result = read_response(stdout);
        let _ = sender.send(result);
    });
    match receiver.recv_timeout(timeout) {
        Ok(result) => {
            if let Ok(mut child) = child.lock() {
                let _ = child.wait();
            }
            result
        }
        Err(_) => {
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
            }
            Err("Plugin host timed out".to_string())
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HostResponse {
    pub protocol_version: u32,
    pub ok: bool,
    pub error: Option<String>,
    pub name: Option<String>,
    pub category: Option<String>,
    pub effect_type: Option<String>,
    #[serde(default)]
    pub properties: Vec<PluginProperty>,
    pub pixel_length: usize,
    #[serde(skip)]
    pub pixels: Vec<u8>,
}

fn read_response(mut stdout: impl Read) -> Result<HostResponse, String> {
    let mut length = [0u8; 4];
    stdout
        .read_exact(&mut length)
        .map_err(|error| error.to_string())?;
    let length = u32::from_le_bytes(length) as usize;
    if length == 0 || length > 1_048_576 {
        return Err("Plugin host returned an invalid header".to_string());
    }
    let mut header = vec![0; length];
    stdout
        .read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    let mut response: HostResponse =
        serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    if response.protocol_version != PROTOCOL_VERSION {
        return Err("Plugin host protocol version mismatch".to_string());
    }
    if !response.ok {
        return Err(response
            .error
            .unwrap_or_else(|| "Plugin host failed".to_string()));
    }
    response.pixels.resize(response.pixel_length, 0);
    stdout
        .read_exact(&mut response.pixels)
        .map_err(|error| error.to_string())?;
    Ok(response)
}

fn save_manifest(package_dir: &Path, manifest: &PluginManifest) -> Result<(), String> {
    let data = serde_json::to_vec_pretty(manifest).map_err(|error| error.to_string())?;
    fs::write(package_dir.join("manifest.json"), data).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_response_rejects_wrong_protocol() {
        let json = br#"{"protocolVersion":99,"ok":true,"error":null,"name":null,"category":null,"effectType":null,"properties":[],"pixelLength":0}"#;
        let mut framed = Vec::new();
        framed.extend_from_slice(&(json.len() as u32).to_le_bytes());
        framed.extend_from_slice(json);
        assert!(
            read_response(framed.as_slice())
                .unwrap_err()
                .contains("version")
        );
    }

    #[test]
    fn supplied_normal_map_plus_dll_runs_when_host_is_available() {
        let plugin_path = Path::new("NormalMapPlus/NormalMapPlus.dll");
        if !plugin_path.is_file() || host_path().is_err() {
            return;
        }

        let description = describe(plugin_path).expect("NormalMapPlus describe failed");
        assert_eq!(description.name.as_deref(), Some("NormalMapPlus"));
        assert_eq!(description.category.as_deref(), Some("Stylize"));
        assert_eq!(description.properties.len(), 3);
        assert_eq!(description.properties[0].name, "X");

        let plugin = PluginManifest {
            profile: PROFILE.to_string(),
            source_file: plugin_path.to_string_lossy().into_owned(),
            sha256: String::new(),
            trusted: true,
            enabled: true,
            name: description.name.unwrap(),
            category: description.category.unwrap(),
            effect_type: description.effect_type.unwrap(),
            properties: description.properties,
            error: None,
        };
        let image = RgbaImage::from_fn(3, 3, |x, y| {
            image::Rgba([(x * 100) as u8, (y * 100) as u8, 50, 255])
        });
        let parameters = Map::from_iter([
            ("X".to_string(), Value::from(0.3)),
            ("Y".to_string(), Value::from(0.5)),
            ("Z".to_string(), Value::from(0.11)),
        ]);
        let output =
            render(&plugin, &image, &parameters, None).expect("NormalMapPlus render failed");
        assert_eq!(output.dimensions(), (3, 3));
        assert_eq!(output.get_pixel(0, 0).0, [83, 53, 221, 255]);
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn windows_only_plugin_reports_its_platform_requirement() {
        let plugin_path = Path::new("RemoveBackground/RemoveBackground.dll");
        if !plugin_path.is_file() || host_path().is_err() {
            return;
        }

        let error = describe(plugin_path).expect_err("Windows-only plugin was accepted");
        assert!(error.contains("Windows-only"), "unexpected error: {error}");
        assert!(
            error.contains("System.Windows.Forms"),
            "unexpected error: {error}"
        );
    }
}
