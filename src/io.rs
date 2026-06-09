use crate::components::dialogs::TiffCompression;
use image::ImageEncoder;
use image::codecs::bmp::BmpEncoder;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::tga::TgaEncoder;
use image::{DynamicImage, ImageError, Rgba, RgbaImage};
use rfd::FileDialog;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::canvas::{
    BlendMode, CHUNK_SIZE, CanvasState, Layer, LayerContent, PixelFormat, TiledImage,
};
use crate::components::dialogs::SaveFormat;
use crate::experimental::{DeepRgbaBuffer, f16_bits_to_f32, reinhard_tone_map_rgba};

/// Minimum frame delay in milliseconds for animated images.
const MIN_FRAME_DELAY_MS: u16 = 10;

/// Common RAW camera file extensions (lowercase).
pub const RAW_EXTENSIONS: &[&str] = &[
    "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "dng", "orf", "rw2", "pef", "raf", "raw",
    "rwl", "srw", "x3f", "3fr", "fff", "iiq", "mrw", "mef", "mos", "kdc", "dcr", "erf",
];

/// Check if a file extension is a known RAW format.
pub fn is_raw_extension(ext: &str) -> bool {
    RAW_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

/// Decode a RAW camera file to an sRGB RgbaImage.
/// Uses rawloader for decoding and imagepipe for demosaicing + colour processing.
pub fn decode_raw_image(path: &Path) -> Result<RgbaImage, String> {
    // Use imagepipe to handle the full decode + demosaicing + color pipeline
    let mut pipeline =
        imagepipe::Pipeline::new_from_file(path).map_err(|e| format!("RAW decode error: {}", e))?;

    let srgb = pipeline
        .output_8bit(None)
        .map_err(|e| format!("RAW processing error: {}", e))?;

    let width = srgb.width;
    let height = srgb.height;
    validate_open_dimensions(width as u32, height as u32)?;

    // srgb.data is Vec<u8> in RGB (3 bytes per pixel) — convert to RGBA
    let pixel_count = width * height;
    if srgb.data.len() < pixel_count * 3 {
        return Err(format!(
            "RAW buffer too short: expected {} bytes, got {}",
            pixel_count * 3,
            srgb.data.len()
        ));
    }
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        rgba.push(srgb.data[i * 3]);
        rgba.push(srgb.data[i * 3 + 1]);
        rgba.push(srgb.data[i * 3 + 2]);
        rgba.push(255);
    }

    RgbaImage::from_raw(width as u32, height as u32, rgba)
        .ok_or_else(|| "Failed to create image from RAW data".to_string())
}

// ============================================================================
// PFE PROJECT FILE FORMAT
// ============================================================================

/// Magic header for the flat legacy format (v0)
const PFE_MAGIC_V0: &str = "PFE0";
/// Magic header for the tiled sparse format (v1)
const PFE_MAGIC_V1: &str = "PFE1";
/// Magic header for the tiled format with text layer support (v2)
const PFE_MAGIC_V2: &str = "PFE2";
/// Magic header for experimental metadata, adjustment and format support (v3)
const PFE_MAGIC_V3: &str = "PFE3";

/// V0 (legacy) serializable project file structure
#[derive(Serialize, Deserialize)]
struct ProjectFileV0 {
    magic: String,
    width: u32,
    height: u32,
    active_layer_index: usize,
    layers: Vec<LayerDataV0>,
}

/// V0 (legacy) serializable layer data — flat pixel buffer
#[derive(Serialize, Deserialize)]
struct LayerDataV0 {
    name: String,
    visible: bool,
    opacity: f32,
    blend_mode: u8,
    pixels: Vec<u8>,
}

/// V1 serializable project file — sparse tiled format
#[derive(Serialize, Deserialize)]
pub(crate) struct ProjectFileV1 {
    magic: String,
    width: u32,
    height: u32,
    active_layer_index: usize,
    layers: Vec<LayerDataV1>,
}

/// V1 serializable layer data — sparse chunks
#[derive(Serialize, Deserialize)]
struct LayerDataV1 {
    name: String,
    visible: bool,
    opacity: f32,
    blend_mode: u8,
    chunks: Vec<ChunkData>,
}

/// A single serialisable chunk (64×64 × 4 bytes = 16 384 bytes of pixel data)
#[derive(Serialize, Deserialize)]
struct ChunkData {
    cx: u32,
    cy: u32,
    pixels: Vec<u8>,
}

/// V2 serializable project file — sparse tiled format with text layer support.
/// Always includes rasterized chunks for backward compatibility (V1 readers).
#[derive(Serialize, Deserialize)]
pub(crate) struct ProjectFileV2 {
    magic: String,
    width: u32,
    height: u32,
    active_layer_index: usize,
    layers: Vec<LayerDataV2>,
}

/// V2 serializable layer data — sparse chunks + optional serialized text data.
#[derive(Serialize, Deserialize)]
struct LayerDataV2 {
    name: String,
    visible: bool,
    opacity: f32,
    blend_mode: u8,
    /// 0 = Raster, 1 = Text
    layer_type: u8,
    /// Rasterized chunks — always present for backward compatibility.
    chunks: Vec<ChunkData>,
    /// Serialized `TextLayerData` (bincode). Only present when `layer_type == 1`.
    text_data: Option<Vec<u8>>,
}

/// V3 serializable project file — experimental feature payloads.
#[derive(Serialize, Deserialize)]
pub(crate) struct ProjectFileV3 {
    magic: String,
    width: u32,
    height: u32,
    active_layer_index: usize,
    #[serde(default)]
    folders: Vec<crate::canvas::LayerFolder>,
    #[serde(default = "default_next_layer_folder_id")]
    next_layer_folder_id: u64,
    layers: Vec<LayerDataV3>,
}

fn default_next_layer_folder_id() -> u64 {
    1
}

/// V3 layer data with backward-compatible raster chunks plus experimental metadata.
#[derive(Serialize, Deserialize)]
struct LayerDataV3 {
    name: String,
    visible: bool,
    #[serde(default)]
    folder_id: Option<u64>,
    opacity: f32,
    blend_mode: u8,
    /// 0 = Raster, 1 = Text, 2 = Adjustment
    layer_type: u8,
    chunks: Vec<ChunkData>,
    content_data: Option<Vec<u8>>,
    pixel_format: crate::canvas::PixelFormat,
    hdr_metadata: crate::canvas::HdrMetadata,
    source_metadata: crate::canvas::ImageMetadata,
    deep_pixels: Option<crate::experimental::DeepRgbaBuffer>,
}

/// Error type for PFE file operations
#[derive(Debug)]
pub enum PfeError {
    Io(std::io::Error),
    Serialize(String),
    InvalidFormat(String),
}

impl std::fmt::Display for PfeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PfeError::Io(e) => write!(f, "I/O error: {}", e),
            PfeError::Serialize(e) => write!(f, "Serialization error: {}", e),
            PfeError::InvalidFormat(e) => write!(f, "Invalid format: {}", e),
        }
    }
}

impl From<std::io::Error> for PfeError {
    fn from(e: std::io::Error) -> Self {
        PfeError::Io(e)
    }
}

impl From<Box<bincode::ErrorKind>> for PfeError {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        PfeError::Serialize(e.to_string())
    }
}

/// Save a CanvasState as a .pfe project file.
/// Uses V2 format when text layers are present, V1 otherwise for backward compatibility.
pub fn save_pfe(state: &CanvasState, path: &Path) -> Result<(), PfeError> {
    let data = build_pfe(state);
    write_pfe(&data, path)
}

/// Version-independent PFE project data — auto-selects V1 or V2 at build time.
pub enum PfeData {
    V1(ProjectFileV1),
    V2(ProjectFileV2),
    V3(ProjectFileV3),
}

/// Build a serializable PFE project, auto-selecting V1 or V2 based on content.
/// Safe to call on the main thread, then move the result to a background thread.
pub fn build_pfe(state: &CanvasState) -> PfeData {
    let has_layer_folders =
        !state.layer_folders.is_empty() || state.layers.iter().any(|l| l.folder_id.is_some());
    let has_experimental_layers = state.layers.iter().any(|l| {
        !matches!(
            l.content,
            crate::canvas::LayerContent::Raster | crate::canvas::LayerContent::Text(_)
        ) || l.pixel_format != crate::canvas::PixelFormat::RgbaU8
            || l.hdr_metadata.enabled
            || !l.source_metadata.png_text_chunks.is_empty()
            || !l.source_metadata.raw_png_chunks.is_empty()
            || l.source_metadata.source_format.is_some()
            || l.deep_pixels.is_some()
    });
    let has_text_layers = state
        .layers
        .iter()
        .any(|l| matches!(l.content, crate::canvas::LayerContent::Text(_)));
    if has_experimental_layers || has_layer_folders {
        PfeData::V3(build_pfe_v3(state))
    } else if has_text_layers {
        PfeData::V2(build_pfe_v2(state))
    } else {
        PfeData::V1(build_pfe_v1(state))
    }
}

/// Write a pre-built PFE project to disk. Safe to call on a background thread.
pub fn write_pfe(data: &PfeData, path: &Path) -> Result<(), PfeError> {
    match data {
        PfeData::V1(project) => write_pfe_v1(project, path),
        PfeData::V2(project) => write_pfe_v2(project, path),
        PfeData::V3(project) => write_pfe_v3(project, path),
    }
}

/// Build the serializable PFE v1 project data from canvas state.
/// This copies pixel data from TiledImage chunks — safe to call on the main
/// thread, then move the result to a background thread for serialization.
pub fn build_pfe_v1(state: &CanvasState) -> ProjectFileV1 {
    let layers: Vec<LayerDataV1> = state
        .layers
        .iter()
        .map(|layer| {
            let chunks: Vec<ChunkData> = layer
                .pixels
                .chunk_keys()
                .map(|(cx, cy)| {
                    let chunk_img = layer.pixels.get_chunk(cx, cy).unwrap();
                    ChunkData {
                        cx,
                        cy,
                        pixels: chunk_img.as_raw().clone(),
                    }
                })
                .collect();

            LayerDataV1 {
                name: layer.name.clone(),
                visible: layer.visible,
                opacity: layer.opacity,
                blend_mode: layer.blend_mode.to_u8(),
                chunks,
            }
        })
        .collect();

    ProjectFileV1 {
        magic: PFE_MAGIC_V1.to_string(),
        width: state.width,
        height: state.height,
        active_layer_index: state.active_layer_index,
        layers,
    }
}

/// Serialize + write a pre-built ProjectFileV1 to disk.
/// Safe to call on a background thread.
pub fn write_pfe_v1(project: &ProjectFileV1, path: &Path) -> Result<(), PfeError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &project)?;
    Ok(())
}

/// Build the serializable PFE v2 project data from canvas state.
/// V2 includes text layer vector data alongside rasterized chunks.
/// Safe to call on the main thread, then move the result to a background thread.
pub fn build_pfe_v2(state: &CanvasState) -> ProjectFileV2 {
    use crate::canvas::LayerContent;

    let layers: Vec<LayerDataV2> = state
        .layers
        .iter()
        .map(|layer| {
            let chunks: Vec<ChunkData> = layer
                .pixels
                .chunk_keys()
                .map(|(cx, cy)| {
                    let chunk_img = layer.pixels.get_chunk(cx, cy).unwrap();
                    ChunkData {
                        cx,
                        cy,
                        pixels: chunk_img.as_raw().clone(),
                    }
                })
                .collect();

            let (layer_type, text_data) = match &layer.content {
                LayerContent::Raster => (0u8, None),
                LayerContent::Text(td) => {
                    // Serialize TextLayerData with bincode
                    let serialized = bincode::serialize(td).ok();
                    (1u8, serialized)
                }
                LayerContent::Adjustment(_) => (0u8, None),
            };

            LayerDataV2 {
                name: layer.name.clone(),
                visible: layer.visible,
                opacity: layer.opacity,
                blend_mode: layer.blend_mode.to_u8(),
                layer_type,
                chunks,
                text_data,
            }
        })
        .collect();

    ProjectFileV2 {
        magic: PFE_MAGIC_V2.to_string(),
        width: state.width,
        height: state.height,
        active_layer_index: state.active_layer_index,
        layers,
    }
}

/// Serialize + write a pre-built ProjectFileV2 to disk.
/// Safe to call on a background thread.
pub fn write_pfe_v2(project: &ProjectFileV2, path: &Path) -> Result<(), PfeError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &project)?;
    Ok(())
}

/// Build the experimental v3 project data from canvas state.
pub fn build_pfe_v3(state: &CanvasState) -> ProjectFileV3 {
    use crate::canvas::LayerContent;

    let layers: Vec<LayerDataV3> = state
        .layers
        .iter()
        .map(|layer| {
            let chunks: Vec<ChunkData> = layer
                .pixels
                .chunk_keys()
                .map(|(cx, cy)| {
                    let chunk_img = layer.pixels.get_chunk(cx, cy).unwrap();
                    ChunkData {
                        cx,
                        cy,
                        pixels: chunk_img.as_raw().clone(),
                    }
                })
                .collect();

            let (layer_type, content_data) = match &layer.content {
                LayerContent::Raster => (0u8, None),
                LayerContent::Text(td) => (1u8, bincode::serialize(td).ok()),
                LayerContent::Adjustment(adj) => (2u8, bincode::serialize(adj).ok()),
            };

            LayerDataV3 {
                name: layer.name.clone(),
                visible: layer.visible,
                folder_id: layer.folder_id,
                opacity: layer.opacity,
                blend_mode: layer.blend_mode.to_u8(),
                layer_type,
                chunks,
                content_data,
                pixel_format: layer.pixel_format,
                hdr_metadata: layer.hdr_metadata.clone(),
                source_metadata: layer.source_metadata.clone(),
                deep_pixels: layer.deep_pixels.clone(),
            }
        })
        .collect();

    ProjectFileV3 {
        magic: PFE_MAGIC_V3.to_string(),
        width: state.width,
        height: state.height,
        active_layer_index: state.active_layer_index,
        folders: state.layer_folders.clone(),
        next_layer_folder_id: state.next_layer_folder_id,
        layers,
    }
}

pub fn write_pfe_v3(project: &ProjectFileV3, path: &Path) -> Result<(), PfeError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &project)?;
    Ok(())
}

/// Load a .pfe project file (supports both v0 flat and v1 tiled formats)
pub fn load_pfe(path: &Path) -> Result<CanvasState, PfeError> {
    // Peek at the first 4 bytes to determine version
    let raw = std::fs::read(path)?;
    if raw.len() < 12 {
        return Err(PfeError::InvalidFormat("File too small".into()));
    }

    // bincode encodes a String as: 8-byte length prefix + UTF-8 data.
    // Our magic strings are 4 chars, so bytes 8..12 hold the magic.
    let magic = std::str::from_utf8(&raw[8..12]).unwrap_or("");

    match magic {
        PFE_MAGIC_V3 => load_pfe_v3(&raw),
        PFE_MAGIC_V2 => load_pfe_v2(&raw),
        PFE_MAGIC_V1 => load_pfe_v1(&raw),
        PFE_MAGIC_V0 => load_pfe_v0(&raw),
        _ => Err(PfeError::InvalidFormat(format!(
            "Unknown magic '{}'",
            magic
        ))),
    }
}

/// Maximum supported canvas dimension in pixels (per axis).
/// Prevents memory exhaustion from crafted project files.
pub const MAX_OPEN_IMAGE_DIM: u32 = 25_000;
const MAX_CANVAS_DIM: u32 = MAX_OPEN_IMAGE_DIM;
/// Maximum number of layers in a project file.
const MAX_LAYERS: usize = 256;

pub fn validate_open_dimensions(width: u32, height: u32) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("Image dimensions cannot be zero".to_string());
    }
    if width > MAX_OPEN_IMAGE_DIM || height > MAX_OPEN_IMAGE_DIM {
        return Err(format!(
            "Image size {}x{} exceeds maximum allowed {}x{}",
            width, height, MAX_OPEN_IMAGE_DIM, MAX_OPEN_IMAGE_DIM
        ));
    }
    Ok(())
}

// ============================================================================
// SYNCHRONOUS IMAGE LOADER (CLI / headless mode)
// ============================================================================

/// Returns the platform-specific directory used for auto-save files.
///
/// `%APPDATA%\PaintFE\autosave\`       (Windows)  
/// `~/.local/share/PaintFE/autosave/`  (Linux)  
/// `~/Library/Application Support/PaintFE/autosave/`  (macOS)
pub fn autosave_dir() -> Option<std::path::PathBuf> {
    crate::assets::AppSettings::settings_path().and_then(|p| p.parent().map(|d| d.join("autosave")))
}

// ============================================================================

fn metadata_for_path(path: &Path) -> crate::canvas::ImageMetadata {
    let mut meta = crate::canvas::ImageMetadata {
        source_format: path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase()),
        source_name: path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string()),
        ..Default::default()
    };

    if meta.source_format.as_deref() != Some("png") {
        return meta;
    }
    let Ok(bytes) = std::fs::read(path) else {
        return meta;
    };
    const PNG_SIG: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if bytes.len() < 12 || &bytes[..8] != PNG_SIG {
        return meta;
    }

    let mut pos = 8usize;
    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let typ_start = pos + 4;
        let data_start = pos + 8;
        let data_end = data_start.saturating_add(len);
        let chunk_end = data_end.saturating_add(4);
        if chunk_end > bytes.len() {
            break;
        }
        let chunk_type = &bytes[typ_start..typ_start + 4];
        if matches!(chunk_type, b"tEXt" | b"iTXt" | b"zTXt") {
            meta.raw_png_chunks.push(bytes[pos..chunk_end].to_vec());
            if chunk_type == b"tEXt" {
                let data = &bytes[data_start..data_end];
                if let Some(split) = data.iter().position(|&b| b == 0) {
                    let key = String::from_utf8_lossy(&data[..split]).to_string();
                    let value = String::from_utf8_lossy(&data[split + 1..]).to_string();
                    meta.png_text_chunks.push((key, value));
                }
            }
        }
        if chunk_type == b"IEND" {
            break;
        }
        pos = chunk_end;
    }

    meta
}

fn dynamic_image_to_rgba_and_deep(
    img: DynamicImage,
) -> (
    RgbaImage,
    crate::canvas::PixelFormat,
    Option<crate::experimental::DeepRgbaBuffer>,
    crate::canvas::HdrMetadata,
) {
    match &img {
        DynamicImage::ImageRgba16(buf) => (
            img.to_rgba8(),
            crate::canvas::PixelFormat::RgbaU16,
            Some(crate::experimental::DeepRgbaBuffer::U16(
                buf.as_raw().clone(),
            )),
            crate::canvas::HdrMetadata::default(),
        ),
        DynamicImage::ImageRgb16(buf) => {
            let mut deep = Vec::with_capacity(buf.as_raw().len() / 3 * 4);
            for px in buf.pixels() {
                deep.extend_from_slice(&[px[0], px[1], px[2], u16::MAX]);
            }
            (
                img.to_rgba8(),
                crate::canvas::PixelFormat::RgbaU16,
                Some(crate::experimental::DeepRgbaBuffer::U16(deep)),
                crate::canvas::HdrMetadata::default(),
            )
        }
        DynamicImage::ImageLuma16(buf) => {
            let mut deep = Vec::with_capacity(buf.as_raw().len() * 4);
            for px in buf.pixels() {
                let v = px[0];
                deep.extend_from_slice(&[v, v, v, u16::MAX]);
            }
            (
                img.to_rgba8(),
                crate::canvas::PixelFormat::RgbaU16,
                Some(crate::experimental::DeepRgbaBuffer::U16(deep)),
                crate::canvas::HdrMetadata::default(),
            )
        }
        DynamicImage::ImageLumaA16(buf) => {
            let mut deep = Vec::with_capacity(buf.as_raw().len() / 2 * 4);
            for px in buf.pixels() {
                let v = px[0];
                deep.extend_from_slice(&[v, v, v, px[1]]);
            }
            (
                img.to_rgba8(),
                crate::canvas::PixelFormat::RgbaU16,
                Some(crate::experimental::DeepRgbaBuffer::U16(deep)),
                crate::canvas::HdrMetadata::default(),
            )
        }
        DynamicImage::ImageRgba32F(buf) => {
            let raw = buf.as_raw().clone();
            let max = raw.iter().copied().fold(0.0_f32, f32::max);
            (
                img.to_rgba8(),
                crate::canvas::PixelFormat::RgbaF32,
                Some(crate::experimental::DeepRgbaBuffer::F32(raw)),
                crate::canvas::HdrMetadata {
                    enabled: max > 1.0,
                    max_luminance_nits: (max > 1.0).then_some(max * 100.0),
                    reference_white_nits: Some(100.0),
                    transfer_function: Some("linear-f32".to_string()),
                },
            )
        }
        DynamicImage::ImageRgb32F(buf) => {
            let mut deep = Vec::with_capacity(buf.as_raw().len() / 3 * 4);
            for px in buf.pixels() {
                deep.extend_from_slice(&[px[0], px[1], px[2], 1.0]);
            }
            let max = deep.iter().copied().fold(0.0_f32, f32::max);
            (
                img.to_rgba8(),
                crate::canvas::PixelFormat::RgbaF32,
                Some(crate::experimental::DeepRgbaBuffer::F32(deep)),
                crate::canvas::HdrMetadata {
                    enabled: max > 1.0,
                    max_luminance_nits: (max > 1.0).then_some(max * 100.0),
                    reference_white_nits: Some(100.0),
                    transfer_function: Some("linear-f32".to_string()),
                },
            )
        }
        _ => (
            img.to_rgba8(),
            crate::canvas::PixelFormat::RgbaU8,
            None,
            crate::canvas::HdrMetadata::default(),
        ),
    }
}

/// Synchronously load any supported image into a single-layer [`CanvasState`].
///
/// Supported inputs:
/// - `.pfe` — PaintFE native project (layers preserved, format returned as-is)
/// - RAW camera files (CR2, NEF, ARW, DNG, etc.) — decoded to 8-bit sRGB RGBA
/// - All standard raster formats supported by the `image` crate (PNG, JPEG, WEBP, BMP, …)
pub fn load_image_sync(path: &Path) -> Result<CanvasState, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Native project file — layers and metadata preserved
    if ext == "pfe" {
        return load_pfe(path).map_err(|e| format!("{:?}", e));
    }

    // Decode to display RGBA while preserving high-depth payload when available.
    let (img, pixel_format, deep_pixels, hdr_metadata) = if is_raw_extension(&ext) {
        (
            decode_raw_image(path)?,
            crate::canvas::PixelFormat::RgbaU8,
            None,
            crate::canvas::HdrMetadata::default(),
        )
    } else {
        if let Ok((w, h)) = image::image_dimensions(path) {
            validate_open_dimensions(w, h)?;
        }
        dynamic_image_to_rgba_and_deep(image::open(path).map_err(|e| e.to_string())?)
    };

    let w = img.width();
    let h = img.height();
    validate_open_dimensions(w, h)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Background")
        .to_string();

    let layer = Layer {
        name,
        visible: true,
        folder_id: None,
        opacity: 1.0,
        blend_mode: BlendMode::Normal,
        pixels: TiledImage::from_rgba_image(&img),
        mask: None,
        mask_enabled: true,
        lod_cache: None,
        gpu_generation: 0,
        content: crate::canvas::LayerContent::Raster,
        pixel_format,
        hdr_metadata,
        source_metadata: metadata_for_path(path),
        deep_pixels,
    };

    Ok(CanvasState {
        width: w,
        height: h,
        layers: vec![layer],
        layer_folders: Vec::new(),
        next_layer_folder_id: 1,
        active_layer_index: 0,
        edit_layer_mask: false,
        composite_cache: None,
        dirty_rect: None,
        show_pixel_grid: true,
        show_guidelines: false,
        mirror_mode: crate::canvas::MirrorMode::None,
        show_wrap_preview: false,
        preview_layer: None,
        preview_blend_mode: BlendMode::Normal,
        preview_force_composite: false,
        preview_is_eraser: false,
        preview_replaces_layer: false,
        preview_targets_mask: false,
        preview_mask_reveal: false,
        dirty_generation: 0,
        selection_mask: None,
        lod_composite_cache: None,
        lod_generation: 0,
        preview_dirty_rect: None,
        preview_texture_cache: None,
        preview_generation: 0,
        preview_stroke_bounds: None,
        preview_flat_buffer: Vec::new(),
        preview_flat_ready: false,
        preview_downscale: 1,
        composite_cpu_buffer: Vec::new(),
        composite_cpu_buffer_back: Vec::new(),
        region_extract_buf: Vec::new(),
        composite_above_buffer: Vec::new(),
        preview_premul_cache: Vec::new(),
        preview_cache_rect: None,
        selection_overlay_texture: None,
        selection_overlay_generation: 0,
        selection_overlay_built_generation: 0,
        selection_overlay_anim_offset: -1.0,
        selection_overlay_bounds: None,
        fill_commit_overlays: Vec::new(),
        selection_border_h_segs: Vec::new(),
        selection_border_v_segs: Vec::new(),
        selection_border_built_generation: u64::MAX,
        cmyk_preview: false,
        text_coverage_buf: Vec::new(),
        text_glyph_cache: Default::default(),
        text_editing_layer: None,
        canvas_widget_id: None,
    })
}

/// Load a v3 tiled project file with experimental feature support.
fn load_pfe_v3(raw: &[u8]) -> Result<CanvasState, PfeError> {
    use crate::canvas::{AdjustmentLayerData, LayerContent};

    let project: ProjectFileV3 = bincode::deserialize(raw)?;

    validate_open_dimensions(project.width, project.height).map_err(PfeError::InvalidFormat)?;
    if project.layers.len() > MAX_LAYERS {
        return Err(PfeError::InvalidFormat(format!(
            "Project contains {} layers, which exceeds the maximum of {}",
            project.layers.len(),
            MAX_LAYERS
        )));
    }

    let expected_chunk_bytes = (CHUNK_SIZE * CHUNK_SIZE * 4) as usize;
    let mut layers = Vec::with_capacity(project.layers.len());

    for ld in project.layers {
        let mut tiled = TiledImage::new(project.width, project.height);
        for cd in ld.chunks {
            if cd.pixels.len() != expected_chunk_bytes {
                return Err(PfeError::InvalidFormat(format!(
                    "Chunk ({},{}) in layer '{}' has {} bytes, expected {}",
                    cd.cx,
                    cd.cy,
                    ld.name,
                    cd.pixels.len(),
                    expected_chunk_bytes,
                )));
            }
            let chunk_img =
                RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, cd.pixels).ok_or_else(|| {
                    PfeError::InvalidFormat(format!(
                        "Failed to reconstruct chunk ({},{}) for layer '{}'",
                        cd.cx, cd.cy, ld.name
                    ))
                })?;
            tiled.set_chunk(cd.cx, cd.cy, chunk_img);
        }

        let content = match ld.layer_type {
            1 => ld
                .content_data
                .as_ref()
                .and_then(|b| bincode::deserialize::<crate::ops::text_layer::TextLayerData>(b).ok())
                .map(|mut td| {
                    td.raster_generation = 1;
                    td.next_block_id = td.blocks.iter().map(|b| b.id).max().unwrap_or(0) + 1;
                    LayerContent::Text(td)
                })
                .unwrap_or(LayerContent::Raster),
            2 => ld
                .content_data
                .as_ref()
                .and_then(|b| bincode::deserialize::<AdjustmentLayerData>(b).ok())
                .map(LayerContent::Adjustment)
                .unwrap_or(LayerContent::Raster),
            _ => LayerContent::Raster,
        };

        layers.push(Layer {
            name: ld.name,
            visible: ld.visible,
            folder_id: ld.folder_id,
            opacity: ld.opacity,
            blend_mode: BlendMode::from_u8(ld.blend_mode),
            pixels: tiled,
            mask: None,
            mask_enabled: true,
            lod_cache: None,
            gpu_generation: 0,
            content,
            pixel_format: ld.pixel_format,
            hdr_metadata: ld.hdr_metadata,
            source_metadata: ld.source_metadata,
            deep_pixels: ld.deep_pixels,
        });
    }

    if layers.is_empty() {
        return Err(PfeError::InvalidFormat("Project contains no layers".into()));
    }

    let active = project.active_layer_index.min(layers.len() - 1);
    Ok(CanvasState {
        width: project.width,
        height: project.height,
        layers,
        layer_folders: project.folders,
        next_layer_folder_id: project.next_layer_folder_id,
        active_layer_index: active,
        edit_layer_mask: false,
        composite_cache: None,
        dirty_rect: None,
        show_pixel_grid: true,
        show_guidelines: false,
        mirror_mode: crate::canvas::MirrorMode::None,
        show_wrap_preview: false,
        preview_layer: None,
        preview_blend_mode: BlendMode::Normal,
        preview_force_composite: false,
        preview_is_eraser: false,
        preview_replaces_layer: false,
        preview_targets_mask: false,
        preview_mask_reveal: false,
        dirty_generation: 0,
        selection_mask: None,
        lod_composite_cache: None,
        lod_generation: 0,
        preview_dirty_rect: None,
        preview_texture_cache: None,
        preview_generation: 0,
        preview_stroke_bounds: None,
        preview_flat_buffer: Vec::new(),
        preview_flat_ready: false,
        preview_downscale: 1,
        composite_cpu_buffer: Vec::new(),
        composite_cpu_buffer_back: Vec::new(),
        region_extract_buf: Vec::new(),
        composite_above_buffer: Vec::new(),
        preview_premul_cache: Vec::new(),
        preview_cache_rect: None,
        selection_overlay_texture: None,
        selection_overlay_generation: 0,
        selection_overlay_built_generation: 0,
        selection_overlay_anim_offset: -1.0,
        selection_overlay_bounds: None,
        fill_commit_overlays: Vec::new(),
        selection_border_h_segs: Vec::new(),
        selection_border_v_segs: Vec::new(),
        selection_border_built_generation: u64::MAX,
        cmyk_preview: false,
        text_coverage_buf: Vec::new(),
        text_glyph_cache: Default::default(),
        text_editing_layer: None,
        canvas_widget_id: None,
    })
}

/// Load a v2 tiled project file with text layer support
fn load_pfe_v2(raw: &[u8]) -> Result<CanvasState, PfeError> {
    use crate::canvas::LayerContent;

    let project: ProjectFileV2 = bincode::deserialize(raw)?;

    validate_open_dimensions(project.width, project.height).map_err(PfeError::InvalidFormat)?;
    if project.layers.len() > MAX_LAYERS {
        return Err(PfeError::InvalidFormat(format!(
            "Project contains {} layers, which exceeds the maximum of {}",
            project.layers.len(),
            MAX_LAYERS
        )));
    }

    let expected_chunk_bytes = (CHUNK_SIZE * CHUNK_SIZE * 4) as usize;

    let mut layers = Vec::with_capacity(project.layers.len());
    for ld in project.layers {
        let mut tiled = TiledImage::new(project.width, project.height);
        for cd in ld.chunks {
            if cd.pixels.len() != expected_chunk_bytes {
                return Err(PfeError::InvalidFormat(format!(
                    "Chunk ({},{}) in layer '{}' has {} bytes, expected {}",
                    cd.cx,
                    cd.cy,
                    ld.name,
                    cd.pixels.len(),
                    expected_chunk_bytes,
                )));
            }
            let chunk_img =
                RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, cd.pixels).ok_or_else(|| {
                    PfeError::InvalidFormat(format!(
                        "Failed to reconstruct chunk ({},{}) for layer '{}'",
                        cd.cx, cd.cy, ld.name
                    ))
                })?;
            tiled.set_chunk(cd.cx, cd.cy, chunk_img);
        }

        // Reconstruct LayerContent from layer_type + text_data
        let content = if ld.layer_type == 1 {
            if let Some(text_bytes) = ld.text_data {
                match bincode::deserialize::<crate::ops::text_layer::TextLayerData>(&text_bytes) {
                    Ok(mut td) => {
                        // Re-initialize transient state after deserialization
                        td.cache_generation = 1;
                        td.raster_generation = 1; // pixels are already up-to-date from chunks
                        // Rebuild next_block_id from max existing id
                        td.next_block_id = td.blocks.iter().map(|b| b.id).max().unwrap_or(0) + 1;
                        LayerContent::Text(td)
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: failed to deserialize text data for layer '{}': {}. Treating as raster.",
                            ld.name, e
                        );
                        LayerContent::Raster
                    }
                }
            } else {
                // layer_type says Text but no text_data — treat as raster
                LayerContent::Raster
            }
        } else {
            LayerContent::Raster
        };

        layers.push(Layer {
            name: ld.name,
            visible: ld.visible,
            folder_id: None,
            opacity: ld.opacity,
            blend_mode: BlendMode::from_u8(ld.blend_mode),
            pixels: tiled,
            mask: None,
            mask_enabled: true,
            lod_cache: None,
            gpu_generation: 0,
            content,
            pixel_format: crate::canvas::PixelFormat::RgbaU8,
            hdr_metadata: crate::canvas::HdrMetadata::default(),
            source_metadata: crate::canvas::ImageMetadata::default(),
            deep_pixels: None,
        });
    }

    if layers.is_empty() {
        return Err(PfeError::InvalidFormat("Project contains no layers".into()));
    }

    let active = project.active_layer_index.min(layers.len() - 1);

    Ok(CanvasState {
        width: project.width,
        height: project.height,
        layers,
        layer_folders: Vec::new(),
        next_layer_folder_id: 1,
        active_layer_index: active,
        edit_layer_mask: false,
        composite_cache: None,
        dirty_rect: None,
        show_pixel_grid: true,
        show_guidelines: false,
        mirror_mode: crate::canvas::MirrorMode::None,
        show_wrap_preview: false,
        preview_layer: None,
        preview_blend_mode: BlendMode::Normal,
        preview_force_composite: false,
        preview_is_eraser: false,
        preview_replaces_layer: false,
        preview_targets_mask: false,
        preview_mask_reveal: false,
        dirty_generation: 0,
        selection_mask: None,
        lod_composite_cache: None,
        lod_generation: 0,
        preview_dirty_rect: None,
        preview_texture_cache: None,
        preview_generation: 0,
        preview_stroke_bounds: None,
        preview_flat_buffer: Vec::new(),
        preview_flat_ready: false,
        preview_downscale: 1,
        composite_cpu_buffer: Vec::new(),
        composite_cpu_buffer_back: Vec::new(),
        region_extract_buf: Vec::new(),
        composite_above_buffer: Vec::new(),
        preview_premul_cache: Vec::new(),
        preview_cache_rect: None,
        selection_overlay_texture: None,
        selection_overlay_generation: 0,
        selection_overlay_built_generation: 0,
        selection_overlay_anim_offset: -1.0,
        selection_overlay_bounds: None,
        fill_commit_overlays: Vec::new(),
        selection_border_h_segs: Vec::new(),
        selection_border_v_segs: Vec::new(),
        selection_border_built_generation: u64::MAX,
        cmyk_preview: false,
        text_coverage_buf: Vec::new(),
        text_glyph_cache: Default::default(),
        text_editing_layer: None,
        canvas_widget_id: None,
    })
}

/// Load a v1 tiled project file
fn load_pfe_v1(raw: &[u8]) -> Result<CanvasState, PfeError> {
    let project: ProjectFileV1 = bincode::deserialize(raw)?;

    validate_open_dimensions(project.width, project.height).map_err(PfeError::InvalidFormat)?;
    if project.layers.len() > MAX_LAYERS {
        return Err(PfeError::InvalidFormat(format!(
            "Project contains {} layers, which exceeds the maximum of {}",
            project.layers.len(),
            MAX_LAYERS
        )));
    }

    let expected_chunk_bytes = (CHUNK_SIZE * CHUNK_SIZE * 4) as usize;

    let mut layers = Vec::with_capacity(project.layers.len());
    for ld in project.layers {
        let mut tiled = TiledImage::new(project.width, project.height);
        for cd in ld.chunks {
            if cd.pixels.len() != expected_chunk_bytes {
                return Err(PfeError::InvalidFormat(format!(
                    "Chunk ({},{}) in layer '{}' has {} bytes, expected {}",
                    cd.cx,
                    cd.cy,
                    ld.name,
                    cd.pixels.len(),
                    expected_chunk_bytes,
                )));
            }
            let chunk_img =
                RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, cd.pixels).ok_or_else(|| {
                    PfeError::InvalidFormat(format!(
                        "Failed to reconstruct chunk ({},{}) for layer '{}'",
                        cd.cx, cd.cy, ld.name
                    ))
                })?;
            tiled.set_chunk(cd.cx, cd.cy, chunk_img);
        }

        layers.push(Layer {
            name: ld.name,
            visible: ld.visible,
            folder_id: None,
            opacity: ld.opacity,
            blend_mode: BlendMode::from_u8(ld.blend_mode),
            pixels: tiled,
            mask: None,
            mask_enabled: true,
            lod_cache: None,
            gpu_generation: 0,
            content: crate::canvas::LayerContent::Raster,
            pixel_format: crate::canvas::PixelFormat::RgbaU8,
            hdr_metadata: crate::canvas::HdrMetadata::default(),
            source_metadata: crate::canvas::ImageMetadata::default(),
            deep_pixels: None,
        });
    }

    if layers.is_empty() {
        return Err(PfeError::InvalidFormat("Project contains no layers".into()));
    }

    let active = project.active_layer_index.min(layers.len() - 1);

    Ok(CanvasState {
        width: project.width,
        height: project.height,
        layers,
        layer_folders: Vec::new(),
        next_layer_folder_id: 1,
        active_layer_index: active,
        edit_layer_mask: false,
        composite_cache: None,
        dirty_rect: None,
        show_pixel_grid: true,
        show_guidelines: false,
        mirror_mode: crate::canvas::MirrorMode::None,
        show_wrap_preview: false,
        preview_layer: None,
        preview_blend_mode: BlendMode::Normal,
        preview_force_composite: false,
        preview_is_eraser: false,
        preview_replaces_layer: false,
        preview_targets_mask: false,
        preview_mask_reveal: false,
        dirty_generation: 0,
        selection_mask: None,
        lod_composite_cache: None,
        lod_generation: 0,
        preview_dirty_rect: None,
        preview_texture_cache: None,
        preview_generation: 0,
        preview_stroke_bounds: None,
        preview_flat_buffer: Vec::new(),
        preview_flat_ready: false,
        preview_downscale: 1,
        composite_cpu_buffer: Vec::new(),
        composite_cpu_buffer_back: Vec::new(),
        region_extract_buf: Vec::new(),
        composite_above_buffer: Vec::new(),
        preview_premul_cache: Vec::new(),
        preview_cache_rect: None,
        selection_overlay_texture: None,
        selection_overlay_generation: 0,
        selection_overlay_built_generation: 0,
        selection_overlay_anim_offset: -1.0,
        selection_overlay_bounds: None,
        fill_commit_overlays: Vec::new(),
        selection_border_h_segs: Vec::new(),
        selection_border_v_segs: Vec::new(),
        selection_border_built_generation: u64::MAX,
        cmyk_preview: false,
        text_coverage_buf: Vec::new(),
        text_glyph_cache: Default::default(),
        text_editing_layer: None,
        canvas_widget_id: None,
    })
}

/// Load a legacy v0 flat project file and convert to tiled
fn load_pfe_v0(raw: &[u8]) -> Result<CanvasState, PfeError> {
    let project: ProjectFileV0 = bincode::deserialize(raw)?;

    validate_open_dimensions(project.width, project.height).map_err(PfeError::InvalidFormat)?;
    if project.layers.len() > MAX_LAYERS {
        return Err(PfeError::InvalidFormat(format!(
            "Project contains {} layers, which exceeds the maximum of {}",
            project.layers.len(),
            MAX_LAYERS
        )));
    }

    let expected_pixel_count = (project.width as usize) * (project.height as usize) * 4;

    let mut layers = Vec::with_capacity(project.layers.len());
    for layer_data in project.layers {
        if layer_data.pixels.len() != expected_pixel_count {
            return Err(PfeError::InvalidFormat(format!(
                "Layer '{}' has {} bytes, expected {} ({}x{}x4)",
                layer_data.name,
                layer_data.pixels.len(),
                expected_pixel_count,
                project.width,
                project.height,
            )));
        }

        let flat = RgbaImage::from_raw(project.width, project.height, layer_data.pixels)
            .ok_or_else(|| {
                PfeError::InvalidFormat(format!(
                    "Failed to reconstruct pixels for layer '{}'",
                    layer_data.name
                ))
            })?;

        layers.push(Layer {
            name: layer_data.name,
            visible: layer_data.visible,
            folder_id: None,
            opacity: layer_data.opacity,
            blend_mode: BlendMode::from_u8(layer_data.blend_mode),
            pixels: TiledImage::from_rgba_image(&flat),
            mask: None,
            mask_enabled: true,
            lod_cache: None,
            gpu_generation: 0,
            content: crate::canvas::LayerContent::Raster,
            pixel_format: crate::canvas::PixelFormat::RgbaU8,
            hdr_metadata: crate::canvas::HdrMetadata::default(),
            source_metadata: crate::canvas::ImageMetadata::default(),
            deep_pixels: None,
        });
    }

    if layers.is_empty() {
        return Err(PfeError::InvalidFormat("Project contains no layers".into()));
    }

    let active = project.active_layer_index.min(layers.len() - 1);

    Ok(CanvasState {
        width: project.width,
        height: project.height,
        layers,
        layer_folders: Vec::new(),
        next_layer_folder_id: 1,
        active_layer_index: active,
        edit_layer_mask: false,
        composite_cache: None,
        dirty_rect: None,
        show_pixel_grid: true,
        show_guidelines: false,
        mirror_mode: crate::canvas::MirrorMode::None,
        show_wrap_preview: false,
        preview_layer: None,
        preview_blend_mode: BlendMode::Normal,
        preview_force_composite: false,
        preview_is_eraser: false,
        preview_replaces_layer: false,
        preview_targets_mask: false,
        preview_mask_reveal: false,
        dirty_generation: 0,
        selection_mask: None,
        lod_composite_cache: None,
        lod_generation: 0,
        preview_dirty_rect: None,
        preview_texture_cache: None,
        preview_generation: 0,
        preview_stroke_bounds: None,
        preview_flat_buffer: Vec::new(),
        preview_flat_ready: false,
        preview_downscale: 1,
        composite_cpu_buffer: Vec::new(),
        composite_cpu_buffer_back: Vec::new(),
        region_extract_buf: Vec::new(),
        composite_above_buffer: Vec::new(),
        preview_premul_cache: Vec::new(),
        preview_cache_rect: None,
        selection_overlay_texture: None,
        selection_overlay_generation: 0,
        selection_overlay_built_generation: 0,
        selection_overlay_anim_offset: -1.0,
        selection_overlay_bounds: None,
        fill_commit_overlays: Vec::new(),
        selection_border_h_segs: Vec::new(),
        selection_border_v_segs: Vec::new(),
        selection_border_built_generation: u64::MAX,
        cmyk_preview: false,
        text_coverage_buf: Vec::new(),
        text_glyph_cache: Default::default(),
        text_editing_layer: None,
        canvas_widget_id: None,
    })
}

// ============================================================================
// THREAD-SAFE IMAGE ENCODING
// ============================================================================

#[derive(Clone, Debug)]
pub enum PreparedExportImage {
    Rgba8(RgbaImage),
    Rgba16 {
        width: u32,
        height: u32,
        pixels: Vec<u16>,
    },
    RgbaF32 {
        width: u32,
        height: u32,
        pixels: Vec<f32>,
    },
}

impl PreparedExportImage {
    fn rgba8(&self) -> RgbaImage {
        match self {
            PreparedExportImage::Rgba8(image) => image.clone(),
            PreparedExportImage::Rgba16 {
                width,
                height,
                pixels,
            } => {
                let data = pixels
                    .iter()
                    .map(|&v| ((v as u32 + 128) / 257) as u8)
                    .collect();
                RgbaImage::from_raw(*width, *height, data)
                    .unwrap_or_else(|| RgbaImage::new(*width, *height))
            }
            PreparedExportImage::RgbaF32 {
                width,
                height,
                pixels,
            } => {
                let mut data = Vec::with_capacity(pixels.len());
                for px in pixels.chunks_exact(4) {
                    let mapped = if px[0] > 1.0 || px[1] > 1.0 || px[2] > 1.0 {
                        reinhard_tone_map_rgba([px[0], px[1], px[2], px[3]], 1.0)
                    } else {
                        Rgba([
                            (px[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (px[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (px[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (px[3].clamp(0.0, 1.0) * 255.0).round() as u8,
                        ])
                    };
                    data.extend_from_slice(&mapped.0);
                }
                RgbaImage::from_raw(*width, *height, data)
                    .unwrap_or_else(|| RgbaImage::new(*width, *height))
            }
        }
    }
}

pub fn prepare_export_image(state: &CanvasState) -> PreparedExportImage {
    if let Some(deep_adjusted) = adjusted_deep_export(state) {
        return deep_adjusted;
    }
    if let Some(deep) = exact_single_layer_deep_export(state) {
        return deep;
    }

    let composite = state.composite();
    if state.layers.iter().enumerate().any(|(idx, layer)| {
        state.layer_effectively_visible(idx)
            && (layer.hdr_metadata.enabled
                || matches!(
                    layer.pixel_format,
                    PixelFormat::RgbaF16 | PixelFormat::RgbaF32
                ))
    }) {
        return PreparedExportImage::RgbaF32 {
            width: composite.width(),
            height: composite.height(),
            pixels: composite
                .as_raw()
                .iter()
                .map(|&v| v as f32 / 255.0)
                .collect(),
        };
    }
    if state.layers.iter().enumerate().any(|(idx, layer)| {
        state.layer_effectively_visible(idx) && layer.pixel_format == PixelFormat::RgbaU16
    }) {
        return PreparedExportImage::Rgba16 {
            width: composite.width(),
            height: composite.height(),
            pixels: composite
                .as_raw()
                .iter()
                .map(|&v| (v as u16) * 257)
                .collect(),
        };
    }
    PreparedExportImage::Rgba8(composite)
}

fn adjusted_deep_export(state: &CanvasState) -> Option<PreparedExportImage> {
    let visible_layers: Vec<&Layer> = state
        .layers
        .iter()
        .enumerate()
        .filter_map(|(idx, layer)| state.layer_effectively_visible(idx).then_some(layer))
        .collect();
    let base = *visible_layers.first()?;
    if visible_layers.len() < 2
        || !matches!(base.content, LayerContent::Raster)
        || base.opacity < 0.999
        || base.blend_mode != BlendMode::Normal
        || base.mask.is_some()
    {
        return None;
    }
    if !visible_layers
        .iter()
        .skip(1)
        .all(|layer| matches!(layer.content, LayerContent::Adjustment(_)))
    {
        return None;
    }
    let mut pixels = deep_buffer_to_f32(base.deep_pixels.as_ref()?, state.width, state.height)?;
    if base
        .deep_pixels
        .as_ref()?
        .to_rgba8(state.width, state.height)?
        != base.pixels.to_rgba_image()
    {
        return None;
    }

    for layer in visible_layers.iter().skip(1) {
        let LayerContent::Adjustment(adj) = &layer.content else {
            return None;
        };
        for px in pixels.chunks_exact_mut(4) {
            let out = adj.apply_to_f32_with_opacity([px[0], px[1], px[2], px[3]], layer.opacity);
            px.copy_from_slice(&out);
        }
    }

    if base.hdr_metadata.enabled
        || matches!(
            base.pixel_format,
            PixelFormat::RgbaF16 | PixelFormat::RgbaF32
        )
    {
        Some(PreparedExportImage::RgbaF32 {
            width: state.width,
            height: state.height,
            pixels,
        })
    } else if base.pixel_format == PixelFormat::RgbaU16 {
        Some(PreparedExportImage::Rgba16 {
            width: state.width,
            height: state.height,
            pixels: pixels
                .iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
                .collect(),
        })
    } else {
        None
    }
}

fn deep_buffer_to_f32(deep: &DeepRgbaBuffer, width: u32, height: u32) -> Option<Vec<f32>> {
    let expected = (width as usize) * (height as usize) * 4;
    match deep {
        DeepRgbaBuffer::U8(values) if values.len() == expected => {
            Some(values.iter().map(|&v| v as f32 / 255.0).collect())
        }
        DeepRgbaBuffer::U16(values) if values.len() == expected => {
            Some(values.iter().map(|&v| v as f32 / 65535.0).collect())
        }
        DeepRgbaBuffer::F16(values) if values.len() == expected => {
            Some(values.iter().map(|&v| f16_bits_to_f32(v)).collect())
        }
        DeepRgbaBuffer::F32(values) if values.len() == expected => Some(values.clone()),
        _ => None,
    }
}

fn exact_single_layer_deep_export(state: &CanvasState) -> Option<PreparedExportImage> {
    let visible_layers: Vec<&Layer> = state
        .layers
        .iter()
        .enumerate()
        .filter_map(|(idx, layer)| state.layer_effectively_visible(idx).then_some(layer))
        .collect();
    let layer = *visible_layers.first()?;
    if visible_layers.len() != 1
        || !matches!(layer.content, LayerContent::Raster)
        || layer.opacity < 0.999
        || layer.blend_mode != BlendMode::Normal
        || layer.mask.is_some()
    {
        return None;
    }

    let deep = layer.deep_pixels.as_ref()?;
    if deep.to_rgba8(state.width, state.height)? != layer.pixels.to_rgba_image() {
        return None;
    }

    match deep {
        DeepRgbaBuffer::U8(pixels) => {
            RgbaImage::from_raw(state.width, state.height, pixels.clone())
                .map(PreparedExportImage::Rgba8)
        }
        DeepRgbaBuffer::U16(pixels) => Some(PreparedExportImage::Rgba16 {
            width: state.width,
            height: state.height,
            pixels: pixels.clone(),
        }),
        DeepRgbaBuffer::F16(pixels) => Some(PreparedExportImage::RgbaF32 {
            width: state.width,
            height: state.height,
            pixels: pixels.iter().map(|&v| f16_bits_to_f32(v)).collect(),
        }),
        DeepRgbaBuffer::F32(pixels) => Some(PreparedExportImage::RgbaF32 {
            width: state.width,
            height: state.height,
            pixels: pixels.clone(),
        }),
    }
}

pub fn encode_prepared_and_write(
    image: PreparedExportImage,
    path: &Path,
    format: SaveFormat,
    quality: u8,
    tiff_compression: TiffCompression,
) -> Result<(), ImageError> {
    match (&image, format) {
        (
            PreparedExportImage::Rgba16 {
                width,
                height,
                pixels,
            },
            SaveFormat::Png,
        ) => return write_png16(*width, *height, pixels, path),
        (
            PreparedExportImage::Rgba16 {
                width,
                height,
                pixels,
            },
            SaveFormat::Tiff,
        ) => return write_tiff16(*width, *height, pixels, path, tiff_compression),
        (
            PreparedExportImage::RgbaF32 {
                width,
                height,
                pixels,
            },
            SaveFormat::Tiff,
        ) => return write_tiff_f32(*width, *height, pixels, path),
        _ => {}
    }

    let rgba8 = image.rgba8();
    encode_and_write(&rgba8, path, format, quality, tiff_compression)
}

pub fn encode_canvas_state_and_write(
    state: &CanvasState,
    path: &Path,
    format: SaveFormat,
    quality: u8,
    tiff_compression: TiffCompression,
) -> Result<(), ImageError> {
    encode_prepared_and_write(
        prepare_export_image(state),
        path,
        format,
        quality,
        tiff_compression,
    )
}

fn write_png16(width: u32, height: u32, pixels: &[u16], path: &Path) -> Result<(), ImageError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut png_writer = encoder
        .write_header()
        .map_err(|e| ImageError::IoError(std::io::Error::other(e.to_string())))?;
    let mut bytes = Vec::with_capacity(pixels.len() * 2);
    for value in pixels {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    png_writer
        .write_image_data(&bytes)
        .map_err(|e| ImageError::IoError(std::io::Error::other(e.to_string())))?;
    Ok(())
}

fn write_tiff16(
    width: u32,
    height: u32,
    pixels: &[u16],
    path: &Path,
    tiff_compression: TiffCompression,
) -> Result<(), ImageError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let err_map = |e: tiff::TiffError| {
        ImageError::IoError(std::io::Error::other(format!("TIFF encode error: {}", e)))
    };
    let mut tiff_enc = tiff::encoder::TiffEncoder::new(&mut writer).map_err(err_map)?;
    match tiff_compression {
        TiffCompression::None => tiff_enc
            .write_image::<tiff::encoder::colortype::RGBA16>(width, height, pixels)
            .map_err(err_map)?,
        TiffCompression::Lzw => tiff_enc
            .write_image_with_compression::<tiff::encoder::colortype::RGBA16, _>(
                width,
                height,
                tiff::encoder::compression::Lzw,
                pixels,
            )
            .map_err(err_map)?,
        TiffCompression::Deflate => tiff_enc
            .write_image_with_compression::<tiff::encoder::colortype::RGBA16, _>(
                width,
                height,
                tiff::encoder::compression::Deflate::default(),
                pixels,
            )
            .map_err(err_map)?,
    }
    Ok(())
}

fn write_tiff_f32(width: u32, height: u32, pixels: &[f32], path: &Path) -> Result<(), ImageError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let err_map = |e: tiff::TiffError| {
        ImageError::IoError(std::io::Error::other(format!("TIFF encode error: {}", e)))
    };
    let mut tiff_enc = tiff::encoder::TiffEncoder::new(&mut writer).map_err(err_map)?;
    tiff_enc
        .write_image::<tiff::encoder::colortype::RGBA32Float>(width, height, pixels)
        .map_err(err_map)?;
    Ok(())
}

/// Encode and write an image to a file.
/// This is a standalone function (no `&mut self`) so it can be called from
/// background threads via `rayon::spawn`.
pub fn encode_and_write(
    image: &RgbaImage,
    path: &Path,
    format: SaveFormat,
    quality: u8,
    tiff_compression: TiffCompression,
) -> Result<(), ImageError> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    match format {
        SaveFormat::Png => {
            let encoder = PngEncoder::new(&mut writer);
            encoder.write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgba8,
            )?;
        }
        SaveFormat::Jpeg => {
            let rgb_image = DynamicImage::ImageRgba8(image.clone()).to_rgb8();
            let encoder = JpegEncoder::new_with_quality(&mut writer, quality);
            encoder.write_image(
                rgb_image.as_raw(),
                rgb_image.width(),
                rgb_image.height(),
                image::ExtendedColorType::Rgb8,
            )?;
        }
        SaveFormat::Webp => {
            let dyn_img = DynamicImage::ImageRgba8(image.clone());
            dyn_img.save(path)?;
        }
        SaveFormat::Bmp => {
            let encoder = BmpEncoder::new(&mut writer);
            encoder.write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgba8,
            )?;
        }
        SaveFormat::Tga => {
            let encoder = TgaEncoder::new(&mut writer);
            encoder.write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgba8,
            )?;
        }
        SaveFormat::Ico => {
            // ICO entries limited to 256×256; scale down if needed
            let dyn_img = if image.width() > 256 || image.height() > 256 {
                let scale = 256.0 / image.width().max(image.height()) as f32;
                let new_w = ((image.width() as f32 * scale) as u32).max(1);
                let new_h = ((image.height() as f32 * scale) as u32).max(1);
                let resized = image::imageops::resize(
                    image,
                    new_w,
                    new_h,
                    image::imageops::FilterType::Lanczos3,
                );
                DynamicImage::ImageRgba8(resized)
            } else {
                DynamicImage::ImageRgba8(image.clone())
            };
            dyn_img.write_to(&mut writer, image::ImageFormat::Ico)?;
        }
        SaveFormat::Tiff => {
            let err_map = |e: tiff::TiffError| {
                ImageError::IoError(std::io::Error::other(format!("TIFF encode error: {}", e)))
            };
            let mut tiff_enc = tiff::encoder::TiffEncoder::new(&mut writer).map_err(err_map)?;
            match tiff_compression {
                TiffCompression::None => {
                    tiff_enc
                        .write_image::<tiff::encoder::colortype::RGBA8>(
                            image.width(),
                            image.height(),
                            image.as_raw(),
                        )
                        .map_err(err_map)?;
                }
                TiffCompression::Lzw => {
                    tiff_enc
                        .write_image_with_compression::<tiff::encoder::colortype::RGBA8, _>(
                            image.width(),
                            image.height(),
                            tiff::encoder::compression::Lzw,
                            image.as_raw(),
                        )
                        .map_err(err_map)?;
                }
                TiffCompression::Deflate => {
                    tiff_enc
                        .write_image_with_compression::<tiff::encoder::colortype::RGBA8, _>(
                            image.width(),
                            image.height(),
                            tiff::encoder::compression::Deflate::default(),
                            image.as_raw(),
                        )
                        .map_err(err_map)?;
                }
            }
        }
        SaveFormat::Pfe => {
            unreachable!("PFE format should be handled via save_pfe(), not encode_and_write()");
        }
        SaveFormat::Gif => {
            // Static GIF: quantize to 256 colors and save single frame
            encode_static_gif(image, path)
                .map_err(|e| ImageError::IoError(std::io::Error::other(e)))?;
        }
    }

    Ok(())
}

// ============================================================================
// FILE HANDLER
// ============================================================================

pub struct FileHandler {
    /// Current file path (None if new/unsaved file)
    pub current_path: Option<PathBuf>,
    /// Last used save format
    pub last_format: SaveFormat,
    /// Last used quality setting
    pub last_quality: u8,
    /// Last used TIFF compression setting
    pub last_tiff_compression: TiffCompression,
    /// Whether last save was animated
    pub last_animated: bool,
    /// Last used animation FPS
    pub last_animation_fps: f32,
    /// Last used GIF color depth (bits: 1-8, 8 = 256 colors)
    pub last_gif_colors: u16,
    /// Last used GIF dithering setting
    pub last_gif_dither: bool,
}

impl Default for FileHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl FileHandler {
    pub fn new() -> Self {
        Self {
            current_path: None,
            last_format: SaveFormat::Png,
            last_quality: 90,
            last_tiff_compression: TiffCompression::None,
            last_animated: false,
            last_animation_fps: 10.0,
            last_gif_colors: 256,
            last_gif_dither: true,
        }
    }

    /// Returns true if there's a current file path (file has been saved before)
    pub fn has_current_path(&self) -> bool {
        self.current_path.is_some()
    }

    /// Show native file dialog to pick one or more file paths (without loading them).
    /// Returns a Vec of selected paths — empty if the dialog was cancelled.
    /// Supports multi-select so Linux/Wayland users can open multiple files without drag-and-drop.
    pub fn pick_file_paths(&self) -> Vec<PathBuf> {
        FileDialog::new()
            .add_filter(
                "All Supported",
                &[
                    "pfe", "png", "jpg", "jpeg", "webp", "bmp", "tga", "gif", "ico", "tiff", "tif",
                    "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "dng", "orf", "rw2", "pef",
                    "raf", "raw", "rwl", "srw", "x3f", "3fr", "fff", "iiq", "mrw", "mef", "mos",
                    "kdc", "dcr", "erf",
                ],
            )
            .add_filter("PaintFE Project", &["pfe"])
            .add_filter(
                "Images",
                &[
                    "png", "jpg", "jpeg", "webp", "bmp", "tga", "gif", "ico", "tiff", "tif",
                ],
            )
            .add_filter(
                "RAW Files",
                &[
                    "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "dng", "orf", "rw2", "pef",
                    "raf", "raw", "rwl", "srw", "x3f", "3fr", "fff", "iiq", "mrw", "mef", "mos",
                    "kdc", "dcr", "erf",
                ],
            )
            .add_filter("All Files", &["*"])
            .pick_files()
            .unwrap_or_default()
    }

    /// Open an image file using native file dialog
    /// Returns the loaded image and its path on success
    pub fn open_image(&mut self) -> Option<(RgbaImage, PathBuf)> {
        let path = FileDialog::new()
            .add_filter(
                "All Supported",
                &[
                    "pfe", "png", "jpg", "jpeg", "webp", "bmp", "tga", "gif", "ico", "tiff", "tif",
                    "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "dng", "orf", "rw2", "pef",
                    "raf", "raw", "rwl", "srw", "x3f", "3fr", "fff", "iiq", "mrw", "mef", "mos",
                    "kdc", "dcr", "erf",
                ],
            )
            .add_filter("PaintFE Project", &["pfe"])
            .add_filter(
                "Images",
                &[
                    "png", "jpg", "jpeg", "webp", "bmp", "tga", "gif", "ico", "tiff", "tif",
                ],
            )
            .add_filter(
                "RAW Files",
                &[
                    "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "dng", "orf", "rw2", "pef",
                    "raf", "raw", "rwl", "srw", "x3f", "3fr", "fff", "iiq", "mrw", "mef", "mos",
                    "kdc", "dcr", "erf",
                ],
            )
            .add_filter("All Files", &["*"])
            .pick_file()?;

        if let Ok((w, h)) = image::image_dimensions(&path)
            && let Err(e) = validate_open_dimensions(w, h)
        {
            eprintln!("{}", e);
            return None;
        }

        match image::open(&path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                if let Err(e) = validate_open_dimensions(rgba.width(), rgba.height()) {
                    eprintln!("{}", e);
                    return None;
                }
                self.current_path = Some(path.clone());

                // Detect format from extension
                if let Some(ext) = path.extension() {
                    self.last_format = match ext.to_string_lossy().to_lowercase().as_str() {
                        "png" => SaveFormat::Png,
                        "jpg" | "jpeg" => SaveFormat::Jpeg,
                        "webp" => SaveFormat::Webp,
                        "bmp" => SaveFormat::Bmp,
                        "tga" => SaveFormat::Tga,
                        "ico" => SaveFormat::Ico,
                        "tiff" | "tif" => SaveFormat::Tiff,
                        "gif" => SaveFormat::Gif,
                        _ => SaveFormat::Png,
                    };
                }

                Some((rgba, path))
            }
            Err(e) => {
                eprintln!("Failed to open image: {}", e);
                None
            }
        }
    }

    /// Save an image to a specific path with format and quality settings
    pub fn save_image(
        &mut self,
        image: &RgbaImage,
        path: &Path,
        format: SaveFormat,
        quality: u8,
        tiff_compression: TiffCompression,
    ) -> Result<(), ImageError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        match format {
            SaveFormat::Png => {
                let encoder = PngEncoder::new(&mut writer);
                encoder.write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgba8,
                )?;
            }
            SaveFormat::Jpeg => {
                // JPEG doesn't support alpha, convert to RGB
                let rgb_image = DynamicImage::ImageRgba8(image.clone()).to_rgb8();
                let encoder = JpegEncoder::new_with_quality(&mut writer, quality);
                encoder.write_image(
                    rgb_image.as_raw(),
                    rgb_image.width(),
                    rgb_image.height(),
                    image::ExtendedColorType::Rgb8,
                )?;
            }
            SaveFormat::Webp => {
                let dyn_img = DynamicImage::ImageRgba8(image.clone());
                dyn_img.save(path)?;
            }
            SaveFormat::Bmp => {
                let encoder = BmpEncoder::new(&mut writer);
                encoder.write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgba8,
                )?;
            }
            SaveFormat::Tga => {
                let encoder = TgaEncoder::new(&mut writer);
                encoder.write_image(
                    image.as_raw(),
                    image.width(),
                    image.height(),
                    image::ExtendedColorType::Rgba8,
                )?;
            }
            SaveFormat::Ico => {
                // ICO entries limited to 256×256; scale down if needed
                let dyn_img = if image.width() > 256 || image.height() > 256 {
                    let scale = 256.0 / image.width().max(image.height()) as f32;
                    let new_w = ((image.width() as f32 * scale) as u32).max(1);
                    let new_h = ((image.height() as f32 * scale) as u32).max(1);
                    let resized = image::imageops::resize(
                        image,
                        new_w,
                        new_h,
                        image::imageops::FilterType::Lanczos3,
                    );
                    DynamicImage::ImageRgba8(resized)
                } else {
                    DynamicImage::ImageRgba8(image.clone())
                };
                dyn_img.write_to(&mut writer, image::ImageFormat::Ico)?;
            }
            SaveFormat::Tiff => {
                let err_map = |e: tiff::TiffError| {
                    ImageError::IoError(std::io::Error::other(format!("TIFF encode error: {}", e)))
                };
                let mut tiff_enc = tiff::encoder::TiffEncoder::new(&mut writer).map_err(err_map)?;
                match tiff_compression {
                    TiffCompression::None => {
                        tiff_enc
                            .write_image::<tiff::encoder::colortype::RGBA8>(
                                image.width(),
                                image.height(),
                                image.as_raw(),
                            )
                            .map_err(err_map)?;
                    }
                    TiffCompression::Lzw => {
                        tiff_enc
                            .write_image_with_compression::<tiff::encoder::colortype::RGBA8, _>(
                                image.width(),
                                image.height(),
                                tiff::encoder::compression::Lzw,
                                image.as_raw(),
                            )
                            .map_err(err_map)?;
                    }
                    TiffCompression::Deflate => {
                        tiff_enc
                            .write_image_with_compression::<tiff::encoder::colortype::RGBA8, _>(
                                image.width(),
                                image.height(),
                                tiff::encoder::compression::Deflate::default(),
                                image.as_raw(),
                            )
                            .map_err(err_map)?;
                    }
                }
            }
            SaveFormat::Pfe => {
                // PFE projects are saved via save_pfe(), not save_image()
                unreachable!("PFE format should be handled via save_pfe(), not save_image()");
            }
            SaveFormat::Gif => {
                // Static GIF: quantize to 256 colors and save single frame
                encode_static_gif(image, path)
                    .map_err(|e| ImageError::IoError(std::io::Error::other(e)))?;
            }
        }

        // Update state
        self.current_path = Some(path.to_path_buf());
        self.last_format = format;
        self.last_quality = quality;
        self.last_tiff_compression = tiff_compression;

        Ok(())
    }

    /// Quick save to current path with last used settings
    /// Returns Err if no current path is set
    pub fn quick_save(&mut self, image: &RgbaImage) -> Result<(), ImageError> {
        if let Some(path) = &self.current_path.clone() {
            self.save_image(
                image,
                path,
                self.last_format,
                self.last_quality,
                self.last_tiff_compression,
            )
        } else {
            Err(ImageError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No file path set for quick save",
            )))
        }
    }

    /// Open an image file and return the data needed to create a Project.
    /// Returns (image, path, file_handler_with_state) on success.
    pub fn open_image_for_project(&mut self) -> Option<(RgbaImage, PathBuf, FileHandler)> {
        let (image, path) = self.open_image()?;

        // Clone the file handler state for the project
        let project_handler = FileHandler {
            current_path: Some(path.clone()),
            last_format: self.last_format,
            last_quality: self.last_quality,
            last_tiff_compression: self.last_tiff_compression,
            last_animated: self.last_animated,
            last_animation_fps: self.last_animation_fps,
            last_gif_colors: self.last_gif_colors,
            last_gif_dither: self.last_gif_dither,
        };

        Some((image, path, project_handler))
    }
}

// ============================================================================
// ANIMATION SUPPORT — GIF & APNG
// ============================================================================

/// Information about an animated file detected from its header.
pub struct AnimationInfo {
    pub is_animated: bool,
    pub frame_count: u32,
    pub avg_delay_ms: u16,
}

/// Decode all frames from an animated GIF file.
/// Returns Vec<(frame_rgba, delay_ms)> where each frame is composited
/// according to GIF disposal methods.
pub fn decode_gif_frames(path: &Path) -> Result<Vec<(RgbaImage, u16)>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open GIF: {}", e))?;
    let mut decoder = gif::DecodeOptions::new();
    decoder.set_color_output(gif::ColorOutput::RGBA);
    let mut decoder = decoder
        .read_info(BufReader::new(file))
        .map_err(|e| format!("Failed to read GIF info: {}", e))?;

    let width = decoder.width() as u32;
    let height = decoder.height() as u32;
    validate_open_dimensions(width, height)?;

    let mut frames: Vec<(RgbaImage, u16)> = Vec::new();
    // Running canvas for frame composition (GIF frames can be partial)
    let mut canvas = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
    // Previous canvas state for RestoreToPrevious disposal
    let mut prev_canvas = canvas.clone();

    while let Some(frame) = decoder
        .read_next_frame()
        .map_err(|e| format!("GIF frame decode error: {}", e))?
    {
        let frame_x = frame.left as u32;
        let frame_y = frame.top as u32;
        let frame_w = frame.width as u32;
        let frame_h = frame.height as u32;
        let delay_ms = ((frame.delay as u32) * 10).min(65535) as u16; // GIF delay is in centiseconds
        let disposal = frame.dispose;

        // Save canvas before applying frame (for RestoreToPrevious)
        if disposal == gif::DisposalMethod::Previous {
            prev_canvas = canvas.clone();
        }

        // Apply frame pixels onto canvas
        let frame_buf = &frame.buffer;
        for fy in 0..frame_h {
            for fx in 0..frame_w {
                let cx = frame_x + fx;
                let cy = frame_y + fy;
                if cx < width && cy < height {
                    let idx = ((fy * frame_w + fx) * 4) as usize;
                    if idx + 3 < frame_buf.len() {
                        let r = frame_buf[idx];
                        let g = frame_buf[idx + 1];
                        let b = frame_buf[idx + 2];
                        let a = frame_buf[idx + 3];
                        if a > 0 {
                            canvas.put_pixel(cx, cy, Rgba([r, g, b, a]));
                        }
                    }
                }
            }
        }

        // Store the composited frame
        frames.push((canvas.clone(), delay_ms.max(MIN_FRAME_DELAY_MS))); // Min 10ms delay

        // Handle disposal
        match disposal {
            gif::DisposalMethod::Background => {
                // Clear the frame area to transparent
                for fy in 0..frame_h {
                    for fx in 0..frame_w {
                        let cx = frame_x + fx;
                        let cy = frame_y + fy;
                        if cx < width && cy < height {
                            canvas.put_pixel(cx, cy, Rgba([0, 0, 0, 0]));
                        }
                    }
                }
            }
            gif::DisposalMethod::Previous => {
                canvas = prev_canvas.clone();
            }
            _ => {
                // Keep / Any — leave canvas as-is
            }
        }
    }

    if frames.is_empty() {
        return Err("GIF contains no frames".to_string());
    }

    Ok(frames)
}

/// Decode all frames from an animated PNG (APNG) file.
/// Returns Vec<(frame_rgba, delay_ms)>.
pub fn decode_apng_frames(path: &Path) -> Result<Vec<(RgbaImage, u16)>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open APNG: {}", e))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    // Expand indexed/paletted, grayscale <8-bit, and tRNS chunks to full RGB(A)
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("Failed to read PNG info: {}", e))?;
    let info = reader.info();

    let width = info.width;
    let height = info.height;
    validate_open_dimensions(width, height)?;
    let _color_type = info.color_type;
    let _bit_depth = info.bit_depth;

    // Check if this is actually animated
    let anim_ctrl = info.animation_control().cloned();
    if anim_ctrl.is_none() {
        // Not animated — just decode as single frame
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let out_info = reader
            .next_frame(&mut buf)
            .map_err(|e| format!("PNG decode error: {}", e))?;
        let rgba = convert_png_buffer_to_rgba(
            &buf[..out_info.buffer_size()],
            width,
            height,
            out_info.color_type,
            out_info.bit_depth,
        )?;
        return Ok(vec![(rgba, 100)]); // 100ms default
    }

    let num_frames = anim_ctrl.unwrap().num_frames;
    let mut frames = Vec::with_capacity(num_frames as usize);
    let mut canvas = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));

    for _ in 0..num_frames {
        let mut buf = vec![0u8; reader.output_buffer_size()];
        match reader.next_frame(&mut buf) {
            Ok(out_info) => {
                let frame_data = &buf[..out_info.buffer_size()];

                // Get frame control info
                let (delay_ms, f_x, f_y, f_w, f_h) = if let Some(fc) = reader.info().frame_control()
                {
                    let delay = if fc.delay_den == 0 {
                        (fc.delay_num as u32 * 10).min(65535) as u16
                    } else {
                        ((fc.delay_num as f64 / fc.delay_den as f64) * 1000.0).min(65535.0) as u16
                    };
                    (
                        delay.max(MIN_FRAME_DELAY_MS),
                        fc.x_offset,
                        fc.y_offset,
                        fc.width,
                        fc.height,
                    )
                } else {
                    (100u16, 0, 0, width, height)
                };

                // Convert frame data to RGBA (use out_info color type after EXPAND transformation)
                let frame_rgba = convert_png_buffer_to_rgba(
                    frame_data,
                    f_w,
                    f_h,
                    out_info.color_type,
                    out_info.bit_depth,
                )?;

                // Composite frame onto canvas
                for fy in 0..f_h {
                    for fx in 0..f_w {
                        let cx = f_x + fx;
                        let cy = f_y + fy;
                        if cx < width && cy < height {
                            let pixel = *frame_rgba.get_pixel(fx, fy);
                            if pixel[3] > 0 {
                                canvas.put_pixel(cx, cy, pixel);
                            }
                        }
                    }
                }

                frames.push((canvas.clone(), delay_ms));
            }
            Err(_) => break,
        }
    }

    if frames.is_empty() {
        return Err("APNG contains no frames".to_string());
    }

    Ok(frames)
}

/// Convert PNG output buffer to RGBA based on color type.
fn convert_png_buffer_to_rgba(
    buf: &[u8],
    width: u32,
    height: u32,
    color_type: png::ColorType,
    _bit_depth: png::BitDepth,
) -> Result<RgbaImage, String> {
    let pixels = width as usize * height as usize;
    match color_type {
        png::ColorType::Rgba => RgbaImage::from_raw(width, height, buf.to_vec())
            .ok_or_else(|| "Failed to create RGBA image from PNG buffer".to_string()),
        png::ColorType::Rgb => {
            if buf.len() < pixels * 3 {
                return Err("RGB buffer too small".to_string());
            }
            let mut rgba = Vec::with_capacity(pixels * 4);
            for chunk in buf[..pixels * 3].chunks_exact(3) {
                rgba.push(chunk[0]);
                rgba.push(chunk[1]);
                rgba.push(chunk[2]);
                rgba.push(255);
            }
            RgbaImage::from_raw(width, height, rgba)
                .ok_or_else(|| "Failed to create RGBA image from RGB PNG buffer".to_string())
        }
        png::ColorType::GrayscaleAlpha => {
            if buf.len() < pixels * 2 {
                return Err("GrayscaleAlpha buffer too small".to_string());
            }
            let mut rgba = Vec::with_capacity(pixels * 4);
            for chunk in buf[..pixels * 2].chunks_exact(2) {
                let g = chunk[0];
                let a = chunk[1];
                rgba.push(g);
                rgba.push(g);
                rgba.push(g);
                rgba.push(a);
            }
            RgbaImage::from_raw(width, height, rgba).ok_or_else(|| {
                "Failed to create RGBA image from GrayscaleAlpha PNG buffer".to_string()
            })
        }
        png::ColorType::Grayscale => {
            if buf.len() < pixels {
                return Err("Grayscale buffer too small".to_string());
            }
            let mut rgba = Vec::with_capacity(pixels * 4);
            for &g in &buf[..pixels] {
                rgba.push(g);
                rgba.push(g);
                rgba.push(g);
                rgba.push(255);
            }
            RgbaImage::from_raw(width, height, rgba)
                .ok_or_else(|| "Failed to create RGBA image from Grayscale PNG buffer".to_string())
        }
        png::ColorType::Indexed => {
            // With EXPAND transformation, indexed PNGs are decoded to RGB or RGBA.
            // If we still reach here, try RGBA first, then RGB.
            if buf.len() >= pixels * 4 {
                RgbaImage::from_raw(width, height, buf[..pixels * 4].to_vec())
                    .ok_or_else(|| "Failed to handle indexed PNG as RGBA".to_string())
            } else if buf.len() >= pixels * 3 {
                let mut rgba = Vec::with_capacity(pixels * 4);
                for chunk in buf[..pixels * 3].chunks_exact(3) {
                    rgba.push(chunk[0]);
                    rgba.push(chunk[1]);
                    rgba.push(chunk[2]);
                    rgba.push(255);
                }
                RgbaImage::from_raw(width, height, rgba)
                    .ok_or_else(|| "Failed to handle indexed PNG as RGB".to_string())
            } else {
                Err(format!(
                    "Indexed PNG buffer too small: {} bytes for {}x{}",
                    buf.len(),
                    width,
                    height
                ))
            }
        }
    }
}

/// Quick-peek a file to detect if it's animated and get frame count.
pub fn detect_animation(path: &Path) -> AnimationInfo {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "gif" => detect_gif_animation(path),
        "png" => detect_png_animation(path),
        _ => AnimationInfo {
            is_animated: false,
            frame_count: 1,
            avg_delay_ms: 100,
        },
    }
}

fn detect_gif_animation(path: &Path) -> AnimationInfo {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            return AnimationInfo {
                is_animated: false,
                frame_count: 1,
                avg_delay_ms: 100,
            };
        }
    };

    let mut decoder = gif::DecodeOptions::new();
    decoder.set_color_output(gif::ColorOutput::RGBA);
    let mut decoder = match decoder.read_info(BufReader::new(file)) {
        Ok(d) => d,
        Err(_) => {
            return AnimationInfo {
                is_animated: false,
                frame_count: 1,
                avg_delay_ms: 100,
            };
        }
    };

    let mut frame_count = 0u32;
    let mut total_delay = 0u32;

    while let Ok(Some(frame)) = decoder.read_next_frame() {
        frame_count += 1;
        total_delay += frame.delay as u32 * 10; // centiseconds to ms
        // Only need to count for detection, but we iterate all frames
        // to get accurate count. For very large GIFs, could stop early.
        if frame_count > 100 {
            // For huge GIFs, extrapolate
            break;
        }
    }

    let avg_delay_ms = total_delay
        .checked_div(frame_count)
        .map(|d| d.max(MIN_FRAME_DELAY_MS as u32) as u16)
        .unwrap_or(100);

    AnimationInfo {
        is_animated: frame_count > 1,
        frame_count,
        avg_delay_ms,
    }
}

fn detect_png_animation(path: &Path) -> AnimationInfo {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            return AnimationInfo {
                is_animated: false,
                frame_count: 1,
                avg_delay_ms: 100,
            };
        }
    };

    let decoder = png::Decoder::new(BufReader::new(file));
    let reader = match decoder.read_info() {
        Ok(r) => r,
        Err(_) => {
            return AnimationInfo {
                is_animated: false,
                frame_count: 1,
                avg_delay_ms: 100,
            };
        }
    };

    if let Some(anim) = reader.info().animation_control() {
        AnimationInfo {
            is_animated: anim.num_frames > 1,
            frame_count: anim.num_frames,
            avg_delay_ms: 100, // Default; APNG delay is per-frame
        }
    } else {
        AnimationInfo {
            is_animated: false,
            frame_count: 1,
            avg_delay_ms: 100,
        }
    }
}

// ============================================================================
// ANIMATION ENCODING
// ============================================================================

/// Encode a single static GIF from an RGBA image.
fn encode_static_gif(image: &RgbaImage, path: &Path) -> Result<(), String> {
    if image.width() > u16::MAX as u32 || image.height() > u16::MAX as u32 {
        return Err("Image dimensions exceed GIF maximum (65535×65535)".to_string());
    }
    let (w, h) = (image.width() as u16, image.height() as u16);
    let file = File::create(path).map_err(|e| format!("Failed to create GIF file: {}", e))?;

    // Quantize to 256 colors
    let (palette, indexed) = quantize_rgba(image, 256);

    let mut encoder = gif::Encoder::new(BufWriter::new(file), w, h, &palette)
        .map_err(|e| format!("GIF encoder init error: {}", e))?;

    let frame = gif::Frame {
        width: w,
        height: h,
        buffer: std::borrow::Cow::Borrowed(&indexed),
        ..Default::default()
    };
    encoder
        .write_frame(&frame)
        .map_err(|e| format!("GIF write error: {}", e))?;

    Ok(())
}

/// Encode multiple frames as an animated GIF.
/// `frames`: RGBA images for each frame (all must be same dimensions).
/// `fps`: target playback speed.
/// `max_colors`: max palette size (2-256).
/// `dither`: whether to apply Floyd-Steinberg dithering.
pub fn encode_animated_gif(
    frames: &[RgbaImage],
    fps: f32,
    max_colors: u16,
    _dither: bool,
    path: &Path,
) -> Result<(), String> {
    if frames.is_empty() {
        return Err("No frames to encode".to_string());
    }

    if frames[0].width() > u16::MAX as u32 || frames[0].height() > u16::MAX as u32 {
        return Err("Image dimensions exceed GIF maximum (65535×65535)".to_string());
    }
    let (w, h) = (frames[0].width() as u16, frames[0].height() as u16);
    let delay_cs = ((100.0 / fps).round() as u16).max(1); // centiseconds

    let file = File::create(path).map_err(|e| format!("Failed to create GIF file: {}", e))?;

    // Build a global palette from the first frame (or could use per-frame palettes)
    let colors = (max_colors as usize).clamp(2, 256);
    let (global_palette, _) = quantize_rgba(&frames[0], colors);

    let mut encoder = gif::Encoder::new(BufWriter::new(file), w, h, &global_palette)
        .map_err(|e| format!("GIF encoder init error: {}", e))?;

    encoder
        .set_repeat(gif::Repeat::Infinite)
        .map_err(|e| format!("GIF set repeat error: {}", e))?;

    for frame_img in frames {
        // Use per-frame local palette for better color accuracy
        let (local_palette, local_indexed) = quantize_rgba(frame_img, colors);
        let frame = gif::Frame {
            width: w,
            height: h,
            delay: delay_cs,
            palette: Some(local_palette),
            buffer: std::borrow::Cow::Owned(local_indexed),
            ..Default::default()
        };

        encoder
            .write_frame(&frame)
            .map_err(|e| format!("GIF frame write error: {}", e))?;
    }

    Ok(())
}

/// Encode multiple frames as an animated PNG (APNG).
/// `frames`: RGBA images for each frame (all must be same dimensions).
/// `fps`: target playback speed.
pub fn encode_animated_png(frames: &[RgbaImage], fps: f32, path: &Path) -> Result<(), String> {
    if frames.is_empty() {
        return Err("No frames to encode".to_string());
    }

    let width = frames[0].width();
    let height = frames[0].height();
    let delay_ms = (1000.0 / fps).round().clamp(1.0, 65535.0) as u16;
    let delay_num = delay_ms;
    let delay_den = 1000u16;

    let file = File::create(path).map_err(|e| format!("Failed to create APNG file: {}", e))?;
    let writer = BufWriter::new(file);

    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .set_animated(frames.len() as u32, 0) // 0 = infinite loop
        .map_err(|e| format!("APNG set_animated error: {}", e))?;

    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("APNG header write error: {}", e))?;

    for frame_img in frames {
        writer
            .set_frame_delay(delay_num, delay_den)
            .map_err(|e| format!("APNG set frame delay error: {}", e))?;
        writer
            .set_dispose_op(png::DisposeOp::Background)
            .map_err(|e| format!("APNG set dispose op error: {}", e))?;
        writer
            .write_image_data(frame_img.as_raw())
            .map_err(|e| format!("APNG frame write error: {}", e))?;
    }

    writer
        .finish()
        .map_err(|e| format!("APNG finish error: {}", e))?;

    Ok(())
}

/// Quantize an RGBA image to indexed color (palette + indices).
/// Returns (flat_palette_rgb: Vec<u8>, indices: Vec<u8>).
/// The palette is in [R,G,B, R,G,B, ...] format as required by the gif crate.
fn quantize_rgba(image: &RgbaImage, max_colors: usize) -> (Vec<u8>, Vec<u8>) {
    let pixels: Vec<u8> = image
        .pixels()
        .flat_map(|p| [p[0], p[1], p[2], p[3]])
        .collect();

    let nq = color_quant::NeuQuant::new(10, max_colors, &pixels);

    let mut palette = Vec::with_capacity(max_colors * 3);
    for i in 0..max_colors {
        if let Some(color) = nq.lookup(i) {
            palette.push(color[0]); // R
            palette.push(color[1]); // G
            palette.push(color[2]); // B
        } else {
            palette.push(0);
            palette.push(0);
            palette.push(0);
        }
    }

    let npixels = (image.width() * image.height()) as usize;
    let mut indices = Vec::with_capacity(npixels);
    for p in image.pixels() {
        let idx = nq.index_of(&[p[0], p[1], p[2], p[3]]) as u8;
        indices.push(idx);
    }

    (palette, indices)
}
