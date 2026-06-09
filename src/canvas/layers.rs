#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Additive,
    Reflect,
    Glow,
    ColorBurn,
    ColorDodge,
    Overlay,
    Difference,
    Negation,
    Lighten,
    Darken,
    Xor,
    Overwrite,
    HardLight,
    SoftLight,
    Exclusion,
    Subtract,
    Divide,
    LinearBurn,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,
}

impl BlendMode {
    /// Returns all blend modes for UI display
    pub fn all() -> &'static [BlendMode] {
        &[
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Additive,
            BlendMode::Overlay,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Lighten,
            BlendMode::Darken,
            BlendMode::ColorBurn,
            BlendMode::ColorDodge,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Negation,
            BlendMode::Reflect,
            BlendMode::Glow,
            BlendMode::Subtract,
            BlendMode::Divide,
            BlendMode::LinearBurn,
            BlendMode::VividLight,
            BlendMode::LinearLight,
            BlendMode::PinLight,
            BlendMode::HardMix,
            BlendMode::Xor,
            BlendMode::Overwrite,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Additive => "Additive",
            BlendMode::Reflect => "Reflect",
            BlendMode::Glow => "Glow",
            BlendMode::ColorBurn => "Color Burn",
            BlendMode::ColorDodge => "Color Dodge",
            BlendMode::Overlay => "Overlay",
            BlendMode::Difference => "Difference",
            BlendMode::Negation => "Negation",
            BlendMode::Lighten => "Lighten",
            BlendMode::Darken => "Darken",
            BlendMode::Xor => "Xor",
            BlendMode::Overwrite => "Overwrite",
            BlendMode::HardLight => "Hard Light",
            BlendMode::SoftLight => "Soft Light",
            BlendMode::Exclusion => "Exclusion",
            BlendMode::Subtract => "Subtract",
            BlendMode::Divide => "Divide",
            BlendMode::LinearBurn => "Linear Burn",
            BlendMode::VividLight => "Vivid Light",
            BlendMode::LinearLight => "Linear Light",
            BlendMode::PinLight => "Pin Light",
            BlendMode::HardMix => "Hard Mix",
        }
    }

    /// Returns the localized display name for UI rendering
    pub fn display_name(&self) -> String {
        match self {
            BlendMode::Normal => t!("blend.normal"),
            BlendMode::Multiply => t!("blend.multiply"),
            BlendMode::Screen => t!("blend.screen"),
            BlendMode::Additive => t!("blend.additive"),
            BlendMode::Reflect => t!("blend.reflect"),
            BlendMode::Glow => t!("blend.glow"),
            BlendMode::ColorBurn => t!("blend.color_burn"),
            BlendMode::ColorDodge => t!("blend.color_dodge"),
            BlendMode::Overlay => t!("blend.overlay"),
            BlendMode::Difference => t!("blend.difference"),
            BlendMode::Negation => t!("blend.negation"),
            BlendMode::Lighten => t!("blend.lighten"),
            BlendMode::Darken => t!("blend.darken"),
            BlendMode::Xor => t!("blend.xor"),
            BlendMode::Overwrite => t!("blend.overwrite"),
            BlendMode::HardLight => t!("blend.hard_light"),
            BlendMode::SoftLight => t!("blend.soft_light"),
            BlendMode::Exclusion => t!("blend.exclusion"),
            BlendMode::Subtract => t!("blend.subtract"),
            BlendMode::Divide => t!("blend.divide"),
            BlendMode::LinearBurn => t!("blend.linear_burn"),
            BlendMode::VividLight => t!("blend.vivid_light"),
            BlendMode::LinearLight => t!("blend.linear_light"),
            BlendMode::PinLight => t!("blend.pin_light"),
            BlendMode::HardMix => t!("blend.hard_mix"),
        }
    }

    /// Convert to a stable u8 for binary serialization
    pub fn to_u8(self) -> u8 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Multiply => 1,
            BlendMode::Screen => 2,
            BlendMode::Additive => 3,
            BlendMode::Reflect => 4,
            BlendMode::Glow => 5,
            BlendMode::ColorBurn => 6,
            BlendMode::ColorDodge => 7,
            BlendMode::Overlay => 8,
            BlendMode::Difference => 9,
            BlendMode::Negation => 10,
            BlendMode::Lighten => 11,
            BlendMode::Darken => 12,
            BlendMode::Xor => 13,
            BlendMode::Overwrite => 14,
            BlendMode::HardLight => 15,
            BlendMode::SoftLight => 16,
            BlendMode::Exclusion => 17,
            BlendMode::Subtract => 18,
            BlendMode::Divide => 19,
            BlendMode::LinearBurn => 20,
            BlendMode::VividLight => 21,
            BlendMode::LinearLight => 22,
            BlendMode::PinLight => 23,
            BlendMode::HardMix => 24,
        }
    }

    /// Reconstruct from a u8 (defaults to Normal for unknown values)
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => BlendMode::Normal,
            1 => BlendMode::Multiply,
            2 => BlendMode::Screen,
            3 => BlendMode::Additive,
            4 => BlendMode::Reflect,
            5 => BlendMode::Glow,
            6 => BlendMode::ColorBurn,
            7 => BlendMode::ColorDodge,
            8 => BlendMode::Overlay,
            9 => BlendMode::Difference,
            10 => BlendMode::Negation,
            11 => BlendMode::Lighten,
            12 => BlendMode::Darken,
            13 => BlendMode::Xor,
            14 => BlendMode::Overwrite,
            15 => BlendMode::HardLight,
            16 => BlendMode::SoftLight,
            17 => BlendMode::Exclusion,
            18 => BlendMode::Subtract,
            19 => BlendMode::Divide,
            20 => BlendMode::LinearBurn,
            21 => BlendMode::VividLight,
            22 => BlendMode::LinearLight,
            23 => BlendMode::PinLight,
            24 => BlendMode::HardMix,
            _ => BlendMode::Normal,
        }
    }
}

/// Experimental storage/display format marker.
///
/// The existing editor path remains RGBA u8. These variants let project files
/// preserve intent and allow conversion tests while the core renderer migrates
/// incrementally.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PixelFormat {
    #[default]
    RgbaU8,
    RgbaU16,
    RgbaF16,
    RgbaF32,
}

impl PixelFormat {
    pub fn name(self) -> &'static str {
        match self {
            PixelFormat::RgbaU8 => "RGBA UI8",
            PixelFormat::RgbaU16 => "RGBA UI16",
            PixelFormat::RgbaF16 => "RGBA FP16",
            PixelFormat::RgbaF32 => "RGBA FP32",
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct HdrMetadata {
    pub enabled: bool,
    pub max_luminance_nits: Option<f32>,
    pub reference_white_nits: Option<f32>,
    pub transfer_function: Option<String>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ImageMetadata {
    pub source_format: Option<String>,
    pub source_name: Option<String>,
    pub color_profile_name: Option<String>,
    pub png_text_chunks: Vec<(String, String)>,
    pub raw_png_chunks: Vec<Vec<u8>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AdjustmentKind {
    Exposure {
        ev: f32,
    },
    BrightnessContrast {
        brightness: f32,
        contrast: f32,
    },
    Invert,
    ChannelMixer {
        red: [f32; 4],
        green: [f32; 4],
        blue: [f32; 4],
        alpha: [f32; 4],
    },
}

impl Default for AdjustmentKind {
    fn default() -> Self {
        AdjustmentKind::Exposure { ev: 0.0 }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct AdjustmentLayerData {
    pub kind: AdjustmentKind,
}

impl AdjustmentLayerData {
    pub fn apply_to_pixel(&self, p: Rgba<u8>) -> Rgba<u8> {
        let [r, g, b, a] = p.0;
        match self.kind {
            AdjustmentKind::Exposure { ev } => {
                let gain = 2.0f32.powf(ev);
                Rgba([
                    ((r as f32) * gain).clamp(0.0, 255.0) as u8,
                    ((g as f32) * gain).clamp(0.0, 255.0) as u8,
                    ((b as f32) * gain).clamp(0.0, 255.0) as u8,
                    a,
                ])
            }
            AdjustmentKind::BrightnessContrast {
                brightness,
                contrast,
            } => {
                let factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));
                let apply = |v: u8| {
                    (factor * (v as f32 + brightness - 128.0) + 128.0).clamp(0.0, 255.0) as u8
                };
                Rgba([apply(r), apply(g), apply(b), a])
            }
            AdjustmentKind::Invert => Rgba([255 - r, 255 - g, 255 - b, a]),
            AdjustmentKind::ChannelMixer {
                red,
                green,
                blue,
                alpha,
            } => {
                let src = [r as f32, g as f32, b as f32, a as f32];
                let mix = |m: [f32; 4]| {
                    (src[0] * m[0] + src[1] * m[1] + src[2] * m[2] + src[3] * m[3])
                        .clamp(0.0, 255.0) as u8
                };
                Rgba([mix(red), mix(green), mix(blue), mix(alpha)])
            }
        }
    }

    pub fn apply_to_pixel_with_opacity(&self, p: Rgba<u8>, opacity: f32) -> Rgba<u8> {
        let adjusted = self.apply_to_pixel(p);
        let t = opacity.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        Rgba([
            (p[0] as f32 * inv + adjusted[0] as f32 * t).round() as u8,
            (p[1] as f32 * inv + adjusted[1] as f32 * t).round() as u8,
            (p[2] as f32 * inv + adjusted[2] as f32 * t).round() as u8,
            (p[3] as f32 * inv + adjusted[3] as f32 * t).round() as u8,
        ])
    }

    pub fn apply_to_f32_with_opacity(&self, p: [f32; 4], opacity: f32) -> [f32; 4] {
        let adjusted = match self.kind {
            AdjustmentKind::Exposure { ev } => {
                let gain = 2.0f32.powf(ev);
                [p[0] * gain, p[1] * gain, p[2] * gain, p[3]]
            }
            AdjustmentKind::BrightnessContrast {
                brightness,
                contrast,
            } => {
                let factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));
                let b = brightness / 255.0;
                let apply = |v: f32| (factor * (v + b - 0.5) + 0.5).max(0.0);
                [apply(p[0]), apply(p[1]), apply(p[2]), p[3]]
            }
            AdjustmentKind::Invert => [1.0 - p[0], 1.0 - p[1], 1.0 - p[2], p[3]],
            AdjustmentKind::ChannelMixer {
                red,
                green,
                blue,
                alpha,
            } => {
                let mix =
                    |m: [f32; 4]| (p[0] * m[0] + p[1] * m[1] + p[2] * m[2] + p[3] * m[3]).max(0.0);
                [mix(red), mix(green), mix(blue), mix(alpha)]
            }
        };
        let t = opacity.clamp(0.0, 1.0);
        let inv = 1.0 - t;
        [
            p[0] * inv + adjusted[0] * t,
            p[1] * inv + adjusted[1] * t,
            p[2] * inv + adjusted[2] * t,
            p[3] * inv + adjusted[3] * t,
        ]
    }
}

/// Discriminant for heterogeneous layer types.
#[derive(Clone, Debug, Default)]
pub enum LayerContent {
    /// Standard raster layer (current behaviour). Pixel data lives in `Layer::pixels`.
    #[default]
    Raster,
    /// Editable text layer. Vector data + cached rasterisation in `Layer::pixels`.
    Text(TextLayerData),
    /// Experimental non-destructive adjustment layer.
    Adjustment(AdjustmentLayerData),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LayerFolder {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub collapsed: bool,
    #[serde(default)]
    pub insert_above_layer: Option<usize>,
    #[serde(default)]
    pub color_index: Option<u8>,
}

pub struct Layer {
    pub name: String,
    pub visible: bool,
    pub folder_id: Option<u64>,
    pub opacity: f32,
    pub blend_mode: BlendMode,
    pub pixels: TiledImage,
    /// Optional live (non-destructive) layer mask.
    /// We encode concealment in alpha: 0 = reveal, 255 = fully hidden.
    pub mask: Option<TiledImage>,
    /// Whether the live mask participates in compositing.
    pub mask_enabled: bool,
    /// Downscaled cache (max 1024px longest edge) for zoomed-out rendering.
    /// Not serialized — rebuilt on demand.
    pub lod_cache: Option<Arc<RgbaImage>>,
    /// Per-layer generation counter for GPU texture synchronisation.
    /// Bumped only when THIS layer's pixels are modified, so unchanged
    /// layers are never re-uploaded to the GPU.
    pub gpu_generation: u64,
    /// Layer type discriminant — `Raster` for normal layers, `Text(..)` for
    /// editable text layers. Default: `Raster`.
    pub content: LayerContent,
    /// Experimental format metadata. Raster pixels are still mirrored in u8.
    pub pixel_format: PixelFormat,
    /// Experimental HDR metadata for import/export and tone-map previews.
    pub hdr_metadata: HdrMetadata,
    /// Experimental source metadata preservation container.
    pub source_metadata: ImageMetadata,
    /// Original high-depth pixel payload for project round-trips.
    pub deep_pixels: Option<crate::experimental::DeepRgbaBuffer>,
}

fn sync_deep_region<T: Copy, F: Fn(u32, u32, Rgba<u8>) -> [T; 4]>(
    values: &mut [T],
    width: u32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    convert: F,
    pixels: &TiledImage,
) {
    for y in y0..y1 {
        for x in x0..x1 {
            let idx = ((y * width + x) as usize) * 4;
            if idx + 3 >= values.len() {
                continue;
            }
            let converted = convert(x, y, *pixels.get_pixel(x, y));
            values[idx..idx + 4].copy_from_slice(&converted);
        }
    }
}

impl Layer {
    pub fn new(name: String, width: u32, height: u32, fill_color: Rgba<u8>) -> Self {
        let pixels = TiledImage::new_filled(width, height, fill_color);

        Self {
            name,
            visible: true,
            folder_id: None,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            pixels,
            mask: None,
            mask_enabled: true,
            lod_cache: None,
            gpu_generation: 0,
            content: LayerContent::Raster,
            pixel_format: PixelFormat::RgbaU8,
            hdr_metadata: HdrMetadata::default(),
            source_metadata: ImageMetadata::default(),
            deep_pixels: None,
        }
    }

    /// Create a new text layer with default empty text data.
    pub fn new_text(name: String, width: u32, height: u32) -> Self {
        Self {
            name,
            visible: true,
            folder_id: None,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
            pixels: TiledImage::new(width, height),
            mask: None,
            mask_enabled: true,
            lod_cache: None,
            gpu_generation: 0,
            content: LayerContent::Text(TextLayerData::default()),
            pixel_format: PixelFormat::RgbaU8,
            hdr_metadata: HdrMetadata::default(),
            source_metadata: ImageMetadata::default(),
            deep_pixels: None,
        }
    }

    pub fn new_adjustment(name: String, width: u32, height: u32, kind: AdjustmentKind) -> Self {
        let mut layer = Self::new(name, width, height, Rgba([0, 0, 0, 0]));
        layer.content = LayerContent::Adjustment(AdjustmentLayerData { kind });
        layer
    }

    /// Returns true if this is a text layer.
    pub fn is_text_layer(&self) -> bool {
        matches!(self.content, LayerContent::Text(_))
    }

    pub fn is_adjustment_layer(&self) -> bool {
        matches!(self.content, LayerContent::Adjustment(_))
    }

    pub fn sync_deep_pixels_from_preview_region(&mut self, x0: u32, y0: u32, x1: u32, y1: u32) {
        if !matches!(self.content, LayerContent::Raster) {
            return;
        }
        let Some(deep) = self.deep_pixels.as_mut() else {
            return;
        };
        let width = self.pixels.width();
        let height = self.pixels.height();
        let x1 = x1.min(width);
        let y1 = y1.min(height);
        if x0 >= x1 || y0 >= y1 {
            return;
        }

        match deep {
            crate::experimental::DeepRgbaBuffer::U8(values) => {
                sync_deep_region(values, width, x0, y0, x1, y1, |_, _, p| p.0, &self.pixels);
            }
            crate::experimental::DeepRgbaBuffer::U16(values) => {
                sync_deep_region(
                    values,
                    width,
                    x0,
                    y0,
                    x1,
                    y1,
                    |_, _, p| {
                        [
                            (p[0] as u16) * 257,
                            (p[1] as u16) * 257,
                            (p[2] as u16) * 257,
                            (p[3] as u16) * 257,
                        ]
                    },
                    &self.pixels,
                );
            }
            crate::experimental::DeepRgbaBuffer::F16(values) => {
                sync_deep_region(
                    values,
                    width,
                    x0,
                    y0,
                    x1,
                    y1,
                    |_, _, p| {
                        [
                            crate::experimental::f32_to_f16_bits(p[0] as f32 / 255.0),
                            crate::experimental::f32_to_f16_bits(p[1] as f32 / 255.0),
                            crate::experimental::f32_to_f16_bits(p[2] as f32 / 255.0),
                            crate::experimental::f32_to_f16_bits(p[3] as f32 / 255.0),
                        ]
                    },
                    &self.pixels,
                );
            }
            crate::experimental::DeepRgbaBuffer::F32(values) => {
                sync_deep_region(
                    values,
                    width,
                    x0,
                    y0,
                    x1,
                    y1,
                    |_, _, p| {
                        [
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0,
                            p[3] as f32 / 255.0,
                        ]
                    },
                    &self.pixels,
                );
            }
        }
    }

    pub fn sync_all_deep_pixels_from_preview(&mut self) {
        self.sync_deep_pixels_from_preview_region(0, 0, self.pixels.width(), self.pixels.height());
    }

    /// Invalidate the LOD cache (call after any pixel modification).
    pub fn invalidate_lod(&mut self) {
        self.lod_cache = None;
    }

    pub fn has_live_mask(&self) -> bool {
        self.mask.is_some()
    }

    pub fn ensure_mask(&mut self) {
        if self.mask.is_none() {
            self.mask = Some(TiledImage::new(self.pixels.width(), self.pixels.height()));
            self.mask_enabled = true;
        }
    }

    #[inline]
    pub fn apply_mask_alpha_at(&self, x: u32, y: u32, src_alpha: u8) -> u8 {
        if !self.mask_enabled {
            return src_alpha;
        }
        let conceal = self
            .mask
            .as_ref()
            .map(|m| m.get_pixel(x, y)[3])
            .unwrap_or(0);
        if conceal == 0 {
            src_alpha
        } else {
            ((src_alpha as u32 * (255 - conceal as u32)) / 255) as u8
        }
    }

    /// Flatten this layer to RGBA, applying the live mask to alpha when enabled.
    pub fn to_masked_rgba_image(&self) -> RgbaImage {
        let mut flat = self.pixels.to_rgba_image();
        if !self.mask_enabled {
            return flat;
        }
        let Some(mask) = &self.mask else {
            return flat;
        };
        let w = flat.width().min(mask.width());
        let h = flat.height().min(mask.height());
        for y in 0..h {
            for x in 0..w {
                let conceal = mask.get_pixel(x, y)[3];
                if conceal == 0 {
                    continue;
                }
                let mut p = *flat.get_pixel(x, y);
                p[3] = ((p[3] as u32 * (255 - conceal as u32)) / 255) as u8;
                flat.put_pixel(x, y, p);
            }
        }
        flat
    }

    /// Return a reference to the downscaled LOD image, generating it lazily.
    /// The thumbnail is at most `LOD_MAX_EDGE` pixels on its longest side.
    pub fn get_lod_image(&mut self) -> Arc<RgbaImage> {
        if let Some(ref cached) = self.lod_cache {
            return Arc::clone(cached);
        }
        let (w, h) = (self.pixels.width(), self.pixels.height());
        let longest = w.max(h);
        let (nw, nh) = if longest <= LOD_MAX_EDGE {
            (w, h) // Already small enough
        } else {
            let scale = LOD_MAX_EDGE as f32 / longest as f32;
            (
                ((w as f32 * scale).round() as u32).max(1),
                ((h as f32 * scale).round() as u32).max(1),
            )
        };
        let flat = self.pixels.to_rgba_image();
        let thumb = image::imageops::resize(&flat, nw, nh, image::imageops::FilterType::Triangle);
        let arc = Arc::new(thumb);
        self.lod_cache = Some(Arc::clone(&arc));
        arc
    }
}
