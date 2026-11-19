//! Physical document resolution. The canonical unit is pixels per metre.
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Resolution([u32; 2]);

impl Resolution {
    pub const MAX_DPI: f64 = 65_535.0;
    const METRES_PER_INCH: f64 = 0.0254;
    const MAX_PPM: u32 = 2_580_118;

    pub fn from_ppm(x: u32, y: u32) -> Option<Self> {
        let resolution = Self([x, y]);
        resolution.valid().then_some(resolution)
    }

    pub fn from_dpi(x: f64, y: f64) -> Option<Self> {
        if ![x, y].into_iter().all(|n| n.is_finite() && n > 0.0) {
            return None;
        }
        Self::from_ppm(
            (x / Self::METRES_PER_INCH).round() as u32,
            (y / Self::METRES_PER_INCH).round() as u32,
        )
    }

    pub fn ppm(self) -> [u32; 2] {
        self.0
    }

    pub fn dpi(self) -> [f64; 2] {
        self.0.map(|n| f64::from(n) * Self::METRES_PER_INCH)
    }

    pub fn inches(self, size: [u32; 2]) -> [f64; 2] {
        let [x, y] = self.dpi();
        [f64::from(size[0]) / x, f64::from(size[1]) / y]
    }

    pub(crate) fn valid(self) -> bool {
        self.0
            .into_iter()
            .all(|n| (39..=Self::MAX_PPM).contains(&n))
    }

    pub(crate) fn transpose(self) -> Self {
        Self([self.0[1], self.0[0]])
    }

    pub(crate) fn png_dimensions(self) -> png::PixelDimensions {
        png::PixelDimensions {
            xppu: self.0[0],
            yppu: self.0[1],
            unit: png::Unit::Meter,
        }
    }
}

pub(crate) fn read_path(path: &Path) -> Option<Resolution> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    if !matches!(
        ext.as_str(),
        "png" | "apng" | "jpeg" | "jpg" | "jpe" | "jfif" | "tif" | "tiff"
    ) {
        return None;
    }
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    match ext.as_str() {
        "png" | "apng" => {
            let decoder = png::Decoder::new(reader).read_info().ok()?;
            let dims = decoder.info().pixel_dims?;
            (dims.unit == png::Unit::Meter)
                .then(|| Resolution::from_ppm(dims.xppu, dims.yppu))
                .flatten()
        }
        "jpeg" | "jpg" | "jpe" | "jfif" => read_jpeg(&mut reader),
        "tif" | "tiff" => read_tiff(reader),
        _ => None,
    }
}

fn read_tiff(reader: impl Read + Seek) -> Option<Resolution> {
    use tiff::decoder::ifd::Value;
    use tiff::tags::Tag;
    let mut decoder = tiff::decoder::Decoder::new(reader).ok()?;
    let multiplier = match decoder
        .get_tag_unsigned::<u16>(Tag::ResolutionUnit)
        .unwrap_or(2)
    {
        2 => 1.0,
        3 => 2.54,
        _ => return None,
    };
    let mut density = |tag| match decoder.get_tag(tag).ok()? {
        Value::Rational(num, den) if den != 0 => Some(f64::from(num) * multiplier / f64::from(den)),
        _ => None,
    };
    Resolution::from_dpi(density(Tag::XResolution)?, density(Tag::YResolution)?)
}

fn read_jpeg(reader: &mut (impl Read + Seek)) -> Option<Resolution> {
    let mut pair = [0; 2];
    reader.read_exact(&mut pair).ok()?;
    if pair != [0xff, 0xd8] {
        return None;
    }
    let mut jfif = None;
    // Inspect only a bounded portion of the JPEG header.
    while reader.stream_position().ok()? < 1024 * 1024 {
        reader.read_exact(&mut pair).ok()?;
        while pair == [0xff, 0xff] {
            reader.read_exact(&mut pair[1..]).ok()?;
        }
        if pair[0] != 0xff || matches!(pair[1], 0xda | 0xd9) {
            break;
        }
        let marker = pair[1];
        reader.read_exact(&mut pair).ok()?;
        let length = usize::from(u16::from_be_bytes(pair)).checked_sub(2)?;
        match marker {
            0xe0 if length >= 14 => {
                let mut header = [0; 14];
                reader.read_exact(&mut header).ok()?;
                if &header[..5] == b"JFIF\0" {
                    let scale = match header[7] {
                        1 => 1.0,
                        2 => 2.54,
                        _ => 0.0,
                    };
                    jfif = Resolution::from_dpi(
                        f64::from(u16::from_be_bytes([header[8], header[9]])) * scale,
                        f64::from(u16::from_be_bytes([header[10], header[11]])) * scale,
                    );
                }
                reader
                    .seek(SeekFrom::Current((length - header.len()) as i64))
                    .ok()?;
            }
            0xe1 if length >= 14 => {
                let mut data = vec![0; length];
                reader.read_exact(&mut data).ok()?;
                if let Some(resolution) = data.strip_prefix(b"Exif\0\0").and_then(read_exif) {
                    return Some(resolution);
                }
            }
            _ => {
                reader.seek(SeekFrom::Current(length as i64)).ok()?;
            }
        }
    }
    jfif
}

fn read_exif(data: &[u8]) -> Option<Resolution> {
    let little = match data.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |offset: usize| {
        let bytes = data.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
        Some(if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    };
    let u32_at = |offset: usize| {
        let bytes = data.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
        Some(if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    };
    if u16_at(2)? != 42 {
        return None;
    }
    let ifd = usize::try_from(u32_at(4)?).ok()?;
    let count = usize::from(u16_at(ifd)?);
    let start = ifd.checked_add(2)?;
    data.get(start..start.checked_add(count * 12)?)?;
    let mut axes = [None; 2];
    let mut unit = 2;
    for entry in (start..start + count * 12).step_by(12) {
        if u32_at(entry + 4)? != 1 {
            continue;
        }
        match (u16_at(entry)?, u16_at(entry + 2)?) {
            (tag @ (282 | 283), 5) => {
                let ptr = usize::try_from(u32_at(entry + 8)?).ok()?;
                let numerator = u32_at(ptr)?;
                let denominator = u32_at(ptr.checked_add(4)?)?;
                if denominator != 0 {
                    axes[usize::from(tag - 282)] =
                        Some(f64::from(numerator) / f64::from(denominator));
                }
            }
            (296, 3) => unit = u16_at(entry + 8)?,
            _ => {}
        }
    }
    let scale = match unit {
        2 => 1.0,
        3 => 2.54,
        _ => return None,
    };
    Resolution::from_dpi(axes[0]? * scale, axes[1]? * scale)
}

pub(crate) struct ChangeResolution {
    pub before: Option<Resolution>,
    pub after: Option<Resolution>,
}

impl crate::components::history::Command for ChangeResolution {
    fn undo(&self, canvas: &mut crate::canvas::CanvasState) {
        canvas.resolution = self.before;
    }
    fn redo(&self, canvas: &mut crate::canvas::CanvasState) {
        canvas.resolution = self.after;
    }
    fn description(&self) -> String {
        crate::t!("resolution.title")
    }
    fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
    fn action(&self) -> crate::components::history::HistoryAction {
        crate::components::history::HistoryAction::PrintResolution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::CanvasState;
    use crate::components::dialogs::SaveFormat;

    #[test]
    fn resolution_units_and_history_leave_pixels_intact() {
        let a = Resolution::from_dpi(300.0, 300.0).unwrap();
        let b = Resolution::from_dpi(150.0, 150.0).unwrap();
        for (value, target) in [(a, [10.0, 8.0]), (b, [20.0, 16.0])] {
            for (actual, expected) in value.inches([3000, 2400]).into_iter().zip(target) {
                assert!((actual - expected).abs() < 0.005);
            }
        }
        assert!(Resolution::from_dpi(f64::NAN, 300.0).is_none());
        assert!(Resolution::from_dpi(0.0, 300.0).is_none());
        assert!(Resolution::from_dpi(70000.0, 300.0).is_none());
        let mut canvas = CanvasState::new(3, 2);
        let original = canvas.composite();
        let mut history = crate::components::history::HistoryManager::new(4);
        canvas.resolution = Some(a);
        history.push(Box::new(ChangeResolution {
            before: None,
            after: Some(a),
        }));
        history.undo(&mut canvas).unwrap();
        assert_eq!(canvas.resolution, None);
        history.redo(&mut canvas).unwrap();
        assert_eq!(canvas.resolution, Some(a));
        assert_eq!(canvas.composite(), original);
    }

    #[test]
    fn physical_resolution_roundtrips_in_projects_and_image_formats() {
        let mut canvas = CanvasState::new(7, 5);
        let base =
            std::env::temp_dir().join(format!("paintfe-resolution-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).unwrap();
        let original = canvas.composite();
        let density = Resolution::from_dpi(300.0, 144.0).unwrap();
        let project = base.join("test.pfe");
        assert!(!matches!(
            super::super::build_pfe(&canvas),
            super::super::PfeData::V6(_)
        ));
        canvas.resolution = Some(density);
        canvas.layers[0].ensure_mask();
        canvas
            .guides
            .add(crate::canvas::guides::Axis::Vertical, 2.0, 7, 5);
        super::super::save_pfe(&canvas, &project).unwrap();
        let reopened = super::super::load_pfe(&project).unwrap();
        assert_eq!(reopened.resolution, Some(density));
        assert_eq!(reopened.composite(), original);
        assert_eq!(reopened.guides, canvas.guides);
        assert!(reopened.layers[0].mask.is_some());

        for format in [SaveFormat::Png, SaveFormat::Jpeg, SaveFormat::Tiff] {
            let path = base.join(format!("test.{}", format.extension()));
            let options = super::super::export::ExportOptions {
                format,
                ..Default::default()
            };
            super::super::export::write_with_resolution(
                super::super::PreparedExportImage::Rgba8(original.clone()),
                &path,
                options,
                Some(density),
            )
            .unwrap();
            assert_eq!(read_path(&path), Some(density), "{format:?}");
            assert_eq!(
                super::super::load_image_sync(&path).unwrap().resolution,
                Some(density)
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = base.join("test.psd");
            super::super::psd::Export::prepare(&canvas)
                .unwrap()
                .write(&path)
                .unwrap();
            assert_eq!(
                super::super::psd::load(&path).unwrap().canvas.resolution,
                Some(density)
            );
        }
        std::fs::remove_dir_all(base).unwrap();
    }
}
