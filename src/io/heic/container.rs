//! Container-only validation. The dependency has no HEVC codec implementation.
use heifer_isobmff::{HeifFile, ItemId, boxes::Property};
pub(super) struct Info {
    pub alpha: bool,
    pub high_depth: bool,
}
fn depth(file: &HeifFile<'_>, id: ItemId, level: u8, budget: &mut usize) -> Result<u8, String> {
    if level > 16 || *budget == 0 {
        return Err("HEIC derived-image metadata is nested too deeply".into());
    }
    *budget -= 1;
    let mut bits = 8;
    for property in file.item_properties(id).map_err(|e| e.to_string())? {
        match property {
            Property::PixelInfo(channels) => {
                bits = bits.max(channels.iter().copied().max().unwrap_or(8))
            }
            Property::HevcConfig(config) => {
                bits = bits.max(config.bit_depth_luma).max(config.bit_depth_chroma)
            }
            _ => {}
        }
    }
    let references = file.referenced_items(id, b"dimg");
    if references.len() > 256 {
        return Err("Too many HEIC grid references".into());
    }
    for &item in references {
        bits = bits.max(depth(file, item, level + 1, budget)?);
    }
    if bits > 16 {
        return Err("Unsupported HEIC sample depth".into());
    }
    Ok(bits)
}
pub(super) fn validate(bytes: &[u8]) -> Result<Info, String> {
    let file = HeifFile::parse(bytes).map_err(|e| format!("Invalid HEIF container: {e}"))?;
    let (width, height) = file
        .image_size(file.primary_id)
        .map_err(|e| e.to_string())?
        .ok_or("Missing HEIC primary dimensions")?;
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 64_000_000 {
        return Err("HEIC image exceeds supported dimensions".into());
    }
    let mut alpha = false;
    let mut budget = 1024;
    let mut bits = depth(&file, file.primary_id, 0, &mut budget)?;
    for item in file.referencing_items(file.primary_id, b"auxl") {
        let is_alpha=file.item_properties(item).map_err(|e|e.to_string())?.any(|property|matches!(property,Property::AuxiliaryType(kind) if kind=="urn:mpeg:hevc:2015:auxid:1" || kind=="urn:mpeg:mpegB:cicp:systems:auxiliary:alpha"));
        if is_alpha {
            if file.image_size(item).map_err(|e| e.to_string())? != Some((width, height)) {
                return Err("HEIC alpha dimensions do not match the primary image".into());
            }
            alpha = true;
            bits = bits.max(depth(&file, item, 0, &mut budget)?);
        }
    }
    Ok(Info {
        alpha,
        high_depth: bits > 8,
    })
}
