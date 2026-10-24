//! PFE4 adds sparse layer masks while reusing the validated PFE3 document loader.
use super::*;

#[derive(Serialize, Deserialize)]
pub(crate) struct ProjectFileV4 {
    magic: String,
    document: ProjectFileV3,
    masks: Vec<LayerMask>,
}

#[derive(Serialize, Deserialize)]
struct LayerMask {
    enabled: bool,
    chunks: Option<Vec<ChunkData>>,
}

pub(super) fn build(state: &CanvasState) -> ProjectFileV4 {
    ProjectFileV4 {
        magic: "PFE4".into(),
        document: build_pfe_v3(state),
        masks: state
            .layers
            .iter()
            .map(|layer| LayerMask {
                enabled: layer.mask_enabled,
                chunks: layer.mask.as_ref().map(|mask| {
                    mask.chunk_keys()
                        .map(|(cx, cy)| ChunkData {
                            cx,
                            cy,
                            pixels: mask.get_chunk(cx, cy).unwrap().as_raw().clone(),
                        })
                        .collect()
                }),
            })
            .collect(),
    }
}

pub(super) fn write(project: &ProjectFileV4, path: &Path) -> Result<(), PfeError> {
    let mut writer = BufWriter::new(File::create(path)?);
    bincode::serialize_into(&mut writer, project)?;
    writer.flush()?;
    Ok(())
}

pub(super) fn load(raw: &[u8]) -> Result<CanvasState, PfeError> {
    let project: ProjectFileV4 = bincode::deserialize(raw)?;
    load_project(project)
}

pub(super) fn load_project(project: ProjectFileV4) -> Result<CanvasState, PfeError> {
    if project.magic != "PFE4" {
        return Err(PfeError::InvalidFormat(
            "Invalid layer-mask project header".into(),
        ));
    }
    if project.masks.len() != project.document.layers.len() {
        return Err(PfeError::InvalidFormat(
            "Layer mask count does not match layers".into(),
        ));
    }
    let mut state = canvas_from_pfe_v3(project.document)?;
    for (layer, mask) in state.layers.iter_mut().zip(project.masks) {
        layer.mask_enabled = mask.enabled;
        if let Some(chunks) = mask.chunks {
            let mut tiled = TiledImage::new(state.width, state.height);
            for chunk in chunks {
                let image =
                    RgbaImage::from_raw(CHUNK_SIZE, CHUNK_SIZE, chunk.pixels).ok_or_else(|| {
                        PfeError::InvalidFormat("Invalid layer mask chunk size".into())
                    })?;
                tiled.set_chunk(chunk.cx, chunk.cy, image);
            }
            layer.mask = Some(tiled);
        }
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn masked_project() -> ProjectFileV4 {
        let mut state = CanvasState::new(4, 4);
        state.layers[0].ensure_mask();
        state.layers[0].mask_enabled = false;
        state.layers[0]
            .mask
            .as_mut()
            .unwrap()
            .put_pixel(2, 1, Rgba([0, 0, 0, 172]));
        build(&state)
    }

    #[test]
    fn disabled_mask_and_sparse_pixels_roundtrip() {
        let bytes = bincode::serialize(&masked_project()).unwrap();
        let state = load_pfe_from_bytes(&bytes).unwrap();
        assert!(!state.layers[0].mask_enabled);
        let mask = state.layers[0].mask.as_ref().unwrap();
        assert_eq!(mask.get_pixel(2, 1)[3], 172);
        assert_eq!(mask.get_pixel(0, 0)[3], 0);
    }

    #[test]
    fn malformed_mask_count_and_chunk_are_rejected() {
        let mut project = masked_project();
        project.masks.clear();
        assert!(load_pfe_from_bytes(&bincode::serialize(&project).unwrap()).is_err());
        let mut project = masked_project();
        project.masks[0].chunks.as_mut().unwrap()[0].pixels.pop();
        assert!(load_pfe_from_bytes(&bincode::serialize(&project).unwrap()).is_err());
    }
}
