//! PFE5 wraps the unchanged PFE4 payload, retaining older project loaders.
use super::*;
#[derive(Serialize, Deserialize)]
pub(crate) struct ProjectFileV5 {
    magic: String,
    document: pfe_masks::ProjectFileV4,
    guides: crate::canvas::guides::Guides,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ProjectFileV6 {
    magic: String,
    document: ProjectFileV5,
    resolution: resolution::Resolution,
}

pub(super) fn build_v6(state: &CanvasState, resolution: resolution::Resolution) -> ProjectFileV6 {
    ProjectFileV6 {
        magic: "PFE6".into(),
        document: build(state),
        resolution,
    }
}

pub(super) fn write_v6(project: &ProjectFileV6, path: &Path) -> Result<(), PfeError> {
    let mut writer = BufWriter::new(File::create(path)?);
    bincode::serialize_into(&mut writer, project)?;
    writer.flush()?;
    Ok(())
}

pub(super) fn load_v6(raw: &[u8]) -> Result<CanvasState, PfeError> {
    use bincode::Options;
    let p: ProjectFileV6 = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(raw.len() as u64)
        .reject_trailing_bytes()
        .deserialize(raw)?;
    if p.magic != "PFE6" || !p.resolution.valid() {
        return Err(PfeError::InvalidFormat(
            "Invalid project print resolution".into(),
        ));
    }
    let mut state = load_project(p.document)?;
    state.resolution = Some(p.resolution);
    Ok(state)
}
pub(super) fn build(state: &CanvasState) -> ProjectFileV5 {
    ProjectFileV5 {
        magic: "PFE5".into(),
        document: pfe_masks::build(state),
        guides: state.guides.clone(),
    }
}
pub(super) fn write(project: &ProjectFileV5, path: &Path) -> Result<(), PfeError> {
    let mut writer = BufWriter::new(File::create(path)?);
    bincode::serialize_into(&mut writer, project)?;
    writer.flush()?;
    Ok(())
}
pub(super) fn load(raw: &[u8]) -> Result<CanvasState, PfeError> {
    use bincode::Options;
    let p: ProjectFileV5 = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(raw.len() as u64)
        .reject_trailing_bytes()
        .deserialize(raw)?;
    load_project(p)
}

fn load_project(p: ProjectFileV5) -> Result<CanvasState, PfeError> {
    if p.magic != "PFE5" {
        return Err(PfeError::InvalidFormat(
            "Invalid guide project header".into(),
        ));
    }
    let mut state = pfe_masks::load_project(p.document)?;
    if !p.guides.valid(state.width, state.height) {
        return Err(PfeError::InvalidFormat(
            "Invalid guide positions or identities".into(),
        ));
    }
    state.guides = p.guides;
    Ok(state)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guides_and_masks_roundtrip_without_changing_exported_pixels() {
        let mut s = CanvasState::new(16, 12);
        s.layers[0].ensure_mask();
        s.guides
            .add(crate::canvas::guides::Axis::Vertical, 7.5, 16, 12);
        let p = build(&s);
        let decoded = load_pfe_from_bytes(&bincode::serialize(&p).unwrap()).unwrap();
        assert_eq!(decoded.guides, s.guides);
        assert!(decoded.layers[0].mask.is_some());
        assert_eq!(
            decoded.layers[0].pixels.get_pixel(0, 0),
            s.layers[0].pixels.get_pixel(0, 0)
        );
        let mut bad = p;
        bad.guides.items[0].position = f32::NAN;
        assert!(load_pfe_from_bytes(&bincode::serialize(&bad).unwrap()).is_err());
    }
}
