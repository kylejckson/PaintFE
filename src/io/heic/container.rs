//! Bounded HEIF metadata inspection, independent of any image codec.
use std::collections::{BTreeMap, BTreeSet};
pub(super) struct Info {
    pub alpha: bool,
    pub high_depth: bool,
}
type Result<T> = std::result::Result<T, String>;
const LIMIT: usize = 8192;
fn invalid() -> String {
    "Invalid or unsupported HEIF metadata".into()
}
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or_else(invalid)?;
        let value = self.bytes.get(self.at..end).ok_or_else(invalid)?;
        self.at = end;
        Ok(value)
    }
    fn number(&mut self, n: usize) -> Result<u32> {
        Ok(self
            .take(n)?
            .iter()
            .fold(0u32, |v, b| (v << 8) | u32::from(*b)))
    }
    fn full(&mut self) -> Result<(u32, u32)> {
        Ok((self.number(1)?, self.number(3)?))
    }
}
fn boxes<'a>(bytes: &'a [u8], budget: &mut usize) -> Result<Vec<([u8; 4], &'a [u8])>> {
    let mut r = Reader::new(bytes);
    let mut out = Vec::new();
    while r.at < bytes.len() {
        if *budget == 0 {
            return Err("Too many HEIF metadata boxes".into());
        }
        *budget -= 1;
        let start = r.at;
        let size = r.number(4)?;
        let kind = r.take(4)?.try_into().map_err(|_| invalid())?;
        let length = match size {
            0 => bytes.len() - start,
            1 => usize::try_from(u64::from_be_bytes(
                r.take(8)?.try_into().map_err(|_| invalid())?,
            ))
            .map_err(|_| invalid())?,
            value => value as usize,
        };
        let header = r.at - start;
        if length < header {
            return Err(invalid());
        }
        out.push((kind, r.take(length - header)?));
    }
    Ok(out)
}
#[derive(Default)]
struct Metadata<'a> {
    primary: Option<u32>,
    properties: Vec<([u8; 4], &'a [u8])>,
    associations: BTreeMap<u32, Vec<usize>>,
    references: BTreeMap<([u8; 4], u32), Vec<u32>>,
}
impl<'a> Metadata<'a> {
    fn associations(&mut self, data: &[u8]) -> Result<()> {
        let mut r = Reader::new(data);
        let (v, f) = r.full()?;
        if v > 1 || f & !1 != 0 {
            return Err(invalid());
        }
        let count = r.number(4)? as usize;
        if count > LIMIT {
            return Err(invalid());
        }
        for _ in 0..count {
            let id = r.number(if v == 0 { 2 } else { 4 })?;
            let count = r.number(1)?;
            let values = self.associations.entry(id).or_default();
            for _ in 0..count {
                let value = r.number(if f & 1 == 0 { 1 } else { 2 })? as usize;
                let index = value & if f & 1 == 0 { 0x7f } else { 0x7fff };
                if index != 0 {
                    values.push(index);
                }
            }
            if values.len() > 256 {
                return Err(invalid());
            }
        }
        if r.at != data.len() {
            return Err(invalid());
        }
        Ok(())
    }
    fn properties(&self, id: u32) -> Result<Vec<([u8; 4], &'a [u8])>> {
        self.associations
            .get(&id)
            .ok_or_else(invalid)?
            .iter()
            .map(|i| self.properties.get(i - 1).copied().ok_or_else(invalid))
            .collect()
    }
    fn size(&self, id: u32) -> Result<(u32, u32)> {
        let mut found = None;
        for (kind, data) in self.properties(id)? {
            if &kind == b"ispe" {
                let mut r = Reader::new(data);
                if r.full()? != (0, 0) {
                    return Err(invalid());
                }
                let size = (r.number(4)?, r.number(4)?);
                if r.at != data.len() || found.replace(size).is_some() {
                    return Err(invalid());
                }
            }
        }
        found.ok_or_else(invalid)
    }
    fn depth(&self, id: u32, path: &mut BTreeSet<u32>, budget: &mut usize) -> Result<u8> {
        if *budget == 0 || path.len() > 16 || !path.insert(id) {
            return Err("HEIF metadata reference cycle or depth limit".into());
        }
        *budget -= 1;
        let mut bits = 8;
        for (kind, data) in self.properties(id)? {
            match &kind {
                b"pixi" => {
                    let mut r = Reader::new(data);
                    if r.full()? != (0, 0) {
                        return Err(invalid());
                    }
                    let n = r.number(1)? as usize;
                    if n == 0 || n > 16 {
                        return Err(invalid());
                    }
                    for &depth in r.take(n)? {
                        if depth == 0 || depth > 16 {
                            return Err(invalid());
                        }
                        bits = bits.max(depth);
                    }
                    if r.at != data.len() {
                        return Err(invalid());
                    }
                }
                b"hvcC" => {
                    if data.len() < 23 || data[0] != 1 {
                        return Err(invalid());
                    }
                    bits = bits.max((data[17] & 7) + 8).max((data[18] & 7) + 8);
                }
                _ => {}
            }
        }
        if let Some(children) = self.references.get(&(*b"dimg", id)) {
            for &child in children {
                bits = bits.max(self.depth(child, path, budget)?);
            }
        }
        path.remove(&id);
        Ok(bits)
    }
}
pub(super) fn validate(bytes: &[u8]) -> Result<Info> {
    let mut budget = LIMIT;
    let mut metadata = None;
    for (kind, data) in boxes(bytes, &mut budget)? {
        if &kind != b"meta" {
            continue;
        }
        if metadata.is_some() {
            return Err(invalid());
        }
        let mut r = Reader::new(data);
        if r.full()? != (0, 0) {
            return Err(invalid());
        }
        let mut m = Metadata::default();
        for (kind, data) in boxes(&data[4..], &mut budget)? {
            match &kind {
                b"pitm" => {
                    let mut r = Reader::new(data);
                    let (v, f) = r.full()?;
                    if v > 1 || f != 0 {
                        return Err(invalid());
                    }
                    let id = r.number(if v == 0 { 2 } else { 4 })?;
                    if r.at != data.len() || m.primary.replace(id).is_some() {
                        return Err(invalid());
                    }
                }
                b"iprp" => {
                    for (kind, data) in boxes(data, &mut budget)? {
                        match &kind {
                            b"ipco" => {
                                if !m.properties.is_empty() {
                                    return Err(invalid());
                                }
                                m.properties = boxes(data, &mut budget)?;
                            }
                            b"ipma" => m.associations(data)?,
                            _ => {}
                        }
                    }
                }
                b"iref" => {
                    let mut r = Reader::new(data);
                    let (v, f) = r.full()?;
                    if v > 1 || f != 0 {
                        return Err(invalid());
                    }
                    for (kind, data) in boxes(&data[4..], &mut budget)? {
                        let mut r = Reader::new(data);
                        let id = r.number(if v == 0 { 2 } else { 4 })?;
                        let count = r.number(2)?;
                        if count > 256 {
                            return Err(invalid());
                        }
                        let list = m.references.entry((kind, id)).or_default();
                        for _ in 0..count {
                            list.push(r.number(if v == 0 { 2 } else { 4 })?);
                        }
                        if r.at != data.len() || list.len() > 256 {
                            return Err(invalid());
                        }
                    }
                }
                _ => {}
            }
        }
        metadata = Some(m);
    }
    let m = metadata.ok_or_else(invalid)?;
    let primary = m.primary.ok_or_else(invalid)?;
    if primary == 0 {
        return Err(invalid());
    }
    for (kind, data) in &m.properties {
        if kind == b"auxC" && (data.len() < 5 || data[..4] != [0; 4] || !data[4..].contains(&0)) {
            return Err(invalid());
        }
    }
    let size = m.size(primary)?;
    super::validate_dimensions(size.0, size.1)?;
    let mut remaining = 1024;
    let mut bits = m.depth(primary, &mut BTreeSet::new(), &mut remaining)?;
    let mut alpha = false;
    for ((kind, id), targets) in &m.references {
        if kind != b"auxl" || !targets.contains(&primary) {
            continue;
        }
        let is_alpha = m.properties(*id)?.iter().any(|(kind, data)| {
            kind == b"auxC"
                && data
                    .get(4..)
                    .and_then(|d| d.split(|b| *b == 0).next())
                    .is_some_and(|v| {
                        v == b"urn:mpeg:hevc:2015:auxid:1"
                            || v == b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha"
                    })
        });
        if is_alpha {
            if m.size(*id)? != size {
                return Err("HEIC alpha dimensions do not match the primary image".into());
            }
            alpha = true;
            bits = bits.max(m.depth(*id, &mut BTreeSet::new(), &mut remaining)?);
        }
    }
    Ok(Info {
        alpha,
        high_depth: bits > 8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_metadata_reader_covers_grid_and_high_depth_fixtures() {
        for (bytes, high) in [
            (
                include_bytes!("../../../tests/fixtures/heif/LA_8__29x100.heif").as_slice(),
                false,
            ),
            (
                include_bytes!("../../../tests/fixtures/heif/RGBA_8__29x100.heif").as_slice(),
                false,
            ),
            (
                include_bytes!("../../../tests/fixtures/heif/LA_8__128x128.heif").as_slice(),
                false,
            ),
            (
                include_bytes!("../../../tests/fixtures/heif/RGBA_10__128x128.heif").as_slice(),
                true,
            ),
            (
                include_bytes!("../../../tests/fixtures/heif/RGBA_12__128x128.heif").as_slice(),
                true,
            ),
        ] {
            let info = validate(bytes).unwrap();
            assert!(info.alpha);
            assert_eq!(info.high_depth, high);
        }
    }
    #[test]
    fn malformed_box_lengths_truncation_and_counts_fail_without_panics() {
        let bytes = include_bytes!("../../../tests/fixtures/heif/RGBA_8__29x100.heif");
        for cut in (0..bytes.len()).step_by(17) {
            assert!(
                validate(&bytes[..cut]).is_err(),
                "accepted truncated length {cut}"
            );
        }
        for bytes in [
            vec![0; 7],
            vec![0, 0, 0, 4, 0, 0, 0, 0],
            vec![255; 16],
            vec![
                0, 0, 0, 1, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,
            ],
        ] {
            assert!(validate(&bytes).is_err());
        }
        let many = b"\0\0\0\x08free".repeat(LIMIT + 1);
        assert!(validate(&many).is_err());
    }
    #[test]
    fn cycles_deep_graphs_and_invalid_property_indices_fail() {
        let mut m = Metadata::default();
        m.associations.insert(1, vec![]);
        m.references.insert((*b"dimg", 1), vec![1]);
        assert!(m.depth(1, &mut BTreeSet::new(), &mut 1024).is_err());
        m.references.clear();
        for id in 1..40 {
            m.associations.insert(id, vec![]);
            m.references.insert((*b"dimg", id), vec![id + 1]);
        }
        assert!(m.depth(1, &mut BTreeSet::new(), &mut 1024).is_err());
        m.associations.insert(1, vec![usize::MAX]);
        assert!(m.size(1).is_err());
    }
}
