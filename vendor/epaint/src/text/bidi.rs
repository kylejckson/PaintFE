// PaintFE patch: UAX #9 ordering after logical line breaking, before tessellation.
fn is_rtl(chr: char) -> bool {
    matches!(
        unicode_bidi::bidi_class(chr),
        unicode_bidi::BidiClass::R
            | unicode_bidi::BidiClass::AL
            | unicode_bidi::BidiClass::RLE
            | unicode_bidi::BidiClass::RLO
            | unicode_bidi::BidiClass::RLI
    )
}

fn reorder_bidi_rows(text: &str, rows: &mut [PlacedRow]) {
    if !text.chars().any(is_rtl) {
        return;
    }
    let bidi = unicode_bidi::BidiInfo::new(text, None);
    let source: Vec<_> = text.char_indices().collect();
    let mut cursor = 0usize;
    for placed in rows {
        let row = Arc::make_mut(&mut placed.row);
        let count = row.glyphs.len();
        let begin = source.get(cursor).map_or(text.len(), |c| c.0);
        let Some(paragraph) = bidi.paragraphs.iter().find(|p| p.range.contains(&begin)) else {
            break;
        };
        let end = source
            .get(cursor + count)
            .map_or(text.len(), |c| c.0)
            .min(paragraph.range.end);
        let resolved = bidi.reordered_levels(paragraph, begin..end);
        let levels: Vec<_> = (0..count)
            .map(|i| {
                source
                    .get(cursor + i)
                    .and_then(|(byte, _)| resolved.get(*byte))
                    .copied()
                    .unwrap_or(paragraph.level)
            })
            .collect();
        cursor += count + usize::from(placed.ends_with_newline);
        if !levels.iter().any(unicode_bidi::Level::is_rtl) {
            continue;
        }
        let mut groups = Vec::new();
        let mut start = 0usize;
        while start < count {
            let end = (start + row.glyphs[start].cluster_len.max(1)).min(count);
            groups.push(start..end);
            start = end;
        }
        let group_levels: Vec<_> = groups.iter().map(|g| levels[g.start]).collect();
        let order = unicode_bidi::BidiInfo::reorder_visual(&group_levels);
        let origin = row.glyphs.first().map_or(0.0, |g| g.pos.x);
        let mut logical_x = vec![origin];
        for glyph in &row.glyphs {
            logical_x.push(logical_x.last().copied().unwrap_or(origin) + glyph.advance_width);
        }
        let mut x = origin;
        for index in order {
            let range = &groups[index];
            let delta = x - logical_x[range.start];
            for glyph in &mut row.glyphs[range.clone()] {
                glyph.pos.x += delta;
            }
            x += logical_x[range.end] - logical_x[range.start];
        }
        let mut carets = vec![0.0; count + 1];
        for group in &groups {
            let glyph = &row.glyphs[group.start];
            for i in group.clone() {
                let fraction = (i - group.start) as f32 / group.len() as f32;
                carets[i] = glyph.pos.x
                    + glyph.advance_width
                        * if levels[group.start].is_rtl() {
                            1.0 - fraction
                        } else {
                            fraction
                        };
            }
            if group.end == count {
                carets[count] = if levels[group.start].is_rtl() {
                    glyph.pos.x
                } else {
                    glyph.max_x()
                };
            }
        }
        row.bidi_carets = Some(carets);
    }
}
