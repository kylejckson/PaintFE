use ab_glyph::{point, Font, FontArc, ScaleFont, GlyphId};
use std::collections::HashMap;

/// Cache for rasterized glyph pixel data. Key: (GlyphId, font_size_bits).
/// Value: (pixels as (u32, u32, f32), bounds_min_x_at_origin_zero, bounds_min_y_at_origin_zero).
pub type GlyphPixelCache = HashMap<(GlyphId, u32), (Vec<(u32, u32, f32)>, f32, f32)>;

/// Text alignment options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
}

impl TextAlignment {
    pub fn label(&self) -> String {
        match self {
            TextAlignment::Left => t!("text_alignment.left"),
            TextAlignment::Center => t!("text_alignment.center"),
            TextAlignment::Right => t!("text_alignment.right"),
        }
    }
}

/// A positioned glyph ready for rasterization.
struct LayoutGlyph {
    id: GlyphId,
    x: f32,
    y: f32,
    scale: f32,
}

/// Lay out a single line of text, returning positioned glyphs and bounding rect.
/// Returns `(glyphs, total_width, ascent, descent, line_height)`.
pub fn layout_text(
    font: &FontArc,
    text: &str,
    font_size: f32,
    alignment: TextAlignment,
) -> (Vec<(GlyphId, f32, f32)>, f32, f32, f32, f32) {
    let scaled = font.as_scaled(font_size);
    let ascent = scaled.ascent();
    let descent = scaled.descent();
    let line_height = scaled.height();

    // First pass: compute glyph positions (left-aligned at x=0)
    let mut glyphs = Vec::new();
    let mut cursor_x = 0.0f32;
    let mut last_glyph: Option<GlyphId> = None;

    for ch in text.chars() {
        let glyph_id = font.glyph_id(ch);
        if let Some(prev) = last_glyph {
            cursor_x += scaled.kern(prev, glyph_id);
        }
        glyphs.push((glyph_id, cursor_x, ascent));
        cursor_x += scaled.h_advance(glyph_id);
        last_glyph = Some(glyph_id);
    }

    let total_width = cursor_x;

    // Apply alignment offset
    let offset = match alignment {
        TextAlignment::Left => 0.0,
        TextAlignment::Center => -total_width * 0.5,
        TextAlignment::Right => -total_width,
    };

    for glyph in &mut glyphs {
        glyph.1 += offset;
    }

    (glyphs, total_width, ascent, descent, line_height)
}

/// Result from rasterize_text including cursor metrics for overlay rendering.
pub struct RasterizedText {
    pub buf: Vec<u8>,
    pub buf_w: u32,
    pub buf_h: u32,
    pub off_x: i32,
    pub off_y: i32,
    /// Per-line cumulative advance widths for cursor positioning.
    /// Maps (line_index, char_count_in_line) -> x_advance from line start.
    pub line_advances: Vec<Vec<f32>>,
    pub line_height: f32,
}

/// Rasterize text glyphs into an RGBA buffer.
///
/// `origin` is the (x, y) position in canvas coordinates where the text baseline starts.
/// Supports multiline text via '\n' characters.
/// `coverage_buf` is an optional reusable buffer to avoid per-call allocation.
pub fn rasterize_text(
    font: &FontArc,
    text: &str,
    font_size: f32,
    alignment: TextAlignment,
    origin_x: f32,
    origin_y: f32,
    color: [u8; 4],
    anti_alias: bool,
    bold: bool,
    italic: bool,
    underline: bool,
    strikethrough: bool,
    canvas_w: u32,
    canvas_h: u32,
    coverage_buf: &mut Vec<f32>,
    glyph_cache: &mut GlyphPixelCache,
) -> RasterizedText {
    let scaled = font.as_scaled(font_size);
    let ascent = scaled.ascent();
    let line_height = scaled.height();

    // Split text into lines and lay out each line, also compute per-line cursor advances
    let lines: Vec<&str> = text.split('\n').collect();
    let mut all_glyphs: Vec<(GlyphId, f32, f32)> = Vec::new();
    let mut line_widths: Vec<f32> = Vec::new();
    let mut line_advances: Vec<Vec<f32>> = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        let y_offset = line_idx as f32 * line_height;
        let (mut glyphs, total_width, _, _, _) = layout_text(font, line, font_size, alignment);
        // Offset y positions for this line
        for glyph in &mut glyphs {
            glyph.2 += y_offset;
        }
        // Compute cumulative advances for cursor positioning
        let mut advances = Vec::with_capacity(line.chars().count() + 1);
        advances.push(0.0f32);
        let mut cursor_x = 0.0f32;
        let mut prev_glyph: Option<GlyphId> = None;
        for ch in line.chars() {
            let gid = font.glyph_id(ch);
            if let Some(prev) = prev_glyph {
                cursor_x += scaled.kern(prev, gid);
            }
            cursor_x += scaled.h_advance(gid);
            advances.push(cursor_x);
            prev_glyph = Some(gid);
        }
        line_advances.push(advances);
        all_glyphs.extend(glyphs);
        line_widths.push(total_width);
    }

    if all_glyphs.is_empty() && !underline && !strikethrough {
        return RasterizedText { buf: Vec::new(), buf_w: 0, buf_h: 0, off_x: 0, off_y: 0, line_advances, line_height };
    }

    // Compute bounding box of all glyphs using fast glyph_bounds (no outlining needed)
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for &(glyph_id, gx, gy) in &all_glyphs {
        let glyph = glyph_id.with_scale_and_position(font_size, point(gx, gy));
        let bounds = font.glyph_bounds(&glyph);
        min_x = min_x.min(bounds.min.x);
        min_y = min_y.min(bounds.min.y);
        max_x = max_x.max(bounds.max.x);
        max_y = max_y.max(bounds.max.y);
    }

    // Handle decorations (extend bounds per line)
    if underline || strikethrough {
        for (line_idx, &line_w) in line_widths.iter().enumerate() {
            let align_offset = match alignment {
                TextAlignment::Left => 0.0,
                TextAlignment::Center => -line_w * 0.5,
                TextAlignment::Right => -line_w,
            };
            min_x = min_x.min(align_offset);
            max_x = max_x.max(align_offset + line_w);
            let y_off = line_idx as f32 * line_height;
            min_y = min_y.min(y_off);
            max_y = max_y.max(y_off + ascent + font_size * 0.2);
        }
    }

    // If we only have empty lines (no glyphs, no decorations with width), bail
    if min_x >= max_x || min_y >= max_y {
        return RasterizedText { buf: Vec::new(), buf_w: 0, buf_h: 0, off_x: 0, off_y: 0, line_advances, line_height };
    }

    // Add padding
    let pad = 2.0;
    min_x -= pad;
    min_y -= pad;
    max_x += pad;
    max_y += pad;

    // Convert to canvas space
    let buf_x0 = (origin_x + min_x).floor() as i32;
    let buf_y0 = (origin_y + min_y).floor() as i32;
    let buf_x1 = (origin_x + max_x).ceil() as i32;
    let buf_y1 = (origin_y + max_y).ceil() as i32;

    // Clamp to canvas
    let x0 = buf_x0.max(0);
    let y0 = buf_y0.max(0);
    let x1 = buf_x1.min(canvas_w as i32);
    let y1 = buf_y1.min(canvas_h as i32);
    let buf_w = (x1 - x0).max(0) as u32;
    let buf_h = (y1 - y0).max(0) as u32;

    if buf_w == 0 || buf_h == 0 {
        return RasterizedText { buf: Vec::new(), buf_w: 0, buf_h: 0, off_x: 0, off_y: 0, line_advances, line_height };
    }

    // Reuse coverage buffer (single channel)
    let needed = buf_w as usize * buf_h as usize;
    coverage_buf.resize(needed, 0.0);
    coverage_buf[..needed].fill(0.0);

    // Rasterize each glyph using glyph pixel cache (Opt 4)
    // Glyphs are cached at position (0,0) — we shift the cached pixels by the actual glyph position.
    let font_size_key = font_size.to_bits();
    for &(glyph_id, gx, gy) in &all_glyphs {
        let draw_x = gx.round();
        let draw_y = gy.round();
        let cache_key = (glyph_id, font_size_key);

        // Populate cache if this glyph hasn't been rasterized yet
        if !glyph_cache.contains_key(&cache_key) {
            let base_glyph = glyph_id.with_scale_and_position(font_size, point(0.0, 0.0));
            let mut px_list = Vec::new();
            let (bx, by) = if let Some(outlined) = font.outline_glyph(base_glyph) {
                let b = outlined.px_bounds();
                outlined.draw(|px, py, cov| {
                    px_list.push((px, py, cov));
                });
                (b.min.x, b.min.y)
            } else {
                (0.0, 0.0)
            };
            glyph_cache.insert(cache_key, (px_list, bx, by));
        }

        // Replay cached glyph pixels at actual position
        if let Some((pixels, base_bx, base_by)) = glyph_cache.get(&cache_key) {
            let actual_bx = *base_bx + draw_x;
            let actual_by = *base_by + draw_y;

            for &(px, py, cov) in pixels.iter() {
                let mut cx = px as f32 + origin_x + actual_bx;
                let cy = py as f32 + origin_y + actual_by;

                // Apply italic shear
                if italic {
                    let baseline_y = origin_y + draw_y;
                    cx += (baseline_y - cy) * 0.2;
                }

                let ix = cx.round() as i32 - x0;
                let iy = cy.round() as i32 - y0;
                if ix >= 0 && iy >= 0 && (ix as u32) < buf_w && (iy as u32) < buf_h {
                    let idx = iy as usize * buf_w as usize + ix as usize;
                    let v = if anti_alias { cov } else { if cov > 0.5 { 1.0 } else { 0.0 } };
                    coverage_buf[idx] = coverage_buf[idx].max(v);
                    if bold && ix + 1 < buf_w as i32 {
                        coverage_buf[idx + 1] = coverage_buf[idx + 1].max(v);
                    }
                }
            }
        }
    }

    // Draw decorations per line
    for (line_idx, &line_w) in line_widths.iter().enumerate() {
        if line_w < 0.1 { continue; }
        let y_off = line_idx as f32 * line_height;
        let thickness = (font_size * 0.06).max(1.0);

        if underline {
            let line_y = origin_y + y_off + ascent + font_size * 0.1;
            draw_decoration_line(coverage_buf, buf_w, buf_h, x0, y0,
                origin_x, line_y, line_w, thickness, alignment);
        }

        if strikethrough {
            let line_y = origin_y + y_off + ascent * 0.6;
            draw_decoration_line(coverage_buf, buf_w, buf_h, x0, y0,
                origin_x, line_y, line_w, thickness, alignment);
        }
    }

    // Convert coverage to RGBA
    let mut buf = vec![0u8; buf_w as usize * buf_h as usize * 4];
    for i in 0..needed {
        let cov = coverage_buf[i];
        if cov > 0.001 {
            let idx = i * 4;
            let a = (color[3] as f32 * cov).round().min(255.0) as u8;
            buf[idx] = color[0];
            buf[idx + 1] = color[1];
            buf[idx + 2] = color[2];
            buf[idx + 3] = a;
        }
    }

    RasterizedText { buf, buf_w, buf_h, off_x: x0, off_y: y0, line_advances, line_height }
}

fn draw_decoration_line(
    coverage: &mut [f32],
    buf_w: u32,
    buf_h: u32,
    x0: i32,
    y0: i32,
    origin_x: f32,
    line_y: f32,
    total_width: f32,
    thickness: f32,
    alignment: TextAlignment,
) {
    let line_start_x = origin_x + match alignment {
        TextAlignment::Left => 0.0,
        TextAlignment::Center => -total_width * 0.5,
        TextAlignment::Right => -total_width,
    };

    let half_t = thickness * 0.5;
    let ly0 = ((line_y - half_t).floor() as i32 - y0).max(0);
    let ly1 = ((line_y + half_t).ceil() as i32 - y0).min(buf_h as i32);
    let lx0 = ((line_start_x).floor() as i32 - x0).max(0);
    let lx1 = ((line_start_x + total_width).ceil() as i32 - x0).min(buf_w as i32);

    for ly in ly0..ly1 {
        for lx in lx0..lx1 {
            let idx = ly as usize * buf_w as usize + lx as usize;
            if idx < coverage.len() {
                coverage[idx] = 1.0;
            }
        }
    }
}

/// Enumerate system font families (family names only, no weight variants).
/// Returns a sorted, deduplicated list of font family names.
/// This is fast — just queries family names without loading any font data.
pub fn enumerate_system_fonts() -> Vec<String> {
    match font_kit::source::SystemSource::new().all_families() {
        Ok(mut families) => {
            families.sort();
            families.dedup();
            families
        }
        Err(_) => {
            #[cfg(target_os = "linux")]
            { vec!["Liberation Sans".to_string(), "DejaVu Sans".to_string(), "Liberation Mono".to_string()] }
            #[cfg(not(target_os = "linux"))]
            { vec!["Arial".to_string(), "Times New Roman".to_string(), "Courier New".to_string()] }
        },
    }
}

/// Enumerate available weight variants for a specific font family.
/// Returns list of (display_name, weight_value) pairs, e.g. [("Regular", 400), ("Light", 300), ("Bold", 700)].
pub fn enumerate_font_weights(family: &str) -> Vec<(String, u16)> {
    use font_kit::source::SystemSource;

    let source = SystemSource::new();
    let family_handle = match source.select_family_by_name(family) {
        Ok(h) => h,
        Err(_) => return vec![("Regular".to_string(), 400)],
    };

    let fonts = family_handle.fonts();
    if fonts.is_empty() {
        return vec![("Regular".to_string(), 400)];
    }

    let mut weights: Vec<(String, u16)> = Vec::new();
    let mut seen_weights = std::collections::HashSet::new();

    for font_handle in fonts {
        if let Ok(font) = font_handle.load() {
            let props = font.properties();
            let weight_val = props.weight.0 as u16;
            let style = props.style;
            let is_italic = style == font_kit::properties::Style::Italic || style == font_kit::properties::Style::Oblique;

            // Skip italic variants — they're handled by the italic toggle
            if is_italic { continue; }

            if seen_weights.contains(&weight_val) { continue; }
            seen_weights.insert(weight_val);

            let weight_name = match weight_val {
                0..=149 => "Thin",
                150..=249 => "ExtraLight",
                250..=349 => "Light",
                350..=449 => "Regular",
                450..=549 => "Medium",
                550..=649 => "SemiBold",
                650..=749 => "Bold",
                750..=849 => "ExtraBold",
                _ => "Black",
            };

            weights.push((weight_name.to_string(), weight_val));
        }
    }

    if weights.is_empty() {
        weights.push(("Regular".to_string(), 400));
    }

    weights.sort_by_key(|w| w.1);
    weights
}

/// Load a font by family name, weight, and style from the system.
/// `weight` is a CSS-style weight value (100=Thin, 400=Regular, 700=Bold, etc.)
/// Returns None if the font cannot be found.
pub fn load_system_font(family: &str, weight: u16, italic: bool) -> Option<FontArc> {
    use font_kit::properties::{Properties, Style, Weight};
    use font_kit::family_name::FamilyName;
    use font_kit::source::SystemSource;

    let mut props = Properties::new();
    props.weight = Weight(weight as f32);
    if italic {
        props.style = Style::Italic;
    }

    let source = SystemSource::new();
    let handle = source
        .select_best_match(&[FamilyName::Title(family.to_string())], &props)
        .ok()?;

    let font_data = handle.load().ok()?;
    let font_data_copy = font_data.copy_font_data()?;
    let bytes: Vec<u8> = (*font_data_copy).clone();
    FontArc::try_from_vec(bytes).ok()
}
