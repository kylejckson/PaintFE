/// Pixel edges have integer coordinates and pixel centres have half coordinates.
pub fn pixel_aligned_brush_center(point: (f32, f32), diameter: f32, snap: bool) -> (f32, f32) {
    if !snap {
        return point;
    }
    let offset = if (diameter.round().max(1.) as u32).is_multiple_of(2) {
        0.
    } else {
        0.5
    };
    (point.0.floor() + offset, point.1.floor() + offset)
}

pub fn geometric_tip_alpha(
    geometry: crate::config::brushes::BuiltinTipGeometry,
    x: f32,
    y: f32,
    radius: f32,
    hardness: f32,
    aa: bool,
) -> f32 {
    use crate::config::brushes::BuiltinTipGeometry;
    let x = x.abs();
    let y = y.abs();
    let (distance, edge) = match geometry {
        BuiltinTipGeometry::Square => {
            let qx = x - radius;
            let qy = y - radius;
            (x.max(y), qx.max(0.).hypot(qy.max(0.)) + qx.max(qy).min(0.))
        }
        BuiltinTipGeometry::Diamond => (x + y, (x + y - radius) / std::f32::consts::SQRT_2),
    };
    let coverage = if aa {
        (0.5 - edge).clamp(0., 1.)
    } else if edge <= 0. {
        1.
    } else {
        0.
    };
    let h = hardness.clamp(0., 1.);
    let t = (distance / radius.max(0.001)).min(1.);
    let material = if t <= h || h >= 0.9999 {
        1.
    } else {
        let u = ((t - h) / (1. - h)).clamp(0., 1.);
        1. - u * u * (3. - 2. * u)
    };
    coverage * material
}

/// Width of the rotated geometric footprint, before antialiasing coverage.
pub fn geometric_tip_span(
    geometry: crate::config::brushes::BuiltinTipGeometry,
    diameter: f32,
    degrees: f32,
) -> f32 {
    let (sin, cos) = degrees.to_radians().sin_cos();
    diameter
        * match geometry {
            crate::config::brushes::BuiltinTipGeometry::Square => sin.abs() + cos.abs(),
            crate::config::brushes::BuiltinTipGeometry::Diamond => sin.abs().max(cos.abs()),
        }
}

pub fn rotated_geometric_tip_alpha(
    geometry: crate::config::brushes::BuiltinTipGeometry,
    point: (f32, f32),
    diameter: f32,
    rotation: (f32, f32),
    hardness: f32,
    aa: bool,
) -> f32 {
    let (sin, cos) = rotation;
    geometric_tip_alpha(
        geometry,
        point.0 * cos - point.1 * sin,
        point.0 * sin + point.1 * cos,
        diameter / 2.,
        hardness,
        aa,
    )
}

/// Physical extent preserves the pixel parity of the final rotated footprint.
pub fn geometric_cursor_extent(
    geometry: crate::config::brushes::BuiltinTipGeometry,
    diameter: f32,
    degrees: f32,
) -> u32 {
    let span = geometric_tip_span(geometry, diameter, degrees).max(1.);
    let parity = span.round() as u32 % 2;
    let mut side = span.ceil() as u32 + 2;
    if side % 2 != parity {
        side += 1;
    }
    side
}

/// Rasterise the final orientation instead of rotating a tiny, already rasterised mask.
pub fn geometric_cursor_mask(
    geometry: crate::config::brushes::BuiltinTipGeometry,
    diameter: f32,
    degrees: f32,
    hardness: f32,
    aa: bool,
) -> (Vec<u8>, u32) {
    let extent = geometric_cursor_extent(geometry, diameter, degrees);
    let side = extent.min(512);
    let scale = extent as f32 / side as f32;
    let half = side as f32 / 2.;
    let rotation = (-degrees).to_radians().sin_cos();
    let mut mask = vec![0; (side * side) as usize];
    for y in 0..side {
        for x in 0..side {
            mask[(y * side + x) as usize] = (rotated_geometric_tip_alpha(
                geometry,
                (
                    (x as f32 + 0.5 - half) * scale,
                    (y as f32 + 0.5 - half) * scale,
                ),
                diameter,
                rotation,
                hardness,
                aa,
            ) * 255.)
                .round() as u8;
        }
    }
    (mask, side)
}
