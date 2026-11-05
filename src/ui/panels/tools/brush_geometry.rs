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
