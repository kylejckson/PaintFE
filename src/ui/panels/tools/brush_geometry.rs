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
