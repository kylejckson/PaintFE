use image::{Rgba, RgbaImage};
use paintfe::ops::transform::warp_reference_mesh;

fn grid() -> Vec<[f32; 2]> {
    (0..=2)
        .flat_map(|y| (0..=2).map(move |x| [x as f32 * 8., y as f32 * 8.]))
        .collect()
}

#[test]
fn editing_reference_preserves_all_pixels_including_transparent_rgb() {
    let image = RgbaImage::from_fn(16, 16, |x, y| {
        Rgba([
            x as u8 * 13,
            y as u8 * 11,
            91,
            if x % 2 == 0 { 0 } else { 255 },
        ])
    });
    let mut points = grid();
    points[4] = [11., 7.];
    assert_eq!(warp_reference_mesh(&image, &points, &points, 2, 2), image);
}

#[test]
fn relocating_an_anchor_changes_the_physical_mapping_location() {
    let mut image = RgbaImage::from_pixel(16, 16, Rgba([0, 0, 0, 255]));
    for y in 7..10 {
        for x in 10..13 {
            image.put_pixel(x, y, Rgba([255, 0, 0, 255]));
        }
    }
    let mut original = grid();
    original[4] = [11., 8.];
    let mut changed = original.clone();
    changed[4] = [13., 8.];
    let result = warp_reference_mesh(&image, &original, &changed, 2, 2);
    assert!(result.get_pixel(12, 8).0[0] > 200);
    assert!(result.get_pixel(10, 8).0[0] < 100);
}

#[test]
fn invalid_and_empty_meshes_do_not_panic_or_change_pixels() {
    let image = RgbaImage::new(16, 16);
    assert_eq!(warp_reference_mesh(&image, &[], &[], 0, 0), image);
    let original = grid();
    let mut changed = original.clone();
    changed[4][0] = f32::NAN;
    assert_eq!(
        warp_reference_mesh(&image, &original, &changed, 2, 2),
        image
    );
    assert_eq!(
        warp_reference_mesh(&RgbaImage::new(0, 0), &original, &changed, 2, 2).dimensions(),
        (0, 0)
    );
}
