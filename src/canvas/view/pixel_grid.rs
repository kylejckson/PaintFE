use egui::{Color32, ColorImage, Painter, Pos2, Rect, TextureHandle, TextureOptions, Vec2};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    spacing: u32,
    dpi: u32,
    outline: Color32,
    center: Color32,
    dashed: bool,
}

struct Cache {
    owner: egui::Context,
    key: Key,
    vertical: TextureHandle,
    horizontal: TextureHandle,
}

#[derive(Default)]
pub(super) struct PixelGrid {
    cache: Option<Cache>,
}

impl PixelGrid {
    pub(super) fn paint(
        &mut self,
        painter: &Painter,
        image: Rect,
        viewport: Rect,
        spacing: f32,
        settings: &crate::config::settings::AppSettings,
    ) {
        let visible = image.intersect(viewport);
        if !spacing.is_finite()
            || spacing < 4.
            || !visible.is_positive()
            || !settings.pixel_grid_opacity.is_finite()
            || settings.pixel_grid_opacity <= 0.
        {
            return;
        }
        let dpi = painter.ctx().pixels_per_point().clamp(0.5, 8.);
        let opacity = settings.pixel_grid_opacity.clamp(0., 1.);
        let key = Key {
            spacing: spacing.to_bits(),
            dpi: dpi.to_bits(),
            outline: settings.pixel_grid_outline_color.gamma_multiply(opacity),
            center: settings.pixel_grid_center_color.gamma_multiply(opacity),
            dashed: settings.pixel_grid_dashed,
        };
        let same_owner = self
            .cache
            .as_ref()
            .is_some_and(|cache| cache.owner == *painter.ctx());
        if !same_owner || self.cache.as_ref().is_none_or(|cache| cache.key != key) {
            let vertical = pattern(key);
            let horizontal = transpose(&vertical);
            if same_owner {
                let cache = self.cache.as_mut().unwrap();
                cache.vertical.set(vertical, TextureOptions::LINEAR_REPEAT);
                cache
                    .horizontal
                    .set(horizontal, TextureOptions::LINEAR_REPEAT);
                cache.key = key;
            } else {
                self.cache = Some(Cache {
                    owner: painter.ctx().clone(),
                    key,
                    vertical: painter.ctx().load_texture(
                        "pixel_grid_vertical",
                        vertical,
                        TextureOptions::LINEAR_REPEAT,
                    ),
                    horizontal: painter.ctx().load_texture(
                        "pixel_grid_horizontal",
                        horizontal,
                        TextureOptions::LINEAR_REPEAT,
                    ),
                });
            }
        }
        let cache = self.cache.as_ref().unwrap();
        let painter = painter.with_clip_rect(visible);
        // Line positions follow the image origin. Dash phase follows the clipped
        // viewport, as in the original line renderer. Panning changes only UVs.
        let vertical_uv = Rect::from_min_size(
            Pos2::new((visible.min.x - image.min.x) / spacing, 0.),
            Vec2::new(visible.width() / spacing, visible.height() / 5.),
        );
        let horizontal_uv = Rect::from_min_size(
            Pos2::new(0., (visible.min.y - image.min.y) / spacing),
            Vec2::new(visible.width() / 5., visible.height() / spacing),
        );
        painter.image(cache.vertical.id(), visible, vertical_uv, Color32::WHITE);
        painter.image(
            cache.horizontal.id(),
            visible,
            horizontal_uv,
            Color32::WHITE,
        );
    }
}

fn coverage(distance: f32, width: f32, feather: f32) -> f32 {
    // Match egui's thin-stroke ridge instead of making subpixel lines opaque.
    if width <= 0.9 * feather {
        (1. - distance / feather).clamp(0., 1.) * (width / feather)
    } else {
        ((width * 0.5 + feather * 0.5 - distance) / feather).clamp(0., 1.)
    }
}

fn over(front: Color32, back: Color32) -> Color32 {
    let f = front.to_array();
    let b = back.to_array();
    let inverse = 255 - u16::from(f[3]);
    let channel =
        |index| (u16::from(f[index]) + (u16::from(b[index]) * inverse + 127) / 255).min(255) as u8;
    Color32::from_rgba_premultiplied(channel(0), channel(1), channel(2), channel(3))
}

fn pattern(key: Key) -> ColorImage {
    let spacing = f32::from_bits(key.spacing);
    let dpi = f32::from_bits(key.dpi);
    let width = ((spacing * dpi * 2.).ceil() as usize).clamp(2, 2048);
    let height = ((5. * dpi * 2.).ceil() as usize).clamp(2, 80);
    let feather = 1. / dpi;
    let outline_width = (1.2 * 8. / spacing).clamp(0.5, 1.2);
    let center_width = (0.6 * 8. / spacing).clamp(0.3, 0.6);
    let mut image = ColorImage::filled([width, height], Color32::TRANSPARENT);
    for y in 0..height {
        let along = (y as f32 + 0.5) * 5. / height as f32;
        let dash = if !key.dashed || along <= 2.5 {
            1.
        } else if outline_width <= 0.9 * feather {
            0.
        } else {
            // Thick egui dash caps extend one feather into the gap.
            (1. - (along - 2.5).min(5. - along) / feather).clamp(0., 1.)
        };
        for x in 0..width {
            let across = (x as f32 + 0.5) * spacing / width as f32;
            let distance = across.min(spacing - across);
            let outline = key
                .outline
                .gamma_multiply(coverage(distance, outline_width, feather) * dash);
            image.pixels[y * width + x] = if key.dashed {
                outline
            } else {
                over(
                    key.center
                        .gamma_multiply(coverage(distance, center_width, feather)),
                    outline,
                )
            };
        }
    }
    image
}

fn transpose(source: &ColorImage) -> ColorImage {
    let [width, height] = source.size;
    let mut image = ColorImage::filled([height, width], Color32::TRANSPARENT);
    for y in 0..height {
        for x in 0..width {
            image.pixels[x * height + y] = source.pixels[y * width + x];
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        grid: &mut PixelGrid,
        ctx: &egui::Context,
        image: Rect,
        zoom: f32,
        settings: &crate::config::settings::AppSettings,
    ) -> egui::FullOutput {
        ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280., 720.))),
                ..Default::default()
            },
            |ui| {
                grid.paint(ui.painter(), image, ui.max_rect(), zoom, settings);
            },
        )
    }

    #[test]
    fn grid_geometry_is_bounded_and_panning_reuses_textures() {
        for dpi in [1., 1.25, 2.] {
            for zoom in [8., 12., 8.375] {
                let ctx = egui::Context::default();
                ctx.set_pixels_per_point(dpi);
                let mut grid = PixelGrid::default();
                let settings = crate::config::settings::AppSettings::default();
                let image =
                    Rect::from_min_size(Pos2::new(-17., -23.), Vec2::new(800. * zoom, 600. * zoom));
                let first = frame(&mut grid, &ctx, image, zoom, &settings);
                let cache = grid.cache.as_ref().unwrap();
                let ids = [cache.vertical.id(), cache.horizontal.id()];
                let size = cache.vertical.size();
                assert_eq!(
                    first
                        .textures_delta
                        .set
                        .iter()
                        .filter(|(id, _)| ids.contains(id))
                        .count(),
                    2
                );
                let moved = frame(
                    &mut grid,
                    &ctx,
                    image.translate(Vec2::new(-3.25, 7.5)),
                    zoom,
                    &settings,
                );
                assert!(
                    moved
                        .textures_delta
                        .set
                        .iter()
                        .all(|(id, _)| !ids.contains(id))
                );
                let meshes = ctx.tessellate(moved.shapes, moved.pixels_per_point);
                let grid_meshes: Vec<_> = meshes
                    .iter()
                    .filter_map(|primitive| {
                        if let egui::epaint::Primitive::Mesh(mesh) = &primitive.primitive {
                            ids.contains(&mesh.texture_id).then_some(mesh)
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(grid_meshes.len(), 2);
                assert_eq!(
                    grid_meshes
                        .iter()
                        .map(|mesh| mesh.vertices.len())
                        .sum::<usize>(),
                    8
                );
                let vertical = grid_meshes
                    .iter()
                    .find(|mesh| mesh.texture_id == ids[0])
                    .unwrap();
                let left = vertical
                    .vertices
                    .iter()
                    .min_by(|a, b| a.pos.x.total_cmp(&b.pos.x))
                    .unwrap();
                let origin = image.min.x - 3.25;
                assert!((left.uv.x - (left.pos.x - origin) / zoom).abs() < 0.0001);
                assert!(size[0] <= 2048 && size[1] <= 80);
            }
        }
    }

    #[test]
    fn settings_invalidate_cache_without_changing_texture_identity() {
        let ctx = egui::Context::default();
        let mut grid = PixelGrid::default();
        let mut settings = crate::config::settings::AppSettings::default();
        let image = Rect::from_min_size(Pos2::ZERO, Vec2::splat(1000.));
        let _ = frame(&mut grid, &ctx, image, 8., &settings);
        let id = grid.cache.as_ref().unwrap().vertical.id();
        settings.pixel_grid_dashed = !settings.pixel_grid_dashed;
        settings.pixel_grid_opacity = 0.25;
        settings.pixel_grid_outline_color = Color32::RED;
        let changed = frame(&mut grid, &ctx, image, 8., &settings);
        assert!(
            changed
                .textures_delta
                .set
                .iter()
                .any(|(texture, _)| *texture == id)
        );
        assert_eq!(grid.cache.as_ref().unwrap().vertical.id(), id);
        settings.pixel_grid_opacity = 0.;
        assert!(
            frame(&mut grid, &ctx, image, 8., &settings)
                .shapes
                .is_empty()
        );
        settings.pixel_grid_opacity = 1.;
        let other = egui::Context::default();
        let output = frame(&mut grid, &other, image, 12., &settings);
        assert!(grid.cache.as_ref().unwrap().owner == other);
        assert_eq!(
            output
                .textures_delta
                .set
                .iter()
                .filter(|(id, _)| *id == grid.cache.as_ref().unwrap().vertical.id()
                    || *id == grid.cache.as_ref().unwrap().horizontal.id())
                .count(),
            2
        );
    }

    #[test]
    fn patterns_preserve_alpha_orientation_and_dash_gaps() {
        for dashed in [false, true] {
            let key = Key {
                spacing: 8f32.to_bits(),
                dpi: 1f32.to_bits(),
                outline: Color32::from_rgba_unmultiplied(20, 60, 100, 90),
                center: Color32::from_rgba_unmultiplied(240, 230, 220, 100),
                dashed,
            };
            let image = pattern(key);
            let horizontal = transpose(&image);
            assert_eq!(horizontal.size, [image.size[1], image.size[0]]);
            for y in 0..image.size[1] {
                for x in 0..image.size[0] {
                    let color = image.pixels[y * image.size[0] + x];
                    assert_eq!(horizontal.pixels[x * image.size[1] + y], color);
                    assert!(
                        color.r() <= color.a() && color.g() <= color.a() && color.b() <= color.a()
                    );
                }
            }
            assert!(image.pixels.iter().any(|pixel| pixel.a() > 0));
            assert!(image.pixels.iter().any(|pixel| pixel.a() == 0));
            if dashed {
                assert!(image.pixels[7 * image.size[0]].a() < image.pixels[image.size[0]].a());
            }
        }
        assert!((coverage(0., 0.6, 1.) - 0.6).abs() < 0.0001);
    }
}
