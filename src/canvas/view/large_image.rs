// Bounded, viewport-only display for documents that cannot use full-size GPU textures.
use super::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
struct DisplayKey {
    generation: u64,
    width: u32,
    height: u32,
    layer: Option<uuid::Uuid>,
    cmyk: bool,
}

#[derive(Default)]
pub(super) struct LargeDisplay {
    pub active: bool,
    options: Option<TextureOptions>,
    key: Option<DisplayKey>,
    tiles: HashMap<(u32, u32, u32), egui::TextureHandle>,
}
impl LargeDisplay {
    pub fn clear(&mut self) {
        self.key = None;
        self.tiles.clear();
    }
    pub fn paint(
        &mut self,
        painter: &egui::Painter,
        image: Rect,
        viewport: Rect,
        state: &CanvasState,
        options: TextureOptions,
    ) {
        let key = DisplayKey {
            generation: state.dirty_generation,
            width: state.width,
            height: state.height,
            layer: state.layers.first().map(|l| l.runtime_id),
            cmyk: state.cmyk_preview,
        };
        if self.key != Some(key) || self.options != Some(options) {
            self.clear();
            self.key = Some(key);
            self.options = Some(options);
        }
        let visible = image.intersect(viewport);
        if !visible.is_positive() || state.width == 0 || state.height == 0 {
            self.tiles.clear();
            return;
        }
        let sx = image.width() / state.width as f32;
        let sy = image.height() / state.height as f32;
        let bounds = Rect::from_min_max(
            Pos2::new(
                ((visible.left() - image.left()) / sx).floor().max(0.0),
                ((visible.top() - image.top()) / sy).floor().max(0.0),
            ),
            Pos2::new(
                ((visible.right() - image.left()) / sx)
                    .ceil()
                    .min(state.width as f32),
                ((visible.bottom() - image.top()) / sy)
                    .ceil()
                    .min(state.height as f32),
            ),
        );
        let mut scale = 1u32;
        let density = sx.min(sy) * painter.ctx().pixels_per_point();
        while scale < 32768
            && (density * (scale as f32) < 0.5
                || bounds.width() * bounds.height() / (scale as f32).powi(2) > 4_194_304.0)
        {
            scale *= 2;
        }
        let edge = painter.ctx().input(|i| i.max_texture_side).clamp(1, 1024) as u32;
        let span = edge * scale;
        let x0 = bounds.left() as u32 / span;
        let y0 = bounds.top() as u32 / span;
        let x1 = (bounds.right().ceil() as u32).div_ceil(span);
        let y1 = (bounds.bottom().ceil() as u32).div_ceil(span);
        self.tiles
            .retain(|&(x, y, s), _| s == scale && x >= x0 && x < x1 && y >= y0 && y < y1);
        let painter = painter.with_clip_rect(viewport);
        for y in y0..y1 {
            for x in x0..x1 {
                let tile_rect = Rect::from_min_max(
                    Pos2::new((x * span) as f32, (y * span) as f32),
                    Pos2::new(
                        ((x + 1) * span).min(state.width) as f32,
                        ((y + 1) * span).min(state.height) as f32,
                    ),
                );
                let tile = self.tiles.entry((x, y, scale)).or_insert_with(|| {
                    let (mut pixels, _) = state.composite_committed_downscaled(tile_rect, scale);
                    if state.cmyk_preview {
                        pixels.pixels = super::apply_cmyk_soft_proof(&pixels.pixels);
                    }
                    painter.ctx().load_texture(
                        format!("large_tile_{x}_{y}_{scale}"),
                        pixels,
                        options,
                    )
                });
                let destination = Rect::from_min_max(
                    image.min + tile_rect.min.to_vec2() * Vec2::new(sx, sy),
                    image.min + tile_rect.max.to_vec2() * Vec2::new(sx, sy),
                );
                // Edge tiles may contain one partial sample. UVs keep coordinates exact.
                let uv = Rect::from_min_max(
                    Pos2::ZERO,
                    Pos2::new(
                        tile_rect.width() / (tile.size()[0] as f32 * scale as f32),
                        tile_rect.height() / (tile.size()[1] as f32 * scale as f32),
                    ),
                );
                painter.image(tile.id(), destination, uv, Color32::WHITE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn committed_tiles_keep_preview_separate_from_the_base() {
        let mut state = CanvasState::new(16, 16);
        state.layers[0]
            .pixels
            .put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        let mut preview = crate::canvas::TiledImage::new(16, 16);
        preview.put_pixel(0, 0, image::Rgba([0, 0, 255, 255]));
        state.preview_layer = Some(preview);
        let rect = Rect::from_min_max(Pos2::ZERO, egui::pos2(16.0, 16.0));
        assert_eq!(
            state.composite_committed_downscaled(rect, 1).0.pixels[0],
            Color32::RED
        );
        assert_eq!(
            state.composite_partial_downscaled(rect, 1).0.pixels[0],
            Color32::BLUE
        );
    }
    #[test]
    fn viewport_tiles_stay_bounded_and_track_pan_and_edits() {
        let mut state = CanvasState::new(8736, 8);
        state.layers[0]
            .pixels
            .put_pixel(8192, 0, image::Rgba([255, 0, 0, 255]));
        state.mark_dirty(None);
        let ctx = egui::Context::default();
        let mut display = LargeDisplay::default();
        let frame = |display: &mut LargeDisplay, state: &CanvasState, offset: f32| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 400.0))),
                    max_texture_side: Some(1024),
                    ..Default::default()
                },
                |ui| {
                    display.paint(
                        ui.painter(),
                        Rect::from_min_size(Pos2::new(offset, 30.0), Vec2::new(8736.0, 8.0)),
                        Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 400.0)),
                        state,
                        TextureOptions::NEAREST,
                    )
                },
            )
        };
        let _ = frame(&mut display, &state, 0.0);
        assert!(display.tiles.contains_key(&(0, 0, 1)));
        let output = frame(&mut display, &state, -8500.0);
        assert!(!display.tiles.contains_key(&(0, 0, 1)));
        assert!(display.tiles.contains_key(&(8, 0, 1)));
        assert!(display.tiles.len() <= 2);
        assert!(
            output.textures_delta.set.iter().all(|(_, d)| d
                .image
                .size()
                .into_iter()
                .all(|v| v <= 1024))
        );
        let id = display.tiles[&(8, 0, 1)].id();
        let pixels = output
            .textures_delta
            .set
            .iter()
            .find(|(texture, _)| *texture == id)
            .unwrap();
        let egui::ImageData::Color(image) = &pixels.1.image;
        assert_eq!(image.pixels[0], Color32::RED);
        state.layers[0]
            .pixels
            .put_pixel(8192, 0, image::Rgba([0, 0, 255, 255]));
        state.mark_dirty(None);
        let output = frame(&mut display, &state, -8500.0);
        let egui::ImageData::Color(image) = &output
            .textures_delta
            .set
            .iter()
            .find(|(id, _)| *id == display.tiles[&(8, 0, 1)].id())
            .unwrap()
            .1
            .image;
        assert_eq!(image.pixels[0], Color32::BLUE);
        assert_eq!((state.width, state.height), (8736, 8));
    }
    #[test]
    fn bounded_previews_preserve_premultiplied_pixels_and_source_dimensions() {
        let ctx = egui::Context::default();
        let color = Color32::from_rgba_premultiplied(50, 25, 10, 128);
        let original = Arc::new(ColorImage::filled([8736, 2], color));
        let bounded =
            super::super::bound_canvas_image(&ctx, egui::ImageData::Color(original.clone()));
        let egui::ImageData::Color(bounded) = bounded;
        assert!(bounded.size[0] <= ctx.input(|i| i.max_texture_side));
        assert_eq!(bounded.pixels[0], color);
        assert_eq!(bounded.source_size, original.source_size);
        assert_eq!(original.size, [8736, 2]);
    }
}
