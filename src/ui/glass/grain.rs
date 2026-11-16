//! One static, repeating texture. No desktop or document pixels are read.
use egui::{Color32, Context, CornerRadius, Id, Rect, Shape, TextureHandle};

pub(super) fn shape(
    ctx: &Context,
    rect: Rect,
    rounding: CornerRadius,
    strength: f32,
) -> Option<Shape> {
    if !strength.is_finite() || strength <= 0.0 || !rect.is_positive() {
        return None;
    }
    let id = Id::new("paintfe_static_glass_grain");
    let texture = ctx
        .data(|data| data.get_temp::<TextureHandle>(id))
        .unwrap_or_else(|| {
            let mut seed = 0x8c37_2d19_u32;
            let pixels = (0..64 * 64)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    Color32::from_gray(if seed & 1 == 0 { 0 } else { 255 })
                })
                .collect();
            let texture = ctx.load_texture(
                "glass-grain",
                egui::ColorImage::new([64, 64], pixels),
                egui::TextureOptions::NEAREST_REPEAT,
            );
            ctx.data_mut(|data| data.insert_temp(id, texture.clone()));
            texture
        });
    let scale = ctx.pixels_per_point() / 64.0;
    let uv = Rect::from_min_max(
        (rect.min.to_vec2() * scale).to_pos2(),
        (rect.max.to_vec2() * scale).to_pos2(),
    );
    let colour = Color32::from_white_alpha((strength.clamp(0.0, 1.0) * 12.0).round() as u8);
    Some(
        egui::epaint::RectShape::filled(rect, rounding, colour)
            .with_texture(texture.id(), uv)
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_grain_uploads_once_and_keeps_rounded_geometry() {
        let ctx = Context::default();
        let rect = Rect::from_min_size(egui::pos2(12., 19.), egui::vec2(400., 300.));
        for frame in 0..4 {
            let output = ctx.run_ui(egui::RawInput::default(), |ui| {
                let Shape::Rect(noise) = shape(&ctx, rect, CornerRadius::same(10), 0.4).unwrap()
                else {
                    panic!("Expected textured rectangle")
                };
                assert_eq!(noise.corner_radius, CornerRadius::same(10));
                assert_eq!(noise.rect, rect);
                assert!(noise.brush.is_some());
                ui.painter().add(noise);
            });
            if frame > 0 {
                assert!(
                    output.textures_delta.set.is_empty(),
                    "Static grain must not upload per frame"
                );
            }
        }
        assert_eq!(
            ctx.data(|data| data.get_temp::<TextureHandle>(Id::new("paintfe_static_glass_grain")))
                .unwrap()
                .size(),
            [64, 64]
        );
    }

    #[test]
    fn zero_invalid_and_disabled_grain_do_not_allocate_textures() {
        let ctx = Context::default();
        let rect = Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100., 100.));
        for strength in [0., -1., f32::NAN, f32::INFINITY] {
            assert!(shape(&ctx, rect, CornerRadius::ZERO, strength).is_none());
        }
        assert!(
            ctx.data(|data| data.get_temp::<TextureHandle>(Id::new("paintfe_static_glass_grain")))
                .is_none()
        );
    }
}
