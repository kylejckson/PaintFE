//! Small generated paintbrush masks and distance-based dab dynamics.
use serde::{Deserialize, Serialize};
mod browser;
pub use browser::{BrowserSettings, SavedPreset};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintTip {
    Oval,
    Flat,
    Bristle,
    Fan,
    RoundPaint,
    FlatPaint,
    Filbert,
    DenseFan,
    Wash,
    Rake,
}

impl PaintTip {
    pub const ALL: [Self; 10] = [
        Self::RoundPaint,
        Self::FlatPaint,
        Self::Filbert,
        Self::DenseFan,
        Self::Bristle,
        Self::Fan,
        Self::Wash,
        Self::Rake,
        Self::Oval,
        Self::Flat,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Oval => "Oval Brush",
            Self::Flat => "Flat Brush",
            Self::Bristle => "Round Bristle",
            Self::Fan => "Fan Bristle",
            Self::RoundPaint => "Round Paint",
            Self::FlatPaint => "Flat Paint",
            Self::Filbert => "Filbert",
            Self::DenseFan => "Dense Fan",
            Self::Wash => "Bristle Wash",
            Self::Rake => "Rake",
        }
    }

    pub fn has_bristles(self) -> bool {
        !matches!(self, Self::Oval | Self::Flat)
    }

    pub fn loaded(self) -> bool {
        matches!(
            self,
            Self::RoundPaint | Self::FlatPaint | Self::Filbert | Self::DenseFan | Self::Wash
        )
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Bristle => "Dry Bristle",
            Self::Fan => "Dry Fan",
            _ => self.name(),
        }
    }

    pub fn flow(self) -> f32 {
        if self == Self::Wash {
            0.4
        } else if self.loaded() {
            1.
        } else if self.has_bristles() {
            0.75
        } else {
            1.
        }
    }

    /// Contact coverage in normalized brush coordinates. Texture changes coverage,
    /// while paint loading and user alpha remain independent deposition controls.
    pub fn contact(self, x: f32, y: f32, settings: BrushEngineSettings, edge: f32) -> f32 {
        let settings = settings.sanitized();
        let y = y / settings.roundness;
        let distance = match self {
            Self::FlatPaint => (x.abs() / 0.9).max(y.abs() / 0.85),
            Self::Filbert => ((x / 0.9).powi(4) + (y / 0.9).powi(4)).sqrt().sqrt(),
            Self::DenseFan => {
                let curve = 0.35 * x * x - 0.15;
                (x.abs() / 0.9).max((y - curve).abs() / 0.55)
            }
            _ => (x * x + y * y).sqrt() / 0.9,
        };
        let body = ((1. - distance) / edge.max(0.005)).clamp(0., 1.);
        let bands = (x * (18. + settings.density * 48.) / (0.5 + settings.bristle_thickness * 1.5)
            + (x * 17.).sin() * 0.25)
            .sin()
            .powi(2);
        body * (1. - settings.texture * bands * 0.65)
    }

    pub fn transform(
        self,
        settings: BrushEngineSettings,
        force: f32,
        speed: f32,
        lag: f32,
    ) -> DabTransform {
        let settings = settings.sanitized();
        if !self.has_bristles() {
            return DabTransform::default();
        }
        let pressure = if settings.pressure_spread {
            settings.pressure(force)
        } else {
            1.
        };
        let pressure =
            pressure * (1. - bounded(speed, 0., 1., 0.) * settings.speed_response * 0.65);
        DabTransform {
            scale_x: 1. + settings.spread * (pressure * 2. - 1.) * 0.55,
            scale_y: 1. - settings.spread * pressure * 0.15,
            shear: bounded(lag, -std::f32::consts::PI, std::f32::consts::PI, 0.).sin()
                * (1. - settings.stiffness)
                * 0.45,
            ..Default::default()
        }
    }

    pub fn settings(self) -> BrushEngineSettings {
        BrushEngineSettings {
            follow_direction: self.has_bristles(),
            roundness: match self {
                Self::Oval => 0.4,
                Self::Flat => 0.22,
                Self::RoundPaint => 1.,
                Self::FlatPaint => 0.38,
                Self::Filbert => 0.55,
                Self::DenseFan => 0.7,
                Self::Rake => 0.4,
                _ => 0.65,
            },
            texture: if self.loaded() { 0.25 } else { 0.8 },
            bristle_thickness: if self == Self::Rake { 0.85 } else { 0.55 },
            paint_load: if self == Self::Wash { 0.45 } else { 1. },
            ..Default::default()
        }
    }

    pub fn fibers(self, settings: BrushEngineSettings) -> Vec<Fiber> {
        if !self.has_bristles() {
            return Vec::new();
        }
        let settings = settings.sanitized();
        let hairs = if self == Self::Rake {
            (4. + settings.density * 8.).round() as usize
        } else {
            (24. + settings.density * 88.).round() as usize
        };
        (0..hairs)
            .map(|hair| {
                let random = ((hair as u32)
                    .wrapping_mul(747796405)
                    .wrapping_add(2891336453)
                    >> 8) as f32
                    / 16777215.;
                let t = (hair as f32 + 0.5) / hairs as f32;
                let (x, y) = if matches!(self, Self::Fan | Self::DenseFan | Self::Rake) {
                    let x = (2. * t - 1.) * 0.83;
                    (x, 0.48 * x * x - 0.24 + (random - 0.5) * 0.08)
                } else {
                    let radius = t.sqrt() * 0.78;
                    let angle = hair as f32 * 2.3999631;
                    (
                        radius * angle.cos(),
                        radius * angle.sin() * settings.roundness,
                    )
                };
                Fiber {
                    x,
                    y,
                    rx: (if self == Self::Fan {
                        0.012 + 0.01 * random
                    } else if self == Self::Rake {
                        0.035
                    } else {
                        0.025 + 0.025 * random
                    }) * (0.5 + settings.bristle_thickness * 1.5),
                    ry: 0.045 + 0.035 * random,
                    weight: 1. - settings.texture * (0.15 + 0.45 * random),
                }
            })
            .collect()
    }

    pub fn mask(self, size: u32, settings: BrushEngineSettings) -> image::GrayImage {
        let size = size.clamp(16, 256);
        let settings = settings.sanitized();
        let mut image = image::GrayImage::new(size, size);
        if !self.has_bristles() {
            for (x, y, pixel) in image.enumerate_pixels_mut() {
                let x = (x as f32 + 0.5) * 2. / size as f32 - 1.;
                let y = ((y as f32 + 0.5) * 2. / size as f32 - 1.) / settings.roundness;
                let distance = if self == Self::Oval {
                    (x * x + y * y).sqrt()
                } else {
                    x.abs().max(y.abs())
                };
                pixel[0] = ((0.92 - distance) * size as f32 * 0.5)
                    .clamp(0., 1.)
                    .mul_add(255., 0.)
                    .round() as u8;
            }
            return image;
        }
        if self.loaded() {
            for (x, y, pixel) in image.enumerate_pixels_mut() {
                pixel[0] = (255.
                    * self.contact(
                        (x as f32 + 0.5) * 2. / size as f32 - 1.,
                        (y as f32 + 0.5) * 2. / size as f32 - 1.,
                        settings,
                        2. / size as f32,
                    ))
                .round() as u8;
            }
            return image;
        }
        for fiber in self.fibers(settings) {
            let (cx, cy, rx, ry, weight) = (fiber.x, fiber.y, fiber.rx, fiber.ry, fiber.weight);
            let min_x = (((cx - rx + 1.) * 0.5 * size as f32).floor() as i32).max(0) as u32;
            let max_x = (((cx + rx + 1.) * 0.5 * size as f32).ceil() as u32).min(size - 1);
            let min_y = (((cy - ry + 1.) * 0.5 * size as f32).floor() as i32).max(0) as u32;
            let max_y = (((cy + ry + 1.) * 0.5 * size as f32).ceil() as u32).min(size - 1);
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let dx = (((x as f32 + 0.5) * 2. / size as f32 - 1.) - cx) / rx;
                    let dy = (((y as f32 + 0.5) * 2. / size as f32 - 1.) - cy) / ry;
                    let alpha =
                        ((1. - dx * dx - dy * dy).clamp(0., 1.) * weight * 255.).round() as u8;
                    image.get_pixel_mut(x, y)[0] = image.get_pixel(x, y)[0].max(alpha);
                }
            }
        }
        image
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fiber {
    pub x: f32,
    pub y: f32,
    pub rx: f32,
    pub ry: f32,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BrushEngineSettings {
    pub paint_load: f32,
    pub texture: f32,
    pub bristle_thickness: f32,
    pub stiffness: f32,
    pub depletion: f32,
    pub follow_direction: bool,
    pub smoothing: f32,
    pub roundness: f32,
    pub density: f32,
    pub spread: f32,
    pub pressure_spread: bool,
    pub pressure_gamma: f32,
    pub speed_response: f32,
}

impl Default for BrushEngineSettings {
    fn default() -> Self {
        Self {
            paint_load: 1.,
            texture: 0.25,
            bristle_thickness: 0.55,
            stiffness: 0.65,
            depletion: 0.,
            follow_direction: false,
            smoothing: 0.35,
            roundness: 0.65,
            density: 0.65,
            spread: 0.35,
            pressure_spread: true,
            pressure_gamma: 1.,
            speed_response: 0.,
        }
    }
}

fn bounded(value: f32, low: f32, high: f32, default: f32) -> f32 {
    if value.is_finite() {
        value.clamp(low, high)
    } else {
        default
    }
}

impl BrushEngineSettings {
    pub fn sanitized(mut self) -> Self {
        self.paint_load = bounded(self.paint_load, 0., 1., 1.);
        self.texture = bounded(self.texture, 0., 1., 0.25);
        self.bristle_thickness = bounded(self.bristle_thickness, 0., 1., 0.55);
        self.stiffness = bounded(self.stiffness, 0., 1., 0.65);
        self.depletion = bounded(self.depletion, 0., 1., 0.);
        self.smoothing = bounded(self.smoothing, 0., 1., 0.35);
        self.roundness = bounded(self.roundness, 0.08, 1., 0.65);
        self.density = bounded(self.density, 0., 1., 0.65);
        self.spread = bounded(self.spread, 0., 1., 0.35);
        self.pressure_gamma = bounded(self.pressure_gamma, 0.25, 4., 1.);
        self.speed_response = bounded(self.speed_response, 0., 1., 0.);
        self
    }

    pub fn pressure(self, force: f32) -> f32 {
        bounded(force, 0., 1., 1.).powf(self.sanitized().pressure_gamma)
    }

    pub fn mask_key(self) -> u64 {
        let settings = self.sanitized();
        {
            use std::hash::{Hash, Hasher};
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            for value in [
                settings.roundness,
                settings.density,
                settings.texture,
                settings.bristle_thickness,
            ] {
                value.to_bits().hash(&mut hash);
            }
            hash.finish()
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BrushMotion {
    pub last: Option<(f32, f32)>,
    angle: Option<f32>,
    pub lag: f32,
}

impl BrushMotion {
    pub fn angle(&self) -> f32 {
        self.angle.unwrap_or(0.)
    }

    pub fn advance(&mut self, position: (f32, f32), diameter: f32, smoothing: f32) -> f32 {
        if !position.0.is_finite() || !position.1.is_finite() {
            return self.angle.unwrap_or(0.);
        }
        if let Some(last) = self.last {
            let dx = position.0 - last.0;
            let dy = position.1 - last.1;
            let distance = dx.hypot(dy);
            if distance > 0.0001 {
                let target = dy.atan2(dx);
                let previous = self.angle.unwrap_or(target);
                let delta = (target - previous + std::f32::consts::PI)
                    .rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI;
                let weight = 1.
                    - (-distance / (diameter.max(1.) * smoothing.clamp(0., 1.) * 0.5).max(0.001))
                        .exp();
                self.angle = Some(previous + delta * weight);
                self.lag = delta * (1. - weight);
            }
        }
        self.last = Some(position);
        self.angle.unwrap_or(0.)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DabTransform {
    pub angle: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub shear: f32,
}

impl Default for DabTransform {
    fn default() -> Self {
        Self {
            angle: 0.,
            scale_x: 1.,
            scale_y: 1.,
            shear: 0.,
        }
    }
}

impl DabTransform {
    pub fn extent(self) -> f32 {
        self.scale_y.max(self.scale_x * (1. + self.shear.abs()))
            * if self.angle.abs() > 0.01 {
                std::f32::consts::SQRT_2
            } else {
                1.
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_masks_are_bounded_distinct_and_repeatable() {
        let mut masks = Vec::new();
        for tip in PaintTip::ALL {
            let mask = tip.mask(128, tip.settings());
            assert_eq!(mask, tip.mask(128, tip.settings()));
            assert_eq!(mask.dimensions(), (128, 128));
            assert!(mask.as_raw().iter().any(|&value| value > 0));
            assert!(mask.as_raw().contains(&0));
            assert!(masks.iter().all(|previous| previous != &mask));
            masks.push(mask);
        }
        assert_eq!(
            PaintTip::Fan.mask(100000, Default::default()).dimensions(),
            (256, 256)
        );
    }

    #[test]
    fn density_roundness_and_pressure_change_the_material() {
        let mut settings = PaintTip::Bristle.settings();
        settings.density = 0.;
        let sparse = PaintTip::Bristle.mask(128, settings);
        settings.density = 1.;
        let dense = PaintTip::Bristle.mask(128, settings);
        assert!(
            dense.as_raw().iter().map(|&v| u32::from(v)).sum::<u32>()
                > sparse.as_raw().iter().map(|&v| u32::from(v)).sum()
        );
        settings.roundness = 0.1;
        let flat = PaintTip::Oval.mask(128, settings);
        settings.roundness = 1.;
        assert_ne!(flat, PaintTip::Oval.mask(128, settings));
        let light = PaintTip::Fan.transform(settings, 0.1, 0., 0.);
        let heavy = PaintTip::Fan.transform(settings, 1., 0., 0.);
        assert!(heavy.scale_x > light.scale_x);
        settings.speed_response = 1.;
        assert!(PaintTip::Fan.transform(settings, 1., 1., 0.).scale_x < heavy.scale_x);
    }

    #[test]
    fn direction_filter_is_distance_based_and_wraps_shortest_way() {
        let mut coarse = BrushMotion::default();
        coarse.advance((0., 0.), 20., 0.5);
        coarse.advance((10., 0.), 20., 0.5);
        let mut fine = coarse;
        let a = coarse.advance((10., 20.), 20., 0.5);
        let mut b = 0.;
        for step in 1..=20 {
            b = fine.advance((10., step as f32), 20., 0.5);
        }
        assert!((a - b).abs() < 0.0001);
        let mut wrap = BrushMotion::default();
        wrap.advance((0., 0.), 20., 0.5);
        let before = wrap.advance((-10., 0.1), 20., 0.5);
        let after = wrap.advance((-20., 0.), 20., 0.5);
        assert!((after - before).abs() < 0.1);
        assert_eq!(wrap.advance((-20., 0.), 20., 0.5), after);
        assert!(wrap.advance((f32::NAN, 0.), 20., 0.5).is_finite());
    }

    #[test]
    fn pressure_curves_and_invalid_settings_are_safe() {
        let settings = BrushEngineSettings {
            pressure_gamma: 2.,
            roundness: f32::NAN,
            density: f32::INFINITY,
            ..Default::default()
        }
        .sanitized();
        assert_eq!(settings.pressure(0.), 0.);
        assert_eq!(settings.pressure(1.), 1.);
        assert!((settings.pressure(0.5) - 0.25).abs() < 0.0001);
        assert!(settings.roundness.is_finite() && settings.density.is_finite());
        let restored: BrushEngineSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(restored, BrushEngineSettings::default());
    }
}
