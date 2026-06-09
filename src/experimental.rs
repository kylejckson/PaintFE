use crate::canvas::PixelFormat;
use image::{Rgba, RgbaImage};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DeepRgbaBuffer {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F16(Vec<u16>),
    F32(Vec<f32>),
}

impl DeepRgbaBuffer {
    pub fn format(&self) -> PixelFormat {
        match self {
            DeepRgbaBuffer::U8(_) => PixelFormat::RgbaU8,
            DeepRgbaBuffer::U16(_) => PixelFormat::RgbaU16,
            DeepRgbaBuffer::F16(_) => PixelFormat::RgbaF16,
            DeepRgbaBuffer::F32(_) => PixelFormat::RgbaF32,
        }
    }

    pub fn from_rgba8(image: &RgbaImage, format: PixelFormat) -> Self {
        match format {
            PixelFormat::RgbaU8 => DeepRgbaBuffer::U8(image.as_raw().clone()),
            PixelFormat::RgbaU16 => {
                DeepRgbaBuffer::U16(image.as_raw().iter().map(|&v| (v as u16) * 257).collect())
            }
            PixelFormat::RgbaF16 => DeepRgbaBuffer::F16(
                image
                    .as_raw()
                    .iter()
                    .map(|&v| f32_to_f16_bits(v as f32 / 255.0))
                    .collect(),
            ),
            PixelFormat::RgbaF32 => {
                DeepRgbaBuffer::F32(image.as_raw().iter().map(|&v| v as f32 / 255.0).collect())
            }
        }
    }

    pub fn to_rgba8(&self, width: u32, height: u32) -> Option<RgbaImage> {
        let data: Vec<u8> = match self {
            DeepRgbaBuffer::U8(v) => v.clone(),
            DeepRgbaBuffer::U16(v) => v.iter().map(|&x| ((x as u32 + 128) / 257) as u8).collect(),
            DeepRgbaBuffer::F16(v) => v
                .iter()
                .map(|&x| (f16_bits_to_f32(x).clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
            DeepRgbaBuffer::F32(v) => v
                .iter()
                .map(|&x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
        };
        RgbaImage::from_raw(width, height, data)
    }
}

pub fn reinhard_tone_map_rgba(pixel: [f32; 4], exposure: f32) -> Rgba<u8> {
    let map = |v: f32| {
        let x = (v * exposure.max(0.0)).max(0.0);
        (x / (1.0 + x) * 255.0).round().clamp(0.0, 255.0) as u8
    };
    Rgba([
        map(pixel[0]),
        map(pixel[1]),
        map(pixel[2]),
        (pixel[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ])
}

pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;

    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = mant | 0x800000;
        let shift = 14 - exp;
        return sign | ((mant >> shift) as u16);
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x03ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut mant_norm = mant;
            let mut exp_norm = -14i32;
            while (mant_norm & 0x0400) == 0 {
                mant_norm <<= 1;
                exp_norm -= 1;
            }
            mant_norm &= 0x03ff;
            sign | (((exp_norm + 127) as u32) << 23) | (mant_norm << 13)
        }
    } else if exp == 31 {
        sign | 0x7f800000 | (mant << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}
