use crate::par_compat::*;
use ab_glyph::{Font, FontArc, ScaleFont};
use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

use crate::canvas::TiledImage;
use crate::ops::text::{self, GlyphPixelCache};

include!("text_layer/core.rs");
include!("text_layer/data_impl.rs");
include!("text_layer/block_impl.rs");
include!("text_layer/selection.rs");
include!("text_layer/raster.rs");
include!("text_layer/layout.rs");
include!("text_layer/warp.rs");
include!("text_layer/effects.rs");
