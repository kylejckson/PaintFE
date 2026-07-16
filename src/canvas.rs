use crate::par_compat::*;
use eframe::egui;
use egui::{Color32, ColorImage, ImageData, Pos2, Rect, TextureFilter, TextureOptions, Vec2};
use image::{GrayImage, Luma, Rgba, RgbaImage};
use std::collections::VecDeque;
use std::sync::Arc;

use crate::ops::text_layer::TextLayerData;

include!("canvas/defs.rs");
include!("canvas/view_full.rs");
