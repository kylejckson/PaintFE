/// Maximum longest-edge dimension for the LOD cache thumbnail.
const LOD_MAX_EDGE: u32 = 1024;
/// Above this many selected bbox pixels, freeze the interior hatch overlay and
/// rely on the cached GPU texture + border only to keep panning responsive.
const LARGE_SELECTION_STATIC_THRESHOLD: u32 = 1_048_576;

pub const CHUNK_SIZE: u32 = 64;
pub const BRUSH_PREVIEW_TILE_SIZE: u32 = 256;

/// A pixel with zero alpha, returned by reference for missing chunks.
static TRANSPARENT_PIXEL: Rgba<u8> = Rgba([0, 0, 0, 0]);

include!("selection.rs");
include!("tiled_image.rs");
include!("layers.rs");
include!("mirror.rs");
include!("canvas_state.rs");
