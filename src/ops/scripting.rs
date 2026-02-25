// ============================================================================
// PaintFE Scripting System â€” Rhai-based sandboxed scripting engine
// ============================================================================
//
// Provides a safe, embedded scripting environment for users to write
// pixel-manipulation scripts. The engine exposes a host API for reading/writing
// pixels, calling built-in effects, and utility functions.

use image::{GrayImage, RgbaImage, imageops};
use rhai::{AST, Array, Dynamic, Engine, EvalAltResult, FnPtr, ImmutableString, Scope};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

// ============================================================================
// Canvas-wide transform operations to replay on non-active layers
// ============================================================================

/// Filter type used by resize_image — stored in CanvasOpRequest.
#[derive(Clone, Copy, Debug)]
pub enum ScriptFilterType {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos,
}

impl ScriptFilterType {
    pub fn to_image_filter(self) -> imageops::FilterType {
        match self {
            ScriptFilterType::Nearest => imageops::FilterType::Nearest,
            ScriptFilterType::Bilinear => imageops::FilterType::Triangle,
            ScriptFilterType::Bicubic => imageops::FilterType::CatmullRom,
            ScriptFilterType::Lanczos => imageops::FilterType::Lanczos3,
        }
    }
}

/// A canvas-wide transform requested from script — replayed on all other layers by app.rs.
#[derive(Clone, Debug)]
pub enum CanvasOpRequest {
    FlipHorizontal,
    FlipVertical,
    Rotate90CW,
    Rotate90CCW,
    Rotate180,
    ResizeImage {
        w: u32,
        h: u32,
        filter: ScriptFilterType,
    },
    ResizeCanvas {
        new_w: u32,
        new_h: u32,
        anchor: (u32, u32),
    },
}

fn parse_script_filter(s: &str) -> ScriptFilterType {
    match s.to_lowercase().as_str() {
        "nearest" | "nn" => ScriptFilterType::Nearest,
        "bicubic" | "catmull" | "catmullrom" => ScriptFilterType::Bicubic,
        "lanczos" | "lanczos3" => ScriptFilterType::Lanczos,
        _ => ScriptFilterType::Bilinear,
    }
}

fn parse_anchor(s: &str) -> (u32, u32) {
    match s.to_lowercase().as_str() {
        "top-left" | "tl" | "nw" => (0, 0),
        "top-center" | "tc" | "n" | "top" => (1, 0),
        "top-right" | "tr" | "ne" => (2, 0),
        "center-left" | "cl" | "w" | "left" => (0, 1),
        "center" | "c" | "middle" => (1, 1),
        "center-right" | "cr" | "e" | "right" => (2, 1),
        "bottom-left" | "bl" | "sw" => (0, 2),
        "bottom-center" | "bc" | "s" | "bottom" => (1, 2),
        "bottom-right" | "br" | "se" => (2, 2),
        _ => (0, 0),
    }
}

// ============================================================================
// Error type
// ============================================================================

#[derive(Debug, Clone)]
pub struct ScriptError {
    pub message: String,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

impl ScriptError {
    /// Error explanation with line/column context and suggestions.
    pub fn friendly_message(&self) -> String {
        let raw = &self.message;
        let mut parts = Vec::new();

        // Location header
        match (self.line, self.column) {
            (Some(line), Some(col)) => {
                parts.push(format!("Error on line {}, column {}:", line, col));
            }
            (Some(line), None) => {
                parts.push(format!("Error on line {}:", line));
            }
            _ => {
                parts.push("Script error:".to_string());
            }
        }

        // Categorize error type
        if raw.contains("Function not found:") {
            // Extract function name and arg types
            // Example: "Function not found: f (i64) (line 7, position 6) in closure call (line 1, pos 1)"
            if let Some(fn_part) = raw.strip_prefix("Function not found: ") {
                let fn_desc = fn_part.split(" (line ").next().unwrap_or(fn_part);
                parts.push(format!("  Could not find function: {}", fn_desc.trim()));

                // Check if this looks like a closure stored in a variable
                let fn_name = fn_desc.split('(').next().unwrap_or("").trim();
                if fn_name.len() <= 3 || fn_name.chars().all(|c| c.is_lowercase() || c == '_') {
                    parts.push(String::new());
                    parts.push(
                        "  Tip: If this is a closure stored in a variable, use .call() syntax:"
                            .to_string(),
                    );
                    parts.push(format!("    let {} = |x| {{ x * 2 }};", fn_name));
                    parts.push(format!("    {}.call(42);   // âœ“ correct", fn_name));
                    parts.push(format!("    {}(42);        // âœ— won't work", fn_name));
                }
            } else {
                parts.push(format!("  {}", raw));
            }
        } else if raw.contains("Variable not found:") {
            if let Some(var_part) = raw.split("Variable not found:").nth(1) {
                let var_name = var_part.split('(').next().unwrap_or(var_part).trim();
                parts.push(format!("  Variable '{}' is not defined.", var_name));
                parts.push(String::new());
                parts.push(
                    "  Tip: Make sure you declared it with 'let' before using it:".to_string(),
                );
                parts.push(format!("    let {} = 0;", var_name));
            } else {
                parts.push(format!("  {}", raw));
            }
        } else if raw.contains("Syntax error") || raw.contains("Expected") {
            parts.push(format!(
                "  Syntax error: {}",
                raw.split(" (line ").next().unwrap_or(raw)
            ));
            parts.push(String::new());
            parts.push(
                "  Tip: Check for missing semicolons, brackets, or typos near this line."
                    .to_string(),
            );
        } else if raw.contains("Data type") && raw.contains("not supported") {
            parts.push(format!(
                "  Type error: {}",
                raw.split(" (line ").next().unwrap_or(raw)
            ));
            parts.push(String::new());
            parts.push(
                "  Tip: Rhai is strict about types. Use .to_float() or .to_int() to convert:"
                    .to_string(),
            );
            parts.push("    let x = 10;           // int".to_string());
            parts.push("    let y = x.to_float(); // now float".to_string());
        } else if raw.contains("Too many operations") {
            parts.push(
                "  Script exceeded the maximum operation limit (50 million ops).".to_string(),
            );
            parts.push(String::new());
            parts
                .push("  Tip: Your script may have an infinite loop, or is processing".to_string());
            parts.push(
                "  too many pixels. Try processing a smaller region with for_region(),".to_string(),
            );
            parts.push("  or use built-in apply_* functions which run natively.".to_string());
        } else if raw.contains("index") && raw.contains("out of") {
            parts.push(format!("  {}", raw.split(" (line ").next().unwrap_or(raw)));
            parts.push(String::new());
            parts.push("  Tip: An array index is out of bounds. Check array lengths".to_string());
            parts.push("  with .len() before accessing elements.".to_string());
        } else if raw.contains("panicked") || raw.contains("internal error") {
            parts.push("  An internal error occurred (this is a bug in PaintFE).".to_string());
            parts.push(String::new());
            parts.push("  Please report this issue.".to_string());
        } else if raw.contains("ErrorTerminated")
            || raw.contains("terminated")
            || raw.contains("Script execution cancelled")
        {
            parts.push("  Script was cancelled.".to_string());
        } else {
            // Generic fallback â€” clean up Rhai's raw message
            let cleaned = raw.split(" (line ").next().unwrap_or(raw);
            parts.push(format!("  {}", cleaned));
        }

        parts.join("\n")
    }
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let (Some(line), Some(col)) = (self.line, self.column) {
            write!(f, "Line {}, Col {}: {}", line, col, self.message)
        } else if let Some(line) = self.line {
            write!(f, "Line {}: {}", line, self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

// ============================================================================
// Messages sent from script worker thread to main thread
// ============================================================================

pub enum ScriptMessage {
    /// Final result (success)
    Completed {
        project_index: usize,
        layer_idx: usize,
        original_pixels: crate::canvas::TiledImage,
        result_pixels: Vec<u8>,
        width: u32,
        height: u32,
        console_output: Vec<String>,
        elapsed_ms: u64,
        /// Canvas-wide transform ops to replay on non-active layers.
        canvas_ops: Vec<CanvasOpRequest>,
    },
    /// Script error
    Error {
        error: ScriptError,
        console_output: Vec<String>,
    },
    /// Intermediate preview frame (from sleep())
    Preview {
        project_index: usize,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Console output line (from print())
    ConsoleOutput(String),
    /// Progress update (0.0 - 1.0)
    Progress(f32),
}

// ============================================================================
// Script context â€” shared mutable state between engine and host functions
// ============================================================================

struct ScriptContext {
    /// Index of the project that spawned this script.
    project_index: usize,
    /// Working pixel buffer (RGBA, row-major)
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    /// Selection mask (same dimensions as pixels, 0=unselected, 255=selected)
    mask: Option<Vec<u8>>,
    console_output: Vec<String>,
    cancelled: Arc<AtomicBool>,
    /// Channel for sending messages to main thread.
    sender: std::sync::mpsc::Sender<ScriptMessage>,
    /// PRNG state (xorshift64)
    rng_state: u64,
    /// Canvas-wide transform ops queued for replay on other layers.
    canvas_ops: Vec<CanvasOpRequest>,
}

type SharedContext = Arc<Mutex<ScriptContext>>;

// ============================================================================
// Engine construction with full sandbox + API registration
// ============================================================================

/// Create a new sandboxed Rhai engine with all PaintFE host functions registered.
fn create_engine(ctx: SharedContext) -> Engine {
    let mut engine = Engine::new();

    // â”€â”€ Sandbox limits â”€â”€
    engine.set_max_operations(50_000_000);
    engine.set_max_call_levels(64);
    engine.set_max_expr_depths(64, 64);
    engine.set_max_string_size(10_000);
    engine.set_max_array_size(10_000);
    engine.set_max_map_size(1_000);

    // â”€â”€ Cancellation check via progress callback â”€â”€
    let cancel_flag = {
        let c = ctx.lock().unwrap_or_else(|e| e.into_inner());
        c.cancelled.clone()
    };
    engine.on_progress(move |_ops| {
        if cancel_flag.load(Ordering::Relaxed) {
            Some(Dynamic::from("Script cancelled by user".to_string()))
        } else {
            None
        }
    });

    // â”€â”€ Register APIs â”€â”€
    register_canvas_api(&mut engine, ctx.clone());
    register_pixel_api(&mut engine, ctx.clone());
    register_effect_api(&mut engine, ctx.clone());
    register_transform_api(&mut engine, ctx.clone());
    register_utility_api(&mut engine, ctx.clone());

    engine
}

// ============================================================================
// Canvas info API
// ============================================================================

fn register_canvas_api(engine: &mut Engine, ctx: SharedContext) {
    let c = ctx.clone();
    engine.register_fn("width", move || -> i64 {
        let lock = c.lock().unwrap_or_else(|e| e.into_inner());
        lock.width as i64
    });

    let c = ctx.clone();
    engine.register_fn("height", move || -> i64 {
        let lock = c.lock().unwrap_or_else(|e| e.into_inner());
        lock.height as i64
    });

    let c = ctx.clone();
    engine.register_fn("is_selected", move |x: i64, y: i64| -> bool {
        let lock = c.lock().unwrap_or_else(|e| e.into_inner());
        if x < 0 || y < 0 || x >= lock.width as i64 || y >= lock.height as i64 {
            return false;
        }
        if let Some(ref mask) = lock.mask {
            let idx = (y as u32 * lock.width + x as u32) as usize;
            idx < mask.len() && mask[idx] > 0
        } else {
            true // no selection = everything selected
        }
    });
}

// ============================================================================
// Pixel access API
// ============================================================================

fn register_pixel_api(engine: &mut Engine, ctx: SharedContext) {
    // get_pixel(x, y) -> [r, g, b, a]
    let c = ctx.clone();
    engine.register_fn("get_pixel", move |x: i64, y: i64| -> Array {
        let lock = c.lock().unwrap_or_else(|e| e.into_inner());
        if x < 0 || y < 0 || x >= lock.width as i64 || y >= lock.height as i64 {
            return vec![
                Dynamic::from(0_i64),
                Dynamic::from(0_i64),
                Dynamic::from(0_i64),
                Dynamic::from(0_i64),
            ];
        }
        let idx = ((y as u32 * lock.width + x as u32) * 4) as usize;
        if idx + 3 < lock.pixels.len() {
            vec![
                Dynamic::from(lock.pixels[idx] as i64),
                Dynamic::from(lock.pixels[idx + 1] as i64),
                Dynamic::from(lock.pixels[idx + 2] as i64),
                Dynamic::from(lock.pixels[idx + 3] as i64),
            ]
        } else {
            vec![
                Dynamic::from(0_i64),
                Dynamic::from(0_i64),
                Dynamic::from(0_i64),
                Dynamic::from(0_i64),
            ]
        }
    });

    // set_pixel(x, y, r, g, b, a)
    let c = ctx.clone();
    engine.register_fn(
        "set_pixel",
        move |x: i64, y: i64, r: i64, g: i64, b: i64, a: i64| {
            let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
            if x < 0 || y < 0 || x >= lock.width as i64 || y >= lock.height as i64 {
                return;
            }
            let idx = ((y as u32 * lock.width + x as u32) * 4) as usize;
            if idx + 3 < lock.pixels.len() {
                lock.pixels[idx] = (r.clamp(0, 255)) as u8;
                lock.pixels[idx + 1] = (g.clamp(0, 255)) as u8;
                lock.pixels[idx + 2] = (b.clamp(0, 255)) as u8;
                lock.pixels[idx + 3] = (a.clamp(0, 255)) as u8;
            }
        },
    );

    // get_r/g/b/a â€” fast single-channel read
    for (name, offset) in [("get_r", 0usize), ("get_g", 1), ("get_b", 2), ("get_a", 3)] {
        let c = ctx.clone();
        engine.register_fn(name, move |x: i64, y: i64| -> i64 {
            let lock = c.lock().unwrap_or_else(|e| e.into_inner());
            if x < 0 || y < 0 || x >= lock.width as i64 || y >= lock.height as i64 {
                return 0;
            }
            let idx = ((y as u32 * lock.width + x as u32) * 4) as usize + offset;
            if idx < lock.pixels.len() {
                lock.pixels[idx] as i64
            } else {
                0
            }
        });
    }

    // set_r/g/b/a â€” fast single-channel write
    for (name, offset) in [("set_r", 0usize), ("set_g", 1), ("set_b", 2), ("set_a", 3)] {
        let c = ctx.clone();
        engine.register_fn(name, move |x: i64, y: i64, v: i64| {
            let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
            if x < 0 || y < 0 || x >= lock.width as i64 || y >= lock.height as i64 {
                return;
            }
            let idx = ((y as u32 * lock.width + x as u32) * 4) as usize + offset;
            if idx < lock.pixels.len() {
                lock.pixels[idx] = (v.clamp(0, 255)) as u8;
            }
        });
    }

    // â”€â”€ Bulk iteration: for_each_pixel â”€â”€
    // Calls a Rhai closure f(x, y, r, g, b, a) for every pixel.
    // If the closure returns an array [r, g, b, a], the pixel is updated.
    // If it returns () (unit), the pixel is left unchanged.
    let c = ctx.clone();
    engine.register_fn(
        "for_each_pixel",
        move |ncc: rhai::NativeCallContext, callback: FnPtr| -> Result<(), Box<EvalAltResult>> {
            // Extract dimensions and pixels under brief lock
            let (w, h, mut pixels) = {
                let lock = c.lock().unwrap_or_else(|e| e.into_inner());
                (lock.width, lock.height, lock.pixels.clone())
            };

            for y in 0..h as i64 {
                for x in 0..w as i64 {
                    let idx = ((y as u32 * w + x as u32) * 4) as usize;
                    let r = pixels[idx] as i64;
                    let g = pixels[idx + 1] as i64;
                    let b = pixels[idx + 2] as i64;
                    let a = pixels[idx + 3] as i64;

                    let result =
                        callback.call_within_context::<Dynamic>(&ncc, (x, y, r, g, b, a))?;

                    // If result is an array of 4 integers, update pixel
                    if result.is_array() {
                        let arr = result.cast::<Array>();
                        if arr.len() >= 4 {
                            pixels[idx] = arr[0].as_int().unwrap_or(r).clamp(0, 255) as u8;
                            pixels[idx + 1] = arr[1].as_int().unwrap_or(g).clamp(0, 255) as u8;
                            pixels[idx + 2] = arr[2].as_int().unwrap_or(b).clamp(0, 255) as u8;
                            pixels[idx + 3] = arr[3].as_int().unwrap_or(a).clamp(0, 255) as u8;
                        }
                    }
                    // else: unit or other â†’ skip (leave pixel unchanged)
                }

                // Check cancellation periodically (every row)
                {
                    let lock = c.lock().unwrap_or_else(|e| e.into_inner());
                    if lock.cancelled.load(Ordering::Relaxed) {
                        return Err(Box::new(EvalAltResult::ErrorSystem(
                            "cancelled".to_string(),
                            "Script cancelled by user".into(),
                        )));
                    }
                }
            }

            // Write back modified pixels
            {
                let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
                lock.pixels = pixels;
            }

            Ok(())
        },
    );

    // â”€â”€ Bulk iteration: for_region(x, y, w, h, closure) â”€â”€
    let c = ctx.clone();
    engine.register_fn(
        "for_region",
        move |ncc: rhai::NativeCallContext,
              rx: i64,
              ry: i64,
              rw: i64,
              rh: i64,
              callback: FnPtr|
              -> Result<(), Box<EvalAltResult>> {
            let (w, h, mut pixels) = {
                let lock = c.lock().unwrap_or_else(|e| e.into_inner());
                (lock.width, lock.height, lock.pixels.clone())
            };

            let x0 = rx.max(0) as u32;
            let y0 = ry.max(0) as u32;
            let x1 = ((rx + rw) as u32).min(w);
            let y1 = ((ry + rh) as u32).min(h);

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = ((y * w + x) * 4) as usize;
                    let r = pixels[idx] as i64;
                    let g = pixels[idx + 1] as i64;
                    let b = pixels[idx + 2] as i64;
                    let a = pixels[idx + 3] as i64;

                    let result = callback
                        .call_within_context::<Dynamic>(&ncc, (x as i64, y as i64, r, g, b, a))?;

                    if result.is_array() {
                        let arr = result.cast::<Array>();
                        if arr.len() >= 4 {
                            pixels[idx] = arr[0].as_int().unwrap_or(r).clamp(0, 255) as u8;
                            pixels[idx + 1] = arr[1].as_int().unwrap_or(g).clamp(0, 255) as u8;
                            pixels[idx + 2] = arr[2].as_int().unwrap_or(b).clamp(0, 255) as u8;
                            pixels[idx + 3] = arr[3].as_int().unwrap_or(a).clamp(0, 255) as u8;
                        }
                    }
                }

                {
                    let lock = c.lock().unwrap_or_else(|e| e.into_inner());
                    if lock.cancelled.load(Ordering::Relaxed) {
                        return Err(Box::new(EvalAltResult::ErrorSystem(
                            "cancelled".to_string(),
                            "Script cancelled by user".into(),
                        )));
                    }
                }
            }

            {
                let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
                lock.pixels = pixels;
            }
            Ok(())
        },
    );

    // â”€â”€ Bulk iteration: map_channels(closure) â”€â”€
    // Calls f(r, g, b, a) for every pixel, expects [r, g, b, a] return.
    // Fastest bulk path â€” no x/y coordinates.
    let c = ctx.clone();
    engine.register_fn(
        "map_channels",
        move |ncc: rhai::NativeCallContext, callback: FnPtr| -> Result<(), Box<EvalAltResult>> {
            let (w, h, mut pixels) = {
                let lock = c.lock().unwrap_or_else(|e| e.into_inner());
                (lock.width, lock.height, lock.pixels.clone())
            };

            let total = (w * h) as usize;
            for i in 0..total {
                let idx = i * 4;
                let r = pixels[idx] as i64;
                let g = pixels[idx + 1] as i64;
                let b = pixels[idx + 2] as i64;
                let a = pixels[idx + 3] as i64;

                let result = callback.call_within_context::<Dynamic>(&ncc, (r, g, b, a))?;

                if result.is_array() {
                    let arr = result.cast::<Array>();
                    if arr.len() >= 4 {
                        pixels[idx] = arr[0].as_int().unwrap_or(r).clamp(0, 255) as u8;
                        pixels[idx + 1] = arr[1].as_int().unwrap_or(g).clamp(0, 255) as u8;
                        pixels[idx + 2] = arr[2].as_int().unwrap_or(b).clamp(0, 255) as u8;
                        pixels[idx + 3] = arr[3].as_int().unwrap_or(a).clamp(0, 255) as u8;
                    }
                }

                // Check cancellation every 10000 pixels
                if i % 10000 == 0 {
                    let lock = c.lock().unwrap_or_else(|e| e.into_inner());
                    if lock.cancelled.load(Ordering::Relaxed) {
                        return Err(Box::new(EvalAltResult::ErrorSystem(
                            "cancelled".to_string(),
                            "Script cancelled by user".into(),
                        )));
                    }
                }
            }

            {
                let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
                lock.pixels = pixels;
            }
            Ok(())
        },
    );
}

// ============================================================================
// Effect API â€” wrap existing _core functions
// ============================================================================

/// Reconstruct an RgbaImage from context pixels, apply a function, write back.
fn apply_effect_to_context(
    ctx: &SharedContext,
    f: impl FnOnce(&RgbaImage, Option<&GrayImage>) -> RgbaImage,
) {
    let mut lock = ctx.lock().unwrap_or_else(|e| e.into_inner());
    let w = lock.width;
    let h = lock.height;
    let img =
        RgbaImage::from_raw(w, h, lock.pixels.clone()).unwrap_or_else(|| RgbaImage::new(w, h));

    let mask = lock
        .mask
        .as_ref()
        .map(|m| GrayImage::from_raw(w, h, m.clone()).unwrap_or_else(|| GrayImage::new(w, h)));

    let result = f(&img, mask.as_ref());
    lock.pixels = result.into_raw();
}

// ============================================================================
// Transform API — layer-only and canvas-wide geometric transforms
// ============================================================================

fn register_transform_api(engine: &mut Engine, ctx: SharedContext) {
    // -- Layer-only (active layer only, no canvas_ops entry) --

    // flip_horizontal() — mirror active layer left↔right
    let c = ctx.clone();
    engine.register_fn("flip_horizontal", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let w = lock.width;
        let h = lock.height;
        if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
            lock.pixels = imageops::flip_horizontal(&img).into_raw();
        }
    });

    // flip_vertical() — mirror active layer top↔bottom
    let c = ctx.clone();
    engine.register_fn("flip_vertical", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let w = lock.width;
        let h = lock.height;
        if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
            lock.pixels = imageops::flip_vertical(&img).into_raw();
        }
    });

    // rotate_180() — rotate active layer 180°
    let c = ctx.clone();
    engine.register_fn("rotate_180", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let w = lock.width;
        let h = lock.height;
        if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
            lock.pixels = imageops::rotate180(&img).into_raw();
        }
    });

    // -- Canvas-wide (all layers, replayed by app.rs) --

    // flip_canvas_horizontal() — mirror every layer left↔right
    let c = ctx.clone();
    engine.register_fn("flip_canvas_horizontal", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let w = lock.width;
        let h = lock.height;
        if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
            lock.pixels = imageops::flip_horizontal(&img).into_raw();
        }
        lock.canvas_ops.push(CanvasOpRequest::FlipHorizontal);
    });

    // flip_canvas_vertical() — mirror every layer top↔bottom
    let c = ctx.clone();
    engine.register_fn("flip_canvas_vertical", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let w = lock.width;
        let h = lock.height;
        if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
            lock.pixels = imageops::flip_vertical(&img).into_raw();
        }
        lock.canvas_ops.push(CanvasOpRequest::FlipVertical);
    });

    // rotate_canvas_90cw() — rotate every layer 90° clockwise (swaps W↔H)
    let c = ctx.clone();
    engine.register_fn("rotate_canvas_90cw", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let w = lock.width;
        let h = lock.height;
        if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
            lock.pixels = imageops::rotate90(&img).into_raw();
        }
        let tmp_w = lock.width;
        lock.width = lock.height;
        lock.height = tmp_w;
        lock.canvas_ops.push(CanvasOpRequest::Rotate90CW);
    });

    // rotate_canvas_90ccw() — rotate every layer 90° counter-clockwise (swaps W↔H)
    let c = ctx.clone();
    engine.register_fn("rotate_canvas_90ccw", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let w = lock.width;
        let h = lock.height;
        if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
            lock.pixels = imageops::rotate270(&img).into_raw();
        }
        let tmp_w = lock.width;
        lock.width = lock.height;
        lock.height = tmp_w;
        lock.canvas_ops.push(CanvasOpRequest::Rotate90CCW);
    });

    // rotate_canvas_180() — rotate every layer 180°
    let c = ctx.clone();
    engine.register_fn("rotate_canvas_180", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let w = lock.width;
        let h = lock.height;
        if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
            lock.pixels = imageops::rotate180(&img).into_raw();
        }
        lock.canvas_ops.push(CanvasOpRequest::Rotate180);
    });

    // resize_image(w, h, method) — scale every layer to new dimensions.
    // method: "nearest", "bilinear" (default), "bicubic", "lanczos"
    let c = ctx.clone();
    engine.register_fn(
        "resize_image",
        move |new_w: i64, new_h: i64, method: ImmutableString| {
            let new_w = (new_w.max(1) as u32).min(32768);
            let new_h = (new_h.max(1) as u32).min(32768);
            let filter = parse_script_filter(&method);
            let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
            let w = lock.width;
            let h = lock.height;
            if new_w == w && new_h == h {
                return;
            }
            if let Some(img) = RgbaImage::from_raw(w, h, lock.pixels.clone()) {
                let resized = imageops::resize(&img, new_w, new_h, filter.to_image_filter());
                lock.pixels = resized.into_raw();
                lock.width = new_w;
                lock.height = new_h;
            }
            lock.canvas_ops.push(CanvasOpRequest::ResizeImage {
                w: new_w,
                h: new_h,
                filter,
            });
        },
    );

    // resize_canvas(w, h, anchor) — extend/crop canvas; content stays at anchor.
    // anchor: "top-left" (default), "top-center", "top-right",
    //         "center-left", "center", "center-right",
    //         "bottom-left", "bottom-center", "bottom-right"
    // Short aliases: "tl", "tc", "tr", "cl", "c", "cr", "bl", "bc", "br"
    let c = ctx.clone();
    engine.register_fn(
        "resize_canvas",
        move |new_w: i64, new_h: i64, anchor: ImmutableString| {
            let new_w = (new_w.max(1) as u32).min(32768);
            let new_h = (new_h.max(1) as u32).min(32768);
            let anchor_tuple = parse_anchor(&anchor);
            let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
            let old_w = lock.width;
            let old_h = lock.height;
            let offset_x: i32 = match anchor_tuple.0 {
                0 => 0,
                1 => ((new_w as i32) - (old_w as i32)) / 2,
                _ => (new_w as i32) - (old_w as i32),
            };
            let offset_y: i32 = match anchor_tuple.1 {
                0 => 0,
                1 => ((new_h as i32) - (old_h as i32)) / 2,
                _ => (new_h as i32) - (old_h as i32),
            };
            if let Some(old_img) = RgbaImage::from_raw(old_w, old_h, lock.pixels.clone()) {
                let mut new_img = RgbaImage::new(new_w, new_h); // zeroed = transparent
                for y in 0..old_h {
                    for x in 0..old_w {
                        let nx = x as i32 + offset_x;
                        let ny = y as i32 + offset_y;
                        if nx >= 0 && ny >= 0 && (nx as u32) < new_w && (ny as u32) < new_h {
                            new_img.put_pixel(nx as u32, ny as u32, *old_img.get_pixel(x, y));
                        }
                    }
                }
                lock.pixels = new_img.into_raw();
                lock.width = new_w;
                lock.height = new_h;
            }
            lock.canvas_ops.push(CanvasOpRequest::ResizeCanvas {
                new_w,
                new_h,
                anchor: anchor_tuple,
            });
        },
    );
}

fn register_effect_api(engine: &mut Engine, ctx: SharedContext) {
    // --- Blur ---
    let c = ctx.clone();
    engine.register_fn("apply_blur", move |sigma: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::filters::blur_with_selection_pub(img, sigma as f32, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_box_blur", move |radius: i64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::box_blur_core(img, radius as f32, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_motion_blur", move |angle: f64, distance: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::motion_blur_core(img, angle as f32, distance as f32, mask)
        });
    });

    // --- Sharpen / Denoise ---
    let c = ctx.clone();
    engine.register_fn("apply_sharpen", move |amount: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::sharpen_core(img, amount as f32, 1.0, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_reduce_noise", move |strength: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::reduce_noise_core(img, strength as f32, 2, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_median", move |radius: i64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::median_core(img, radius.max(1) as u32, mask)
        });
    });

    // --- Color / Adjustment ---
    let c = ctx.clone();
    engine.register_fn("apply_invert", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let len = lock.pixels.len();
        let mut i = 0;
        while i + 3 < len {
            lock.pixels[i] = 255 - lock.pixels[i];
            lock.pixels[i + 1] = 255 - lock.pixels[i + 1];
            lock.pixels[i + 2] = 255 - lock.pixels[i + 2];
            // alpha unchanged
            i += 4;
        }
    });

    let c = ctx.clone();
    engine.register_fn("apply_desaturate", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let len = lock.pixels.len();
        let mut i = 0;
        while i + 3 < len {
            let r = lock.pixels[i] as u32;
            let g = lock.pixels[i + 1] as u32;
            let b = lock.pixels[i + 2] as u32;
            let gray = ((r * 299 + g * 587 + b * 114) / 1000) as u8;
            lock.pixels[i] = gray;
            lock.pixels[i + 1] = gray;
            lock.pixels[i + 2] = gray;
            i += 4;
        }
    });

    let c = ctx.clone();
    engine.register_fn("apply_sepia", move || {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let len = lock.pixels.len();
        let mut i = 0;
        while i + 3 < len {
            let r = lock.pixels[i] as f32;
            let g = lock.pixels[i + 1] as f32;
            let b = lock.pixels[i + 2] as f32;
            let nr = (r * 0.393 + g * 0.769 + b * 0.189).min(255.0) as u8;
            let ng = (r * 0.349 + g * 0.686 + b * 0.168).min(255.0) as u8;
            let nb = (r * 0.272 + g * 0.534 + b * 0.131).min(255.0) as u8;
            lock.pixels[i] = nr;
            lock.pixels[i + 1] = ng;
            lock.pixels[i + 2] = nb;
            i += 4;
        }
    });

    // apply_sepia(strength) â€” with blend strength 0.0..1.0
    let c = ctx.clone();
    engine.register_fn("apply_sepia", move |strength: f64| {
        let strength = strength.clamp(0.0, 1.0) as f32;
        let inv = 1.0 - strength;
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let len = lock.pixels.len();
        let mut i = 0;
        while i + 3 < len {
            let r = lock.pixels[i] as f32;
            let g = lock.pixels[i + 1] as f32;
            let b = lock.pixels[i + 2] as f32;
            let sr = (r * 0.393 + g * 0.769 + b * 0.189).min(255.0);
            let sg = (r * 0.349 + g * 0.686 + b * 0.168).min(255.0);
            let sb = (r * 0.272 + g * 0.534 + b * 0.131).min(255.0);
            lock.pixels[i] = (r * inv + sr * strength) as u8;
            lock.pixels[i + 1] = (g * inv + sg * strength) as u8;
            lock.pixels[i + 2] = (b * inv + sb * strength) as u8;
            i += 4;
        }
    });

    let c = ctx.clone();
    engine.register_fn(
        "apply_brightness_contrast",
        move |brightness: f64, contrast: f64| {
            let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
            let factor = (259.0 * (contrast as f32 + 255.0)) / (255.0 * (259.0 - contrast as f32));
            let bright = brightness as f32;
            let len = lock.pixels.len();
            let mut i = 0;
            while i + 3 < len {
                let r = lock.pixels[i] as f32;
                let g = lock.pixels[i + 1] as f32;
                let b = lock.pixels[i + 2] as f32;
                lock.pixels[i] = (factor * (r + bright - 128.0) + 128.0).clamp(0.0, 255.0) as u8;
                lock.pixels[i + 1] =
                    (factor * (g + bright - 128.0) + 128.0).clamp(0.0, 255.0) as u8;
                lock.pixels[i + 2] =
                    (factor * (b + bright - 128.0) + 128.0).clamp(0.0, 255.0) as u8;
                i += 4;
            }
        },
    );

    let c = ctx.clone();
    engine.register_fn("apply_hsl", move |hue: f64, sat: f64, light: f64| {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let hue_shift = hue as f32;
        let sat_factor = 1.0 + sat as f32 / 100.0;
        let light_offset = light as f32 * 255.0 / 100.0;
        let len = lock.pixels.len();
        let mut i = 0;
        while i + 3 < len {
            let r = lock.pixels[i] as f32 / 255.0;
            let g = lock.pixels[i + 1] as f32 / 255.0;
            let b = lock.pixels[i + 2] as f32 / 255.0;
            let cmax = r.max(g).max(b);
            let cmin = r.min(g).min(b);
            let l = (cmax + cmin) / 2.0;
            let (h, s) = if (cmax - cmin).abs() < 1e-10 {
                (0.0, 0.0)
            } else {
                let d = cmax - cmin;
                let s = if l > 0.5 {
                    d / (2.0 - cmax - cmin)
                } else {
                    d / (cmax + cmin)
                };
                let h = if (cmax - r).abs() < 1e-10 {
                    (g - b) / d + if g < b { 6.0 } else { 0.0 }
                } else if (cmax - g).abs() < 1e-10 {
                    (b - r) / d + 2.0
                } else {
                    (r - g) / d + 4.0
                } / 6.0;
                (h, s)
            };
            let nh = (h + hue_shift / 360.0).rem_euclid(1.0);
            let ns = (s * sat_factor).clamp(0.0, 1.0);
            // HSL to RGB
            let (nr, ng, nb) = if ns.abs() < 1e-10 {
                (l, l, l)
            } else {
                let q = if l < 0.5 {
                    l * (1.0 + ns)
                } else {
                    l + ns - l * ns
                };
                let p = 2.0 * l - q;
                let hue2rgb = |p: f32, q: f32, mut t: f32| -> f32 {
                    if t < 0.0 {
                        t += 1.0;
                    }
                    if t > 1.0 {
                        t -= 1.0;
                    }
                    if t < 1.0 / 6.0 {
                        return p + (q - p) * 6.0 * t;
                    }
                    if t < 1.0 / 2.0 {
                        return q;
                    }
                    if t < 2.0 / 3.0 {
                        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
                    }
                    p
                };
                (
                    hue2rgb(p, q, nh + 1.0 / 3.0),
                    hue2rgb(p, q, nh),
                    hue2rgb(p, q, nh - 1.0 / 3.0),
                )
            };
            lock.pixels[i] = (nr * 255.0 + light_offset).clamp(0.0, 255.0) as u8;
            lock.pixels[i + 1] = (ng * 255.0 + light_offset).clamp(0.0, 255.0) as u8;
            lock.pixels[i + 2] = (nb * 255.0 + light_offset).clamp(0.0, 255.0) as u8;
            i += 4;
        }
    });

    let c = ctx.clone();
    engine.register_fn("apply_exposure", move |ev: f64| {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let gain = 2.0f32.powf(ev as f32);
        let len = lock.pixels.len();
        let mut i = 0;
        while i + 3 < len {
            lock.pixels[i] = (lock.pixels[i] as f32 * gain).clamp(0.0, 255.0) as u8;
            lock.pixels[i + 1] = (lock.pixels[i + 1] as f32 * gain).clamp(0.0, 255.0) as u8;
            lock.pixels[i + 2] = (lock.pixels[i + 2] as f32 * gain).clamp(0.0, 255.0) as u8;
            i += 4;
        }
    });

    let c = ctx.clone();
    engine.register_fn("apply_levels", move |black: f64, white: f64, gamma: f64| {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let in_black = black as f32;
        let in_white = white as f32;
        let in_range = (in_white - in_black).max(1.0);
        let inv_gamma = 1.0 / (gamma as f32).max(0.01);
        // Build LUT
        let mut lut = [0u8; 256];
        for (i, item) in lut.iter_mut().enumerate() {
            let normalized = ((i as f32 - in_black) / in_range).clamp(0.0, 1.0);
            let gamma_corrected = normalized.powf(inv_gamma);
            *item = (gamma_corrected * 255.0).clamp(0.0, 255.0) as u8;
        }
        let len = lock.pixels.len();
        let mut i = 0;
        while i + 3 < len {
            lock.pixels[i] = lut[lock.pixels[i] as usize];
            lock.pixels[i + 1] = lut[lock.pixels[i + 1] as usize];
            lock.pixels[i + 2] = lut[lock.pixels[i + 2] as usize];
            i += 4;
        }
    });

    // --- Noise ---
    let c = ctx.clone();
    engine.register_fn("apply_noise", move |amount: f64, monochrome: bool| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::add_noise_core(
                img,
                amount as f32,
                crate::ops::effects::NoiseType::Gaussian,
                monochrome,
                42,
                1.0,
                1,
                mask,
            )
        });
    });

    // --- Distort ---
    let c = ctx.clone();
    engine.register_fn("apply_pixelate", move |size: i64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::pixelate_core(img, size.max(1) as u32, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_crystallize", move |size: i64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::crystallize_core(img, size.max(1) as f32, 42, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_bulge", move |amount: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::bulge_core(img, amount as f32, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_twist", move |angle: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::twist_core(img, angle as f32, mask)
        });
    });

    // --- Stylize ---
    let c = ctx.clone();
    engine.register_fn("apply_glow", move |radius: f64, intensity: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::glow_core(img, radius as f32, intensity as f32, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_vignette", move |strength: f64, softness: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::vignette_core(img, strength as f32, softness as f32, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_halftone", move |dot_size: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::halftone_core(
                img,
                dot_size as f32,
                45.0,
                crate::ops::effects::HalftoneShape::Circle,
                mask,
            )
        });
    });

    // --- Artistic ---
    let c = ctx.clone();
    engine.register_fn("apply_ink", move |strength: f64, threshold: f64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::ink_core(img, strength as f32, threshold as f32, mask)
        });
    });

    let c = ctx.clone();
    engine.register_fn("apply_oil_painting", move |radius: i64| {
        apply_effect_to_context(&c, |img, mask| {
            crate::ops::effects::oil_painting_core(img, radius.max(1) as u32, 20, mask)
        });
    });
}

// ============================================================================
// Utility API
// ============================================================================

fn register_utility_api(engine: &mut Engine, ctx: SharedContext) {
    // print(msg) â€” output to script console
    let c = ctx.clone();
    engine.register_fn("print_line", move |msg: ImmutableString| {
        let lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let _ = lock
            .sender
            .send(ScriptMessage::ConsoleOutput(msg.to_string()));
    });
    // Also override built-in print
    let c = ctx.clone();
    engine.on_print(move |msg| {
        let lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let _ = lock
            .sender
            .send(ScriptMessage::ConsoleOutput(msg.to_string()));
    });

    // sleep(ms) â€” pause and send intermediate preview
    let c = ctx.clone();
    engine.register_fn("sleep", move |ms: i64| {
        let ms_clamped = ms.clamp(0, 10_000) as u64;
        // Send current pixel state as preview
        {
            let lock = c.lock().unwrap_or_else(|e| e.into_inner());
            let _ = lock.sender.send(ScriptMessage::Preview {
                project_index: lock.project_index,
                pixels: lock.pixels.clone(),
                width: lock.width,
                height: lock.height,
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(ms_clamped));
    });

    // progress(fraction) â€” update progress bar
    let c = ctx.clone();
    engine.register_fn("progress", move |frac: f64| {
        let lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let _ = lock
            .sender
            .send(ScriptMessage::Progress(frac.clamp(0.0, 1.0) as f32));
    });

    // Random number generation (xorshift64 PRNG, properly seeded per-script)
    let c = ctx.clone();
    engine.register_fn("rand_int", move |min: i64, max: i64| -> i64 {
        if min >= max {
            return min;
        }
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        // xorshift64
        let mut s = lock.rng_state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        lock.rng_state = s;
        let range = (max - min) as u64;
        min + ((s % range.max(1)) as i64)
    });

    let c = ctx.clone();
    engine.register_fn("rand_float", move |min: f64, max: f64| -> f64 {
        if min >= max {
            return min;
        }
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = lock.rng_state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        lock.rng_state = s;
        min + ((s as f64) / (u64::MAX as f64)) * (max - min)
    });

    // rand_float() with no args returns 0.0..1.0
    let c = ctx.clone();
    engine.register_fn("rand_float", move || -> f64 {
        let mut lock = c.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = lock.rng_state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        lock.rng_state = s;
        (s as f64) / (u64::MAX as f64)
    });

    // Math utilities
    engine.register_fn("clamp", |v: i64, lo: i64, hi: i64| -> i64 {
        v.clamp(lo, hi)
    });
    engine.register_fn("clamp_f", |v: f64, lo: f64, hi: f64| -> f64 {
        v.clamp(lo, hi)
    });
    engine.register_fn("lerp", |a: f64, b: f64, t: f64| -> f64 { a + (b - a) * t });
    engine.register_fn("distance", |x1: f64, y1: f64, x2: f64, y2: f64| -> f64 {
        ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
    });
    // abs: Rhai built-in covers integers; add float overload
    engine.register_fn("abs", |x: f64| -> f64 { x.abs() });
    engine.register_fn("abs", |x: i64| -> i64 { x.abs() });
    // min/max: register both int and float overloads with short names
    engine.register_fn("min", |a: i64, b: i64| -> i64 { a.min(b) });
    engine.register_fn("max", |a: i64, b: i64| -> i64 { a.max(b) });
    engine.register_fn("min", |a: f64, b: f64| -> f64 { a.min(b) });
    engine.register_fn("max", |a: f64, b: f64| -> f64 { a.max(b) });
    // Keep suffixed variants for backward compatibility
    engine.register_fn("abs_i", |x: i64| -> i64 { x.abs() });
    engine.register_fn("min_i", |a: i64, b: i64| -> i64 { a.min(b) });
    engine.register_fn("max_i", |a: i64, b: i64| -> i64 { a.max(b) });
    engine.register_fn("min_f", |a: f64, b: f64| -> f64 { a.min(b) });
    engine.register_fn("max_f", |a: f64, b: f64| -> f64 { a.max(b) });
    engine.register_fn("floor", |x: f64| -> f64 { x.floor() });
    engine.register_fn("ceil", |x: f64| -> f64 { x.ceil() });
    engine.register_fn("round", |x: f64| -> f64 { x.round() });
    engine.register_fn("sqrt", |x: f64| -> f64 { x.sqrt() });
    engine.register_fn("pow", |x: f64, y: f64| -> f64 { x.powf(y) });
    engine.register_fn("sin", |x: f64| -> f64 { x.sin() });
    engine.register_fn("cos", |x: f64| -> f64 { x.cos() });
    engine.register_fn("tan", |x: f64| -> f64 { x.tan() });
    engine.register_fn("atan2", |y: f64, x: f64| -> f64 { y.atan2(x) });
    engine.register_fn("PI", || -> f64 { std::f64::consts::PI });

    // Color space conversion
    engine.register_fn("rgb_to_hsl", |r: i64, g: i64, b: i64| -> Array {
        let rf = r.clamp(0, 255) as f64 / 255.0;
        let gf = g.clamp(0, 255) as f64 / 255.0;
        let bf = b.clamp(0, 255) as f64 / 255.0;
        let max = rf.max(gf).max(bf);
        let min = rf.min(gf).min(bf);
        let l = (max + min) / 2.0;
        if (max - min).abs() < 1e-10 {
            return vec![
                Dynamic::from(0.0_f64),
                Dynamic::from(0.0_f64),
                Dynamic::from(l * 100.0),
            ];
        }
        let d = max - min;
        let s = if l > 0.5 {
            d / (2.0 - max - min)
        } else {
            d / (max + min)
        };
        let h = if (max - rf).abs() < 1e-10 {
            (gf - bf) / d + if gf < bf { 6.0 } else { 0.0 }
        } else if (max - gf).abs() < 1e-10 {
            (bf - rf) / d + 2.0
        } else {
            (rf - gf) / d + 4.0
        } * 60.0;
        vec![
            Dynamic::from(h),
            Dynamic::from(s * 100.0),
            Dynamic::from(l * 100.0),
        ]
    });

    engine.register_fn("hsl_to_rgb", |h: f64, s: f64, l: f64| -> Array {
        let s = s / 100.0;
        let l = l / 100.0;
        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let h2 = h / 60.0;
        let x = c * (1.0 - (h2 % 2.0 - 1.0).abs());
        let (r1, g1, b1) = match h2 as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        let m = l - c / 2.0;
        vec![
            Dynamic::from(((r1 + m) * 255.0).round() as i64),
            Dynamic::from(((g1 + m) * 255.0).round() as i64),
            Dynamic::from(((b1 + m) * 255.0).round() as i64),
        ]
    });
}

// ============================================================================
// Public execution API
// ============================================================================

/// Compile a script and return the AST, or a ScriptError.
pub fn compile_script(source: &str) -> Result<AST, ScriptError> {
    // Use a temp engine just for compilation (no context needed)
    let engine = Engine::new();
    engine.compile(source).map_err(|e| {
        let pos = e.position();
        ScriptError {
            message: e.to_string(),
            line: if pos.line().unwrap_or(0) > 0 {
                Some(pos.line().unwrap_or(0))
            } else {
                None
            },
            column: if pos.position().unwrap_or(0) > 0 {
                Some(pos.position().unwrap_or(0))
            } else {
                None
            },
        }
    })
}

/// Execute a script on a background thread.
/// Takes the active layer's pixel data, runs the script, and sends results via channel.
pub fn execute_script(
    source: String,
    project_index: usize,
    layer_idx: usize,
    original_pixels: crate::canvas::TiledImage,
    flat_pixels: Vec<u8>,
    width: u32,
    height: u32,
    mask: Option<Vec<u8>>,
    cancel_flag: Arc<AtomicBool>,
    sender: std::sync::mpsc::Sender<ScriptMessage>,
) {
    let sender_clone = sender.clone();
    rayon::spawn(move || {
        let start = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let rng_seed = {
                use std::time::SystemTime;
                let t = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default();
                t.as_nanos() as u64 ^ 0x517cc1b727220a95
            };
            let ctx = Arc::new(Mutex::new(ScriptContext {
                project_index,
                pixels: flat_pixels,
                width,
                height,
                mask,
                console_output: Vec::new(),
                cancelled: cancel_flag,
                sender: sender_clone.clone(),
                rng_state: rng_seed,
                canvas_ops: Vec::new(),
            }));

            let engine = create_engine(ctx.clone());
            let mut scope = Scope::new();

            let ast = engine.compile(&source).map_err(|e| {
                let pos = e.position();
                ScriptError {
                    message: e.to_string(),
                    line: if pos.line().unwrap_or(0) > 0 {
                        Some(pos.line().unwrap_or(0))
                    } else {
                        None
                    },
                    column: if pos.position().unwrap_or(0) > 0 {
                        Some(pos.position().unwrap_or(0))
                    } else {
                        None
                    },
                }
            })?;

            engine.run_ast_with_scope(&mut scope, &ast).map_err(|e| {
                let pos = e.position();
                ScriptError {
                    message: e.to_string(),
                    line: if pos.line().unwrap_or(0) > 0 {
                        Some(pos.line().unwrap_or(0))
                    } else {
                        None
                    },
                    column: if pos.position().unwrap_or(0) > 0 {
                        Some(pos.position().unwrap_or(0))
                    } else {
                        None
                    },
                }
            })?;

            // Extract final pixels, canvas ops, and console output
            let lock = ctx.lock().unwrap_or_else(|e| e.into_inner());
            Ok::<(Vec<u8>, u32, u32, Vec<String>, Vec<CanvasOpRequest>), ScriptError>((
                lock.pixels.clone(),
                lock.width,
                lock.height,
                lock.console_output.clone(),
                lock.canvas_ops.clone(),
            ))
        }));

        let elapsed_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(Ok((pixels, final_w, final_h, console_output, canvas_ops))) => {
                let _ = sender.send(ScriptMessage::Completed {
                    project_index,
                    layer_idx,
                    original_pixels,
                    result_pixels: pixels,
                    width: final_w,
                    height: final_h,
                    console_output,
                    elapsed_ms,
                    canvas_ops,
                });
            }
            Ok(Err(error)) => {
                let _ = sender.send(ScriptMessage::Error {
                    error,
                    console_output: Vec::new(),
                });
            }
            Err(_panic) => {
                let _ = sender.send(ScriptMessage::Error {
                    error: ScriptError {
                        message: "Script panicked (internal error)".to_string(),
                        line: None,
                        column: None,
                    },
                    console_output: Vec::new(),
                });
            }
        }
    });
}

// ============================================================================
// Shared canvas-op replay helper (used by both GUI handler and CLI)
// ============================================================================

/// Apply a list of canvas-wide transform operations to every layer in `state`
/// **except** `active_layer_idx` (which the caller has already updated).
///
/// Updates `state.width` and `state.height` to the final post-op dimensions.
pub fn apply_canvas_ops(
    state: &mut crate::canvas::CanvasState,
    active_layer_idx: usize,
    canvas_ops: &[CanvasOpRequest],
) {
    let mut cur_w = state.width;
    let mut cur_h = state.height;

    for op in canvas_ops {
        let n = state.layers.len();
        for i in 0..n {
            if i == active_layer_idx {
                continue; // active layer already has the correct result
            }
            let flat = state.layers[i]
                .pixels
                .extract_region_rgba(0, 0, cur_w, cur_h);
            if let Some(img) = RgbaImage::from_raw(cur_w, cur_h, flat) {
                let new_img: RgbaImage = match op {
                    CanvasOpRequest::FlipHorizontal => imageops::flip_horizontal(&img),
                    CanvasOpRequest::FlipVertical => imageops::flip_vertical(&img),
                    CanvasOpRequest::Rotate90CW => imageops::rotate90(&img),
                    CanvasOpRequest::Rotate90CCW => imageops::rotate270(&img),
                    CanvasOpRequest::Rotate180 => imageops::rotate180(&img),
                    CanvasOpRequest::ResizeImage { w, h, filter } => {
                        imageops::resize(&img, *w, *h, filter.to_image_filter())
                    }
                    CanvasOpRequest::ResizeCanvas {
                        new_w,
                        new_h,
                        anchor,
                    } => {
                        let offset_x: i32 = match anchor.0 {
                            0 => 0,
                            1 => ((*new_w as i32) - (cur_w as i32)) / 2,
                            _ => (*new_w as i32) - (cur_w as i32),
                        };
                        let offset_y: i32 = match anchor.1 {
                            0 => 0,
                            1 => ((*new_h as i32) - (cur_h as i32)) / 2,
                            _ => (*new_h as i32) - (cur_h as i32),
                        };
                        let mut out = RgbaImage::new(*new_w, *new_h);
                        for y in 0..cur_h {
                            for x in 0..cur_w {
                                let nx = x as i32 + offset_x;
                                let ny = y as i32 + offset_y;
                                if nx >= 0
                                    && ny >= 0
                                    && (nx as u32) < *new_w
                                    && (ny as u32) < *new_h
                                {
                                    out.put_pixel(nx as u32, ny as u32, *img.get_pixel(x, y));
                                }
                            }
                        }
                        out
                    }
                };
                state.layers[i].pixels = crate::canvas::TiledImage::from_rgba_image(&new_img);
            }
        }

        // Advance tracked dimensions after each op
        match op {
            CanvasOpRequest::Rotate90CW | CanvasOpRequest::Rotate90CCW => {
                std::mem::swap(&mut cur_w, &mut cur_h)
            }
            CanvasOpRequest::ResizeImage { w, h, .. } => {
                cur_w = *w;
                cur_h = *h;
            }
            CanvasOpRequest::ResizeCanvas { new_w, new_h, .. } => {
                cur_w = *new_w;
                cur_h = *new_h;
            }
            _ => {}
        }
    }

    // Write final dimensions back to state
    state.width = cur_w;
    state.height = cur_h;
}

// ============================================================================
// Synchronous script executor (CLI / headless mode)
// ============================================================================

/// Execute a script synchronously on the **calling thread** (no rayon spawn).
///
/// Returns `(result_pixels, final_w, final_h, console_output, canvas_ops)`.
/// Used by the CLI batch processor — no GUI, no channel polling required.
pub fn execute_script_sync(
    source: &str,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    mask: Option<Vec<u8>>,
) -> Result<(Vec<u8>, u32, u32, Vec<String>, Vec<CanvasOpRequest>), ScriptError> {
    let cancel_flag = Arc::new(AtomicBool::new(false));

    // Dummy channel — sync mode ignores progress/preview messages
    let (dummy_tx, _dummy_rx) = std::sync::mpsc::channel::<ScriptMessage>();

    let rng_seed = {
        use std::time::SystemTime;
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        t.as_nanos() as u64 ^ 0x517cc1b727220a95
    };

    let ctx = Arc::new(Mutex::new(ScriptContext {
        project_index: 0,
        pixels,
        width,
        height,
        mask,
        console_output: Vec::new(),
        cancelled: cancel_flag,
        sender: dummy_tx,
        rng_state: rng_seed,
        canvas_ops: Vec::new(),
    }));

    let engine = create_engine(ctx.clone());
    let mut scope = Scope::new();

    let ast = engine.compile(source).map_err(|e| {
        let pos = e.position();
        ScriptError {
            message: e.to_string(),
            line: if pos.line().unwrap_or(0) > 0 {
                Some(pos.line().unwrap_or(0))
            } else {
                None
            },
            column: if pos.position().unwrap_or(0) > 0 {
                Some(pos.position().unwrap_or(0))
            } else {
                None
            },
        }
    })?;

    engine.run_ast_with_scope(&mut scope, &ast).map_err(|e| {
        let pos = e.position();
        ScriptError {
            message: e.to_string(),
            line: if pos.line().unwrap_or(0) > 0 {
                Some(pos.line().unwrap_or(0))
            } else {
                None
            },
            column: if pos.position().unwrap_or(0) > 0 {
                Some(pos.position().unwrap_or(0))
            } else {
                None
            },
        }
    })?;

    let lock = ctx.lock().unwrap_or_else(|e| e.into_inner());
    Ok((
        lock.pixels.clone(),
        lock.width,
        lock.height,
        lock.console_output.clone(),
        lock.canvas_ops.clone(),
    ))
}
