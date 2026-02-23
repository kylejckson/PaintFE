// ============================================================================
// PaintFE CLI — headless batch processing via command-line arguments
// ============================================================================
//
// Usage examples:
//   paintfe --input photo.png --script blur.rhai --output result.png
//   paintfe -i photo.jpg -o out.png                    (format inferred from output ext)
//   paintfe -i *.jpg --script invert.rhai --output-dir processed/ --format png
//   paintfe -i project.pfe --output flat.jpg --quality 85
//   paintfe -i a.png b.png c.png --output-dir out/
//
// No GUI is opened in CLI mode. All processing runs synchronously on the
// current thread (no rayon, no wgpu) using CPU-only paths.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;

use crate::io::{load_image_sync, encode_and_write, save_pfe};
use crate::components::dialogs::{SaveFormat, TiffCompression};
use crate::ops::scripting::{execute_script_sync, apply_canvas_ops};

// ============================================================================
// CLI argument definition (clap Derive)
// ============================================================================

/// PaintFE headless image processor.
///
/// Process images with Rhai scripts and convert between formats — no GUI required.
#[derive(Parser, Debug)]
#[command(
    name = "paintfe",
    about = "PaintFE headless batch image processor",
    long_about = "Run Rhai scripts on image files and convert between formats without\n\
                  opening the GUI. Supports PNG, JPEG, WEBP, BMP, TGA, ICO, TIFF,\n\
                  GIF (static), and PFE project files.\n\n\
                  Example:\n  \
                  paintfe --input photo.png --script blur.rhai --output result.png\n  \
                  paintfe -i *.jpg --script adjust.rhai --output-dir out/ --format png"
)]
pub struct CliArgs {
    /// Input file(s). Glob patterns accepted (e.g. "*.png", "shots/*.jpg").
    /// PFE project files retain all layers; all other formats load as one layer.
    #[arg(short, long, required = true, num_args = 1..)]
    pub input: Vec<String>,

    /// Rhai script file to execute on each input image.
    /// If omitted, images are only loaded and re-saved (useful for format conversion).
    #[arg(short, long, value_name = "SCRIPT.rhai")]
    pub script: Option<PathBuf>,

    /// Output file path. Only valid for single-file input.
    /// For batch input use --output-dir instead.
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Output directory for batch processing.
    /// Files are written here with the original stem and the target format's extension.
    #[arg(long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Output format: png, jpeg, webp, bmp, tga, ico, tiff, gif, pfe.
    /// When omitted, the format is inferred from --output's extension, defaulting to png.
    #[arg(short, long, value_name = "FORMAT")]
    pub format: Option<String>,

    /// JPEG / WEBP quality (1–100, default 90).
    #[arg(short, long, default_value_t = 90, value_name = "1-100")]
    pub quality: u8,

    /// TIFF compression mode: none, lzw, deflate (default: none).
    #[arg(long, default_value = "none", value_name = "MODE")]
    pub tiff_compression: String,

    /// Flatten all visible layers before saving.
    /// Always true for raster formats; PFE output preserves layers regardless.
    #[arg(long, default_value_t = true)]
    pub flatten: bool,

    /// Print script console output and per-file timing information.
    #[arg(short, long)]
    pub verbose: bool,
}

impl CliArgs {
    /// Returns `true` when any CLI-mode flag is present in the real process arguments.
    /// Used by `main()` to route before creating an eframe window.
    pub fn is_cli_mode() -> bool {
        std::env::args().any(|a| a == "--input" || a == "-i")
    }
}

// ============================================================================
// Public entry point
// ============================================================================

/// Run all CLI processing and return an OS exit code.
/// `0` = all files succeeded, `1` = one or more files failed.
pub fn run(args: CliArgs) -> ExitCode {
    // Resolve glob patterns / literal paths → concrete PathBufs
    let inputs = resolve_inputs(&args.input);
    if inputs.is_empty() {
        eprintln!("error: no input files matched the given pattern(s).");
        return ExitCode::FAILURE;
    }

    // Multiple inputs require --output-dir, not --output
    if inputs.len() > 1 && args.output.is_some() && args.output_dir.is_none() {
        eprintln!(
            "error: {} input files given but --output only accepts a single file path.\n\
             Use --output-dir to specify a destination directory for batch processing.",
            inputs.len()
        );
        return ExitCode::FAILURE;
    }

    // Parse format and compression settings
    let save_format = parse_format(args.format.as_deref(), args.output.as_deref());
    let tiff_compression = match args.tiff_compression.to_lowercase().as_str() {
        "lzw"     => TiffCompression::Lzw,
        "deflate" => TiffCompression::Deflate,
        _         => TiffCompression::None,
    };

    // Load script source if provided
    let script_source: Option<String> = match &args.script {
        Some(path) => {
            match std::fs::read_to_string(path) {
                Ok(src) => Some(src),
                Err(e) => {
                    eprintln!(
                        "error: could not read script '{}': {}",
                        path.display(), e
                    );
                    return ExitCode::FAILURE;
                }
            }
        }
        None => None,
    };

    // Create output directory if specified
    if let Some(dir) = &args.output_dir {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!(
                "error: could not create output directory '{}': {}",
                dir.display(), e
            );
            return ExitCode::FAILURE;
        }
    }

    let total = inputs.len();
    let multi = total > 1;
    let mut any_failure = false;

    for (idx, input_path) in inputs.iter().enumerate() {
        if multi || args.verbose {
            println!("[{}/{}] {}", idx + 1, total, input_path.display());
        }

        let file_start = Instant::now();

        // Determine output path
        let output_path = match build_output_path(
            input_path,
            args.output.as_deref(),
            args.output_dir.as_deref(),
            save_format,
        ) {
            Some(p) => p,
            None => {
                eprintln!(
                    "  error: cannot determine output path for '{}'.",
                    input_path.display()
                );
                any_failure = true;
                continue;
            }
        };

        match run_one(
            input_path,
            &output_path,
            script_source.as_deref(),
            save_format,
            args.quality,
            tiff_compression,
            args.flatten,
            args.verbose,
        ) {
            Ok(()) => {
                if args.verbose || multi {
                    println!(
                        "  → {} ({:.0}ms)",
                        output_path.display(),
                        file_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
            Err(e) => {
                eprintln!("  error: {}", e);
                any_failure = true;
            }
        }
    }

    if any_failure { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}

// ============================================================================
// Per-file processing pipeline
// ============================================================================

fn run_one(
    input:            &Path,
    output:           &Path,
    script:           Option<&str>,
    format:           SaveFormat,
    quality:          u8,
    tiff_compression: TiffCompression,
    flatten:          bool,
    verbose:          bool,
) -> Result<(), String> {
    // -- Step 1: Load ----------------------------------------------------
    let mut state = load_image_sync(input)
        .map_err(|e| format!("load failed: {}", e))?;

    // -- Step 2: Apply script (optional) ---------------------------------
    if let Some(src) = script {
        let layer_idx = state.active_layer_index;
        let flat  = state.layers[layer_idx].pixels
            .extract_region_rgba(0, 0, state.width, state.height);
        let w     = state.width;
        let h     = state.height;
        let mask  = state.selection_mask.as_ref().map(|m| m.as_raw().clone());

        let (result_pixels, new_w, new_h, console_output, canvas_ops) =
            execute_script_sync(src, flat, w, h, mask)
                .map_err(|e| format!("script error: {}", e.friendly_message()))?;

        if verbose {
            for line in &console_output {
                println!("  [script] {}", line);
            }
        }

        // Apply result to the active layer
        let result_img = image::RgbaImage::from_raw(new_w, new_h, result_pixels)
            .ok_or_else(|| "script produced invalid pixel dimensions".to_string())?;
        state.layers[layer_idx].pixels =
            crate::canvas::TiledImage::from_rgba_image(&result_img);

        if !canvas_ops.is_empty() {
            // Canvas-wide ops (resize, rotate 90°, …) — replay on all other layers.
            // apply_canvas_ops also updates state.width / state.height.
            apply_canvas_ops(&mut state, layer_idx, &canvas_ops);
        } else {
            // Layer-only ops (flip, rotate 180°) keep same dimensions.
            state.width  = new_w;
            state.height = new_h;
        }
    }

    // -- Step 3: Save ----------------------------------------------------
    match format {
        SaveFormat::Pfe => {
            save_pfe(&state, output)
                .map_err(|e| format!("PFE save failed: {:?}", e))?;
        }
        _ => {
            let flat_img = if flatten && state.layers.len() > 1 {
                // Composite all visible layers
                state.composite()
            } else {
                // Use the active layer directly (single-layer or flatten disabled)
                let layer = &state.layers[state.active_layer_index];
                let raw = layer.pixels
                    .extract_region_rgba(0, 0, state.width, state.height);
                image::RgbaImage::from_raw(state.width, state.height, raw)
                    .unwrap_or_else(|| image::RgbaImage::new(state.width, state.height))
            };

            encode_and_write(&flat_img, output, format, quality, tiff_compression)
                .map_err(|e| format!("save failed: {}", e))?;
        }
    }

    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

/// Expand glob patterns and literal paths into a deduplicated, ordered list.
fn resolve_inputs(patterns: &[String]) -> Vec<PathBuf> {
    let mut result: Vec<PathBuf> = Vec::new();

    for pattern in patterns {
        let as_path = Path::new(pattern);

        if as_path.exists() {
            // Literal path — use directly
            if !result.iter().any(|p| p.as_path() == as_path) {
                result.push(as_path.to_path_buf());
            }
            continue;
        }

        // Treat as glob pattern
        match glob::glob(pattern) {
            Ok(entries) => {
                let mut matched = false;
                for entry in entries.flatten() {
                    if !result.contains(&entry) {
                        result.push(entry);
                    }
                    matched = true;
                }
                if !matched {
                    eprintln!("warning: pattern '{}' matched no files.", pattern);
                }
            }
            Err(e) => {
                eprintln!("warning: invalid glob '{}': {}", pattern, e);
            }
        }
    }

    result
}

/// Choose the [`SaveFormat`] from the `--format` string or infer it from the
/// output file extension. Defaults to PNG when neither is known.
fn parse_format(format_arg: Option<&str>, output: Option<&Path>) -> SaveFormat {
    if let Some(f) = format_arg {
        return match f.to_lowercase().as_str() {
            "jpeg" | "jpg" => SaveFormat::Jpeg,
            "webp"         => SaveFormat::Webp,
            "bmp"          => SaveFormat::Bmp,
            "tga"          => SaveFormat::Tga,
            "ico"          => SaveFormat::Ico,
            "tiff" | "tif" => SaveFormat::Tiff,
            "gif"          => SaveFormat::Gif,
            "pfe"          => SaveFormat::Pfe,
            _              => SaveFormat::Png,
        };
    }

    if let Some(out) = output {
        return match out
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
            .as_str()
        {
            "jpg" | "jpeg" => SaveFormat::Jpeg,
            "webp"         => SaveFormat::Webp,
            "bmp"          => SaveFormat::Bmp,
            "tga"          => SaveFormat::Tga,
            "ico"          => SaveFormat::Ico,
            "tiff" | "tif" => SaveFormat::Tiff,
            "gif"          => SaveFormat::Gif,
            "pfe"          => SaveFormat::Pfe,
            _              => SaveFormat::Png,
        };
    }

    SaveFormat::Png
}

/// Compute the output path for a single input file.
///
/// Priority:
/// 1. `--output` (explicit path, used for single-file input)
/// 2. `--output-dir` (batch directory, derives filename from input stem)
/// 3. Fallback: same directory as input, same stem, new extension
///    (appends `_out` to stem if it would collide with the input path)
fn build_output_path(
    input:      &Path,
    output:     Option<&Path>,
    output_dir: Option<&Path>,
    format:     SaveFormat,
) -> Option<PathBuf> {
    // Explicit output path
    if let Some(out) = output {
        return Some(out.to_path_buf());
    }

    let ext  = format.extension();
    let stem = input.file_stem()?.to_string_lossy().into_owned();

    if let Some(dir) = output_dir {
        return Some(dir.join(format!("{}.{}", stem, ext)));
    }

    // Write next to the input file
    let parent = input.parent().unwrap_or(Path::new("."));
    let candidate = parent.join(format!("{}.{}", stem, ext));

    // Avoid silent overwrite of the input
    if candidate == input {
        Some(parent.join(format!("{}_out.{}", stem, ext)))
    } else {
        Some(candidate)
    }
}
