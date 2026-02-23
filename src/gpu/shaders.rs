// ============================================================================
// GPU SHADERS — all WGSL code kept inline for containment
// ============================================================================

// ============================================================================
// DISPLAY SHADER — Renders composite directly to viewport with hardware pan/zoom
// ============================================================================
//
// This shader handles all pan/zoom/aspect transformations in the vertex shader.
// The CPU only needs to send: offset (vec2), scale (f32), viewport_size (vec2),
// and image_size (vec2).  Zero per-frame pixel math on the CPU.
//
// The composite texture is sampled and displayed directly — no readback needed.
pub const DISPLAY_SHADER: &str = r#"
struct DisplayUniforms {
    offset: vec2<f32>,       // Pan offset in screen pixels
    scale: f32,              // Zoom factor
    _pad0: f32,
    viewport_size: vec2<f32>, // Viewport dimensions in pixels
    image_size: vec2<f32>,    // Source image dimensions in pixels
};

@group(0) @binding(0) var<uniform> u: DisplayUniforms;
@group(1) @binding(0) var display_tex: texture_2d<f32>;
@group(1) @binding(1) var display_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_display(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Unit quad (0..1)
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    let unit_pos = positions[vi];

    // Convert to world space: scale the image
    let scaled_size = u.image_size * u.scale;

    // Center in viewport, then apply pan offset
    let center_offset = (u.viewport_size - scaled_size) * 0.5;
    let world_pos = unit_pos * scaled_size + center_offset + u.offset;

    // Convert to NDC (-1..1), with Y flipped for wgpu conventions
    let ndc = vec2<f32>(
        (world_pos.x / u.viewport_size.x) * 2.0 - 1.0,
        1.0 - (world_pos.y / u.viewport_size.y) * 2.0
    );

    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = unit_pos;
    return out;
}

@fragment
fn fs_display(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(display_tex, display_samp, in.uv);
}
"#;

/// Legacy vertex + fragment shader for simple alpha compositing.
/// Kept for the checkerboard pipeline and any simple draws.
pub const COMPOSITE_SHADER: &str = r#"
struct ViewUniforms {
    view_proj: mat4x4<f32>,
    opacity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> view: ViewUniforms;
@group(1) @binding(0) var layer_texture: texture_2d<f32>;
@group(1) @binding(1) var layer_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );

    let pos = positions[vi];
    var out: VertexOutput;
    out.position = view.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(pos.x, pos.y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(layer_texture, layer_sampler, in.uv);
    color.a *= view.opacity;
    color = vec4<f32>(color.rgb * color.a, color.a);
    return color;
}
"#;

// ============================================================================
// UBER-COMPOSITOR SHADER
// ============================================================================
//
// Custom fragment shader that implements ALL blend modes.  Instead of relying
// on fixed-function hardware blending (which can only do Normal), the shader
// samples BOTH the background accumulator and the foreground layer, applies
// the requested blend mode per-pixel, and outputs the composited result.
//
// Used with ping-pong rendering: two textures alternate as read/write targets
// so each layer is composited on top of the previous result.
//
// Blend mode IDs match BlendMode::to_u8() in canvas.rs:
//   0 = Normal,  1 = Multiply,  2 = Screen,  3 = Additive,
//   4 = Reflect, 5 = Glow,      6 = ColorBurn, 7 = ColorDodge,
//   8 = Overlay, 9 = Difference, 10 = Negation, 11 = Lighten,
//  12 = Darken,  13 = Xor,      14 = Overwrite,
//  15 = HardLight, 16 = SoftLight, 17 = Exclusion,
//  18 = Subtract, 19 = Divide, 20 = LinearBurn,
//  21 = VividLight, 22 = LinearLight, 23 = PinLight, 24 = HardMix
// ============================================================================

pub const UBER_COMPOSITE_SHADER: &str = r#"
struct BlendUniforms {
    view_proj: mat4x4<f32>,
    opacity:    f32,
    blend_mode: u32,
    _pad0:      f32,
    _pad1:      f32,
};

@group(0) @binding(0) var<uniform> u: BlendUniforms;

// Foreground layer
@group(1) @binding(0) var fg_tex: texture_2d<f32>;
@group(1) @binding(1) var fg_samp: sampler;

// Background accumulator
@group(2) @binding(0) var bg_tex: texture_2d<f32>;
@group(2) @binding(1) var bg_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blend(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    let pos = positions[vi];
    var out: VertexOutput;
    out.position = u.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos;
    return out;
}

// ---- Blend mode helper functions ----

fn overlay_ch(base: f32, top: f32) -> f32 {
    if (base < 0.5) {
        return 2.0 * base * top;
    } else {
        return 1.0 - 2.0 * (1.0 - base) * (1.0 - top);
    }
}

fn color_burn_ch(base: f32, top: f32) -> f32 {
    if (top == 0.0) { return 0.0; }
    return max(1.0 - (1.0 - base) / top, 0.0);
}

fn color_dodge_ch(base: f32, top: f32) -> f32 {
    if (top >= 1.0) { return 1.0; }
    return min(base / (1.0 - top), 1.0);
}

fn reflect_ch(base: f32, top: f32) -> f32 {
    if (top >= 1.0) { return 1.0; }
    return min(base * base / (1.0 - top), 1.0);
}

// Soft Light (W3C formula)
fn soft_light_ch(base: f32, top: f32) -> f32 {
    if (top <= 0.5) {
        return base - (1.0 - 2.0 * top) * base * (1.0 - base);
    } else {
        var d: f32;
        if (base <= 0.25) {
            d = ((16.0 * base - 12.0) * base + 4.0) * base;
        } else {
            d = sqrt(base);
        }
        return base + (2.0 * top - 1.0) * (d - base);
    }
}

fn divide_ch(base: f32, top: f32) -> f32 {
    if (top <= 0.0) { return 1.0; }
    return min(base / top, 1.0);
}

fn vivid_light_ch(base: f32, top: f32) -> f32 {
    if (top <= 0.5) {
        let t2 = 2.0 * top;
        if (t2 <= 0.0) { return 0.0; }
        return max(1.0 - (1.0 - base) / t2, 0.0);
    } else {
        let t2 = 2.0 * (top - 0.5);
        if (t2 >= 1.0) { return 1.0; }
        return min(base / (1.0 - t2), 1.0);
    }
}

fn pin_light_ch(base: f32, top: f32) -> f32 {
    if (top <= 0.5) {
        return min(base, 2.0 * top);
    } else {
        return max(base, 2.0 * (top - 0.5));
    }
}

@fragment
fn fs_blend(in: VertexOutput) -> @location(0) vec4<f32> {
    let fg_raw = textureSample(fg_tex, fg_samp, in.uv);
    let bg     = textureSample(bg_tex, bg_samp, in.uv);

    // Apply layer opacity to foreground alpha.
    let fg_a = fg_raw.a * u.opacity;

    // ---- Overwrite: replace entirely ----
    if (u.blend_mode == 14u) {
        // Output premultiplied alpha
        return vec4<f32>(fg_raw.rgb * fg_a, fg_a);
    }

    // ---- Xor ----
    if (u.blend_mode == 13u) {
        let xor_a = bg.a * (1.0 - fg_a) + fg_a * (1.0 - bg.a);
        if (xor_a <= 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
        // bg.rgb is already premultiplied from previous pass
        let xor_rgb = (bg.rgb * (1.0 - fg_a) + fg_raw.rgb * fg_a * (1.0 - bg.a)) / xor_a;
        // Output premultiplied alpha
        let xor_premul = clamp(xor_rgb * xor_a, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(xor_premul, xor_a);
    }

    // If foreground is fully transparent, pass through background.
    if (fg_a <= 0.0) { return bg; }

    // ---- Compute blended RGB (blend mode math on straight-alpha colors) ----
    var blended: vec3<f32>;

    switch (u.blend_mode) {
        // Normal
        case 0u {
            blended = fg_raw.rgb;
        }
        // Multiply
        case 1u {
            blended = fg_raw.rgb * bg.rgb;
        }
        // Screen
        case 2u {
            blended = vec3<f32>(1.0) - (vec3<f32>(1.0) - fg_raw.rgb) * (vec3<f32>(1.0) - bg.rgb);
        }
        // Additive
        case 3u {
            blended = min(bg.rgb + fg_raw.rgb, vec3<f32>(1.0));
        }
        // Reflect
        case 4u {
            blended = vec3<f32>(
                reflect_ch(bg.r, fg_raw.r),
                reflect_ch(bg.g, fg_raw.g),
                reflect_ch(bg.b, fg_raw.b),
            );
        }
        // Glow (reflect with swapped args)
        case 5u {
            blended = vec3<f32>(
                reflect_ch(fg_raw.r, bg.r),
                reflect_ch(fg_raw.g, bg.g),
                reflect_ch(fg_raw.b, bg.b),
            );
        }
        // ColorBurn
        case 6u {
            blended = vec3<f32>(
                color_burn_ch(bg.r, fg_raw.r),
                color_burn_ch(bg.g, fg_raw.g),
                color_burn_ch(bg.b, fg_raw.b),
            );
        }
        // ColorDodge
        case 7u {
            blended = vec3<f32>(
                color_dodge_ch(bg.r, fg_raw.r),
                color_dodge_ch(bg.g, fg_raw.g),
                color_dodge_ch(bg.b, fg_raw.b),
            );
        }
        // Overlay
        case 8u {
            blended = vec3<f32>(
                overlay_ch(bg.r, fg_raw.r),
                overlay_ch(bg.g, fg_raw.g),
                overlay_ch(bg.b, fg_raw.b),
            );
        }
        // Difference
        case 9u {
            blended = abs(bg.rgb - fg_raw.rgb);
        }
        // Negation
        case 10u {
            blended = vec3<f32>(1.0) - abs(vec3<f32>(1.0) - bg.rgb - fg_raw.rgb);
        }
        // Lighten
        case 11u {
            blended = max(bg.rgb, fg_raw.rgb);
        }
        // Darken
        case 12u {
            blended = min(bg.rgb, fg_raw.rgb);
        }
        // HardLight (overlay with swapped base/top)
        case 15u {
            blended = vec3<f32>(
                overlay_ch(fg_raw.r, bg.r),
                overlay_ch(fg_raw.g, bg.g),
                overlay_ch(fg_raw.b, bg.b),
            );
        }
        // SoftLight (W3C formula)
        case 16u {
            blended = vec3<f32>(
                soft_light_ch(bg.r, fg_raw.r),
                soft_light_ch(bg.g, fg_raw.g),
                soft_light_ch(bg.b, fg_raw.b),
            );
        }
        // Exclusion
        case 17u {
            blended = bg.rgb + fg_raw.rgb - 2.0 * bg.rgb * fg_raw.rgb;
        }
        // Subtract
        case 18u {
            blended = max(bg.rgb - fg_raw.rgb, vec3<f32>(0.0));
        }
        // Divide
        case 19u {
            blended = vec3<f32>(
                divide_ch(bg.r, fg_raw.r),
                divide_ch(bg.g, fg_raw.g),
                divide_ch(bg.b, fg_raw.b),
            );
        }
        // Linear Burn
        case 20u {
            blended = max(bg.rgb + fg_raw.rgb - vec3<f32>(1.0), vec3<f32>(0.0));
        }
        // Vivid Light
        case 21u {
            blended = vec3<f32>(
                vivid_light_ch(bg.r, fg_raw.r),
                vivid_light_ch(bg.g, fg_raw.g),
                vivid_light_ch(bg.b, fg_raw.b),
            );
        }
        // Linear Light
        case 22u {
            blended = clamp(bg.rgb + 2.0 * fg_raw.rgb - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0));
        }
        // Pin Light
        case 23u {
            blended = vec3<f32>(
                pin_light_ch(bg.r, fg_raw.r),
                pin_light_ch(bg.g, fg_raw.g),
                pin_light_ch(bg.b, fg_raw.b),
            );
        }
        // Hard Mix
        case 24u {
            blended = vec3<f32>(
                select(0.0, 1.0, bg.r + fg_raw.r >= 1.0),
                select(0.0, 1.0, bg.g + fg_raw.g >= 1.0),
                select(0.0, 1.0, bg.b + fg_raw.b >= 1.0),
            );
        }
        // Fallback: Normal
        default {
            blended = fg_raw.rgb;
        }
    }

    // ---- Alpha compositing (Porter-Duff source-over, premultiplied) ----
    // bg.rgb is already premultiplied from previous pass, so:
    // result_a = fg_a + bg_a * (1 - fg_a)
    // result_rgb = (blended * fg_a + bg_premul_rgb * (1 - fg_a)) / result_a
    let out_a = fg_a + bg.a * (1.0 - fg_a);
    if (out_a <= 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }

    let out_rgb = (blended * fg_a + bg.rgb * (1.0 - fg_a)) / out_a;
    
    // Output PREMULTIPLIED alpha: multiply RGB by alpha before output.
    // This prevents color desaturation when blending over light backgrounds.
    let premul_rgb = clamp(out_rgb * out_a, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(premul_rgb, out_a);
}
"#;

/// Checkerboard shader — renders the transparency checkerboard pattern without
/// uploading any texture data.  Pure math in the fragment shader.
pub const CHECKERBOARD_SHADER: &str = r#"
struct ViewUniforms {
    view_proj: mat4x4<f32>,
    opacity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> view: ViewUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );
    let pos = positions[vi];
    var out: VertexOutput;
    out.position = view.view_proj * vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos;
    return out;
}

@fragment
fn fs_checker(in: VertexOutput) -> @location(0) vec4<f32> {
    let checker_size = 8.0; // pixels per checker square
    let cx = floor(in.uv.x * 1024.0 / checker_size);
    let cy = floor(in.uv.y * 1024.0 / checker_size);
    let checker = ((cx + cy) % 2.0);
    let gray = select(0.8, 0.9, checker > 0.5);
    return vec4<f32>(gray, gray, gray, 1.0);
}
"#;

/// Gaussian blur compute shader — optimised with workgroup shared memory.
///
/// Dispatched twice: once for horizontal, once for vertical (two-pass separable).
/// Each workgroup loads a tile + apron into shared memory, reducing global
/// texture reads from O(pixels × radius) to O(pixels + apron).
pub const GAUSSIAN_BLUR_SHADER: &str = r#"
struct BlurParams {
    radius: u32,
    direction: u32,  // 0 = horizontal, 1 = vertical
    width: u32,
    height: u32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: BlurParams;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

// Workgroup tile: 256 threads in a 256×1 (horizontal) or 1×256 (vertical)
// configuration.  Shared memory holds the tile + apron pixels.
// Max supported radius = 127 → shared array max = 256 + 2*127 = 510.
const TILE_W: u32 = 256u;
const MAX_SHARED: u32 = 512u;

var<workgroup> shared_tile: array<vec4<f32>, MAX_SHARED>;

@compute @workgroup_size(256, 1, 1)
fn cs_blur(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>) {
    let radius = i32(params.radius);
    let tile_start = i32(wid.x) * i32(TILE_W);
    let local_idx = i32(lid.x);

    if (params.direction == 0u) {
        // ---- HORIZONTAL PASS ----
        let y = gid.y;
        if (y >= params.height) { return; }

        // Load tile + left/right apron into shared memory.
        let apron_size = i32(TILE_W) + 2 * radius;
        var i = local_idx;
        while (i < apron_size) {
            let gx = clamp(tile_start + i - radius, 0, i32(params.width) - 1);
            shared_tile[i] = textureLoad(input_tex, vec2<u32>(u32(gx), y), 0);
            i = i + i32(TILE_W);
        }
        workgroupBarrier();

        // Compute blurred value for this pixel.
        let x = tile_start + local_idx;
        if (x >= i32(params.width)) { return; }

        var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        for (var k: i32 = -radius; k <= radius; k = k + 1) {
            let shared_idx = local_idx + radius + k;
            let weight = kernel[u32(k + radius)];
            color = color + shared_tile[shared_idx] * weight;
        }
        textureStore(output_tex, vec2<u32>(u32(x), y), color);
    } else {
        // ---- VERTICAL PASS ----
        let x = gid.y;  // gid.y carries the column index
        if (x >= params.width) { return; }

        let apron_size = i32(TILE_W) + 2 * radius;
        var i = local_idx;
        while (i < apron_size) {
            let gy = clamp(tile_start + i - radius, 0, i32(params.height) - 1);
            shared_tile[i] = textureLoad(input_tex, vec2<u32>(x, u32(gy)), 0);
            i = i + i32(TILE_W);
        }
        workgroupBarrier();

        let y = tile_start + local_idx;
        if (y >= i32(params.height)) { return; }

        var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        for (var k: i32 = -radius; k <= radius; k = k + 1) {
            let shared_idx = local_idx + radius + k;
            let weight = kernel[u32(k + radius)];
            color = color + shared_tile[shared_idx] * weight;
        }
        textureStore(output_tex, vec2<u32>(x, u32(y)), color);
    }
}
"#;

/// Mipmap generation compute shader — generates a single mip level by
/// averaging 2×2 blocks from the source level.
pub const MIPMAP_SHADER: &str = r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn cs_mipmap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(dst_tex);
    if (gid.x >= dst_size.x || gid.y >= dst_size.y) {
        return;
    }

    let src_x = gid.x * 2u;
    let src_y = gid.y * 2u;

    let p00 = textureLoad(src_tex, vec2<u32>(src_x, src_y), 0);
    let p10 = textureLoad(src_tex, vec2<u32>(src_x + 1u, src_y), 0);
    let p01 = textureLoad(src_tex, vec2<u32>(src_x, src_y + 1u), 0);
    let p11 = textureLoad(src_tex, vec2<u32>(src_x + 1u, src_y + 1u), 0);

    let avg = (p00 + p10 + p01 + p11) * 0.25;
    textureStore(dst_tex, vec2<u32>(gid.x, gid.y), avg);
}
"#;

// ============================================================================
// GPU COMPUTE FILTERS
// ============================================================================

/// Brightness/Contrast compute shader.
///
/// Same formula as the CPU version:
///   factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
///   out = factor * (pixel + brightness - 128) + 128
pub const BRIGHTNESS_CONTRAST_SHADER: &str = r#"
struct BcParams {
    width:      u32,
    height:     u32,
    brightness: f32,
    contrast:   f32,
};

@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: BcParams;

@compute @workgroup_size(16, 16)
fn cs_brightness_contrast(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let px = textureLoad(input_tex, vec2<u32>(gid.x, gid.y), 0);

    // Work in 0-255 space to match CPU
    let r = px.r * 255.0;
    let g = px.g * 255.0;
    let b = px.b * 255.0;

    let factor = (259.0 * (params.contrast + 255.0)) / (255.0 * (259.0 - params.contrast));

    let nr = clamp((factor * (r + params.brightness - 128.0) + 128.0) / 255.0, 0.0, 1.0);
    let ng = clamp((factor * (g + params.brightness - 128.0) + 128.0) / 255.0, 0.0, 1.0);
    let nb = clamp((factor * (b + params.brightness - 128.0) + 128.0) / 255.0, 0.0, 1.0);

    textureStore(output_tex, vec2<u32>(gid.x, gid.y), vec4<f32>(nr, ng, nb, px.a));
}
"#;

/// Hue/Saturation/Lightness compute shader.
///
/// Matches CPU logic: RGB→HSL→shift→HSL→RGB + lightness offset.
pub const HSL_ADJUST_SHADER: &str = r#"
struct HslParams {
    width:       u32,
    height:      u32,
    hue_shift:   f32,   // -180..180 → normalised to -0.5..0.5 in 0-1 space
    sat_factor:  f32,   // 1.0 + saturation/100
    light_offset: f32,  // lightness * 255 / 100 → in 0-1 space: /255
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: HslParams;

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> vec3<f32> {
    let cmax = max(max(r, g), b);
    let cmin = min(min(r, g), b);
    let delta = cmax - cmin;
    let l = (cmax + cmin) * 0.5;

    var h: f32 = 0.0;
    var s: f32 = 0.0;

    if (delta > 0.0) {
        if (l < 0.5) {
            s = delta / (cmax + cmin);
        } else {
            s = delta / (2.0 - cmax - cmin);
        }

        if (cmax == r) {
            h = (g - b) / delta;
            if (g < b) { h = h + 6.0; }
        } else if (cmax == g) {
            h = (b - r) / delta + 2.0;
        } else {
            h = (r - g) / delta + 4.0;
        }
        h = h / 6.0;
    }

    return vec3<f32>(h, s, l);
}

fn hue_to_rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    if (t < 0.0) { t = t + 1.0; }
    if (t > 1.0) { t = t - 1.0; }
    if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
    if (t < 0.5)        { return q; }
    if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    if (s <= 0.0) {
        return vec3<f32>(l, l, l);
    }
    var q: f32;
    if (l < 0.5) {
        q = l * (1.0 + s);
    } else {
        q = l + s - l * s;
    }
    let p = 2.0 * l - q;
    let r = hue_to_rgb(p, q, h + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h);
    let b = hue_to_rgb(p, q, h - 1.0 / 3.0);
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(16, 16)
fn cs_hsl_adjust(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let px = textureLoad(input_tex, vec2<u32>(gid.x, gid.y), 0);

    let hsl = rgb_to_hsl(px.r, px.g, px.b);
    var h = hsl.x + params.hue_shift;
    if (h < 0.0) { h = h + 1.0; }
    if (h > 1.0) { h = h - 1.0; }
    let s = clamp(hsl.y * params.sat_factor, 0.0, 1.0);

    let rgb = hsl_to_rgb(h, s, hsl.z);
    let light = params.light_offset;
    let nr = clamp(rgb.r + light, 0.0, 1.0);
    let ng = clamp(rgb.g + light, 0.0, 1.0);
    let nb = clamp(rgb.b + light, 0.0, 1.0);

    textureStore(output_tex, vec2<u32>(gid.x, gid.y), vec4<f32>(nr, ng, nb, px.a));
}
"#;

/// Invert colors compute shader.  Inverts R, G, B; preserves alpha.
pub const INVERT_SHADER: &str = r#"
struct InvParams {
    width:  u32,
    height: u32,
    _pad0:  u32,
    _pad1:  u32,
};

@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: InvParams;

@compute @workgroup_size(16, 16)
fn cs_invert(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let px = textureLoad(input_tex, vec2<u32>(gid.x, gid.y), 0);
    textureStore(output_tex, vec2<u32>(gid.x, gid.y),
        vec4<f32>(1.0 - px.r, 1.0 - px.g, 1.0 - px.b, px.a));
}
"#;

/// Median filter compute shader.
///
/// Each thread gathers pixels in a (2r+1)² window, sorts per-channel via
/// partial insertion sort, and outputs the median.  Radius is limited to 7
/// on GPU (window 15×15 = 225 samples) to keep register pressure sane.
/// For larger radii the CPU path is used.
pub const MEDIAN_SHADER: &str = r#"
struct MedianParams {
    width:  u32,
    height: u32,
    radius: u32,
    _pad0:  u32,
};

@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: MedianParams;

// Workgroup-local scratch for sorting (per-thread).
// Max window = (2*7+1)^2 = 225.  We sort each channel independently.
// WGSL does not have dynamic arrays in functions, so we use a fixed-size
// private array and loop up to the actual count.

const MAX_WINDOW: u32 = 225u;

// Simple insertion-sort a private array of floats up to `count` elements,
// then return the median.
fn sort_and_median(arr: ptr<function, array<f32, 225>>, count: u32) -> f32 {
    for (var i: u32 = 1u; i < count; i = i + 1u) {
        let key = (*arr)[i];
        var j: i32 = i32(i) - 1;
        while (j >= 0 && (*arr)[u32(j)] > key) {
            (*arr)[u32(j + 1)] = (*arr)[u32(j)];
            j = j - 1;
        }
        (*arr)[u32(j + 1)] = key;
    }
    return (*arr)[count / 2u];
}

@compute @workgroup_size(16, 16)
fn cs_median(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let r = i32(params.radius);
    let window = u32((2 * r + 1) * (2 * r + 1));

    var ch_r: array<f32, 225>;
    var ch_g: array<f32, 225>;
    var ch_b: array<f32, 225>;
    var ch_a: array<f32, 225>;

    var idx: u32 = 0u;
    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        let sy = clamp(i32(gid.y) + dy, 0, i32(params.height) - 1);
        for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
            let sx = clamp(i32(gid.x) + dx, 0, i32(params.width) - 1);
            let s = textureLoad(input_tex, vec2<u32>(u32(sx), u32(sy)), 0);
            ch_r[idx] = s.r;
            ch_g[idx] = s.g;
            ch_b[idx] = s.b;
            ch_a[idx] = s.a;
            idx = idx + 1u;
        }
    }

    let mr = sort_and_median(&ch_r, window);
    let mg = sort_and_median(&ch_g, window);
    let mb = sort_and_median(&ch_b, window);
    let ma = sort_and_median(&ch_a, window);

    textureStore(output_tex, vec2<u32>(gid.x, gid.y), vec4<f32>(mr, mg, mb, ma));
}
"#;
// ============================================================================
// GRADIENT GENERATOR — generates gradient pixels on the GPU (no input texture)
// ============================================================================
//
// Binding 0: output storage texture (write-only, rgba8unorm)
// Binding 1: uniform params (shape, repeat, is_eraser, start, end, dimensions)
// Binding 2: storage buffer (LUT: 256 × 4 u32-packed RGBA entries)
//
// Workgroup size: 16×16
pub const GRADIENT_SHADER: &str = r#"
struct GradientParams {
    start: vec2<f32>,     // gradient start point (canvas coords)
    end: vec2<f32>,       // gradient end point
    width: u32,           // canvas width
    height: u32,          // canvas height
    shape: u32,           // 0=Linear, 1=LinearReflected, 2=Radial, 3=Diamond
    repeat: u32,          // 0=clamp, 1=repeat
    is_eraser: u32,       // 0=color mode, 1=transparency/eraser mode
    _pad0: u32,
};

@group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: GradientParams;
@group(0) @binding(2) var<storage, read> lut: array<u32>; // 256 packed RGBA values

// Unpack a u32 into vec4<f32> in 0..1 range (little-endian RGBA)
fn unpack_rgba(packed: u32) -> vec4<f32> {
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

@compute @workgroup_size(16, 16)
fn cs_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let px = f32(gid.x) + 0.5;
    let py = f32(gid.y) + 0.5;
    let ax = params.start.x;
    let ay = params.start.y;

    let dx = params.end.x - ax;
    let dy = params.end.y - ay;
    let len_sq = dx * dx + dy * dy;
    let len = sqrt(len_sq);
    let inv_len = select(0.0, 1.0 / len, len > 1e-6);
    let inv_len_sq = select(0.0, 1.0 / len_sq, len_sq > 1e-6);

    let rx = px - ax;
    let ry = py - ay;

    var t: f32;

    switch (params.shape) {
        case 0u: { // Linear
            let raw = (rx * dx + ry * dy) * inv_len_sq;
            if (params.repeat != 0u) {
                t = raw - floor(raw); // rem_euclid
            } else {
                t = clamp(raw, 0.0, 1.0);
            }
        }
        case 1u: { // LinearReflected
            let raw = (rx * dx + ry * dy) * inv_len_sq;
            if (params.repeat != 0u) {
                let t_mod = raw - floor(raw / 2.0) * 2.0; // rem_euclid(2.0)
                t = select(t_mod, 2.0 - t_mod, t_mod > 1.0);
            } else {
                t = 1.0 - abs(2.0 * clamp(raw, 0.0, 1.0) - 1.0);
            }
        }
        case 2u: { // Radial
            let dist = sqrt(rx * rx + ry * ry) * inv_len;
            if (params.repeat != 0u) {
                t = dist - floor(dist);
            } else {
                t = clamp(dist, 0.0, 1.0);
            }
        }
        case 3u: { // Diamond
            let ux = dx * inv_len;
            let uy = dy * inv_len;
            let proj = abs(rx * ux + ry * uy) * inv_len;
            let perp = abs(rx * (-uy) + ry * ux) * inv_len;
            let dist = proj + perp;
            if (params.repeat != 0u) {
                t = dist - floor(dist);
            } else {
                t = clamp(dist, 0.0, 1.0);
            }
        }
        default: {
            t = 0.0;
        }
    }

    // LUT lookup
    let idx = u32(t * 255.0);
    var color = unpack_rgba(lut[idx]);

    // Transparency/eraser mode: compute luminance → mask alpha
    if (params.is_eraser != 0u) {
        let lum = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
        let mask_a = lum * color.a; // factor in stop alpha
        color = vec4<f32>(1.0, 1.0, 1.0, mask_a);
    }

    if (color.a > 0.0) {
        textureStore(output_tex, vec2<u32>(gid.x, gid.y), color);
    } else {
        textureStore(output_tex, vec2<u32>(gid.x, gid.y), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
"#;

// ============================================================================
// LIQUIFY WARP — GPU displacement warp for the Liquify tool
// ============================================================================

pub const LIQUIFY_WARP_SHADER: &str = r#"
struct LiquifyParams {
    width: u32,
    height: u32,
};

@group(0) @binding(0) var source_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> displacement: array<f32>;
@group(0) @binding(3) var<uniform> params: LiquifyParams;

// Safe texel load — returns transparent black for out-of-bounds coords.
fn load_safe(x: i32, y: i32, w: i32, h: i32) -> vec4<f32> {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return textureLoad(source_tex, vec2<i32>(x, y), 0);
}

@compute @workgroup_size(16, 16)
fn cs_liquify_warp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let w = i32(params.width);
    let h = i32(params.height);

    // Read displacement (dx, dy) for this pixel
    let idx = (y * params.width + x) * 2u;
    let dx = displacement[idx];
    let dy = displacement[idx + 1u];

    // Source coordinate = destination - displacement
    let sx = f32(x) - dx;
    let sy = f32(y) - dy;

    // Integer part for texel fetches
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));

    // Fast path: entirely out of bounds
    if (x0 < -1 || y0 < -1 || x0 >= w || y0 >= h) {
        textureStore(output_tex, vec2<u32>(x, y), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Fractional part for bilinear interpolation
    let fx = sx - floor(sx);
    let fy = sy - floor(sy);

    // Fetch the four neighbours
    let tl = load_safe(x0,     y0,     w, h);
    let tr = load_safe(x0 + 1, y0,     w, h);
    let bl = load_safe(x0,     y0 + 1, w, h);
    let br = load_safe(x0 + 1, y0 + 1, w, h);

    // Bilinear blend
    let top = mix(tl, tr, fx);
    let bot = mix(bl, br, fx);
    let color = mix(top, bot, fy);

    textureStore(output_tex, vec2<u32>(x, y), color);
}
"#;

// ============================================================================
// MESH WARP DISPLACEMENT SHADER — generates displacement field from Catmull-Rom
// spline surface evaluated on deformed grid control points (GPU compute).
// ============================================================================

pub const MESH_WARP_DISPLACEMENT_SHADER: &str = r#"
struct MeshWarpParams {
    width: u32,
    height: u32,
    grid_cols: u32,
    grid_rows: u32,
};

// Deformed control points — row-major (grid_rows+1) × (grid_cols+1), packed as vec2<f32>.
@group(0) @binding(0) var<storage, read> deformed_points: array<vec2<f32>>;
// Output displacement field — flat array of (dx, dy) pairs, length = width * height * 2.
@group(0) @binding(1) var<storage, read_write> displacement_out: array<f32>;
// Uniform parameters.
@group(0) @binding(2) var<uniform> params: MeshWarpParams;

/// Catmull-Rom basis weights for parameter t in [0,1].
fn cr_weights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    );
}

/// Evaluate bicubic Catmull-Rom surface at (u_global, v_global).
/// points are row-major with (grid_rows+1) rows of (grid_cols+1) points.
fn catmull_rom_surface(u_global: f32, v_global: f32) -> vec2<f32> {
    let cols = params.grid_cols;
    let rows = params.grid_rows;
    let pts_per_row = cols + 1u;
    let num_rows = rows + 1u;

    // Determine cell and local parameters
    let col_f = clamp(u_global, 0.0, f32(cols) - 0.0001);
    let row_f = clamp(v_global, 0.0, f32(rows) - 0.0001);
    let ci = min(u32(col_f), cols - 1u);
    let ri = min(u32(row_f), rows - 1u);
    let u_local = col_f - f32(ci);
    let v_local = row_f - f32(ri);

    let wu = cr_weights(u_local);
    let wv = cr_weights(v_local);

    // Clamped column indices for the 4 neighbours in u
    var cu0: u32;
    if (ci == 0u) { cu0 = 0u; } else { cu0 = ci - 1u; }
    let cu1 = ci;
    let cu2 = min(ci + 1u, pts_per_row - 1u);
    let cu3 = min(ci + 2u, pts_per_row - 1u);

    // Clamped row indices for the 4 neighbours in v
    var rv0: u32;
    if (ri == 0u) { rv0 = 0u; } else { rv0 = ri - 1u; }
    let rv1 = ri;
    let rv2 = min(ri + 1u, num_rows - 1u);
    let rv3 = min(ri + 2u, num_rows - 1u);

    let row_indices = array<u32, 4>(rv0, rv1, rv2, rv3);

    // Unrolled: for each of 4 v-rows, evaluate Catmull-Rom in u, then blend in v.
    // (WGSL forbids dynamic indexing of local arrays, so we unroll manually.)

    // j=0: row rv0
    var base = rv0 * pts_per_row;
    var p0 = deformed_points[base + cu0];
    var p1 = deformed_points[base + cu1];
    var p2 = deformed_points[base + cu2];
    var p3 = deformed_points[base + cu3];
    var u_val = wu[0] * p0 + wu[1] * p1 + wu[2] * p2 + wu[3] * p3;
    var result = wv[0] * u_val;

    // j=1: row rv1
    base = rv1 * pts_per_row;
    p0 = deformed_points[base + cu0];
    p1 = deformed_points[base + cu1];
    p2 = deformed_points[base + cu2];
    p3 = deformed_points[base + cu3];
    u_val = wu[0] * p0 + wu[1] * p1 + wu[2] * p2 + wu[3] * p3;
    result = result + wv[1] * u_val;

    // j=2: row rv2
    base = rv2 * pts_per_row;
    p0 = deformed_points[base + cu0];
    p1 = deformed_points[base + cu1];
    p2 = deformed_points[base + cu2];
    p3 = deformed_points[base + cu3];
    u_val = wu[0] * p0 + wu[1] * p1 + wu[2] * p2 + wu[3] * p3;
    result = result + wv[2] * u_val;

    // j=3: row rv3
    base = rv3 * pts_per_row;
    p0 = deformed_points[base + cu0];
    p1 = deformed_points[base + cu1];
    p2 = deformed_points[base + cu2];
    p3 = deformed_points[base + cu3];
    u_val = wu[0] * p0 + wu[1] * p1 + wu[2] * p2 + wu[3] * p3;
    result = result + wv[3] * u_val;

    return result;
}

@compute @workgroup_size(16, 16)
fn cs_mesh_warp_displacement(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    // Global parametric coords for this pixel
    let u_global = (f32(x) + 0.5) / f32(params.width) * f32(params.grid_cols);
    let v_global = (f32(y) + 0.5) / f32(params.height) * f32(params.grid_rows);

    let deformed_pos = catmull_rom_surface(u_global, v_global);

    // Displacement = deformed - identity (original uniform grid maps to pixel pos)
    let idx = (y * params.width + x) * 2u;
    displacement_out[idx]      = deformed_pos.x - (f32(x) + 0.5);
    displacement_out[idx + 1u] = deformed_pos.y - (f32(y) + 0.5);
}
"#;