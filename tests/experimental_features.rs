use image::{Rgba, RgbaImage};
use paintfe::canvas::{AdjustmentKind, Layer, LayerContent, PixelFormat};
use paintfe::components::dialogs::{SaveFormat, TiffCompression};
use paintfe::experimental::{DeepRgbaBuffer, reinhard_tone_map_rgba};
use paintfe::io::{encode_canvas_state_and_write, load_image_sync, load_pfe, save_pfe};
use paintfe::ops::canvas_ops::{
    ImageChannel, extract_channel_to_layer, replace_channel_from_layer,
};

#[test]
fn deep_pixel_formats_round_trip_to_preview_rgba8() {
    let mut img = RgbaImage::new(1, 1);
    img.put_pixel(0, 0, Rgba([17, 128, 251, 255]));

    for format in [
        PixelFormat::RgbaU8,
        PixelFormat::RgbaU16,
        PixelFormat::RgbaF16,
        PixelFormat::RgbaF32,
    ] {
        let deep = DeepRgbaBuffer::from_rgba8(&img, format);
        assert_eq!(deep.format(), format);
        let out = deep.to_rgba8(1, 1).unwrap();
        let px = out.get_pixel(0, 0);
        assert!((px[0] as i16 - 17).abs() <= 1);
        assert!((px[1] as i16 - 128).abs() <= 1);
        assert!((px[2] as i16 - 251).abs() <= 1);
        assert_eq!(px[3], 255);
    }
}

#[test]
fn hdr_reinhard_tone_map_compresses_overbright_values() {
    let px = reinhard_tone_map_rgba([4.0, 1.0, 0.0, 0.5], 1.0);
    assert!(px[0] > px[1]);
    assert_eq!(px[2], 0);
    assert_eq!(px[3], 128);
    assert!(px[0] < 255);
}

#[test]
fn adjustment_layer_changes_composite_without_editing_source_layer() {
    let mut state = paintfe::canvas::CanvasState::new(1, 1);
    state.layers[0]
        .pixels
        .put_pixel(0, 0, Rgba([10, 20, 30, 255]));
    state.layers.push(Layer::new_adjustment(
        "Invert".to_string(),
        1,
        1,
        AdjustmentKind::Invert,
    ));

    let out = state.composite();
    assert_eq!(out.get_pixel(0, 0).0, [245, 235, 225, 255]);
    assert_eq!(state.layers[0].pixels.get_pixel(0, 0).0, [10, 20, 30, 255]);
}

#[test]
fn adjustment_layer_opacity_blends_adjusted_result() {
    let mut state = paintfe::canvas::CanvasState::new(1, 1);
    state.layers[0]
        .pixels
        .put_pixel(0, 0, Rgba([10, 20, 30, 255]));
    let mut adj = Layer::new_adjustment("Invert".to_string(), 1, 1, AdjustmentKind::Invert);
    adj.opacity = 0.5;
    state.layers.push(adj);

    assert_eq!(state.composite().get_pixel(0, 0).0, [128, 128, 128, 255]);
}

#[test]
fn channel_extract_and_replace_work_between_layers() {
    let mut state = paintfe::canvas::CanvasState::new(1, 1);
    state.layers[0]
        .pixels
        .put_pixel(0, 0, Rgba([10, 20, 30, 255]));
    extract_channel_to_layer(&mut state, 0, ImageChannel::Blue);
    assert_eq!(state.layers[1].pixels.get_pixel(0, 0).0, [30, 30, 30, 255]);

    replace_channel_from_layer(&mut state, 0, 1, ImageChannel::Red, ImageChannel::Red);
    assert_eq!(state.layers[0].pixels.get_pixel(0, 0).0, [30, 20, 30, 255]);
}

#[test]
fn pfe3_round_trips_experimental_layer_data() {
    let mut state = paintfe::canvas::CanvasState::new(1, 1);
    state.layers[0].pixel_format = PixelFormat::RgbaF32;
    state.layers[0].hdr_metadata.enabled = true;
    state.layers[0]
        .source_metadata
        .png_text_chunks
        .push(("parameters".to_string(), "prompt data".to_string()));
    state.layers.push(Layer::new_adjustment(
        "Exposure".to_string(),
        1,
        1,
        AdjustmentKind::Exposure { ev: 1.0 },
    ));

    let path =
        std::env::temp_dir().join(format!("paintfe_experimental_{}.pfe", std::process::id()));
    save_pfe(&state, &path).unwrap();
    let loaded = load_pfe(&path).unwrap();
    let _ = std::fs::remove_file(path);

    assert_eq!(loaded.layers[0].pixel_format, PixelFormat::RgbaF32);
    assert!(loaded.layers[0].hdr_metadata.enabled);
    assert_eq!(
        loaded.layers[0].source_metadata.png_text_chunks[0],
        ("parameters".to_string(), "prompt data".to_string())
    );
    assert!(matches!(
        loaded.layers[1].content,
        LayerContent::Adjustment(_)
    ));
}

#[test]
fn imports_16bit_png_and_preserves_deep_values_through_pfe() {
    let png_path =
        std::env::temp_dir().join(format!("paintfe_16bit_source_{}.png", std::process::id()));
    let pfe_path = std::env::temp_dir().join(format!(
        "paintfe_16bit_roundtrip_{}.pfe",
        std::process::id()
    ));

    let values = [
        0x1234u16, 0x4567, 0x89ab, 0xcdef, 0xffff, 0x8000, 0x0001, 0x2222,
    ];
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    {
        let file = std::fs::File::create(&png_path).unwrap();
        let writer = std::io::BufWriter::new(file);
        let mut enc = png::Encoder::new(writer, 2, 1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Sixteen);
        let mut png_writer = enc.write_header().unwrap();
        png_writer.write_image_data(&bytes).unwrap();
    }

    let state = load_image_sync(&png_path).unwrap();
    assert_eq!(state.layers[0].pixel_format, PixelFormat::RgbaU16);
    assert_eq!(
        state.layers[0].deep_pixels,
        Some(DeepRgbaBuffer::U16(values.to_vec()))
    );

    save_pfe(&state, &pfe_path).unwrap();
    let loaded = load_pfe(&pfe_path).unwrap();
    let _ = std::fs::remove_file(&png_path);
    let _ = std::fs::remove_file(&pfe_path);

    assert_eq!(loaded.layers[0].pixel_format, PixelFormat::RgbaU16);
    assert_eq!(
        loaded.layers[0].deep_pixels,
        Some(DeepRgbaBuffer::U16(values.to_vec()))
    );
}

#[test]
fn exports_16bit_png_and_reopens_exact_deep_values() {
    let values = vec![
        0x1234u16, 0x4567, 0x89ab, 0xcdef, 0xffff, 0x8000, 0x0001, 0x2222,
    ];
    let mut state = paintfe::canvas::CanvasState::new(2, 1);
    let deep = DeepRgbaBuffer::U16(values.clone());
    state.layers[0].pixel_format = PixelFormat::RgbaU16;
    state.layers[0].deep_pixels = Some(deep.clone());
    state.layers[0].pixels =
        paintfe::canvas::TiledImage::from_rgba_image(&deep.to_rgba8(2, 1).unwrap());

    let path =
        std::env::temp_dir().join(format!("paintfe_export_16bit_{}.png", std::process::id()));
    encode_canvas_state_and_write(&state, &path, SaveFormat::Png, 95, TiffCompression::None)
        .unwrap();

    let loaded = load_image_sync(&path).unwrap();
    let _ = std::fs::remove_file(path);

    assert_eq!(loaded.layers[0].pixel_format, PixelFormat::RgbaU16);
    assert_eq!(
        loaded.layers[0].deep_pixels,
        Some(DeepRgbaBuffer::U16(values))
    );
}

#[test]
fn editing_16bit_layer_updates_only_dirty_deep_region() {
    let values = vec![
        0x1234u16, 0x4567, 0x89ab, 0xcdef, 0xffff, 0x8000, 0x0001, 0x2222,
    ];
    let mut state = paintfe::canvas::CanvasState::new(2, 1);
    let deep = DeepRgbaBuffer::U16(values.clone());
    state.layers[0].pixel_format = PixelFormat::RgbaU16;
    state.layers[0].deep_pixels = Some(deep.clone());
    state.layers[0].pixels =
        paintfe::canvas::TiledImage::from_rgba_image(&deep.to_rgba8(2, 1).unwrap());

    state.layers[0]
        .pixels
        .put_pixel(0, 0, Rgba([10, 20, 30, 40]));
    state.mark_dirty(Some(egui::Rect::from_min_max(
        egui::pos2(0.0, 0.0),
        egui::pos2(1.0, 1.0),
    )));

    let Some(DeepRgbaBuffer::U16(edited)) = &state.layers[0].deep_pixels else {
        panic!("expected edited u16 payload");
    };
    assert_eq!(&edited[0..4], &[2570, 5140, 7710, 10280]);
    assert_eq!(&edited[4..8], &values[4..8]);
}

#[test]
fn edited_16bit_layer_exports_updated_deep_pixels() {
    let values = vec![
        0x1234u16, 0x4567, 0x89ab, 0xcdef, 0xffff, 0x8000, 0x0001, 0x2222,
    ];
    let mut state = paintfe::canvas::CanvasState::new(2, 1);
    let deep = DeepRgbaBuffer::U16(values.clone());
    state.layers[0].pixel_format = PixelFormat::RgbaU16;
    state.layers[0].deep_pixels = Some(deep.clone());
    state.layers[0].pixels =
        paintfe::canvas::TiledImage::from_rgba_image(&deep.to_rgba8(2, 1).unwrap());
    state.layers[0]
        .pixels
        .put_pixel(0, 0, Rgba([10, 20, 30, 40]));
    state.mark_dirty(Some(egui::Rect::from_min_max(
        egui::pos2(0.0, 0.0),
        egui::pos2(1.0, 1.0),
    )));

    let path = std::env::temp_dir().join(format!(
        "paintfe_edited_export_16bit_{}.png",
        std::process::id()
    ));
    encode_canvas_state_and_write(&state, &path, SaveFormat::Png, 95, TiffCompression::None)
        .unwrap();
    let loaded = load_image_sync(&path).unwrap();
    let _ = std::fs::remove_file(path);

    let Some(DeepRgbaBuffer::U16(exported)) = &loaded.layers[0].deep_pixels else {
        panic!("expected exported u16 payload");
    };
    assert_eq!(&exported[0..4], &[2570, 5140, 7710, 10280]);
    assert_eq!(&exported[4..8], &values[4..8]);
}

#[test]
fn editing_fp32_hdr_layer_preserves_untouched_overbright_samples() {
    let values = vec![2.5_f32, 1.25, 0.5, 1.0, 4.0, 0.25, 0.125, 0.75];
    let mut state = paintfe::canvas::CanvasState::new(2, 1);
    let deep = DeepRgbaBuffer::F32(values.clone());
    state.layers[0].pixel_format = PixelFormat::RgbaF32;
    state.layers[0].deep_pixels = Some(deep.clone());
    state.layers[0].hdr_metadata.enabled = true;
    state.layers[0].pixels =
        paintfe::canvas::TiledImage::from_rgba_image(&deep.to_rgba8(2, 1).unwrap());

    state.layers[0]
        .pixels
        .put_pixel(0, 0, Rgba([64, 128, 255, 255]));
    state.mark_dirty(Some(egui::Rect::from_min_max(
        egui::pos2(0.0, 0.0),
        egui::pos2(1.0, 1.0),
    )));

    let Some(DeepRgbaBuffer::F32(edited)) = &state.layers[0].deep_pixels else {
        panic!("expected edited f32 payload");
    };
    assert_eq!(&edited[4..8], &values[4..8]);
    assert!((edited[0] - 64.0 / 255.0).abs() < 0.0001);
    assert!((edited[1] - 128.0 / 255.0).abs() < 0.0001);
    assert_eq!(edited[2], 1.0);
    assert_eq!(edited[3], 1.0);
}

#[test]
fn exports_16bit_tiff_and_reopens_exact_deep_values() {
    let values = vec![
        0x0102u16, 0x1112, 0x2122, 0x3132, 0x4142, 0x5152, 0x6162, 0x7172,
    ];
    let mut state = paintfe::canvas::CanvasState::new(2, 1);
    let deep = DeepRgbaBuffer::U16(values.clone());
    state.layers[0].pixel_format = PixelFormat::RgbaU16;
    state.layers[0].deep_pixels = Some(deep.clone());
    state.layers[0].pixels =
        paintfe::canvas::TiledImage::from_rgba_image(&deep.to_rgba8(2, 1).unwrap());

    let path =
        std::env::temp_dir().join(format!("paintfe_export_16bit_{}.tiff", std::process::id()));
    encode_canvas_state_and_write(&state, &path, SaveFormat::Tiff, 95, TiffCompression::None)
        .unwrap();

    let loaded = load_image_sync(&path).unwrap();
    let _ = std::fs::remove_file(path);

    assert_eq!(loaded.layers[0].pixel_format, PixelFormat::RgbaU16);
    assert_eq!(
        loaded.layers[0].deep_pixels,
        Some(DeepRgbaBuffer::U16(values))
    );
}

#[test]
fn exports_fp32_hdr_tiff_with_float_samples() {
    let values = vec![2.5_f32, 1.25, 0.5, 1.0, 4.0, 0.25, 0.125, 0.75];
    let mut state = paintfe::canvas::CanvasState::new(2, 1);
    let deep = DeepRgbaBuffer::F32(values.clone());
    state.layers[0].pixel_format = PixelFormat::RgbaF32;
    state.layers[0].deep_pixels = Some(deep.clone());
    state.layers[0].hdr_metadata.enabled = true;
    state.layers[0].pixels =
        paintfe::canvas::TiledImage::from_rgba_image(&deep.to_rgba8(2, 1).unwrap());

    let path =
        std::env::temp_dir().join(format!("paintfe_export_fp32_{}.tiff", std::process::id()));
    encode_canvas_state_and_write(&state, &path, SaveFormat::Tiff, 95, TiffCompression::None)
        .unwrap();

    let file = std::fs::File::open(&path).unwrap();
    let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
    let decoded = decoder.read_image().unwrap();
    let _ = std::fs::remove_file(path);

    let tiff::decoder::DecodingResult::F32(samples) = decoded else {
        panic!("expected RGBA32Float TIFF samples");
    };
    assert_eq!(samples, values);
}

#[test]
fn exports_fp32_hdr_adjustment_layer_in_float_space() {
    let values = vec![2.0_f32, 0.5, 0.25, 1.0];
    let mut state = paintfe::canvas::CanvasState::new(1, 1);
    let deep = DeepRgbaBuffer::F32(values.clone());
    state.layers[0].pixel_format = PixelFormat::RgbaF32;
    state.layers[0].deep_pixels = Some(deep.clone());
    state.layers[0].hdr_metadata.enabled = true;
    state.layers[0].pixels =
        paintfe::canvas::TiledImage::from_rgba_image(&deep.to_rgba8(1, 1).unwrap());
    state.layers.push(Layer::new_adjustment(
        "Exposure".to_string(),
        1,
        1,
        AdjustmentKind::Exposure { ev: 1.0 },
    ));

    let path = std::env::temp_dir().join(format!(
        "paintfe_export_fp32_adjusted_{}.tiff",
        std::process::id()
    ));
    encode_canvas_state_and_write(&state, &path, SaveFormat::Tiff, 95, TiffCompression::None)
        .unwrap();

    let file = std::fs::File::open(&path).unwrap();
    let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
    let decoded = decoder.read_image().unwrap();
    let _ = std::fs::remove_file(path);

    let tiff::decoder::DecodingResult::F32(samples) = decoded else {
        panic!("expected adjusted RGBA32Float TIFF samples");
    };
    assert_eq!(samples, vec![4.0_f32, 1.0, 0.5, 1.0]);
}

#[test]
fn pfe3_round_trips_fp32_hdr_payload() {
    let mut state = paintfe::canvas::CanvasState::new(1, 1);
    let values = vec![2.5_f32, 1.25, 0.5, 1.0];
    state.layers[0].pixel_format = PixelFormat::RgbaF32;
    state.layers[0].deep_pixels = Some(DeepRgbaBuffer::F32(values.clone()));
    state.layers[0].hdr_metadata.enabled = true;
    state.layers[0].hdr_metadata.max_luminance_nits = Some(250.0);

    let path = std::env::temp_dir().join(format!("paintfe_fp32_hdr_{}.pfe", std::process::id()));
    save_pfe(&state, &path).unwrap();
    let loaded = load_pfe(&path).unwrap();
    let _ = std::fs::remove_file(path);

    assert_eq!(loaded.layers[0].pixel_format, PixelFormat::RgbaF32);
    assert_eq!(
        loaded.layers[0].deep_pixels,
        Some(DeepRgbaBuffer::F32(values))
    );
    assert!(loaded.layers[0].hdr_metadata.enabled);
    assert_eq!(
        loaded.layers[0].hdr_metadata.max_luminance_nits,
        Some(250.0)
    );
}
