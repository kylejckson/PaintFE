//! Explicit contracts for supported segmentation exports. No model downloads.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelProfile {
    #[default]
    Auto,
    BiRefNet,
    U2Net,
    ISNet,
    ISNetRembg,
    ISNetClearCut,
    ISNetTransformers,
}

impl ModelProfile {
    pub const ALL: [Self; 7] = [
        Self::Auto,
        Self::BiRefNet,
        Self::U2Net,
        Self::ISNet,
        Self::ISNetRembg,
        Self::ISNetClearCut,
        Self::ISNetTransformers,
    ];

    pub fn description(self) -> &'static str {
        match self {
            Self::Auto => "Automatic",
            Self::BiRefNet => "BiRefNet",
            Self::U2Net => "U²-Net",
            Self::ISNet => "IS-Net (original)",
            Self::ISNetRembg => "IS-Net (rembg)",
            Self::ISNetClearCut => "IS-Net (ClearCut)",
            Self::ISNetTransformers => "IS-Net (Transformers.js)",
        }
    }

    pub(super) fn resolve(
        self,
        path: &str,
        input_name: &str,
        size: u32,
    ) -> Result<Self, OnnxError> {
        if self != Self::Auto {
            return Ok(self);
        }
        let name = path.to_ascii_lowercase();
        if name.contains("transformers") || name.contains("britishwerewolf") {
            return Ok(Self::ISNetTransformers);
        }
        if name.contains("clearcut") || name.contains("sacrednoir") {
            return Ok(Self::ISNetClearCut);
        }
        // A downloaded model may have the generic name model.onnx. Recognize
        // the accompanying processor config when the publisher supplies it.
        if let Some(parent) = Path::new(path).parent() {
            let config = parent.join("preprocessor_config.json");
            if std::fs::metadata(&config).is_ok_and(|m| m.len() <= 64 * 1024)
                && let Ok(bytes) = std::fs::read(config)
                && let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes)
                && config["image_processor_type"] == "U2NetImageProcessor"
                && config["pad_size"]["width"] == 1024
                && config["image_std"] == serde_json::json!([1.0, 1.0, 1.0])
            {
                return Ok(Self::ISNetTransformers);
            }
        }
        if name.contains("birefnet") {
            Ok(Self::BiRefNet)
        } else if name.contains("u2net") || size == 320 {
            Ok(Self::U2Net)
        } else if name.contains("isnet") || name.contains("is-net") {
            Ok(if input_name == "input_image" {
                Self::ISNetRembg
            } else {
                Self::ISNet
            })
        } else {
            Err(OnnxError::InferenceFailed(
                "Couldn't identify this model. Select its model profile in Remove Background."
                    .into(),
            ))
        }
    }

    pub(super) fn preferred_output_index(self, count: usize) -> usize {
        match self {
            Self::BiRefNet => count.saturating_sub(1),
            _ => 0,
        }
    }
}

pub(super) fn preprocess_image(
    input: &RgbaImage,
    width: u32,
    height: u32,
    profile: ModelProfile,
) -> (Vec<f32>, (u32, u32)) {
    let (cw, ch) = if profile == ModelProfile::ISNetTransformers {
        let scale =
            (width as f64 / input.width() as f64).min(height as f64 / input.height() as f64);
        (
            ((input.width() as f64 * scale).round() as u32).clamp(1, width),
            ((input.height() as f64 * scale).round() as u32).clamp(1, height),
        )
    } else {
        (width, height)
    };
    let resized = image::imageops::resize(input, cw, ch, image::imageops::FilterType::Lanczos3);
    let (mean, std) = match profile {
        ModelProfile::ISNet | ModelProfile::ISNetRembg | ModelProfile::ISNetClearCut => {
            ([0.5; 3], [1.0; 3])
        }
        ModelProfile::ISNetTransformers => (IMAGENET_MEAN, [1.0; 3]),
        _ => (IMAGENET_MEAN, IMAGENET_STD),
    };
    let divisor = if profile == ModelProfile::ISNetRembg || profile == ModelProfile::U2Net {
        resized
            .pixels()
            .flat_map(|p| p.0[..3].iter().copied())
            .max()
            .unwrap_or(255)
            .max(1) as f32
    } else if profile == ModelProfile::ISNetClearCut {
        256.0
    } else {
        255.0
    };
    let n = width as usize * height as usize;
    // Transformers pads the normalized image with zero on the bottom and right.
    let mut tensor = vec![0.0; n * 3];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let i = y as usize * width as usize + x as usize;
        for c in 0..3 {
            tensor[c * n + i] = (pixel[c] as f32 / divisor - mean[c]) / std[c];
        }
    }
    (tensor, (cw, ch))
}

#[cfg(not(target_arch = "wasm32"))]
type ElementTypeFn =
    unsafe extern "C" fn(*const OrtTensorTypeAndShapeInfo, *mut u32) -> *mut OrtStatus;

#[cfg(not(target_arch = "wasm32"))]
pub(super) unsafe fn session_element_type(
    api: &OrtApi,
    session: *mut OrtSession,
    input: bool,
) -> Result<u32, OnnxError> {
    let get_info: SessionGetInputTypeInfoFn = api.get_fn(if input { 33 } else { 34 });
    let info = ort_owned(api, api.release_type_info(), |out| {
        get_info(session, 0, out)
    })
    .map_err(OnnxError::InvalidOutput)?;
    let mut tensor_info = std::ptr::null();
    status_to_result(
        api,
        (api.cast_type_info_to_tensor_info())(info.ptr, &mut tensor_info),
    )
    .map_err(OnnxError::InvalidOutput)?;
    if tensor_info.is_null() {
        return Err(OnnxError::InvalidOutput("Expected a tensor".into()));
    }
    let mut element = 0;
    let get_type: ElementTypeFn = api.get_fn(60);
    status_to_result(api, get_type(tensor_info, &mut element)).map_err(OnnxError::InvalidOutput)?;
    if input {
        let mut rank = 0;
        status_to_result(api, (api.get_dimensions_count())(tensor_info, &mut rank))
            .map_err(OnnxError::InvalidOutput)?;
        if rank != 4 {
            return Err(OnnxError::InferenceFailed(
                "Expected a model input shaped [1, 3, height, width]".into(),
            ));
        }
        let mut dims = [0; 4];
        status_to_result(
            api,
            (api.get_dimensions())(tensor_info, dims.as_mut_ptr(), 4),
        )
        .map_err(OnnxError::InvalidOutput)?;
        if (dims[0] > 0 && dims[0] != 1) || (dims[1] > 0 && dims[1] != 3) {
            return Err(OnnxError::InferenceFailed(
                "Only single-image RGB NCHW models are supported".into(),
            ));
        }
    }
    Ok(element)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) unsafe fn input_element_type(
    api: &OrtApi,
    session: *mut OrtSession,
) -> Result<u32, OnnxError> {
    session_element_type(api, session, true)
}

fn mask_shape(dims: &[i64]) -> Result<(u32, u32, usize), OnnxError> {
    let spatial = match dims {
        [1, 1, h, w] | [1, h, w] | [h, w] if *h > 0 && *w > 0 => (*h, *w),
        _ => {
            return Err(OnnxError::InvalidOutput(format!(
                "Expected a single-channel mask, got {dims:?}"
            )));
        }
    };
    let h = u32::try_from(spatial.0)
        .map_err(|_| OnnxError::InvalidOutput("Mask height overflow".into()))?;
    let w = u32::try_from(spatial.1)
        .map_err(|_| OnnxError::InvalidOutput("Mask width overflow".into()))?;
    let count = (h as usize)
        .checked_mul(w as usize)
        .filter(|&n| n <= 64 * 1024 * 1024)
        .ok_or_else(|| OnnxError::InvalidOutput("Mask exceeds supported size".into()))?;
    Ok((h, w, count))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) unsafe fn read_mask(
    api: &OrtApi,
    tensor: *mut OrtValue,
) -> Result<(Vec<f32>, u32, u32, u32), OnnxError> {
    if tensor.is_null() {
        return Err(OnnxError::InvalidOutput("Missing mask tensor".into()));
    }
    let info = ort_owned(api, api.release_tensor_type_and_shape_info(), |out| {
        (api.get_tensor_type_and_shape())(tensor, out)
    })
    .map_err(OnnxError::InvalidOutput)?;
    let mut rank = 0;
    status_to_result(api, (api.get_dimensions_count())(info.ptr, &mut rank))
        .map_err(OnnxError::InvalidOutput)?;
    if !(2..=4).contains(&rank) {
        return Err(OnnxError::InvalidOutput("Unsupported mask rank".into()));
    }
    let mut dims = vec![0; rank];
    status_to_result(
        api,
        (api.get_dimensions())(info.ptr, dims.as_mut_ptr(), rank),
    )
    .map_err(OnnxError::InvalidOutput)?;
    let (h, w, count) = mask_shape(&dims)?;
    let get_count: unsafe extern "C" fn(
        *const OrtTensorTypeAndShapeInfo,
        *mut usize,
    ) -> *mut OrtStatus = api.get_fn(64);
    let mut actual_count = 0;
    status_to_result(api, get_count(info.ptr, &mut actual_count))
        .map_err(OnnxError::InvalidOutput)?;
    if count != actual_count {
        return Err(OnnxError::InvalidOutput(
            "Mask element count mismatch".into(),
        ));
    }
    let get_type: ElementTypeFn = api.get_fn(60);
    let mut element = 0;
    status_to_result(api, get_type(info.ptr, &mut element)).map_err(OnnxError::InvalidOutput)?;
    let mut data = std::ptr::null_mut();
    status_to_result(api, (api.get_tensor_mutable_data())(tensor, &mut data))
        .map_err(OnnxError::InvalidOutput)?;
    if data.is_null() {
        return Err(OnnxError::InvalidOutput("Empty mask data".into()));
    }
    let values = match element {
        1 => std::slice::from_raw_parts(data.cast::<f32>(), count).to_vec(),
        10 => std::slice::from_raw_parts(data.cast::<u16>(), count)
            .iter()
            .map(|&v| half::f16::from_bits(v).to_f32())
            .collect(),
        2 => std::slice::from_raw_parts(data.cast::<u8>(), count)
            .iter()
            .map(|&v| v as f32 / 255.0)
            .collect(),
        _ => {
            return Err(OnnxError::InvalidOutput(format!(
                "Unsupported mask element type {element}"
            )));
        }
    };
    Ok((values, h, w, element))
}

pub(super) fn normalize_mask(
    values: &mut [f32],
    profile: ModelProfile,
    element: u32,
) -> Result<(), OnnxError> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return Err(OnnxError::InvalidOutput(
            "Mask contains invalid values".into(),
        ));
    }
    if element == 2 {
        return Ok(());
    }
    if matches!(
        profile,
        ModelProfile::ISNet
            | ModelProfile::ISNetRembg
            | ModelProfile::ISNetClearCut
            | ModelProfile::ISNetTransformers
            | ModelProfile::U2Net
    ) {
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max - min > f32::EPSILON {
            for v in values {
                *v = (*v - min) / (max - min);
            }
        } else {
            // Avoid division by zero and artificial foreground on empty masks.
            for v in values {
                *v = v.clamp(0.0, 1.0);
            }
        }
    } else {
        let probability_space = is_probability_space(values);
        for v in values {
            *v = to_probability(*v, probability_space);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn soft_threshold_preserves_default_alpha_and_transparent_endpoints() {
        let image = RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]));
        let default = postprocess_mask(
            &[0.0, 0.25, 0.75, 1.0],
            2,
            2,
            &image,
            &RemoveBgSettings::default(),
        );
        assert_eq!(
            default.pixels().map(|p| p[3]).collect::<Vec<_>>(),
            [0, 12, 243, 255]
        );
        let settings = RemoveBgSettings {
            threshold: 0.05,
            ..Default::default()
        };
        let low = postprocess_mask(&[0.0, 0.25, 0.75, 1.0], 2, 2, &image, &settings);
        assert_eq!(low.get_pixel(0, 0)[3], 0);
        assert_eq!(low.get_pixel(1, 1)[3], 255);
    }
    #[test]
    fn mask_quantization_preserves_opaque_interiors_and_source_alpha() {
        for profile in [
            ModelProfile::BiRefNet,
            ModelProfile::U2Net,
            ModelProfile::ISNet,
        ] {
            let settings = RemoveBgSettings {
                model_profile: profile,
                ..Default::default()
            };
            let image = RgbaImage::from_pixel(3, 1, image::Rgba([10, 20, 30, 255]));
            let result = postprocess_mask(&[0.0, 0.999, 1.0], 1, 3, &image, &settings);
            assert_eq!(
                result.pixels().map(|p| p[3]).collect::<Vec<_>>(),
                [0, 255, 255]
            );
            for alpha in 0..=255 {
                let image = RgbaImage::from_pixel(1, 1, image::Rgba([10, 20, 30, alpha]));
                let result = postprocess_mask(&[1.0], 1, 1, &image, &settings);
                assert_eq!(result.get_pixel(0, 0).0, [10, 20, 30, alpha]);
            }
        }
    }

    #[test]
    fn edge_curve_keeps_threshold_at_half_opacity_across_slider_range() {
        let image = RgbaImage::from_pixel(3, 1, image::Rgba([10, 20, 30, 255]));
        for threshold in [0.05, 0.4, 0.5, 0.95] {
            for profile in [
                ModelProfile::BiRefNet,
                ModelProfile::U2Net,
                ModelProfile::ISNet,
            ] {
                let settings = RemoveBgSettings {
                    threshold,
                    model_profile: profile,
                    ..Default::default()
                };
                let result = postprocess_mask(&[0.0, threshold, 1.0], 1, 3, &image, &settings);
                assert_eq!(
                    result.pixels().map(|p| p[3]).collect::<Vec<_>>(),
                    [0, 128, 255]
                );
            }
        }
    }

    #[test]
    fn isnet_retains_soft_mask_while_other_models_keep_edge_contrast() {
        let image = RgbaImage::from_pixel(3, 1, image::Rgba([10, 20, 30, 255]));
        let settings = RemoveBgSettings {
            model_profile: ModelProfile::ISNet,
            ..Default::default()
        };
        let result = postprocess_mask(&[0.25, 0.5, 0.75], 1, 3, &image, &settings);
        assert_eq!(
            result.pixels().map(|p| p[3]).collect::<Vec<_>>(),
            [64, 128, 191]
        );
        for profile in [ModelProfile::BiRefNet, ModelProfile::U2Net] {
            let settings = RemoveBgSettings {
                model_profile: profile,
                ..settings.clone()
            };
            let result = postprocess_mask(&[0.25, 0.5, 0.75], 1, 3, &image, &settings);
            assert_eq!(
                result.pixels().map(|p| p[3]).collect::<Vec<_>>(),
                [12, 128, 243]
            );
        }
    }
    #[test]
    fn profiles_use_distinct_normalization_and_keep_rectangular_shape() {
        let image = RgbaImage::from_pixel(2, 1, image::Rgba([255, 0, 128, 255]));
        let (isnet, _) = preprocess_image(&image, 2, 1, ModelProfile::ISNet);
        assert_eq!(isnet.len(), 6);
        assert_eq!(isnet[0], 0.5);
        assert_eq!(isnet[2], -0.5);
        let (birefnet, _) = preprocess_image(&image, 2, 1, ModelProfile::BiRefNet);
        assert!((birefnet[0] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        let (padded, content) = preprocess_image(&image, 2, 2, ModelProfile::ISNetTransformers);
        assert_eq!(content, (2, 1));
        assert_eq!(padded[2], 0.0);
    }
    #[test]
    fn masks_validate_shape_and_normalize_without_nan() {
        assert!(mask_shape(&[1, 3, 4, 4]).is_err());
        assert!(mask_shape(&[1, 1, -1, 4]).is_err());
        assert_eq!(mask_shape(&[1, 4, 4]).unwrap(), (4, 4, 16));
        let mut values = [0.2, 0.4, 0.6];
        normalize_mask(&mut values, ModelProfile::ISNet, 1).unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[2], 1.0);
        let mut blank = [0.0; 4];
        normalize_mask(&mut blank, ModelProfile::ISNet, 1).unwrap();
        assert_eq!(blank, [0.0; 4]);
        assert!(normalize_mask(&mut [f32::NAN], ModelProfile::ISNet, 1).is_err());
    }
}
