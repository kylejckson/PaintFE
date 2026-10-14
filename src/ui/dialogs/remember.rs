//! Parameter-only dialog history. Preview buffers and document references stay local.
use super::*;
use egui::{Context, Id};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DialogPreferences {
    pub enabled: BTreeSet<String>,
    pub values: BTreeMap<String, Value>,
}

#[derive(Clone, Default)]
struct Session {
    values: BTreeMap<String, Value>,
    defaults: Value,
    name: String,
    enabled: bool,
    supported: bool,
    reset: bool,
    changed: bool,
    applied: bool,
}

fn id() -> Id {
    Id::new("dialog_parameter_session")
}

pub(crate) fn is_parameter(name: &str) -> bool {
    !matches!(
        name,
        "first_open"
            | "previous_wrap_preview"
            | "wrap_preview_enabled"
            | "sample_x"
            | "sample_y"
            | "island_x"
            | "island_y"
            | "remove_seeds"
            | "protect_seeds"
    )
}

pub(crate) fn valid_parameter(name: &str, value: &Value) -> bool {
    match name {
        "active_channel" => value.as_u64().is_some_and(|n| n < 5),
        "selected_band" => value.as_u64().is_some_and(|n| n < 6),
        "active_zone" => value.as_u64().is_some_and(|n| n < 3),
        "channels" => value.as_array().is_some_and(|channels| {
            channels.len() == 5
                && channels.iter().all(|channel| {
                    channel
                        .get("points")
                        .and_then(Value::as_array)
                        .is_some_and(|points| {
                            (2..=256).contains(&points.len())
                                && points.iter().all(|point| {
                                    point.as_array().is_some_and(|pair| {
                                        pair.len() == 2
                                            && pair.iter().all(|v| {
                                                v.as_f64()
                                                    .is_some_and(|n| (0.0..=255.0).contains(&n))
                                            })
                                    })
                                })
                        })
                })
        }),
        _ => true,
    }
}

#[macro_export]
macro_rules! impl_dialog_parameters {
    ($name:ident { $($field:ident),* $(,)? }) => {
        impl $name {
            pub(crate) fn parameters(&self) -> serde_json::Value {
                let mut map = serde_json::Map::new();
                $(if $crate::ui::dialogs::remember::is_parameter(stringify!($field)) {
                    if let Ok(value) = serde_json::to_value(&self.$field) { map.insert(stringify!($field).into(), value); }
                })*
                serde_json::Value::Object(map)
            }
            pub(crate) fn restore_parameters(&mut self, value: &serde_json::Value) {
                $(if $crate::ui::dialogs::remember::is_parameter(stringify!($field)) {
                    if let Some(value) = value.get(stringify!($field)).filter(|value| $crate::ui::dialogs::remember::valid_parameter(stringify!($field), value)) {
                        if let Ok(value) = serde_json::from_value(value.clone()) { self.$field = value; }
                    }
                })*
            }
        }
    };
}

pub(crate) fn begin(
    ctx: &Context,
    dialog: &mut ActiveDialog,
    preferences: &DialogPreferences,
    opening: bool,
) {
    let mut state = ctx.data_mut(|d| d.get_temp::<Session>(id()).unwrap_or_default());
    if opening {
        state.name = dialog.name().into();
        state.defaults = dialog.parameters();
        state.supported = !state.defaults.is_null();
        state.enabled = preferences.enabled.contains(&state.name);
        state.reset = false;
        state.changed = false;
        let saved = state
            .values
            .get(&state.name)
            .or_else(|| {
                state
                    .enabled
                    .then(|| preferences.values.get(&state.name))
                    .flatten()
            })
            .cloned();
        if let Some(saved) = saved {
            dialog.restore_parameters(&saved);
            state.values.insert(state.name.clone(), dialog.parameters());
            state.changed = dialog.parameters() != state.defaults;
        }
    }
    if state.reset {
        dialog.restore_parameters(&state.defaults);
        state.reset = false;
        state.changed = true;
    }
    state.applied = false;
    ctx.data_mut(|d| d.insert_temp(id(), state));
}

pub(crate) fn mark_applied(ctx: &Context) {
    ctx.data_mut(|d| {
        let mut s = d.get_temp::<Session>(id()).unwrap_or_default();
        s.applied = true;
        d.insert_temp(id(), s);
    });
}

pub(crate) fn show_result<T>(
    ctx: &Context,
    result: super::DialogResult<T>,
) -> super::DialogResult<T> {
    let changed = ctx.data_mut(|d| {
        let mut s = d.get_temp::<Session>(id()).unwrap_or_default();
        let changed = s.changed;
        s.changed = false;
        d.insert_temp(id(), s);
        changed
    });
    if changed && matches!(result, super::DialogResult::Open) {
        super::DialogResult::Changed
    } else {
        result
    }
}

pub(crate) fn finish(
    ctx: &Context,
    dialog: &ActiveDialog,
    preferences: &mut DialogPreferences,
) -> bool {
    let mut state = ctx.data_mut(|d| d.get_temp::<Session>(id()).unwrap_or_default());
    if !state.supported {
        return false;
    }
    if !state.applied && preferences.enabled.contains(&state.name) == state.enabled {
        return false;
    }
    let old = preferences.clone();
    if state.applied {
        let values = dialog.parameters();
        if !values.is_null() {
            state.values.insert(state.name.clone(), values);
        }
    }
    if state.enabled {
        preferences.enabled.insert(state.name.clone());
        if let Some(values) = state.values.get(&state.name) {
            preferences
                .values
                .insert(state.name.clone(), values.clone());
        }
    } else {
        preferences.enabled.remove(&state.name);
        preferences.values.remove(&state.name);
    }
    ctx.data_mut(|d| d.insert_temp(id(), state));
    *preferences != old
}

pub(crate) fn header(ui: &mut egui::Ui, rect: egui::Rect, height: f32) {
    let mut state = ui
        .ctx()
        .data_mut(|d| d.get_temp::<Session>(id()).unwrap_or_default());
    if !state.supported {
        return;
    }
    let controls = egui::Rect::from_min_max(
        egui::pos2(rect.right() - height * 3.0, rect.top()),
        egui::pos2(rect.right() - height, rect.bottom()),
    );
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(controls)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
        |ui| {
            if ui
                .small_button("↺")
                .on_hover_text(crate::t!("ui.reset_tool_settings"))
                .clicked()
            {
                state.reset = true;
            }
            let checkbox = ui
                .checkbox(&mut state.enabled, "")
                .on_hover_text(crate::t!("ui.remember_settings_between_launches"));
            checkbox.widget_info(|| {
                egui::WidgetInfo::labeled(
                    egui::WidgetType::Checkbox,
                    ui.is_enabled(),
                    crate::t!("ui.remember_settings_between_launches"),
                )
            });
        },
    );
    ui.ctx().data_mut(|d| d.insert_temp(id(), state));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::dialogs::RemoveBackgroundDialog;

    #[test]
    fn old_saved_model_override_does_not_replace_automatic_detection() {
        let mut dialog = ActiveDialog::RemoveBackground(RemoveBackgroundDialog::new());
        dialog.restore_parameters(&serde_json::json!({"model_profile": "ISNet", "threshold": 0.4}));
        if let ActiveDialog::RemoveBackground(d) = dialog {
            assert_eq!(d.model_profile, crate::ops::ai::ModelProfile::Auto);
            assert_eq!(d.threshold, 0.4);
        } else {
            panic!("Expected background dialog");
        }
    }

    #[test]
    fn apply_cancel_reset_and_restart_keep_only_parameters() {
        let ctx = Context::default();
        let mut preferences = DialogPreferences::default();
        let mut dialog = ActiveDialog::RemoveBackground(RemoveBackgroundDialog::new());
        begin(&ctx, &mut dialog, &preferences, true);
        if let ActiveDialog::RemoveBackground(d) = &mut dialog {
            d.threshold = 0.3;
        }
        mark_applied(&ctx);
        finish(&ctx, &dialog, &mut preferences);
        let mut reopened = ActiveDialog::RemoveBackground(RemoveBackgroundDialog::new());
        begin(&ctx, &mut reopened, &preferences, true);
        assert_eq!(
            reopened.parameters()["threshold"],
            serde_json::json!(0.3f32)
        );
        if let ActiveDialog::RemoveBackground(d) = &mut reopened {
            d.threshold = 0.8;
        }
        finish(&ctx, &reopened, &mut preferences); // Cancel must not replace 0.3.
        dialog = ActiveDialog::RemoveBackground(RemoveBackgroundDialog::new());
        begin(&ctx, &mut dialog, &preferences, true);
        assert_eq!(dialog.parameters()["threshold"], serde_json::json!(0.3f32));
        ctx.data_mut(|d| {
            let mut s = d.get_temp::<Session>(id()).unwrap();
            s.enabled = true;
            d.insert_temp(id(), s);
        });
        finish(&ctx, &dialog, &mut preferences);
        let saved = serde_json::to_string(&preferences).unwrap();
        let loaded = serde_json::from_str(&saved).unwrap();
        let restarted = Context::default();
        let mut fresh = ActiveDialog::RemoveBackground(RemoveBackgroundDialog::new());
        begin(&restarted, &mut fresh, &loaded, true);
        assert_eq!(fresh.parameters()["threshold"], serde_json::json!(0.3f32));
        ctx.data_mut(|d| {
            let mut s = d.get_temp::<Session>(id()).unwrap();
            s.reset = true;
            d.insert_temp(id(), s);
        });
        begin(&ctx, &mut dialog, &preferences, false);
        assert_eq!(dialog.parameters()["threshold"], serde_json::json!(0.5f32));
    }
}

crate::impl_dialog_parameters!(BrightnessContrastDialog {
    brightness,
    contrast
});
crate::impl_dialog_parameters!(HueSaturationDialog {
    hue,
    saturation,
    lightness,
    per_band,
    selected_band,
    bands
});
crate::impl_dialog_parameters!(ExposureDialog { exposure });
crate::impl_dialog_parameters!(HighlightsShadowsDialog {
    shadows,
    highlights
});
crate::impl_dialog_parameters!(LevelsDialog {
    ch_input_black,
    ch_input_white,
    ch_gamma,
    ch_output_black,
    ch_output_white,
    active_channel
});
crate::impl_dialog_parameters!(CurvesDialog {
    channels,
    active_channel
});
crate::impl_dialog_parameters!(TemperatureTintDialog { temperature, tint });
crate::impl_dialog_parameters!(ThresholdDialog { level });
crate::impl_dialog_parameters!(PosterizeDialog { levels });
crate::impl_dialog_parameters!(ColorBalanceDialog {
    shadows,
    midtones,
    highlights,
    active_zone
});
crate::impl_dialog_parameters!(GradientMapDialog {
    shadow_color,
    highlight_color,
    preset
});
crate::impl_dialog_parameters!(BlackAndWhiteDialog {
    r_weight,
    g_weight,
    b_weight
});
crate::impl_dialog_parameters!(VibranceDialog { amount });
crate::impl_dialog_parameters!(ResizeImageDialog {
    lock_aspect,
    interpolation,
    resize_method,
    retarget
});
crate::impl_dialog_parameters!(ResizeCanvasDialog {
    lock_aspect,
    anchor,
    fill_transparent
});
crate::impl_dialog_parameters!(GaussianBlurDialog {
    sigma,
    advanced_blur
});
crate::impl_dialog_parameters!(LayerTransformDialog {
    rotation_z,
    rotation_x,
    rotation_y,
    scale_percent,
    offset_x,
    offset_y
});
crate::impl_dialog_parameters!(AlignLayerDialog {
    anchor_x,
    anchor_y,
    align_to_selection
});
crate::impl_dialog_parameters!(ColorRangeDialog {
    hue_center,
    hue_tolerance,
    sat_min,
    fuzziness,
    mode
});
crate::impl_dialog_parameters!(RemoveBackgroundDialog {
    threshold,
    edge_feather,
    mask_expansion,
    smooth_edges,
    fill_holes
});

impl ActiveDialog {
    pub(crate) fn parameters(&self) -> Value {
        match self {
            Self::ResizeImage(d) => d.parameters(),
            Self::ResizeCanvas(d) => d.parameters(),
            Self::AlignLayer(d) => d.parameters(),
            Self::GaussianBlur(d) => d.parameters(),
            Self::LayerTransform(d) => d.parameters(),
            Self::BrightnessContrast(d) => d.parameters(),
            Self::HueSaturation(d) => d.parameters(),
            Self::Exposure(d) => d.parameters(),
            Self::HighlightsShadows(d) => d.parameters(),
            Self::Levels(d) => d.parameters(),
            Self::Curves(d) => d.parameters(),
            Self::TemperatureTint(d) => d.parameters(),
            Self::BokehBlur(d) => d.parameters(),
            Self::MotionBlur(d) => d.parameters(),
            Self::BoxBlur(d) => d.parameters(),
            Self::ZoomBlur(d) => d.parameters(),
            Self::Crystallize(d) => d.parameters(),
            Self::Dents(d) => d.parameters(),
            Self::Pixelate(d) => d.parameters(),
            Self::Bulge(d) => d.parameters(),
            Self::Twist(d) => d.parameters(),
            Self::AddNoise(d) => d.parameters(),
            Self::ReduceNoise(d) => d.parameters(),
            Self::Median(d) => d.parameters(),
            Self::Glow(d) => d.parameters(),
            Self::Sharpen(d) => d.parameters(),
            Self::Vignette(d) => d.parameters(),
            Self::Halftone(d) => d.parameters(),
            Self::Grid(d) => d.parameters(),
            Self::DropShadow(d) => d.parameters(),
            Self::Outline(d) => d.parameters(),
            Self::PixelDrag(d) => d.parameters(),
            Self::RgbDisplace(d) => d.parameters(),
            Self::Ink(d) => d.parameters(),
            Self::OilPainting(d) => d.parameters(),
            Self::ColorFilter(d) => d.parameters(),
            Self::ColorToAlpha(d) => d.parameters(),
            Self::RecoverTransparency(d) => d.parameters(),
            Self::CanvasBorder(d) => d.parameters(),
            Self::SeamlessTexture(d) => d.parameters(),
            Self::Contours(d) => d.parameters(),
            Self::RemoveBackground(d) => d.parameters(),
            Self::Threshold(d) => d.parameters(),
            Self::Posterize(d) => d.parameters(),
            Self::ColorBalance(d) => d.parameters(),
            Self::GradientMap(d) => d.parameters(),
            Self::BlackAndWhite(d) => d.parameters(),
            Self::Vibrance(d) => d.parameters(),
            Self::ColorRange(d) => d.parameters(),
            _ => Value::Null,
        }
    }
    pub(crate) fn restore_parameters(&mut self, value: &Value) {
        match self {
            Self::ResizeImage(d) => d.restore_parameters(value),
            Self::ResizeCanvas(d) => d.restore_parameters(value),
            Self::AlignLayer(d) => d.restore_parameters(value),
            Self::GaussianBlur(d) => d.restore_parameters(value),
            Self::LayerTransform(d) => d.restore_parameters(value),
            Self::BrightnessContrast(d) => d.restore_parameters(value),
            Self::HueSaturation(d) => d.restore_parameters(value),
            Self::Exposure(d) => d.restore_parameters(value),
            Self::HighlightsShadows(d) => d.restore_parameters(value),
            Self::Levels(d) => {
                let mut value = value.clone();
                if value
                    .get("active_channel")
                    .and_then(Value::as_u64)
                    .is_some_and(|n| n >= 4)
                    && let Some(map) = value.as_object_mut()
                {
                    map.remove("active_channel");
                }
                d.restore_parameters(&value);
            }
            Self::Curves(d) => d.restore_parameters(value),
            Self::TemperatureTint(d) => d.restore_parameters(value),
            Self::BokehBlur(d) => d.restore_parameters(value),
            Self::MotionBlur(d) => d.restore_parameters(value),
            Self::BoxBlur(d) => d.restore_parameters(value),
            Self::ZoomBlur(d) => d.restore_parameters(value),
            Self::Crystallize(d) => d.restore_parameters(value),
            Self::Dents(d) => d.restore_parameters(value),
            Self::Pixelate(d) => d.restore_parameters(value),
            Self::Bulge(d) => d.restore_parameters(value),
            Self::Twist(d) => d.restore_parameters(value),
            Self::AddNoise(d) => d.restore_parameters(value),
            Self::ReduceNoise(d) => d.restore_parameters(value),
            Self::Median(d) => d.restore_parameters(value),
            Self::Glow(d) => d.restore_parameters(value),
            Self::Sharpen(d) => d.restore_parameters(value),
            Self::Vignette(d) => d.restore_parameters(value),
            Self::Halftone(d) => d.restore_parameters(value),
            Self::Grid(d) => d.restore_parameters(value),
            Self::DropShadow(d) => d.restore_parameters(value),
            Self::Outline(d) => d.restore_parameters(value),
            Self::PixelDrag(d) => d.restore_parameters(value),
            Self::RgbDisplace(d) => d.restore_parameters(value),
            Self::Ink(d) => d.restore_parameters(value),
            Self::OilPainting(d) => d.restore_parameters(value),
            Self::ColorFilter(d) => d.restore_parameters(value),
            Self::ColorToAlpha(d) => d.restore_parameters(value),
            Self::RecoverTransparency(d) => d.restore_parameters(value),
            Self::CanvasBorder(d) => d.restore_parameters(value),
            Self::SeamlessTexture(d) => d.restore_parameters(value),
            Self::Contours(d) => d.restore_parameters(value),
            Self::RemoveBackground(d) => d.restore_parameters(value),
            Self::Threshold(d) => d.restore_parameters(value),
            Self::Posterize(d) => d.restore_parameters(value),
            Self::ColorBalance(d) => d.restore_parameters(value),
            Self::GradientMap(d) => d.restore_parameters(value),
            Self::BlackAndWhite(d) => d.restore_parameters(value),
            Self::Vibrance(d) => d.restore_parameters(value),
            Self::ColorRange(d) => d.restore_parameters(value),
            _ => {}
        }
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn every_supported_dialog_roundtrips_only_parameters() {
        let state = crate::canvas::CanvasState::new(8, 8);
        let mut dialogs = vec![
            ActiveDialog::ResizeImage(ResizeImageDialog::new(&state)),
            ActiveDialog::ResizeCanvas(ResizeCanvasDialog::new(&state)),
            ActiveDialog::AlignLayer(AlignLayerDialog::new(&state)),
            ActiveDialog::GaussianBlur(GaussianBlurDialog::new(&state)),
            ActiveDialog::LayerTransform(LayerTransformDialog::new(&state)),
            ActiveDialog::BrightnessContrast(BrightnessContrastDialog::new(&state)),
            ActiveDialog::HueSaturation(HueSaturationDialog::new(&state)),
            ActiveDialog::Exposure(ExposureDialog::new(&state)),
            ActiveDialog::HighlightsShadows(HighlightsShadowsDialog::new(&state)),
            ActiveDialog::Levels(LevelsDialog::new(&state)),
            ActiveDialog::Curves(CurvesDialog::new(&state)),
            ActiveDialog::TemperatureTint(TemperatureTintDialog::new(&state)),
            ActiveDialog::BokehBlur(BokehBlurDialog::new(&state)),
            ActiveDialog::MotionBlur(MotionBlurDialog::new(&state)),
            ActiveDialog::BoxBlur(BoxBlurDialog::new(&state)),
            ActiveDialog::ZoomBlur(ZoomBlurDialog::new(&state)),
            ActiveDialog::Crystallize(CrystallizeDialog::new(&state)),
            ActiveDialog::Dents(DentsDialog::new(&state)),
            ActiveDialog::Pixelate(PixelateDialog::new(&state)),
            ActiveDialog::Bulge(BulgeDialog::new(&state)),
            ActiveDialog::Twist(TwistDialog::new(&state)),
            ActiveDialog::AddNoise(AddNoiseDialog::new(&state)),
            ActiveDialog::ReduceNoise(ReduceNoiseDialog::new(&state)),
            ActiveDialog::Median(MedianDialog::new(&state)),
            ActiveDialog::Glow(GlowDialog::new(&state)),
            ActiveDialog::Sharpen(SharpenDialog::new(&state)),
            ActiveDialog::Vignette(VignetteDialog::new(&state)),
            ActiveDialog::Halftone(HalftoneDialog::new(&state)),
            ActiveDialog::Grid(GridDialog::new(&state)),
            ActiveDialog::DropShadow(DropShadowDialog::new(&state)),
            ActiveDialog::Outline(OutlineDialog::new(&state)),
            ActiveDialog::PixelDrag(PixelDragDialog::new(&state)),
            ActiveDialog::RgbDisplace(RgbDisplaceDialog::new(&state)),
            ActiveDialog::Ink(InkDialog::new(&state)),
            ActiveDialog::OilPainting(OilPaintingDialog::new(&state)),
            ActiveDialog::ColorFilter(ColorFilterDialog::new(&state)),
            ActiveDialog::ColorToAlpha(ColorToAlphaDialog::new(&state)),
            ActiveDialog::RecoverTransparency(RecoverTransparencyDialog::new(&state)),
            ActiveDialog::CanvasBorder(CanvasBorderDialog::new(&state)),
            ActiveDialog::SeamlessTexture(SeamlessTextureDialog::new(&state)),
            ActiveDialog::Contours(ContoursDialog::new(&state)),
            ActiveDialog::RemoveBackground(RemoveBackgroundDialog::new()),
            ActiveDialog::Threshold(ThresholdDialog::new(&state)),
            ActiveDialog::Posterize(PosterizeDialog::new(&state)),
            ActiveDialog::ColorBalance(ColorBalanceDialog::new(&state)),
            ActiveDialog::GradientMap(GradientMapDialog::new(&state)),
            ActiveDialog::BlackAndWhite(BlackAndWhiteDialog::new(&state)),
            ActiveDialog::Vibrance(VibranceDialog::new(&state)),
            ActiveDialog::ColorRange(ColorRangeDialog::new(&state)),
        ];
        for dialog in &mut dialogs {
            let before = dialog.parameters();
            assert!(before.is_object(), "{}", dialog.name());
            for key in [
                "original_pixels",
                "original_flat",
                "layer_idx",
                "pending_flat",
                "remove_seeds",
                "protect_seeds",
            ] {
                assert!(before.get(key).is_none(), "{} leaked {key}", dialog.name());
            }
            dialog.restore_parameters(&before);
            assert_eq!(dialog.parameters(), before, "{}", dialog.name());
        }
        let mut levels = ActiveDialog::Levels(LevelsDialog::new(&state));
        levels.restore_parameters(&serde_json::json!({"active_channel":999}));
        assert_eq!(levels.parameters()["active_channel"], 0);
        let resize = ActiveDialog::ResizeImage(ResizeImageDialog::new(&state));
        assert!(resize.parameters().get("width").is_none());
        assert!(resize.parameters().get("height").is_none());
    }
}
