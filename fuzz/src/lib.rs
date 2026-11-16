//! Exercise production parser sources without linking the GUI, GPU or codecs.
#[path = "../../src/io/heic/container.rs"]
mod container;
#[path = "../../src/ui/numeric/expression.rs"]
mod expression;

// The container reader calls its parent for the image allocation limit. Match
// the production limit file and dimension predicate; no codec is involved.
fn validate_dimensions(width: u32, height: u32) -> Result<(), String> {
    let limit: u64 = include_str!("../../src/io/heic/max_pixels.txt")
        .trim()
        .parse()
        .expect("Valid production limit");
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > limit {
        Err("Image dimensions exceed the processing limit".into())
    } else {
        Ok(())
    }
}

pub fn check_metadata(bytes: &[u8]) {
    if bytes.len() <= 65_536
        && let Ok(info) = container::validate(bytes)
    {
        std::hint::black_box((info.alpha, info.high_depth));
    }
}

pub fn check_expression(bytes: &[u8]) {
    if bytes.len() <= 4096
        && let Ok(text) = std::str::from_utf8(bytes)
        && let Some(value) = expression::evaluate_dimension_expression(text)
    {
        assert!(value.is_finite());
    }
}
