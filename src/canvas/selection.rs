// ============================================================================
// SELECTION SYSTEM
// ============================================================================

/// How a new selection shape interacts with the existing mask.
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum SelectionMode {
    /// Clear any existing selection, then set the new shape.
    #[default]
    Replace,
    /// Union – add to the existing mask.
    Add,
    /// Difference – subtract from the existing mask.
    Subtract,
    /// Keep only pixels present in both the existing mask AND the new shape.
    Intersect,
}

impl SelectionMode {
    pub fn label(&self) -> String {
        match self {
            SelectionMode::Replace => t!("selection_mode.normal"),
            SelectionMode::Add => t!("selection_mode.add"),
            SelectionMode::Subtract => t!("selection_mode.subtract"),
            SelectionMode::Intersect => t!("selection_mode.intersect"),
        }
    }

    pub fn all() -> &'static [SelectionMode] {
        &[
            SelectionMode::Replace,
            SelectionMode::Add,
            SelectionMode::Subtract,
            SelectionMode::Intersect,
        ]
    }
}

fn selection_overlay_should_animate(bounds: Option<(u32, u32, u32, u32)>) -> bool {
    if let Some((min_x, min_y, max_x, max_y)) = bounds {
        let width = max_x.saturating_sub(min_x).saturating_add(1);
        let height = max_y.saturating_sub(min_y).saturating_add(1);
        width.saturating_mul(height) <= LARGE_SELECTION_STATIC_THRESHOLD
    } else {
        true
    }
}

/// Shape used during a selection drag.
#[derive(Clone, Debug)]
pub enum SelectionShape {
    Rectangle {
        min_x: u32,
        min_y: u32,
        max_x: u32,
        max_y: u32,
    },
    Ellipse {
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
    },
}

impl SelectionShape {
    /// Returns 255 if the pixel (x, y) is inside the shape, 0 otherwise.
    pub fn contains(&self, x: u32, y: u32) -> u8 {
        match self {
            SelectionShape::Rectangle {
                min_x,
                min_y,
                max_x,
                max_y,
            } => {
                if x >= *min_x && x <= *max_x && y >= *min_y && y <= *max_y {
                    255
                } else {
                    0
                }
            }
            SelectionShape::Ellipse { cx, cy, rx, ry } => {
                if *rx <= 0.0 || *ry <= 0.0 {
                    return 0;
                }
                let dx = (x as f32 - cx) / rx;
                let dy = (y as f32 - cy) / ry;
                if dx * dx + dy * dy <= 1.0 { 255 } else { 0 }
            }
        }
    }

    /// Bounding box in pixel coordinates (clamped to canvas).
    pub fn bounds(&self, canvas_w: u32, canvas_h: u32) -> (u32, u32, u32, u32) {
        match self {
            SelectionShape::Rectangle {
                min_x,
                min_y,
                max_x,
                max_y,
            } => (
                *min_x,
                *min_y,
                (*max_x).min(canvas_w.saturating_sub(1)),
                (*max_y).min(canvas_h.saturating_sub(1)),
            ),
            SelectionShape::Ellipse { cx, cy, rx, ry } => {
                let min_x = (cx - rx).max(0.0).floor() as u32;
                let min_y = (cy - ry).max(0.0).floor() as u32;
                let max_x = ((cx + rx).ceil() as u32).min(canvas_w.saturating_sub(1));
                let max_y = ((cy + ry).ceil() as u32).min(canvas_h.saturating_sub(1));
                (min_x, min_y, max_x, max_y)
            }
        }
    }
}

