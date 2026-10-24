//! Document-space alignment aids. Never part of a pixel composite.
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    Horizontal,
    Vertical,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Guide {
    pub id: u64,
    pub axis: Axis,
    pub position: f32,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Guides {
    pub items: Vec<Guide>,
    pub next_id: u64,
    pub visible: bool,
    pub rulers: bool,
    pub locked: bool,
    pub snap: bool,
}
impl Default for Guides {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            next_id: 1,
            visible: true,
            rulers: false,
            locked: false,
            snap: false,
        }
    }
}
impl Guides {
    pub fn add(&mut self, axis: Axis, position: f32, width: u32, height: u32) -> bool {
        let limit = match axis {
            Axis::Horizontal => height,
            Axis::Vertical => width,
        } as f32;
        if self.items.len() >= 1024
            || !position.is_finite()
            || !(0.0..=limit).contains(&position)
            || self.next_id == u64::MAX
        {
            return false;
        }
        self.items.push(Guide {
            id: self.next_id,
            axis,
            position,
        });
        self.next_id += 1;
        true
    }
    pub fn valid(&self, width: u32, height: u32) -> bool {
        let mut ids = std::collections::HashSet::new();
        self.items.len() <= 1024
            && self.items.iter().all(|g| {
                g.id > 0
                    && g.id < self.next_id
                    && ids.insert(g.id)
                    && g.position.is_finite()
                    && g.position >= 0.0
                    && g.position
                        <= match g.axis {
                            Axis::Horizontal => height,
                            Axis::Vertical => width,
                        } as f32
            })
    }
    /// A viewport-constant tolerance; independent of zoom and image dimensions.
    pub fn snap_delta(&self, bounds: egui::Rect, zoom: f32) -> egui::Vec2 {
        if !self.snap || !self.visible || zoom <= 0.0 {
            return egui::Vec2::ZERO;
        }
        let tolerance = 6.0 / zoom;
        let mut result = egui::Vec2::ZERO;
        for axis in [Axis::Horizontal, Axis::Vertical] {
            let values = match axis {
                Axis::Horizontal => [bounds.top(), bounds.center().y, bounds.bottom()],
                Axis::Vertical => [bounds.left(), bounds.center().x, bounds.right()],
            };
            let nearest = self
                .items
                .iter()
                .filter(|g| g.axis == axis)
                .flat_map(|g| values.map(|v| g.position - v))
                .filter(|d| d.abs() <= tolerance)
                .min_by(|a, b| a.abs().total_cmp(&b.abs()));
            if let Some(delta) = nearest {
                match axis {
                    Axis::Horizontal => result.y = delta,
                    Axis::Vertical => result.x = delta,
                }
            }
        }
        result
    }
    pub fn scale(&mut self, x: f32, y: f32) {
        for g in &mut self.items {
            g.position *= match g.axis {
                Axis::Horizontal => y,
                Axis::Vertical => x,
            };
        }
    }
    pub fn translate_clip(&mut self, x: f32, y: f32, width: u32, height: u32) {
        for g in &mut self.items {
            g.position += match g.axis {
                Axis::Horizontal => y,
                Axis::Vertical => x,
            };
        }
        self.items.retain(|g| {
            g.position >= 0.0
                && g.position
                    <= match g.axis {
                        Axis::Horizontal => height,
                        Axis::Vertical => width,
                    } as f32
        });
    }
    pub fn flip(&mut self, horizontal: bool, width: u32, height: u32) {
        for g in &mut self.items {
            if (g.axis == Axis::Vertical) == horizontal {
                g.position = if horizontal { width } else { height } as f32 - g.position;
            }
        }
    }
    pub fn rotate_quarter(&mut self, clockwise: bool, width: u32, height: u32) {
        for g in &mut self.items {
            let (axis, p) = match (clockwise, g.axis) {
                (true, Axis::Horizontal) => (Axis::Vertical, height as f32 - g.position),
                (true, Axis::Vertical) => (Axis::Horizontal, g.position),
                (false, Axis::Horizontal) => (Axis::Vertical, g.position),
                (false, Axis::Vertical) => (Axis::Horizontal, width as f32 - g.position),
            };
            g.axis = axis;
            g.position = p;
        }
    }
}

pub struct GuideCommand {
    pub before: Guides,
    pub after: Guides,
}
impl crate::components::history::Command for GuideCommand {
    fn undo(&self, state: &mut super::CanvasState) {
        state.guides = self.before.clone();
        state.mark_dirty(None);
    }
    fn redo(&self, state: &mut super::CanvasState) {
        state.guides = self.after.clone();
        state.mark_dirty(None);
    }
    fn description(&self) -> String {
        crate::t!("guides.edit")
    }
    fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + (self.before.items.len() + self.after.items.len()) * std::mem::size_of::<Guide>()
    }
    fn action(&self) -> crate::components::history::HistoryAction {
        crate::components::history::HistoryAction::Guides
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canvas_operations_and_snapshot_restore_keep_guides_aligned() {
        let mut s = crate::canvas::CanvasState::new(100, 80);
        s.guides.add(Axis::Vertical, 20.0, 100, 80);
        s.guides.add(Axis::Horizontal, 30.0, 100, 80);
        let before = crate::components::history::CanvasSnapshot::capture(&s);
        crate::ops::transform::flip_canvas_horizontal(&mut s);
        assert_eq!(s.guides.items[0].position, 80.0);
        crate::ops::transform::rotate_canvas_90cw(&mut s);
        assert_eq!((s.width, s.height), (80, 100));
        crate::ops::transform::resize_image(
            &mut s,
            40,
            50,
            crate::ops::transform::Interpolation::Nearest,
        );
        assert_eq!(
            (s.guides.items[0].axis, s.guides.items[0].position),
            (Axis::Horizontal, 40.0)
        );
        assert_eq!(s.guides.items[1].position, 25.0);
        s.selection_mask = Some(image::GrayImage::from_fn(40, 50, |x, y| {
            image::Luma([if (10..30).contains(&x) && (10..40).contains(&y) {
                255
            } else {
                0
            }])
        }));
        crate::ops::adjustments::crop_to_selection(&mut s);
        assert_eq!((s.width, s.height), (20, 30));
        assert_eq!(s.guides.items[0].position, 30.0);
        assert_eq!(s.guides.items[1].position, 15.0);
        before.restore_into(&mut s);
        assert_eq!((s.width, s.height), (100, 80));
        assert_eq!(s.guides.items[0].position, 20.0);
    }
    #[test]
    fn positions_ids_and_malformed_values_are_validated() {
        let mut g = Guides::default();
        assert!(!g.add(Axis::Vertical, f32::NAN, 100, 80));
        assert!(!g.add(Axis::Horizontal, 81.0, 100, 80));
        assert!(g.add(Axis::Vertical, 0.0, 100, 80));
        assert!(g.add(Axis::Horizontal, 80.0, 100, 80));
        assert!(g.valid(100, 80));
        g.items[1].id = g.items[0].id;
        assert!(!g.valid(100, 80));
    }
    #[test]
    fn transforms_and_screen_constant_snapping() {
        let mut g = Guides {
            snap: true,
            ..Default::default()
        };
        g.add(Axis::Vertical, 20.0, 100, 80);
        g.add(Axis::Horizontal, 30.0, 100, 80);
        let b = egui::Rect::from_min_size(egui::pos2(18.0, 28.0), egui::vec2(1.0, 1.0));
        assert_eq!(g.snap_delta(b, 1.0), egui::vec2(1.0, 1.0));
        assert_eq!(g.snap_delta(b, 10.0), egui::Vec2::ZERO);
        g.rotate_quarter(true, 100, 80);
        assert_eq!(
            (g.items[0].axis, g.items[0].position),
            (Axis::Horizontal, 20.0)
        );
        assert_eq!(g.items[1].position, 50.0);
        g.rotate_quarter(false, 80, 100);
        assert_eq!(g.items[0].position, 20.0);
        assert_eq!(g.items[1].position, 30.0);
        g.scale(2.0, 2.0);
        assert_eq!(g.items[0].position, 40.0);
        g.translate_clip(-50.0, 0.0, 100, 80);
        assert_eq!(g.items.len(), 1);
    }
}
