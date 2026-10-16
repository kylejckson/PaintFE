impl PaintFEApp {
    fn paste_placement(
        image: egui::Vec2,
        canvas: egui::Rect,
        source: Option<egui::Pos2>,
        cursor: Option<egui::Pos2>,
        visible: Option<egui::Rect>,
        prefer_source: bool,
        keep_visible: bool,
    ) -> egui::Pos2 {
        // A full-document copy must retain its pixel alignment.
        if image == canvas.size() {
            return canvas.center();
        }
        let visible = visible
            .map(|r| r.intersect(canvas))
            .filter(|r| r.is_positive())
            .unwrap_or(canvas);
        let source = source.filter(|p| p.is_finite());
        let cursor = cursor.filter(|p| canvas.contains(*p));
        let desired = if prefer_source {
            source.or(cursor)
        } else {
            cursor
        }
        .unwrap_or(visible.center());
        if !keep_visible {
            return desired;
        }
        fn axis(value: f32, min: f32, max: f32, size: f32) -> f32 {
            if size >= max - min {
                (min + max) * 0.5
            } else {
                value.clamp(min + size * 0.5, max - size * 0.5)
            }
        }
        egui::pos2(
            axis(desired.x, visible.left(), visible.right(), image.x),
            axis(desired.y, visible.top(), visible.bottom(), image.y),
        )
    }
}
