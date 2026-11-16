// Automation composes the same raster paths as interactive brush and pencil input.
impl ToolsPanel {
    pub fn render_automation_stroke(
        width: u32,
        height: u32,
        pixels: &[u8],
        mask: Option<image::GrayImage>,
        points: &[[f32; 3]],
        colour: [u8; 4],
        size: f32,
        pencil: bool,
        erase: bool,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Vec<u8>, String> {
        let mut state = CanvasState::new(width, height);
        state.layers[0].pixels = TiledImage::from_raw_rgba(width, height, pixels);
        state.selection_mask = mask;
        state.preview_layer = Some(TiledImage::new(width, height));
        let mut tools=Self{active_tool:if pencil{Tool::Pencil}else if erase{Tool::Eraser}else{Tool::Brush},..Default::default()};
        tools.properties.size = size;
        tools.properties.hardness = 1.;
        tools.properties.pressure_size = true;
        tools.properties.anti_aliased = !pencil;
        tools.reset_brush_pointer_state();
        let c = colour.map(|v| v as f32 / 255.);
        let mut previous = points[0];
        for next in points {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            tools.tool_state.current_pressure = next[2];
            if pencil {
                tools.draw_pixel_line_and_get_bounds(
                    &mut state,
                    (previous[0], previous[1]),
                    (next[0], next[1]),
                    false,
                    c,
                    c,
                );
            } else {
                let mask = state.selection_mask.clone();
                tools.draw_line_no_dirty(
                    state.preview_layer.as_mut().unwrap(),
                    width,
                    height,
                    (previous[0], previous[1]),
                    (next[0], next[1]),
                    erase,
                    false,
                    c,
                    c,
                    mask.as_ref(),
                );
            }
            previous = *next;
        }
        if erase {
            tools.commit_eraser_to_layer(&mut state);
        } else {
            tools.commit_bezier_to_layer(&mut state, c);
        }
        Ok(state.layers[0]
            .pixels
            .extract_region_rgba(0, 0, width, height))
    }
}
