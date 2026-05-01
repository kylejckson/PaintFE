impl PaintFEApp {
    fn process_brush_tip_dialog(&mut self, ctx: &egui::Context, dialog: &mut ActiveDialog) -> bool {
        if !matches!(dialog, ActiveDialog::AddBrushTip(_) | ActiveDialog::AddShape(_)) {
            return false;
        }

        match dialog {
            ActiveDialog::AddBrushTip(dlg) => {
                match dlg.show(ctx) {
                    Some(result) => {
                        // Load the brush tip into assets
                        self.assets.load_brush_tip(
                            ctx,
                            &result.name,
                            &result.category,
                            &result.png_data,
                        );
                        // Persist to settings (base64-encoded)
                        use base64::Engine;
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&result.png_data);
                        self.settings.custom_brush_tips.push((
                            result.name.clone(),
                            result.category.clone(),
                            b64,
                        ));
                        self.settings.save();
                        // If the current brush tip was circle, switch to the new one
                        if self.tools_panel.properties.brush_tip.is_circle() {
                            self.tools_panel.properties.brush_tip =
                                crate::components::tools::BrushTip::Image(result.name.clone());
                        }
                        self.active_dialog = ActiveDialog::None;
                        true
                    }
                    None => {
                        // Dialog still open
                        false
                    }
                }
            }
            ActiveDialog::AddShape(dlg) => {
                match dlg.show(ctx) {
                    Some(result) => {
                        if self
                            .assets
                            .load_custom_shape(
                                ctx,
                                &result.name,
                                "Custom",
                                &result.svg_path_data,
                            )
                            .is_ok()
                        {
                            use base64::Engine;
                            let b64 = base64::engine::general_purpose::STANDARD
                                .encode(result.svg_path_data.as_bytes());
                            self.settings.custom_shapes.push((
                                result.name.clone(),
                                "Custom".to_string(),
                                b64,
                            ));
                            self.settings.save();
                            self.tools_panel.shapes_state.selected_custom_shape =
                                Some(result.name.clone());
                            self.tools_panel.shapes_state.selected_custom_shape_data = self
                                .assets
                                .get_custom_shape_data(&result.name)
                                .map(Into::into);
                        }
                        self.active_dialog = ActiveDialog::None;
                        true
                    }
                    None => false,
                }
            }
            _ => false,
        }
    }
}
