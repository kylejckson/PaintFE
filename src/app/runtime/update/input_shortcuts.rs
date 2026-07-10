impl PaintFEApp {
    fn update_runtime_input(&mut self, ctx: &egui::Context) -> bool {
        // --- Drag-and-Drop: open dropped image files as new projects ---
        {
            let shortcut_paste_present = ctx.input(|i| {
                i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Key {
                            key: egui::Key::V,
                            pressed: true,
                            modifiers,
                            ..
                        } if modifiers.command || modifiers.ctrl
                    )
                })
            });
            if !shortcut_paste_present {
                let dropped: Vec<egui::DroppedFile> = ctx.input(|i| i.raw.dropped_files.clone());
                for file in dropped {
                    if let Some(path) = file.path.clone() {
                        if path.is_file() {
                            self.open_file_by_path(path, ctx.input(|i| i.time));
                            continue;
                        }

                        // Some Linux/Wayland stacks can surface a dropped URI-like
                        // string in the path field (e.g. file:///...).
                        let parsed = Self::parse_file_uri_list(&path.to_string_lossy());
                        if !parsed.is_empty() {
                            for path in parsed {
                                self.open_file_by_path(path, ctx.input(|i| i.time));
                            }
                            continue;
                        }
                    }

                    if !file.name.is_empty() {
                        let parsed = Self::parse_file_uri_list(&file.name);
                        if !parsed.is_empty() {
                            for path in parsed {
                                self.open_file_by_path(path, ctx.input(|i| i.time));
                            }
                            continue;
                        }

                        let named_path = PathBuf::from(file.name.clone());
                        if named_path.is_file() {
                            self.open_file_by_path(named_path, ctx.input(|i| i.time));
                            continue;
                        }
                    }

                    if let Some(bytes) = file.bytes.as_ref() {
                        let name_hint = if file.name.is_empty() {
                            None
                        } else {
                            Some(file.name.clone())
                        };
                        self.open_image_from_bytes(bytes.as_ref(), name_hint);
                    }
                }
            }
        }

        // Some Linux/Wayland desktop flows surface file drags as text/uri-list
        // paste events instead of dropped file paths.
        self.handle_file_uri_paste_events(ctx);

        // Determine if a modal dialog is open — block all shortcuts and canvas interaction.
        let modal_open = self.save_file_dialog.open
            || self.new_file_dialog.open
            || !matches!(self.active_dialog, ActiveDialog::None)
            || self.pending_paste_request.is_some();

        let global_probe = ctx.input(|i| {
            let cmd = i.modifiers.ctrl || i.modifiers.command;
            let c_down = i.key_down(egui::Key::C);
            let x_down = i.key_down(egui::Key::X);
            let v_down = i.key_down(egui::Key::V);
            let enter_down = i.key_down(egui::Key::Enter);
            let escape_down = i.key_down(egui::Key::Escape);
            let events_len = i.events.len();
            let copy_evt = i.events.iter().any(|e| matches!(e, egui::Event::Copy));
            let cut_evt = i.events.iter().any(|e| matches!(e, egui::Event::Cut));
            let paste_evt = i.events.iter().any(|e| matches!(e, egui::Event::Paste(_)));
            (
                cmd,
                c_down,
                x_down,
                v_down,
                enter_down,
                escape_down,
                events_len,
                copy_evt,
                cut_evt,
                paste_evt,
                i.pointer.hover_pos().is_some(),
            )
        });
        let vk_probe = crate::windows_key_probe::snapshot();
        let linux_probe = crate::linux_key_probe::snapshot();

        // Pre-claim canvas focus when the text tool is actively editing text.
        // This must happen BEFORE any context-bar or panel widgets render (they render
        // before the CentralPanel canvas). Without this, DragValues in the text tool
        // options bar (letter spacing, scale, etc.) can enter text-edit mode and consume
        // Event::Text keystokes before handle_input ever sees them.
        if !modal_open
            && self.tools_panel.text_state.is_editing
            && !self.tools_panel.text_state.font_popup_open
            && let Some(canvas_id) = self.canvas.canvas_widget_id
        {
            ctx.memory_mut(|m| m.request_focus(canvas_id));
        }

        // Handle scroll wheel zoom — only when mouse is over the canvas and NOT over a widget.
        let mut should_zoom = false;
        let mut zoom_amount = 0.0;
        // On Wayland with a tablet, also check for active Touch events since
        // is_pointer_over_egui() relies on interact_pos() which isn't updated by Touch.
        let pointer_over_widget = ctx.is_pointer_over_egui()
            || ctx.input(|i| {
                i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Touch {
                            phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                            ..
                        }
                    )
                })
            });

        if !modal_open {
            ctx.input_mut(|i| {
                if i.smooth_scroll_delta.y.abs() > 0.1 {
                    let mouse_over_canvas = i.pointer.hover_pos().is_some_and(|pos| {
                        self.canvas
                            .last_canvas_rect
                            .is_some_and(|rect| rect.contains(pos))
                    });
                    if mouse_over_canvas && !pointer_over_widget {
                        should_zoom = true;
                        zoom_amount = i.smooth_scroll_delta.y;
                        i.smooth_scroll_delta.y = 0.0;
                    }
                }
            });
        }

        if should_zoom {
            let zoom_factor = 1.0 + zoom_amount * 0.005;
            // Zoom around the mouse cursor so the point under the pointer stays fixed.
            let mouse_pos = ctx.input(|i| i.pointer.hover_pos());
            if let (Some(pos), Some(rect)) = (mouse_pos, self.canvas.last_canvas_rect) {
                self.canvas.zoom_around_screen_point(zoom_factor, pos, rect);
            } else {
                self.canvas.apply_zoom(zoom_factor);
            }
        }

        // --- Paste Overlay Keyboard Shortcuts ---
        // Delete/Backspace cancels paste overlay (same as Escape)
        if self.paste_overlay.is_some() {
            let delete_pressed = ctx.input(|i| i.key_pressed(egui::Key::Delete));
            let backspace_pressed = ctx.input(|i| i.key_pressed(egui::Key::Backspace));
            if delete_pressed || backspace_pressed {
                self.cancel_paste_overlay();
            }
        }

        // --- Selection Keyboard Shortcuts ---
        if !modal_open {
            let delete_pressed = ctx.input(|i| i.key_pressed(egui::Key::Delete));
            let backspace_pressed = ctx.input(|i| i.key_pressed(egui::Key::Backspace));

            if delete_pressed {
                if self.is_pointer_over_layers_panel {
                    let project_idx = self.active_project_index;
                    if let Some(project) = self.projects.get_mut(project_idx) {
                        self.layers_panel.delete_active_layer_from_app(
                            &mut project.canvas_state,
                            &mut project.history,
                        );
                    }
                } else {
                    // Delete selected pixels (make transparent) on active layer
                    let has_sel = self
                        .active_project()
                        .is_some_and(|p| p.canvas_state.has_selection());
                    if has_sel {
                        self.do_snapshot_op("Delete Selection", |s| {
                            s.delete_selected_pixels();
                        });
                    }
                }
            }

            if backspace_pressed {
                // Fill selected area with primary colour on active layer
                let has_sel = self
                    .active_project()
                    .is_some_and(|p| p.canvas_state.has_selection());
                if has_sel {
                    let pc = self.colors_panel.get_primary_color();
                    let fill = image::Rgba([pc.r(), pc.g(), pc.b(), pc.a()]);
                    self.do_snapshot_op("Fill Selection", |s| {
                        s.fill_selected_pixels(fill);
                    });
                }
            }
        }

        // --- Keyboard Shortcuts (uses editable keybindings system) ---
        //
        // The KeyBindings::is_pressed method uses consume_key internally,
        // so egui will NOT forward consumed keys to text widgets.
        // Clone keybindings to avoid borrow conflicts with &mut self methods.
        //
        // Skip all shortcut processing while the settings window is waiting
        // for a keybind combo, so the rebinding handler sees the raw events.
        let is_rebinding = self.settings_window.rebinding_action.is_some();
        if !modal_open && !is_rebinding {
            self.tools_panel.brush_resize_drag_binding = self
                .settings
                .keybindings
                .get(BindableAction::BrushResizeDragModifier)
                .cloned()
                .unwrap_or_else(|| KeyCombo::modifiers_only(false, true, false));
            use crate::assets::BindableAction;
            let ctrl = ctx.input(|i| i.modifiers.command);
            let kb = self.settings.keybindings.clone();

            // NOTE: Check Ctrl+Shift combos before plain Ctrl combos
            // so that e.g. Ctrl+Shift+S is not consumed as Ctrl+S.

            // Ctrl+Shift+S — Save As
            if kb.is_pressed(ctx, BindableAction::SaveAs) {
                // Trigger Save As dialog (mirrors File > Save As menu logic)
                let save_as_data = if self.active_project_index < self.projects.len() {
                    let project = &mut self.projects[self.active_project_index];
                    project.canvas_state.ensure_all_text_layers_rasterized();
                    let composite = project.canvas_state.composite();
                    let frame_images: Option<Vec<image::RgbaImage>> =
                        if project.canvas_state.layers.len() > 1 {
                            Some(
                                project
                                    .canvas_state
                                    .layers
                                    .iter()
                                    .map(|l| l.pixels.to_rgba_image())
                                    .collect(),
                            )
                        } else {
                            None
                        };
                    let was_animated = project.was_animated;
                    let animation_fps = project.animation_fps;
                    let path = project.path.clone();
                    Some((composite, frame_images, was_animated, animation_fps, path))
                } else {
                    None
                };
                self.save_file_dialog.reset();
                if let Some((composite, frame_images, was_animated, animation_fps, path)) =
                    save_as_data
                {
                    self.save_file_dialog.set_source_image(&composite);
                    if let Some(frames) = frame_images.as_ref() {
                        self.save_file_dialog.set_source_animated(
                            frames,
                            was_animated,
                            animation_fps,
                        );
                    }
                    if let Some(ref p) = path {
                        self.save_file_dialog.set_from_path(p);
                    }
                }
                self.save_file_dialog.open = true;
            }

            // Ctrl+Shift+F — Flatten All Layers
            if kb.is_pressed(ctx, BindableAction::FlattenLayers) {
                self.do_snapshot_op("Flatten All Layers", |s| {
                    crate::ops::transform::flatten_image(s);
                });
            }

            // Ctrl+R — Resize Image
            if kb.is_pressed(ctx, BindableAction::ResizeImage)
                && let Some(project) = self.active_project()
            {
                let mut dlg = crate::ops::dialogs::ResizeImageDialog::new(&project.canvas_state);
                dlg.lock_aspect = self.settings.persist_resize_lock_aspect;
                self.active_dialog = ActiveDialog::ResizeImage(dlg);
            }

            // Ctrl+Shift+R — Resize Canvas
            if kb.is_pressed(ctx, BindableAction::ResizeCanvas)
                && let Some(project) = self.active_project()
            {
                let mut dlg = crate::ops::dialogs::ResizeCanvasDialog::new(&project.canvas_state);
                dlg.lock_aspect = self.settings.persist_resize_lock_aspect;
                self.active_dialog = ActiveDialog::ResizeCanvas(dlg);
            }

            // Ctrl++ / Ctrl+= — Zoom In
            // egui 0.24 requires exact shift match in consume_key, so Ctrl+=
            // (shift=false) and Ctrl++ (shift=true, physical Shift+=) need
            // separate handling. We scan raw events and require only ctrl.
            let zoom_in_pressed = kb.is_pressed(ctx, BindableAction::ViewZoomIn)
                || ctx.input_mut(|i| {
                    let ctrl = i.modifiers.ctrl || i.modifiers.command;
                    if !ctrl {
                        return false;
                    }
                    let pos = i.events.iter().position(|e| {
                        matches!(
                            e,
                            egui::Event::Key {
                                key: egui::Key::Equals,
                                pressed: true,
                                ..
                            }
                        )
                    });
                    if let Some(idx) = pos {
                        i.events.remove(idx);
                        true
                    } else {
                        false
                    }
                });
            if zoom_in_pressed {
                self.canvas.zoom_in();
            }

            // Ctrl+- — Zoom Out
            if kb.is_pressed(ctx, BindableAction::ViewZoomOut) {
                self.canvas.zoom_out();
            }

            // Ctrl+0 — Fit to Window
            if kb.is_pressed(ctx, BindableAction::ViewFitToWindow) {
                self.canvas.reset_zoom();
            }

            // Ctrl+N — New File
            if kb.is_pressed(ctx, BindableAction::NewFile) {
                self.new_file_dialog.load_clipboard_dimensions();
                self.new_file_dialog
                    .set_lock_aspect_ratio(self.settings.persist_new_file_lock_aspect);
                self.new_file_dialog.open_dialog();
            }

            // Ctrl+O — Open File
            if kb.is_pressed(ctx, BindableAction::OpenFile) {
                self.handle_open_file(ctx.input(|i| i.time));
            }

            // Ctrl+W — Close current project
            if kb.is_pressed(ctx, BindableAction::CloseProject)
                && self.active_project_index < self.projects.len()
            {
                self.close_project(self.active_project_index);
            }

            // Ctrl+S — Save
            if kb.is_pressed(ctx, BindableAction::Save) {
                self.handle_save(ctx.input(|i| i.time));
            }

            // Ctrl+Alt+S — Save All
            if kb.is_pressed(ctx, BindableAction::SaveAll) {
                self.handle_save_all(ctx.input(|i| i.time));
            }

            // Ctrl+Z — Undo
            if kb.is_pressed(ctx, BindableAction::Undo) {
                if let Some(overlay) = self.paste_overlay.as_mut() {
                    if let Some(prev) = self.paste_transform_undo.pop() {
                        let current = overlay.transform();
                        overlay.set_transform(prev);
                        self.paste_transform_redo.push(current);
                        if let Some(project) = self.active_project_mut() {
                            project.history.undo(&mut project.canvas_state);
                        }
                    } else {
                        self.cancel_paste_overlay();
                        if let Some(project) = self.active_project_mut() {
                            project.canvas_state.clear_selection();
                        }
                    }
                } else if self.tools_panel.has_active_tool_preview() {
                    // Cancel in-progress tool operation instead of undoing
                    if let Some(project) = self.projects.get_mut(self.active_project_index) {
                        self.tools_panel
                            .cancel_active_tool(&mut project.canvas_state);
                    }
                } else {
                    self.commit_pending_tool_history();
                    if let Some(project) = self.active_project_mut() {
                        project.history.undo(&mut project.canvas_state);
                    }
                }
            }

            // Ctrl+Y — Redo (also Ctrl+Shift+Z for Linux muscle memory)
            let redo_from_binding = kb.is_pressed(ctx, BindableAction::Redo);
            let redo_from_ctrl_shift_z = ctx.input_mut(|i| {
                i.consume_key(
                    egui::Modifiers::CTRL | egui::Modifiers::SHIFT,
                    egui::Key::Z,
                )
            });
            if redo_from_binding || redo_from_ctrl_shift_z {
                if let Some(overlay) = self.paste_overlay.as_mut()
                    && let Some(next) = self.paste_transform_redo.pop()
                {
                    let current = overlay.transform();
                    overlay.set_transform(next);
                    self.paste_transform_undo.push(current);
                    if let Some(project) = self.active_project_mut() {
                        project.history.redo(&mut project.canvas_state);
                    }
                } else if let Some(project) = self.active_project_mut() {
                    project.history.redo(&mut project.canvas_state);
                }
            }

            // Ctrl+C — Copy
            let copy_from_binding = kb.is_pressed(ctx, BindableAction::Copy);
            let copy_from_event = ctx.input_mut(|i| {
                let pos = i.events.iter().position(|e| matches!(e, egui::Event::Copy));
                if let Some(idx) = pos {
                    i.events.remove(idx);
                    true
                } else {
                    false
                }
            });
            let copy_from_raw = ctx.input_mut(|i| {
                let pos = i.events.iter().position(|e| {
                    matches!(
                        e,
                        egui::Event::Key {
                            key,
                            physical_key,
                            pressed: true,
                            modifiers,
                            ..
                        } if (*key == egui::Key::C || *physical_key == Some(egui::Key::C))
                            && (modifiers.command || modifiers.ctrl)
                    )
                });
                if let Some(idx) = pos {
                    i.events.remove(idx);
                    true
                } else {
                    false
                }
            });
            let copy_from_ctrl_char = ctx.input_mut(|i| {
                let mut found = false;
                i.events.retain(|ev| {
                    if !found
                        && let egui::Event::Text(t) = ev
                        && t.chars().any(|c| c == '\u{3}')
                    {
                        found = true;
                        return false;
                    }
                    true
                });
                found
            });
            let copy_pressed =
                copy_from_binding || copy_from_event || copy_from_raw || copy_from_ctrl_char;
            let poll_ctrl_c_down =
                (global_probe.0 && global_probe.1) || (linux_probe.ctrl_down && linux_probe.c_down);
            let copy_poll_edge = poll_ctrl_c_down && !self.prev_ctrl_c_down;
            let copy_vk_edge =
                vk_probe.ctrl_down && vk_probe.c_press_count != self.prev_vk_c_press_count;
            if copy_pressed || copy_poll_edge || copy_vk_edge {
                self.copy_active_selection_or_overlay();
            }

            // Ctrl+X — Cut
            let cut_from_binding = kb.is_pressed(ctx, BindableAction::Cut);
            let cut_from_event = ctx.input_mut(|i| {
                let pos = i.events.iter().position(|e| matches!(e, egui::Event::Cut));
                if let Some(idx) = pos {
                    i.events.remove(idx);
                    true
                } else {
                    false
                }
            });
            let cut_from_raw = ctx.input_mut(|i| {
                let pos = i.events.iter().position(|e| {
                    matches!(
                        e,
                        egui::Event::Key {
                            key,
                            physical_key,
                            pressed: true,
                            modifiers,
                            ..
                        } if (*key == egui::Key::X || *physical_key == Some(egui::Key::X))
                            && (modifiers.command || modifiers.ctrl)
                    )
                });
                if let Some(idx) = pos {
                    i.events.remove(idx);
                    true
                } else {
                    false
                }
            });
            let cut_from_ctrl_char = ctx.input_mut(|i| {
                let mut found = false;
                i.events.retain(|ev| {
                    if !found
                        && let egui::Event::Text(t) = ev
                        && t.chars().any(|c| c == '\u{18}')
                    {
                        found = true;
                        return false;
                    }
                    true
                });
                found
            });
            let cut_pressed =
                cut_from_binding || cut_from_event || cut_from_raw || cut_from_ctrl_char;
            let poll_ctrl_x_down =
                (global_probe.0 && global_probe.2) || (linux_probe.ctrl_down && linux_probe.x_down);
            let cut_poll_edge = poll_ctrl_x_down && !self.prev_ctrl_x_down;
            let cut_vk_edge =
                vk_probe.ctrl_down && vk_probe.x_press_count != self.prev_vk_x_press_count;
            if cut_pressed || cut_poll_edge || cut_vk_edge {
                let has_sel = self
                    .active_project()
                    .is_some_and(|p| p.canvas_state.has_selection());
                let transparent_cutout = self.settings.clipboard_copy_transparent_cutout;
                let mut cut_applied = false;
                if has_sel {
                    self.do_snapshot_op("Cut Selection", |s| {
                        cut_applied = crate::ops::clipboard::cut_selection(s, transparent_cutout);
                    });
                }
            }

            // Ctrl+V — Paste
            let paste_from_binding = kb.is_pressed(ctx, BindableAction::Paste);
            let paste_from_event = ctx.input_mut(|i| {
                let pos = i
                    .events
                    .iter()
                    .position(|e| matches!(e, egui::Event::Paste(_)));
                if let Some(idx) = pos {
                    i.events.remove(idx);
                    true
                } else {
                    false
                }
            });
            let paste_from_raw = ctx.input_mut(|i| {
                let pos = i.events.iter().position(|e| {
                    matches!(
                        e,
                        egui::Event::Key {
                            key,
                            physical_key,
                            pressed: true,
                            modifiers,
                            ..
                        } if (*key == egui::Key::V || *physical_key == Some(egui::Key::V))
                            && (modifiers.command || modifiers.ctrl)
                    )
                });
                if let Some(idx) = pos {
                    i.events.remove(idx);
                    true
                } else {
                    false
                }
            });
            let paste_from_release = ctx.input_mut(|i| {
                let pos = i.events.iter().position(|e| {
                    matches!(
                        e,
                        egui::Event::Key {
                            key,
                            physical_key,
                            pressed: false,
                            modifiers,
                            ..
                        } if (*key == egui::Key::V || *physical_key == Some(egui::Key::V))
                            && (modifiers.command || modifiers.ctrl)
                    )
                });
                if let Some(idx) = pos {
                    i.events.remove(idx);
                    true
                } else {
                    false
                }
            });
            let paste_from_ctrl_char = ctx.input_mut(|i| {
                let mut found = false;
                i.events.retain(|ev| {
                    if !found
                        && let egui::Event::Text(t) = ev
                        && t.chars().any(|c| c == '\u{16}')
                    {
                        found = true;
                        return false;
                    }
                    true
                });
                found
            });
            let paste_from_state = ctx.input(|i| {
                i.key_pressed(egui::Key::V) && (i.modifiers.ctrl || i.modifiers.command)
            });
            let paste_from_raw_input = ctx.input(|i| {
                i.raw.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Key {
                            key,
                            physical_key,
                            pressed: true,
                            modifiers,
                            ..
                        } if (*key == egui::Key::V || *physical_key == Some(egui::Key::V))
                            && (modifiers.command || modifiers.ctrl)
                    ) || matches!(e, egui::Event::Paste(_))
                        || matches!(e, egui::Event::Text(t) if t.chars().any(|c| c == '\u{16}'))
                })
            });
            let paste_primary_trigger = paste_from_binding
                || paste_from_event
                || paste_from_raw
                || paste_from_ctrl_char
                || paste_from_state
                || paste_from_raw_input;
            // Release fallback is only for sessions where press-path signals are missing.
            let poll_ctrl_v_down =
                (global_probe.0 && global_probe.3) || (linux_probe.ctrl_down && linux_probe.v_down);
            let paste_poll_edge = poll_ctrl_v_down && !self.prev_ctrl_v_down;
            let paste_vk_edge =
                vk_probe.ctrl_down && vk_probe.v_press_count != self.prev_vk_v_press_count;
            let now = ctx.input(|i| i.time);
            let recent_paste =
                self.last_paste_trigger_time >= 0.0 && (now - self.last_paste_trigger_time) < 0.25;
            let release_only_trigger = paste_from_release
                && !paste_primary_trigger
                && !paste_poll_edge
                && !paste_vk_edge
                && !recent_paste;
            if paste_primary_trigger || paste_poll_edge || paste_vk_edge || release_only_trigger {
                self.last_paste_trigger_time = now;
                // Compute cursor canvas position before mutable borrow
                let cursor_canvas = if self.active_project_index < self.projects.len() {
                    ctx.input(|i| i.pointer.latest_pos())
                        .and_then(|screen_pos| {
                            self.canvas.last_canvas_rect.and_then(|rect| {
                                let state = &self.projects[self.active_project_index].canvas_state;
                                self.canvas
                                    .screen_to_canvas_f32_pub(screen_pos, rect, state)
                            })
                        })
                } else {
                    None
                };
                self.queue_paste_from_clipboard(cursor_canvas);
            }

            let enter_vk_edge = vk_probe.enter_press_count != self.prev_vk_enter_press_count;
            let escape_vk_edge = vk_probe.escape_press_count != self.prev_vk_escape_press_count;

            // Enter — Commit paste (not rebindable)
            if (ctx.input(|i| i.key_pressed(egui::Key::Enter)) || enter_vk_edge)
                && self.paste_overlay.is_some()
            {
                self.commit_paste_overlay();
            }
            // Escape — Cancel paste (not rebindable)
            if (ctx.input(|i| i.key_pressed(egui::Key::Escape)) || escape_vk_edge)
                && self.paste_overlay.is_some()
            {
                self.cancel_paste_overlay();
            }

            // Raw Enter/Escape fallback for overlay tools (e.g. Mesh Warp) so
            // commit/cancel still works even if a focused widget consumes key state.
            let enter_or_escape = ctx.input_mut(|i| {
                let mut enter = false;
                let mut escape = false;
                let mut consumed_enter = false;
                let mut consumed_escape = false;
                i.events.retain(|e| match e {
                    egui::Event::Key {
                        key: egui::Key::Enter,
                        pressed: true,
                        ..
                    } if !consumed_enter => {
                        enter = true;
                        consumed_enter = true;
                        false
                    }
                    egui::Event::Key {
                        key: egui::Key::Escape,
                        pressed: true,
                        ..
                    } if !consumed_escape => {
                        escape = true;
                        consumed_escape = true;
                        false
                    }
                    _ => true,
                });
                (enter, escape)
            });
            let enter_poll_edge = global_probe.4 && !self.prev_enter_down;
            let escape_poll_edge = global_probe.5 && !self.prev_escape_down;
            if enter_or_escape.0 || enter_poll_edge || enter_vk_edge {
                self.tools_panel.injected_enter_pressed = true;
                if self.tools_panel.active_tool == crate::components::tools::Tool::MeshWarp
                    && self.tools_panel.mesh_warp_state.is_active
                {
                    self.tools_panel.mesh_warp_state.commit_pending = true;
                    self.tools_panel.mesh_warp_state.commit_pending_frame = 0;
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.mark_dirty(None);
                    }
                }
                if self.tools_panel.active_tool == crate::components::tools::Tool::Liquify
                    && self.tools_panel.liquify_state.is_active
                {
                    self.tools_panel.liquify_state.commit_pending = true;
                    self.tools_panel.liquify_state.commit_pending_frame = 0;
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.mark_dirty(None);
                    }
                }
            }
            if enter_or_escape.1 || escape_poll_edge || escape_vk_edge {
                self.tools_panel.injected_escape_pressed = true;
                if self.tools_panel.active_tool == crate::components::tools::Tool::MeshWarp
                    && self.tools_panel.mesh_warp_state.is_active
                {
                    self.tools_panel.mesh_warp_state.points =
                        self.tools_panel.mesh_warp_state.original_points.clone();
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.mark_dirty(None);
                    }
                }
                if self.tools_panel.active_tool == crate::components::tools::Tool::Liquify
                    && self.tools_panel.liquify_state.is_active
                {
                    self.tools_panel.liquify_state.is_active = false;
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.clear_preview_state();
                        project.canvas_state.mark_dirty(None);
                    }
                }
            }

            self.prev_ctrl_c_down = poll_ctrl_c_down;
            self.prev_ctrl_x_down = poll_ctrl_x_down;
            self.prev_ctrl_v_down = poll_ctrl_v_down;
            self.prev_enter_down = global_probe.4;
            self.prev_escape_down = global_probe.5;
            self.prev_vk_c_press_count = vk_probe.c_press_count;
            self.prev_vk_x_press_count = vk_probe.x_press_count;
            self.prev_vk_v_press_count = vk_probe.v_press_count;
            self.prev_vk_enter_press_count = vk_probe.enter_press_count;
            self.prev_vk_escape_press_count = vk_probe.escape_press_count;

            // Tab — Center active transform on canvas (paste/move-pixels, placed shape, line edit)
            let tab_applies = self.paste_overlay.is_some()
                || (self.tools_panel.active_tool == crate::components::tools::Tool::Shapes
                    && self.tools_panel.shapes_state.placed.is_some())
                || (self.tools_panel.active_tool == crate::components::tools::Tool::Line
                    && self.tools_panel.line_state.line_tool.stage
                        == crate::components::tools::LineStage::Editing);
            let tab_pressed = ctx.input(|i| i.key_pressed(egui::Key::Tab));
            if tab_applies && tab_pressed {
                let (cw, ch) = self
                    .active_project()
                    .map(|p| (p.canvas_state.width as f32, p.canvas_state.height as f32))
                    .unwrap_or((0.0, 0.0));
                if cw > 0.0 {
                    let cx = cw / 2.0;
                    let cy = ch / 2.0;
                    if let Some(ref mut overlay) = self.paste_overlay {
                        overlay.center = egui::Pos2::new(cx, cy);
                        overlay.snap_center_to_pixel();
                    } else if self.tools_panel.active_tool == crate::components::tools::Tool::Shapes
                    {
                        if let Some(ref mut placed) = self.tools_panel.shapes_state.placed {
                            placed.cx = cx;
                            placed.cy = cy;
                        }
                        // Re-rasterize the shape preview at its new position.
                        let primary = self.colors_panel.get_primary_color_f32();
                        let secondary = self.colors_panel.get_secondary_color_f32();
                        let canvas_ptr: Option<*mut crate::canvas::CanvasState> = self
                            .active_project_mut()
                            .map(|p| &mut p.canvas_state as *mut _);
                        if let Some(ptr) = canvas_ptr {
                            // SAFETY: same-frame, single-threaded, no other borrow active.
                            let canvas = unsafe { &mut *ptr };
                            self.tools_panel
                                .render_shape_preview(canvas, primary, secondary);
                        }
                    } else {
                        // Line tool — translate control-point bounding box to canvas center
                        let cps = &mut self.tools_panel.line_state.line_tool.control_points;
                        let min_x = cps.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
                        let max_x = cps.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
                        let min_y = cps.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
                        let max_y = cps.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
                        let dx = cx - (min_x + max_x) / 2.0;
                        let dy = cy - (min_y + max_y) / 2.0;
                        for pt in cps.iter_mut() {
                            pt.x = (pt.x + dx).clamp(0.0, cw - 1.0);
                            pt.y = (pt.y + dy).clamp(0.0, ch - 1.0);
                        }
                        // Re-rasterize the bezier preview at its new position.
                        let canvas_ptr: Option<*mut crate::canvas::CanvasState> = self
                            .active_project_mut()
                            .map(|p| &mut p.canvas_state as *mut _);
                        let cps = self.tools_panel.line_state.line_tool.control_points;
                        let last_bounds = self.tools_panel.line_state.line_tool.last_bounds;
                        let pattern = self.tools_panel.line_state.line_tool.options.pattern;
                        let cap = self.tools_panel.line_state.line_tool.options.cap_style;
                        let color = self.tools_panel.properties.color;
                        let new_bounds = self
                            .tools_panel
                            .get_bezier_bounds(cps, cw as u32, ch as u32);
                        self.tools_panel.stroke_tracker.expand_bounds(new_bounds);
                        self.tools_panel.line_state.line_tool.last_bounds = Some(new_bounds);
                        if let Some(ptr) = canvas_ptr {
                            let canvas = unsafe { &mut *ptr };
                            self.tools_panel.rasterize_bezier(
                                canvas,
                                cps,
                                color,
                                pattern,
                                cap,
                                last_bounds,
                            );
                            let dirty = last_bounds.map_or(new_bounds, |lb| lb.union(new_bounds));
                            canvas.mark_preview_changed_rect(dirty);
                        }
                    }
                    if let Some(project) = self.active_project_mut() {
                        project.canvas_state.mark_dirty(None);
                    }
                    ctx.request_repaint();
                }
            }

            // Ctrl+A — Select All (skip when a text editor has focus)
            let script_editor_open = self.window_visibility.script_editor;
            let text_edit_focused = ctx.text_edit_focused();
            if !script_editor_open && !text_edit_focused && kb.is_pressed(ctx, BindableAction::SelectAll) {
                self.select_all_canvas();
            }

            // Ctrl+D — Commit active overlay/tool state, then deselect.
            let deselect_pressed = kb.is_pressed(ctx, BindableAction::Deselect);
            if deselect_pressed {
                if self.paste_overlay.is_some() {
                    self.commit_paste_overlay();
                }
                self.tools_panel.injected_enter_pressed = true;
                if self.tools_panel.active_tool == crate::components::tools::Tool::MeshWarp
                    && self.tools_panel.mesh_warp_state.is_active
                {
                    self.tools_panel.mesh_warp_state.commit_pending = true;
                    self.tools_panel.mesh_warp_state.commit_pending_frame = 0;
                }
                if self.tools_panel.active_tool == crate::components::tools::Tool::Liquify
                    && self.tools_panel.liquify_state.is_active
                {
                    self.tools_panel.liquify_state.commit_pending = true;
                    self.tools_panel.liquify_state.commit_pending_frame = 0;
                }
            }
            if deselect_pressed && let Some(project) = self.active_project_mut() {
                let sel_before = project.canvas_state.selection_mask.clone();
                project.canvas_state.clear_selection();
                project.canvas_state.mark_dirty(None);
                if sel_before.is_some() {
                    project.history.push(Box::new(SelectionCommand::new(
                        "Deselect", sel_before, None,
                    )));
                }
            }

            // Arrow keys — Move paste overlay (not rebindable)
            if self.paste_overlay.is_some() {
                let shift = ctx.input(|i| i.modifiers.shift);
                let arrows = [
                    (egui::Key::ArrowUp, 0.0f32, -1.0f32),
                    (egui::Key::ArrowDown, 0.0, 1.0),
                    (egui::Key::ArrowLeft, -1.0, 0.0),
                    (egui::Key::ArrowRight, 1.0, 0.0),
                ];
                for (key, dx_dir, dy_dir) in &arrows {
                    if ctx.input(|i| i.key_pressed(*key))
                        && let Some(ref mut overlay) = self.paste_overlay
                    {
                        let (step_x, step_y) = if shift {
                            let sw = overlay.source.width() as f32 * overlay.scale_x;
                            let sh = overlay.source.height() as f32 * overlay.scale_y;
                            (sw * dx_dir.abs(), sh * dy_dir.abs())
                        } else if ctrl {
                            (100.0, 100.0)
                        } else {
                            (1.0, 1.0)
                        };
                        overlay.center.x += dx_dir * step_x;
                        overlay.center.y += dy_dir * step_y;
                        overlay.snap_center_to_pixel();
                    }
                }
            }

            // Arrow keys — Move selection mask (MoveSelection tool, no paste overlay)
            if self.paste_overlay.is_none()
                && self.tools_panel.active_tool == crate::components::tools::Tool::MoveSelection
            {
                let shift = ctx.input(|i| i.modifiers.shift);
                let arrows = [
                    (egui::Key::ArrowUp, 0i32, -1i32),
                    (egui::Key::ArrowDown, 0, 1),
                    (egui::Key::ArrowLeft, -1, 0),
                    (egui::Key::ArrowRight, 1, 0),
                ];
                for (key, dx_dir, dy_dir) in &arrows {
                    if ctx.input(|i| i.key_pressed(*key))
                        && let Some(project) = self.active_project_mut()
                    {
                        let (step_x, step_y) = if shift {
                            (10, 10)
                        } else if ctrl {
                            (100, 100)
                        } else {
                            (1, 1)
                        };
                        project
                            .canvas_state
                            .translate_selection(dx_dir * step_x, dy_dir * step_y);
                    }
                }
            }

            // Arrow keys — Nudge line tool endpoints while in Editing stage
            // Shift = line bounding-box dimension in move direction (tiling, mirrors paste overlay),
            // Ctrl = 100px, plain = 1px
            if self.paste_overlay.is_none()
                && self.tools_panel.active_tool == crate::components::tools::Tool::Line
                && self.tools_panel.line_state.line_tool.stage
                    == crate::components::tools::LineStage::Editing
            {
                let shift = ctx.input(|i| i.modifiers.shift);
                let arrows = [
                    (egui::Key::ArrowUp, 0.0f32, -1.0f32),
                    (egui::Key::ArrowDown, 0.0, 1.0),
                    (egui::Key::ArrowLeft, -1.0, 0.0),
                    (egui::Key::ArrowRight, 1.0, 0.0),
                ];
                let any_pressed = arrows
                    .iter()
                    .any(|(k, _, _)| ctx.input(|i| i.key_pressed(*k)));
                if any_pressed {
                    // Obtain a raw canvas_state pointer so we can free the mutable
                    // borrow on `self` before calling tools_panel methods.
                    let canvas_ptr: Option<*mut crate::canvas::CanvasState> = self
                        .active_project_mut()
                        .map(|p| &mut p.canvas_state as *mut _);
                    let (cw, ch) = self
                        .active_project_mut()
                        .map(|p| (p.canvas_state.width as f32, p.canvas_state.height as f32))
                        .unwrap_or((0.0, 0.0));

                    // Pre-compute bounding box for Shift tiling step
                    let cps_for_bounds = self.tools_panel.line_state.line_tool.control_points;
                    let (bbox_w, bbox_h) = if shift && cw > 0.0 {
                        let b = self.tools_panel.get_bezier_bounds(
                            cps_for_bounds,
                            cw as u32,
                            ch as u32,
                        );
                        (b.width().max(1.0), b.height().max(1.0))
                    } else {
                        (1.0, 1.0)
                    };

                    if cw > 0.0 {
                        for (key, dx_dir, dy_dir) in &arrows {
                            if ctx.input(|i| i.key_pressed(*key)) {
                                // Shift: move by bounding-box size in that axis (tiling)
                                // Ctrl: 100px, plain: 1px
                                let step_x = if shift {
                                    bbox_w
                                } else if ctrl {
                                    100.0
                                } else {
                                    1.0
                                };
                                let step_y = if shift {
                                    bbox_h
                                } else if ctrl {
                                    100.0
                                } else {
                                    1.0
                                };
                                let dx = dx_dir * step_x;
                                let dy = dy_dir * step_y;

                                // Translate all control points
                                for pt in self
                                    .tools_panel
                                    .line_state
                                    .line_tool
                                    .control_points
                                    .iter_mut()
                                {
                                    pt.x = (pt.x + dx).clamp(0.0, cw - 1.0);
                                    pt.y = (pt.y + dy).clamp(0.0, ch - 1.0);
                                }

                                let cps = self.tools_panel.line_state.line_tool.control_points;
                                let last_bounds = self.tools_panel.line_state.line_tool.last_bounds;
                                let pattern = self.tools_panel.line_state.line_tool.options.pattern;
                                let cap = self.tools_panel.line_state.line_tool.options.cap_style;
                                let color = self.tools_panel.properties.color;
                                let new_bounds = self
                                    .tools_panel
                                    .get_bezier_bounds(cps, cw as u32, ch as u32);
                                self.tools_panel.stroke_tracker.expand_bounds(new_bounds);
                                self.tools_panel.line_state.line_tool.last_bounds =
                                    Some(new_bounds);

                                // SAFETY: canvas_ptr was obtained from self.active_project_mut() in this same
                                // frame, no other code touches canvas_state between these two points, and we
                                // ensure the tools_panel borrow ends before any further project access.
                                if let Some(ptr) = canvas_ptr {
                                    let canvas = unsafe { &mut *ptr };
                                    self.tools_panel.rasterize_bezier(
                                        canvas,
                                        cps,
                                        color,
                                        pattern,
                                        cap,
                                        last_bounds,
                                    );
                                    let dirty =
                                        last_bounds.map_or(new_bounds, |lb| lb.union(new_bounds));
                                    canvas.mark_preview_changed_rect(dirty);
                                }
                                ctx.request_repaint();
                            }
                        }
                    }
                }
            } else if tab_pressed
                && !(self.tools_panel.active_tool == crate::components::tools::Tool::Text
                    && self.tools_panel.text_state.editing_text_layer)
            {
                self.colors_panel.swap_colors();
            }

            // ================================================================
            // TOOL SWITCHING SHORTCUTS (rebindable single letter keys)
            // Only active when not typing into Text tool
            // ================================================================
            let text_tool_active = self.tools_panel.active_tool
                == crate::components::tools::Tool::Text
                && self.tools_panel.text_state.is_editing;
            // Read touch state outside memory() closure to avoid RwLock deadlock.
            let touch_active = ctx.input(|i| {
                i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Touch {
                            phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                            ..
                        }
                    )
                })
            });
            // If any egui TextEdit widget has focus, always block shortcuts.
            // This covers dialog text fields (font search, file name, settings values)
            // that may not be caught by the canvas_widget_id check on Wayland.
            let text_edit_has_focus = ctx.text_edit_focused();
            let other_widget_focused = ctx.memory(|m| {
                let focused = m.focused();
                if focused.is_none() {
                    // No widget has keyboard focus — check if a Touch event is active
                    // (Wayland tablet fallback: touch-emulated clicks may not set focus
                    // properly, so don't block shortcuts in this case).
                    // However, if a TextEdit is focused, always block regardless of touch.
                    text_edit_has_focus || !touch_active
                } else {
                    focused.is_some_and(|id| self.canvas.canvas_widget_id != Some(id))
                }
            });
            if !text_tool_active && !other_widget_focused && self.paste_overlay.is_none() {
                use crate::components::tools::Tool;
                let tool_actions: &[(BindableAction, Tool)] = &[
                    (BindableAction::ToolBrush, Tool::Brush),
                    (BindableAction::ToolEraser, Tool::Eraser),
                    (BindableAction::ToolPencil, Tool::Pencil),
                    (BindableAction::ToolLine, Tool::Line),
                    (BindableAction::ToolGradient, Tool::Gradient),
                    (BindableAction::ToolFill, Tool::Fill),
                    (BindableAction::ToolMagicWand, Tool::MagicWand),
                    (BindableAction::ToolColorPicker, Tool::ColorPicker),
                    (BindableAction::ToolMovePixels, Tool::MovePixels),
                    (BindableAction::ToolRectSelect, Tool::RectangleSelect),
                    (BindableAction::ToolText, Tool::Text),
                    (BindableAction::ToolZoom, Tool::Zoom),
                    (BindableAction::ToolPan, Tool::Pan),
                    (BindableAction::ToolCloneStamp, Tool::CloneStamp),
                    (BindableAction::ToolShapes, Tool::Shapes),
                    (BindableAction::ToolLasso, Tool::Lasso),
                    (BindableAction::ToolColorRemover, Tool::ColorRemover),
                    (BindableAction::ToolMeshWarp, Tool::MeshWarp),
                ];
                for (action, tool) in tool_actions {
                    if kb.is_pressed(ctx, *action) {
                        self.tools_panel.change_tool(*tool);
                        break;
                    }
                }
            }

            // [ / ] — Decrease / Increase brush size
            if !text_tool_active && !other_widget_focused {
                use crate::components::tools::Tool;
                let brush_tool = matches!(
                    self.tools_panel.active_tool,
                    Tool::Brush
                        | Tool::Eraser
                        | Tool::CloneStamp
                        | Tool::ContentAwareBrush
                        | Tool::Liquify
                );
                if brush_tool {
                    if kb.is_pressed(ctx, BindableAction::BrushSizeDecrease) {
                        self.tools_panel.properties.size =
                            (self.tools_panel.properties.size - 1.0).max(1.0);
                    }
                    if kb.is_pressed(ctx, BindableAction::BrushSizeIncrease) {
                        self.tools_panel.properties.size =
                            (self.tools_panel.properties.size + 1.0).min(500.0);
                    }
                }
            }

            // --- Color, Filter and Generate keyboard shortcuts ---
            let has_project = self.active_project().is_some();
            let no_dialog_open = matches!(self.active_dialog, ActiveDialog::None);

            // Color — instant adjustments (no dialog)
            if has_project && !text_tool_active && !other_widget_focused {
                if kb.is_pressed(ctx, BindableAction::ColorAutoLevels) {
                    self.do_layer_snapshot_op("Auto Levels", |s| {
                        let idx = s.active_layer_index;
                        crate::ops::adjustments::auto_levels(s, idx);
                    });
                }
                if kb.is_pressed(ctx, BindableAction::ColorDesaturate) {
                    self.do_layer_snapshot_op("Desaturate", |s| {
                        let idx = s.active_layer_index;
                        crate::ops::filters::desaturate_layer(s, idx);
                    });
                }
                if kb.is_pressed(ctx, BindableAction::ColorInvertColors) {
                    self.do_gpu_snapshot_op("Invert Colors", |s, gpu| {
                        let idx = s.active_layer_index;
                        crate::ops::adjustments::invert_colors_gpu(s, idx, gpu);
                    });
                }
                if kb.is_pressed(ctx, BindableAction::ColorInvertAlpha) {
                    self.do_layer_snapshot_op("Invert Alpha", |s| {
                        let idx = s.active_layer_index;
                        crate::ops::adjustments::invert_alpha(s, idx);
                    });
                }
                if kb.is_pressed(ctx, BindableAction::ColorSepiaTone) {
                    self.do_layer_snapshot_op("Sepia Tone", |s| {
                        let idx = s.active_layer_index;
                        crate::ops::adjustments::sepia(s, idx);
                    });
                }
            }

            // Color, Filter and Generate — dialog openers
            if has_project && no_dialog_open && !text_tool_active && !other_widget_focused {
                // Color dialogs
                if kb.is_pressed(ctx, BindableAction::ColorBrightnessContrast)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::BrightnessContrast(
                        crate::ops::dialogs::BrightnessContrastDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorCurves)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Curves(
                        crate::ops::dialogs::CurvesDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorExposure)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Exposure(
                        crate::ops::dialogs::ExposureDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorHighlightsShadows)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::HighlightsShadows(
                        crate::ops::dialogs::HighlightsShadowsDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorHueSaturation)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::HueSaturation(
                        crate::ops::dialogs::HueSaturationDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorLevels)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Levels(
                        crate::ops::dialogs::LevelsDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorTemperatureTint)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::TemperatureTint(
                        crate::ops::dialogs::TemperatureTintDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorVibrance)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Vibrance(
                        crate::ops::dialogs::VibranceDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorThreshold)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Threshold(
                        crate::ops::dialogs::ThresholdDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorPosterize)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Posterize(
                        crate::ops::dialogs::PosterizeDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorBalance)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::ColorBalance(
                        crate::ops::dialogs::ColorBalanceDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorGradientMap)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::GradientMap(
                        crate::ops::dialogs::GradientMapDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::ColorBlackAndWhite)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::BlackAndWhite(
                        crate::ops::dialogs::BlackAndWhiteDialog::new(&project.canvas_state),
                    );
                }
                // Filter — blur
                if kb.is_pressed(ctx, BindableAction::FilterGaussianBlur)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::GaussianBlur(
                        crate::ops::dialogs::GaussianBlurDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterBokehBlur)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::BokehBlur(
                        crate::ops::effect_dialogs::BokehBlurDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterMotionBlur)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::MotionBlur(
                        crate::ops::effect_dialogs::MotionBlurDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterBoxBlur)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::BoxBlur(
                        crate::ops::effect_dialogs::BoxBlurDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterZoomBlur)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::ZoomBlur(
                        crate::ops::effect_dialogs::ZoomBlurDialog::new(&project.canvas_state),
                    );
                }
                // Filter — sharpen / noise
                if kb.is_pressed(ctx, BindableAction::FilterSharpen)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Sharpen(
                        crate::ops::effect_dialogs::SharpenDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterReduceNoise)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::ReduceNoise(
                        crate::ops::effect_dialogs::ReduceNoiseDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterAddNoise)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::AddNoise(
                        crate::ops::effect_dialogs::AddNoiseDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterMedian)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Median(
                        crate::ops::effect_dialogs::MedianDialog::new(&project.canvas_state),
                    );
                }
                // Filter — distort
                if kb.is_pressed(ctx, BindableAction::FilterCrystallize)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Crystallize(
                        crate::ops::effect_dialogs::CrystallizeDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterDents)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Dents(
                        crate::ops::effect_dialogs::DentsDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterPixelate)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Pixelate(
                        crate::ops::effect_dialogs::PixelateDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterBulge)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Bulge(
                        crate::ops::effect_dialogs::BulgeDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterTwist)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Twist(
                        crate::ops::effect_dialogs::TwistDialog::new(&project.canvas_state),
                    );
                }
                // Filter — stylize
                if kb.is_pressed(ctx, BindableAction::FilterGlow)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Glow(
                        crate::ops::effect_dialogs::GlowDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterVignette)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Vignette(
                        crate::ops::effect_dialogs::VignetteDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterHalftone)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Halftone(
                        crate::ops::effect_dialogs::HalftoneDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterInk)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Ink(
                        crate::ops::effect_dialogs::InkDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterOilPainting)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::OilPainting(
                        crate::ops::effect_dialogs::OilPaintingDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterColorFilter)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::ColorFilter(
                        crate::ops::effect_dialogs::ColorFilterDialog::new(&project.canvas_state),
                    );
                }
                // Filter — glitch
                if kb.is_pressed(ctx, BindableAction::FilterPixelDrag)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::PixelDrag(
                        crate::ops::effect_dialogs::PixelDragDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::FilterRgbDisplace)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::RgbDisplace(
                        crate::ops::effect_dialogs::RgbDisplaceDialog::new(&project.canvas_state),
                    );
                }
                // Filter — AI (requires ONNX runtime)
                if kb.is_pressed(ctx, BindableAction::FilterRemoveBackground) && self.onnx_available
                {
                    self.active_dialog = ActiveDialog::RemoveBackground(
                        crate::ops::effect_dialogs::RemoveBackgroundDialog::new(),
                    );
                }
                // Generate
                if kb.is_pressed(ctx, BindableAction::GenerateGrid)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Grid(
                        crate::ops::effect_dialogs::GridDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::GenerateDropShadow)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::DropShadow(
                        crate::ops::effect_dialogs::DropShadowDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::GenerateOutline)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Outline(
                        crate::ops::effect_dialogs::OutlineDialog::new(&project.canvas_state),
                    );
                }
                if kb.is_pressed(ctx, BindableAction::GenerateContours)
                    && let Some(project) = self.active_project()
                {
                    self.active_dialog = ActiveDialog::Contours(
                        crate::ops::effect_dialogs::ContoursDialog::new(&project.canvas_state),
                    );
                }
            }
        }

        // -- Move Pixels tool: activate paste overlay on first click --
        if !modal_open
            && self.tools_panel.active_tool == crate::components::tools::Tool::MovePixels
            && self.paste_overlay.is_none()
        {
            let primary_pressed = ctx.input(|i| i.pointer.primary_pressed());
            let canvas_rect = self.canvas.last_canvas_rect;
            let over_canvas = ctx.input(|i| {
                i.pointer
                    .hover_pos()
                    .is_some_and(|pos| canvas_rect.is_some_and(|r| r.contains(pos)))
            });
            // On Wayland with a tablet, also check for active Touch events since
            // is_pointer_over_egui() relies on interact_pos() which isn't updated by Touch.
            let over_ui = ctx.is_pointer_over_egui()
                || ctx.input(|i| {
                    i.events.iter().any(|e| {
                        matches!(
                            e,
                            egui::Event::Touch {
                                phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                                ..
                            }
                        )
                    })
                });

            if primary_pressed && over_canvas && !over_ui {
                // Extract pixels into overlay and blank the source area.
                let mut overlay_out: Option<crate::ops::clipboard::PasteOverlay> = None;
                let mut before_snapshot: Option<crate::components::history::CanvasSnapshot> = None;
                if let Some(project) = self.active_project_mut() {
                    before_snapshot = Some(crate::components::history::CanvasSnapshot::capture(
                        &project.canvas_state,
                    ));
                    if let Some(overlay) =
                        crate::ops::clipboard::extract_to_overlay(&mut project.canvas_state)
                    {
                        overlay_out = Some(overlay);
                        project.mark_dirty();
                    }
                }
                if let Some(mut overlay) = overlay_out {
                    overlay.interpolation = self.tools_panel.move_interpolation;
                    overlay.anti_aliasing = self.tools_panel.move_anti_aliasing;
                    self.paste_overlay = Some(overlay);
                    self.move_pixels_before = before_snapshot;
                    self.is_move_pixels_active = true;
                }
            }
        }

        // -- Move Selection tool: drag to translate the selection mask --
        if !modal_open
            && self.tools_panel.active_tool == crate::components::tools::Tool::MoveSelection
            && self.paste_overlay.is_none()
        {
            let primary_pressed = ctx.input(|i| i.pointer.primary_pressed());
            let primary_down = ctx.input(|i| i.pointer.primary_down());
            let primary_released = ctx.input(|i| i.pointer.primary_released());
            let canvas_rect = self.canvas.last_canvas_rect;
            let over_canvas = ctx.input(|i| {
                i.pointer
                    .hover_pos()
                    .is_some_and(|pos| canvas_rect.is_some_and(|r| r.contains(pos)))
            });
            // On Wayland with a tablet, also check for active Touch events since
            // is_pointer_over_egui() relies on interact_pos() which isn't updated by Touch.
            let over_ui = ctx.is_pointer_over_egui()
                || ctx.input(|i| {
                    i.events.iter().any(|e| {
                        matches!(
                            e,
                            egui::Event::Touch {
                                phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                                ..
                            }
                        )
                    })
                });

            // Compute current canvas position from mouse (without borrowing self mutably).
            let cur_canvas_pos: Option<(i32, i32)> =
                if self.active_project_index < self.projects.len() {
                    ctx.input(|i| i.pointer.hover_pos()).and_then(|pos| {
                        canvas_rect.and_then(|rect| {
                            self.canvas
                                .screen_to_canvas_pub(
                                    pos,
                                    rect,
                                    &self.projects[self.active_project_index].canvas_state,
                                )
                                .map(|(x, y)| (x as i32, y as i32))
                        })
                    })
                } else {
                    None
                };

            if primary_pressed
                && over_canvas
                && !over_ui
                && let Some(cp) = cur_canvas_pos
            {
                self.move_sel_dragging = true;
                self.move_sel_last_canvas = Some(cp);
                self.move_sel_start_mask = self
                    .active_project()
                    .and_then(|p| p.canvas_state.selection_mask.clone());
                self.move_sel_start_bounds = self
                    .move_sel_start_mask
                    .as_ref()
                    .and_then(Self::selection_mask_bounds_for_move);
                self.move_sel_handle = self
                    .move_sel_start_bounds
                    .and_then(|b| Self::hit_selection_handle(cp, b, self.canvas.zoom));
            }

            if primary_down
                && self.move_sel_dragging
                && let Some((cx, cy)) = cur_canvas_pos
                && let Some((lx, ly)) = self.move_sel_last_canvas
            {
                if self.move_sel_handle.unwrap_or(crate::ops::shapes::ShapeHandle::Move)
                    == crate::ops::shapes::ShapeHandle::Move
                {
                    let dx = cx - lx;
                    let dy = cy - ly;
                    if dx != 0 || dy != 0 {
                        if let Some(project) = self.active_project_mut() {
                            project.canvas_state.translate_selection(dx, dy);
                        }
                        self.move_sel_last_canvas = Some((cx, cy));
                    }
                } else if let (Some(start_mask), Some(start_bounds), Some(handle)) = (
                    self.move_sel_start_mask.clone(),
                    self.move_sel_start_bounds,
                    self.move_sel_handle,
                ) {
                    let shift = ctx.input(|i| i.modifiers.shift);
                    if let Some(new_mask) =
                        Self::resize_selection_mask_from_handle(
                            &start_mask,
                            start_bounds,
                            handle,
                            (cx, cy),
                            shift,
                            self.tools_panel.move_anti_aliasing,
                        )
                        && let Some(project) = self.active_project_mut()
                    {
                        project.canvas_state.selection_mask = Some(new_mask);
                        project.canvas_state.invalidate_selection_overlay();
                        project.canvas_state.mark_dirty(None);
                    }
                }
            }

            if primary_released {
                if let Some(before) = self.move_sel_start_mask.take()
                    && let Some(project) = self.active_project_mut()
                {
                    let after = project.canvas_state.selection_mask.clone();
                    if after.as_ref() != Some(&before) {
                        project.history.push(Box::new(
                            crate::components::history::SelectionCommand::new(
                                "Transform Selection",
                                Some(before),
                                after,
                            ),
                        ));
                    }
                }
                self.move_sel_dragging = false;
                self.move_sel_last_canvas = None;
                self.move_sel_handle = None;
                self.move_sel_start_bounds = None;
            }
        }

        modal_open
    }

    fn selection_mask_bounds_for_move(mask: &image::GrayImage) -> Option<(u32, u32, u32, u32)> {
        let (w, h) = mask.dimensions();
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0;
        let mut max_y = 0;
        for y in 0..h {
            for x in 0..w {
                if mask.get_pixel(x, y)[0] > 0 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x + 1);
                    max_y = max_y.max(y + 1);
                }
            }
        }
        (min_x < max_x && min_y < max_y).then_some((min_x, min_y, max_x, max_y))
    }

    fn hit_selection_handle(
        pos: (i32, i32),
        bounds: (u32, u32, u32, u32),
        zoom: f32,
    ) -> Option<crate::ops::shapes::ShapeHandle> {
        let (x0, y0, x1, y1) = bounds;
        let px = pos.0 as f32;
        let py = pos.1 as f32;
        let tol = (8.0 / zoom.max(1.0)).max(1.5);
        let pts = [
            (crate::ops::shapes::ShapeHandle::TopLeft, x0 as f32, y0 as f32),
            (crate::ops::shapes::ShapeHandle::Top, (x0 + x1) as f32 * 0.5, y0 as f32),
            (crate::ops::shapes::ShapeHandle::TopRight, x1 as f32, y0 as f32),
            (crate::ops::shapes::ShapeHandle::Right, x1 as f32, (y0 + y1) as f32 * 0.5),
            (crate::ops::shapes::ShapeHandle::BottomRight, x1 as f32, y1 as f32),
            (crate::ops::shapes::ShapeHandle::Bottom, (x0 + x1) as f32 * 0.5, y1 as f32),
            (crate::ops::shapes::ShapeHandle::BottomLeft, x0 as f32, y1 as f32),
            (crate::ops::shapes::ShapeHandle::Left, x0 as f32, (y0 + y1) as f32 * 0.5),
        ];
        for (h, x, y) in pts {
            if (px - x).abs() <= tol && (py - y).abs() <= tol {
                return Some(h);
            }
        }
        if px >= x0 as f32 && px <= x1 as f32 && py >= y0 as f32 && py <= y1 as f32 {
            Some(crate::ops::shapes::ShapeHandle::Move)
        } else {
            None
        }
    }

    fn resize_selection_mask_from_handle(
        start_mask: &image::GrayImage,
        bounds: (u32, u32, u32, u32),
        handle: crate::ops::shapes::ShapeHandle,
        pos: (i32, i32),
        shift: bool,
        anti_alias: bool,
    ) -> Option<image::GrayImage> {
        use image::imageops;
        let (cw, ch) = start_mask.dimensions();
        let (x0, y0, x1, y1) = bounds;
        let mut nx0 = x0 as i32;
        let mut ny0 = y0 as i32;
        let mut nx1 = x1 as i32;
        let mut ny1 = y1 as i32;
        match handle {
            crate::ops::shapes::ShapeHandle::TopLeft => { nx0 = pos.0; ny0 = pos.1; }
            crate::ops::shapes::ShapeHandle::TopRight => { nx1 = pos.0; ny0 = pos.1; }
            crate::ops::shapes::ShapeHandle::BottomRight => { nx1 = pos.0; ny1 = pos.1; }
            crate::ops::shapes::ShapeHandle::BottomLeft => { nx0 = pos.0; ny1 = pos.1; }
            crate::ops::shapes::ShapeHandle::Top => ny0 = pos.1,
            crate::ops::shapes::ShapeHandle::Right => nx1 = pos.0,
            crate::ops::shapes::ShapeHandle::Bottom => ny1 = pos.1,
            crate::ops::shapes::ShapeHandle::Left => nx0 = pos.0,
            _ => return None,
        }
        if shift && matches!(handle, crate::ops::shapes::ShapeHandle::TopLeft | crate::ops::shapes::ShapeHandle::TopRight | crate::ops::shapes::ShapeHandle::BottomLeft | crate::ops::shapes::ShapeHandle::BottomRight) {
            let w = (nx1 - nx0).abs();
            let h = (ny1 - ny0).abs();
            let s = w.max(h).max(1);
            match handle {
                crate::ops::shapes::ShapeHandle::TopLeft => { nx0 = nx1 - s; ny0 = ny1 - s; }
                crate::ops::shapes::ShapeHandle::TopRight => { nx1 = nx0 + s; ny0 = ny1 - s; }
                crate::ops::shapes::ShapeHandle::BottomRight => { nx1 = nx0 + s; ny1 = ny0 + s; }
                crate::ops::shapes::ShapeHandle::BottomLeft => { nx0 = nx1 - s; ny1 = ny0 + s; }
                _ => {}
            }
        }
        let left = nx0.min(nx1).clamp(0, cw as i32);
        let right = nx0.max(nx1).clamp(0, cw as i32);
        let top = ny0.min(ny1).clamp(0, ch as i32);
        let bottom = ny0.max(ny1).clamp(0, ch as i32);
        if right <= left || bottom <= top {
            return None;
        }
        let src = imageops::crop_imm(start_mask, x0, y0, x1 - x0, y1 - y0).to_image();
        let resized = imageops::resize(
            &src,
            (right - left) as u32,
            (bottom - top) as u32,
            if anti_alias {
                imageops::FilterType::Triangle
            } else {
                imageops::FilterType::Nearest
            },
        );
        let mut out = image::GrayImage::from_pixel(cw, ch, image::Luma([0u8]));
        imageops::overlay(&mut out, &resized, left as i64, top as i64);
        Some(out)
    }
}

