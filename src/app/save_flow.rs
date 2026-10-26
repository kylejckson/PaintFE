// All interactive saves share one loss check, snapshot and completion path.
#[derive(Clone)]
struct PendingSave {
    project_id: uuid::Uuid,
    action: crate::components::dialogs::SaveAction,
    update_path: bool,
}

#[derive(Clone)]
struct SaveWarning {
    request: PendingSave,
    /// Computing alpha/depth losses can prepare a large export image. Do it once.
    losses: Vec<String>,
}

struct SaveJob {
    request: PendingSave,
    revision: (u64, u64),
    editable_loss: bool,
}

struct SaveSequence {
    remaining: std::collections::VecDeque<uuid::Uuid>,
    waiting: Option<uuid::Uuid>,
    approved: std::collections::HashMap<uuid::Uuid, (u64, u64)>,
    exit: bool,
    close: bool,
}

enum SavePayload {
    Project(crate::io::PfeData),
    Image(crate::io::PreparedExportImage),
    Frames(
        Vec<image::RgbaImage>,
        Vec<crate::canvas::WebpFrameCompression>,
    ),
}

impl Project {
    fn save_revision(&self) -> (u64, u64) {
        (self.edit_revision, self.canvas_state.dirty_generation)
    }

    fn has_editable_structure(&self) -> bool {
        !self.canvas_state.guides.items.is_empty()
            || self.canvas_state.layers.len() > 1
            || !self.canvas_state.layer_folders.is_empty()
            || self.canvas_state.layers.iter().any(|layer| {
                !matches!(layer.content, crate::canvas::LayerContent::Raster)
                    || layer.mask.is_some()
                    || !layer.visible
                    || layer.opacity != 1.0
                    || layer.blend_mode != BlendMode::Normal
            })
    }

    fn save_losses(&mut self, action: &crate::components::dialogs::SaveAction) -> Vec<String> {
        if action.format == SaveFormat::Pfe {
            return Vec::new();
        }
        let mut losses = Vec::new();
        if !self.canvas_state.guides.items.is_empty() {
            losses.push(crate::t!("guides.save_loss"));
        }
        if self.has_editable_structure() {
            losses.push(crate::t!("save.loss_structure"));
        }
        if self.was_animated && !(action.animated && action.format.supports_animation()) {
            losses.push(crate::t!("save.loss_animation"));
        }
        if matches!(action.format, SaveFormat::Jpeg | SaveFormat::Gif)
            && self.canvas_state.composite().pixels().any(|pixel| {
                if action.format == SaveFormat::Gif {
                    pixel[3] > 0 && pixel[3] < 255
                } else {
                    pixel[3] < 255
                }
            })
        {
            losses.push(crate::t!("save.loss_alpha"));
        }
        let deep = self.canvas_state.layers.iter().any(|layer| {
            layer
                .deep_pixels
                .as_ref()
                .is_some_and(|pixels| !matches!(pixels, crate::experimental::DeepRgbaBuffer::U8(_)))
        });
        if deep {
            let prepared = crate::io::prepare_export_image(&self.canvas_state);
            let preserves_depth = !action.animated
                && matches!(
                    (&prepared, action.format),
                    (
                        crate::io::PreparedExportImage::Rgba16 { .. },
                        SaveFormat::Png | SaveFormat::Tiff | SaveFormat::Jxl(_)
                    ) | (
                        crate::io::PreparedExportImage::RgbaF32 { .. },
                        SaveFormat::Tiff
                    )
                );
            if !preserves_depth {
                losses.push(crate::t!("save.loss_depth"));
            }
        }
        losses
    }
}

impl PaintFEApp {
    fn quick_save_action(&self, idx: usize) -> Option<crate::components::dialogs::SaveAction> {
        let handler = &self.projects.get(idx)?.file_handler;
        Some(crate::components::dialogs::SaveAction {
            path: handler.current_path.clone()?,
            format: handler.last_format,
            quality: handler.last_quality,
            webp_lossless: handler.last_webp_lossless,
            tiff_compression: handler.last_tiff_compression,
            animated: handler.last_animated,
            animation_fps: handler.last_animation_fps,
            gif_colors: handler.last_gif_colors,
            gif_dither: handler.last_gif_dither,
        })
    }

    fn request_save(&mut self, request: PendingSave, current_time: f64) {
        self.finish_pending_line_edit();
        let Some(idx) = self
            .projects
            .iter()
            .position(|p| p.id == request.project_id)
        else {
            return;
        };
        if self.save_jobs.contains_key(&request.project_id) {
            return;
        }
        let losses = self.projects[idx].save_losses(&request.action);
        if !losses.is_empty() {
            self.save_warning = Some(SaveWarning { request, losses });
        } else {
            self.launch_save(request, current_time);
        }
    }

    fn launch_save(&mut self, request: PendingSave, current_time: f64) {
        let Some(project) = self
            .projects
            .iter_mut()
            .find(|p| p.id == request.project_id)
        else {
            return;
        };
        if self.save_jobs.contains_key(&project.id) {
            return;
        }
        project.canvas_state.ensure_all_text_layers_rasterized();
        let action = request.action.clone();
        let editable_loss = action.format != SaveFormat::Pfe && project.has_editable_structure();
        let payload = if action.format == SaveFormat::Pfe {
            SavePayload::Project(crate::io::build_pfe(&project.canvas_state))
        } else if action.animated && action.format.supports_animation() {
            SavePayload::Frames(
                project
                    .canvas_state
                    .layers
                    .iter()
                    .map(|l| l.pixels.to_rgba_image())
                    .collect(),
                project
                    .canvas_state
                    .layers
                    .iter()
                    .map(|l| l.webp_frame_compression)
                    .collect(),
            )
        } else {
            SavePayload::Image(crate::io::prepare_export_image(&project.canvas_state))
        };
        let project_id = project.id;
        self.save_jobs.insert(
            project_id,
            SaveJob {
                revision: project.save_revision(),
                request,
                editable_loss,
            },
        );
        if self.pending_io_ops == 0 {
            self.io_ops_start_time = Some(current_time);
        }
        self.pending_io_ops += 1;
        let sender = self.io_sender.clone();
        crate::par_compat::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| write_save_atomically(&action, payload)))
                .unwrap_or_else(|_| Err("The image encoder failed unexpectedly. Your original file has not been replaced.".into()));
            let _ = sender.send(IoResult::SaveFinished { project_id, result });
        });
    }

    fn finish_save(&mut self, project_id: uuid::Uuid, result: Result<(), String>) {
        let Some(job) = self.save_jobs.remove(&project_id) else {
            return;
        };
        if let Err(error) = result {
            self.save_sequence = None;
            self.force_exit = false;
            self.filter_error = Some(format!(
                "{}\n{}\n{}",
                crate::t!("save.failed"),
                job.request.action.path.display(),
                error
            ));
            return;
        }
        let action = job.request.action;
        let unchanged = self
            .projects
            .iter()
            .find(|p| p.id == project_id)
            .is_some_and(|project| project.save_revision() == job.revision);
        if let Some(project) = self.projects.iter_mut().find(|p| p.id == project_id) {
            // An export must not replace an existing editable project's save target.
            let retain_project =
                job.editable_loss && project.file_handler.last_format == SaveFormat::Pfe;
            if !retain_project {
                project.file_handler.current_path = Some(action.path.clone());
                project.file_handler.last_format = action.format;
                project.file_handler.last_quality = action.quality;
                project.file_handler.last_webp_lossless = action.webp_lossless;
                project.file_handler.last_tiff_compression = action.tiff_compression;
                project.file_handler.last_animated = action.animated;
                project.file_handler.last_animation_fps = action.animation_fps;
                project.file_handler.last_gif_colors = action.gif_colors;
                project.file_handler.last_gif_dither = action.gif_dither;
                if action.animated {
                    project.was_animated = true;
                    project.animation_fps = action.animation_fps;
                }
                if job.request.update_path {
                    project.path = Some(action.path.clone());
                    project.update_name_from_path();
                }
            }
            if unchanged && !job.editable_loss {
                project.mark_clean();
            }
        }
        self.settings.last_export_format = action.format.extension().into();
        self.settings.last_export_directory = action
            .path
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let recent = Self::normalize_open_path(&action.path)
            .to_string_lossy()
            .into_owned();
        self.settings.recent_files.retain(|p| p != &recent);
        self.settings.recent_files.insert(0, recent);
        self.settings.recent_files.truncate(20);
        self.settings.save();
        if self
            .save_sequence
            .as_ref()
            .is_some_and(|sequence| sequence.waiting == Some(project_id))
        {
            if !unchanged {
                self.save_sequence = None;
                self.filter_error = Some(crate::t!("save.changed"));
            } else if let Some(sequence) = &mut self.save_sequence {
                sequence.waiting = None;
                sequence.approved.insert(project_id, job.revision);
                if sequence.close
                    && let Some(idx) = self.projects.iter().position(|p| p.id == project_id)
                {
                    self.force_close_project(idx);
                }
            }
        }
    }

    fn begin_save_sequence(&mut self, ids: Vec<uuid::Uuid>, exit: bool, close: bool) {
        if self.save_sequence.is_none()
            && self.save_warning.is_none()
            && !self.save_file_dialog.open
        {
            self.save_sequence = Some(SaveSequence {
                remaining: ids.into(),
                waiting: None,
                approved: Default::default(),
                exit,
                close,
            });
        }
    }

    fn advance_save_sequence(&mut self, ctx: &egui::Context) {
        if self.save_warning.is_some() || self.save_file_dialog.open || self.filter_error.is_some()
        {
            return;
        }
        let Some(sequence) = &mut self.save_sequence else {
            return;
        };
        if sequence.waiting.is_some() {
            return;
        }
        if let Some(id) = sequence.remaining.pop_front() {
            let Some(idx) = self.projects.iter().position(|p| p.id == id) else {
                ctx.request_repaint();
                return;
            };
            if !self.projects[idx].is_dirty && !sequence.close {
                ctx.request_repaint();
                return;
            }
            sequence.waiting = Some(id);
            if self.save_jobs.contains_key(&id) {
                return;
            }
            if !self.prepare_heic_save(idx) {
                self.save_sequence = None;
                return;
            }
            if let Some(action) = self.quick_save_action(idx) {
                self.request_save(
                    PendingSave {
                        project_id: id,
                        action,
                        update_path: false,
                    },
                    ctx.input(|i| i.time),
                );
            } else {
                self.open_save_as_for_project(idx);
            }
        } else {
            let sequence = self.save_sequence.take().unwrap();
            if sequence.exit {
                let safe = self.save_jobs.is_empty()
                    && self.projects.iter().all(|p| {
                        !p.is_dirty || sequence.approved.get(&p.id) == Some(&p.save_revision())
                    });
                if safe {
                    self.force_exit = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                } else {
                    self.filter_error = Some(crate::t!("save.changed"));
                }
            }
        }
        ctx.request_repaint();
    }

    fn resolve_save_warning(&mut self, choice: usize, current_time: f64) {
        let Some(warning) = self.save_warning.take() else {
            return;
        };
        let request = warning.request;
        match choice {
            0 => {
                if let Some(idx) = self
                    .projects
                    .iter()
                    .position(|p| p.id == request.project_id)
                {
                    self.open_save_as_for_project(idx);
                    self.save_file_dialog
                        .set_from_path(&request.action.path.with_extension("pfe"));
                }
            }
            1 => self.launch_save(request, current_time),
            _ => {
                self.save_sequence = None;
            }
        }
    }

    fn show_save_warning(&mut self, ctx: &egui::Context) {
        let Some(warning) = self.save_warning.clone() else {
            return;
        };
        let request = warning.request;
        let Some(idx) = self
            .projects
            .iter()
            .position(|p| p.id == request.project_id)
        else {
            self.save_warning = None;
            return;
        };
        let losses = warning.losses;
        let name = self.projects[idx].name.clone();
        let mut choice = None;
        let response = crate::ui::polish::window(ctx, "save_loss_warning")
            .id(egui::Id::new("save_loss_warning"))
            .title_bar(false)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .frame(self.theme.dialog_frame())
            .show(ctx, |ui| {
                ui.set_width(470.0);
                if crate::ops::dialogs::dialog_card_header(
                    ui,
                    &self.assets,
                    Icon::DialogUnsavedWarning,
                    &crate::t!("save.warning"),
                ) {
                    choice = Some(2);
                }
                ui.label(format!("{} — {}", name, request.action.format.label()));
                ui.label(crate::t!("save.loss_intro"));
                for loss in &losses {
                    ui.label(format!("• {loss}"));
                }
                ui.label(crate::t!("save.keep_project"));
                ui.add_space(8.0);
                use crate::ops::dialogs::{ConfirmButton, confirm_row};
                let buttons = [
                    ConfirmButton::new(crate::t!("save.project"))
                        .icon(Icon::DialogSave)
                        .primary(),
                    ConfirmButton::new(crate::t!("save.image")),
                    ConfirmButton::new(crate::t!("common.cancel")),
                ];
                if let Some(selected) = confirm_row(ui, &self.assets, &buttons) {
                    choice = Some(selected);
                }
            });
        if let Some(response) = response {
            self.remember_ui_cursor_rect(response.response.rect);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            choice = Some(2);
        }
        if let Some(choice) = choice {
            self.resolve_save_warning(choice, ctx.input(|i| i.time));
        }
    }
}

fn write_save_atomically(
    action: &crate::components::dialogs::SaveAction,
    payload: SavePayload,
) -> Result<(), String> {
    let parent = action
        .path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let temporary = parent.join(format!(
        ".paintfe-save-{}.{}",
        uuid::Uuid::new_v4(),
        action.format.extension()
    ));
    // Reserve our own temporary name and never truncate the user's destination on failure.
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match payload {
            SavePayload::Project(data) => {
                crate::io::write_pfe(&data, &temporary).map_err(|e| e.to_string())?
            }
            SavePayload::Image(image) => crate::io::encode_prepared_and_write(
                image,
                &temporary,
                action.format,
                action.quality,
                action.tiff_compression,
                action.webp_lossless,
            )
            .map_err(|e| e.to_string())?,
            SavePayload::Frames(frames, modes) => match action.format {
                SaveFormat::Gif => crate::io::encode_animated_gif(
                    &frames,
                    action.animation_fps,
                    action.gif_colors,
                    action.gif_dither,
                    &temporary,
                ),
                SaveFormat::Png => {
                    crate::io::encode_animated_png(&frames, action.animation_fps, &temporary)
                }
                SaveFormat::Webp => crate::io::encode_animated_webp(
                    &frames,
                    &modes,
                    action.animation_fps,
                    action.quality,
                    &temporary,
                ),
                _ => Err("Format does not support animation".into()),
            }?,
        }
        std::fs::OpenOptions::new()
            .write(true)
            .open(&temporary)
            .and_then(|file| file.sync_all())
            .map_err(|e| e.to_string())?;
        std::fs::rename(&temporary, &action.path).map_err(|e| e.to_string())
    }))
    .unwrap_or_else(|_| {
        Err(
            "The image encoder failed unexpectedly. Your original file has not been replaced."
                .into(),
        )
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}
