//! Floating panel placement, snapping and portable workspace profiles.
use egui::{Context, Id, Pos2, Rect, Vec2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn guide_color(ctx: &Context) -> egui::Color32 {
    if ctx.global_style().visuals.dark_mode {
        egui::Color32::from_rgba_premultiplied(32, 32, 32, 44)
    } else {
        egui::Color32::from_rgba_premultiplied(14, 14, 14, 36)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkspaceSettings {
    pub snapping: bool,
    pub guides: bool,
    pub snap_distance: f32,
    pub panel_gap: f32,
    pub remember_positions: bool,
}
impl Default for WorkspaceSettings {
    fn default() -> Self {
        Self {
            snapping: true,
            guides: true,
            snap_distance: 8.0,
            panel_gap: 8.0,
            remember_positions: true,
        }
    }
}
impl WorkspaceSettings {
    pub fn sanitize(&mut self) {
        self.snap_distance = if self.snap_distance.is_finite() {
            self.snap_distance.clamp(3.0, 20.0)
        } else {
            8.0
        };
        self.panel_gap = if self.panel_gap.is_finite() {
            self.panel_gap.clamp(0.0, 24.0)
        } else {
            8.0
        };
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PanelPlacement {
    pub panel: String,
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub visible: bool,
    pub right_anchored: bool,
    #[serde(default)]
    pub expanded: Option<bool>,
    #[serde(default)]
    pub section_mask: Option<u8>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceLayout {
    pub version: u32,
    pub name: String,
    pub viewport: [f32; 2],
    pub panels: Vec<PanelPlacement>,
}
impl WorkspaceLayout {
    pub fn validate(&mut self) -> Result<(), String> {
        if self.version != 1 {
            return Err("Unsupported workspace version".into());
        }
        self.name = self.name.trim().chars().take(48).collect();
        if self.name.is_empty() {
            return Err("Enter a workspace name".into());
        }
        if !self.viewport.iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err("Invalid workspace dimensions".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        self.panels.retain(|p| {
            matches!(
                p.panel.as_str(),
                "Tools" | "Layers" | "History" | "Colors" | "Palette" | "ScriptEditor"
            )
        });
        if self.panels.len() > 6
            || self.panels.iter().any(|p| {
                !seen.insert(p.panel.clone())
                    || !p.position.iter().all(|v| v.is_finite())
                    || !p
                        .size
                        .iter()
                        .all(|v| v.is_finite() && *v > 0.0 && *v <= 8192.0)
            })
        {
            return Err("Invalid panel placement".into());
        }
        Ok(())
    }
    pub fn panel_position(&self, panel: &PanelPlacement, viewport: Rect) -> Pos2 {
        let x = if panel.right_anchored {
            viewport.right() - (self.viewport[0] - panel.position[0])
        } else {
            viewport.left() + panel.position[0]
        };
        reachable(
            Pos2::new(x, viewport.top() + panel.position[1]),
            Vec2::from(panel.size),
            viewport,
        )
    }
}

#[derive(Clone, Default)]
struct Snapshot {
    viewport: [f32; 2],
    panels: BTreeMap<String, PanelPlacement>,
}
#[derive(Clone, Default)]
pub enum Command {
    Apply(WorkspaceLayout),
    #[default]
    Reset,
}
pub fn request(ctx: &Context, command: Command) {
    ctx.data_mut(|d| d.insert_temp(Id::new("paintfe_workspace_command"), command));
    ctx.request_repaint();
}
pub fn take_command(ctx: &Context) -> Option<Command> {
    ctx.data_mut(|d| d.remove_temp(Id::new("paintfe_workspace_command")))
}
pub fn begin_frame(ctx: &Context, settings: &WorkspaceSettings, visibility: &[(&str, bool)]) {
    let viewport = ctx.content_rect();
    ctx.data_mut(|d| {
        d.insert_temp(Id::new("paintfe_workspace_settings"), settings.clone());
        let mut snapshot = d
            .get_temp::<Snapshot>(Id::new("paintfe_workspace_snapshot"))
            .unwrap_or_default();
        snapshot.viewport = [viewport.width(), viewport.height()];
        for panel in snapshot.panels.values_mut() {
            panel.visible = visibility
                .iter()
                .any(|(id, visible)| *id == panel.panel && *visible);
        }
        d.insert_temp(Id::new("paintfe_workspace_snapshot"), snapshot);
    });
}
pub fn register(ctx: &Context, panel: &str, rect: Rect) {
    let viewport = ctx.content_rect();
    ctx.data_mut(|d| {
        let mut snapshot = d
            .get_temp::<Snapshot>(Id::new("paintfe_workspace_snapshot"))
            .unwrap_or_default();
        snapshot.panels.insert(
            panel.to_string(),
            PanelPlacement {
                panel: panel.to_string(),
                position: [rect.left() - viewport.left(), rect.top() - viewport.top()],
                size: [rect.width(), rect.height()],
                visible: true,
                right_anchored: viewport.right() - rect.right() < rect.left() - viewport.left(),
                expanded: None,
                section_mask: None,
            },
        );
        d.insert_temp(Id::new("paintfe_workspace_snapshot"), snapshot);
    });
}
pub fn color_sections(ctx: &Context, expanded: bool, mask: u8) {
    ctx.data_mut(|d| {
        if let Some(mut snapshot) = d.get_temp::<Snapshot>(Id::new("paintfe_workspace_snapshot")) {
            if let Some(panel) = snapshot.panels.get_mut("Colors") {
                panel.expanded = Some(expanded);
                panel.section_mask = Some(mask & 7);
            }
            d.insert_temp(Id::new("paintfe_workspace_snapshot"), snapshot);
        }
    });
}

pub fn clear_gestures(ctx: &Context) {
    ctx.data_mut(|d| {
        d.remove::<Snapshot>(Id::new("paintfe_workspace_snapshot"));
        for panel in [
            "Tools",
            "Layers",
            "History",
            "Colors",
            "Palette",
            "ScriptEditor",
        ] {
            d.remove::<SnapState>(Id::new(("paintfe_panel_snap", panel)));
            d.remove::<Vec2>(Id::new(("floating_header_drag", panel)));
            d.remove::<Pos2>(Id::new(("floating_header_pointer", panel)));
            d.remove::<(Pos2, Vec2)>(Id::new(("floating_resize_capture", panel)));
        }
    });
}

pub fn capture(ctx: &Context, name: String) -> Result<WorkspaceLayout, String> {
    let snapshot = ctx
        .data(|d| d.get_temp::<Snapshot>(Id::new("paintfe_workspace_snapshot")))
        .unwrap_or_default();
    let mut layout = WorkspaceLayout {
        version: 1,
        name,
        viewport: snapshot.viewport,
        panels: snapshot.panels.into_values().collect(),
    };
    layout.validate()?;
    Ok(layout)
}

pub fn reachable(position: Pos2, size: Vec2, viewport: Rect) -> Pos2 {
    // Keep the entire header reachable even when a panel is taller than the viewport.
    let left = viewport.left() + 8.0;
    let top = (viewport.top() + 72.0)
        .min(viewport.bottom() - 28.0)
        .max(viewport.top());
    Pos2::new(
        position
            .x
            .clamp(left, (viewport.right() - size.x - 8.0).max(left)),
        position.y.clamp(top, (viewport.bottom() - 28.0).max(top)),
    )
}

#[derive(Clone, Copy, Default)]
struct SnapState {
    x: Option<f32>,
    y: Option<f32>,
    raw: Option<Pos2>,
}
fn closest(value: f32, targets: &[f32], distance: f32, locked: Option<f32>) -> Option<f32> {
    if let Some(target) = locked
        && (value - target).abs() < distance * 1.6
    {
        return Some(target);
    }
    targets
        .iter()
        .copied()
        .filter(|t| (*t - value).abs() <= distance)
        .min_by(|a, b| (a - value).abs().total_cmp(&(b - value).abs()))
}
pub fn place(ctx: &Context, panel: &str, position: Pos2, size: Vec2) -> Pos2 {
    let _profile = super::perf::Scope::new(14);
    let viewport = ctx.content_rect();
    let size = ctx
        .data(|d| d.get_temp::<Snapshot>(Id::new("paintfe_workspace_snapshot")))
        .and_then(|s| s.panels.get(panel).map(|p| Vec2::from(p.size)))
        .unwrap_or(size);
    let delta = ctx
        .data_mut(|d| d.remove_temp::<Vec2>(Id::new(("floating_header_drag", panel))))
        .unwrap_or_default();
    let state_id = Id::new(("paintfe_panel_snap", panel));
    let mut state = ctx
        .data(|d| d.get_temp::<SnapState>(state_id))
        .unwrap_or_default();
    let active = ctx
        .data(|d| d.get_temp::<Pos2>(Id::new(("floating_header_pointer", panel))))
        .is_some();
    let mut raw = if active || delta != Vec2::ZERO {
        state.raw.unwrap_or(position) + delta
    } else {
        position
    };
    raw = reachable(raw, size, viewport);
    let config = ctx
        .data(|d| d.get_temp::<WorkspaceSettings>(Id::new("paintfe_workspace_settings")))
        .unwrap_or_default();
    let bypass = ctx.input(|i| i.modifiers.alt);
    let mut output = raw;
    if config.snapping && !bypass && (active || delta != Vec2::ZERO) {
        let snapshot = ctx
            .data(|d| d.get_temp::<Snapshot>(Id::new("paintfe_workspace_snapshot")))
            .unwrap_or_default();
        let mut xs = vec![viewport.left() + 12.0, viewport.right() - size.x - 12.0];
        let mut ys = vec![viewport.top() + 128.0, viewport.bottom() - size.y - 12.0];
        for other in snapshot
            .panels
            .values()
            .filter(|p| p.panel != panel && p.visible)
        {
            let x = viewport.left() + other.position[0];
            let y = viewport.top() + other.position[1];
            xs.extend([
                x,
                x + other.size[0] - size.x,
                x + other.size[0] + config.panel_gap,
                x - size.x - config.panel_gap,
            ]);
            ys.extend([
                y,
                y + other.size[1] - size.y,
                y + other.size[1] + config.panel_gap,
                y - size.y - config.panel_gap,
            ]);
        }
        state.x = closest(raw.x, &xs, config.snap_distance, state.x);
        state.y = closest(raw.y, &ys, config.snap_distance, state.y);
        output.x = state.x.unwrap_or(raw.x);
        output.y = state.y.unwrap_or(raw.y);
        if config.guides {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                Id::new("panel_alignment_guides"),
            ));
            let color = guide_color(ctx);
            if let Some(x) = state.x {
                painter.line_segment(
                    [
                        Pos2::new(x, viewport.top() + 70.0),
                        Pos2::new(x, viewport.bottom()),
                    ],
                    egui::Stroke::new(1.0, color),
                );
            }
            if let Some(y) = state.y {
                painter.line_segment(
                    [
                        Pos2::new(viewport.left(), y),
                        Pos2::new(viewport.right(), y),
                    ],
                    egui::Stroke::new(1.0, color),
                );
            }
        }
    } else {
        state.x = None;
        state.y = None;
    }
    let settle = super::polish::animate(
        ctx,
        Id::new(("panel_snap_feedback", panel)),
        if state.x.is_some() || state.y.is_some() {
            1.0
        } else {
            0.0
        },
        super::polish::MotionKind::Snapping,
    );
    if settle > 0.0 {
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            Id::new("panel_snap_feedback"),
        ));
        let color = guide_color(ctx).gamma_multiply(settle);
        painter.rect_stroke(
            Rect::from_min_size(output, size).expand(1.0),
            6.0,
            egui::Stroke::new(1.0, color),
            egui::StrokeKind::Outside,
        );
    }
    state.raw = (active || delta != Vec2::ZERO).then_some(raw);
    ctx.data_mut(|d| d.insert_temp(state_id, state));
    // The snapped target follows input immediately; only its border feedback settles.
    output
}

pub fn preferences(
    ui: &mut egui::Ui,
    settings: &mut WorkspaceSettings,
    profiles: &mut Vec<WorkspaceLayout>,
) -> bool {
    let before = settings.clone();
    let mut changed = false;
    ui.checkbox(
        &mut settings.snapping,
        "Snap panels to edges and each other",
    );
    ui.checkbox(&mut settings.guides, "Show alignment guides while dragging");
    ui.checkbox(
        &mut settings.remember_positions,
        "Remember panel positions and sizes",
    );
    ui.add(
        egui::Slider::new(&mut settings.snap_distance, 3.0..=20.0)
            .text("Snap distance")
            .suffix(" px"),
    );
    ui.add(
        egui::Slider::new(&mut settings.panel_gap, 0.0..=24.0)
            .text("Panel spacing")
            .suffix(" px"),
    );
    ui.horizontal(|ui| {
        let name_id = Id::new("workspace_new_name");
        let mut name = ui
            .ctx()
            .data(|d| d.get_temp::<String>(name_id))
            .unwrap_or_default();
        ui.add(
            egui::TextEdit::singleline(&mut name)
                .hint_text("Workspace name")
                .desired_width(140.0),
        );
        if ui.button("Save Current Layout").clicked() {
            match capture(ui.ctx(), name.clone()) {
                Ok(profile) => {
                    changed |= store_profile(profiles, profile);
                }
                Err(error) => {
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(Id::new("workspace_error"), error));
                }
            }
        }
        ui.ctx().data_mut(|d| d.insert_temp(name_id, name));
    });
    let mut remove = None;
    for (index, profile) in profiles.iter_mut().enumerate() {
        ui.push_id(index, |ui| {
            ui.horizontal_wrapped(|ui| {
                let old_name = profile.name.clone();
                let edit =
                    ui.add(egui::TextEdit::singleline(&mut profile.name).desired_width(140.0));
                if profile.name != old_name {
                    profile.name = profile.name.chars().take(48).collect();
                    changed = true;
                }
                if edit.lost_focus() && profile.name.trim().is_empty() {
                    profile.name = format!("Workspace {}", index + 1);
                    changed = true;
                }
                if ui.button("Apply").clicked() {
                    request(ui.ctx(), Command::Apply(profile.clone()));
                }
                if ui.button("Delete").clicked() {
                    remove = Some(index);
                }
                #[cfg(not(target_arch = "wasm32"))]
                if ui.button("Export…").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("PaintFE Workspace", &["json"])
                        .set_file_name("workspace.json")
                        .save_file()
                {
                    let result = serde_json::to_string_pretty(profile)
                        .map_err(|e| e.to_string())
                        .and_then(|s| std::fs::write(path, s).map_err(|e| e.to_string()));
                    if let Err(error) = result {
                        ui.ctx()
                            .data_mut(|d| d.insert_temp(Id::new("workspace_error"), error));
                    }
                }
            });
        });
    }
    if let Some(index) = remove {
        profiles.remove(index);
        changed = true;
    }
    ui.horizontal(|ui| {
        if ui.button("Restore Default Layout").clicked() {
            request(ui.ctx(), Command::Reset);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if ui.button("Import Layout…").clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("PaintFE Workspace", &["json"])
                .pick_file()
        {
            let result = std::fs::metadata(&path)
                .map_err(|e| e.to_string())
                .and_then(|m| {
                    if m.len() <= 65536 {
                        std::fs::read_to_string(path).map_err(|e| e.to_string())
                    } else {
                        Err("Workspace file is too large".into())
                    }
                })
                .and_then(|s| {
                    serde_json::from_str::<WorkspaceLayout>(&s).map_err(|e| e.to_string())
                })
                .and_then(|mut p| {
                    p.validate()?;
                    Ok(p)
                });
            match result {
                Ok(profile) => {
                    changed |= store_profile(profiles, profile);
                }
                Err(error) => {
                    ui.ctx()
                        .data_mut(|d| d.insert_temp(Id::new("workspace_error"), error));
                }
            }
        }
    });
    if let Some(error) = ui
        .ctx()
        .data(|d| d.get_temp::<String>(Id::new("workspace_error")))
    {
        ui.colored_label(ui.visuals().error_fg_color, error);
        if ui.small_button("Dismiss").clicked() {
            ui.ctx()
                .data_mut(|d| d.remove::<String>(Id::new("workspace_error")));
        }
    }
    settings.sanitize();
    changed || *settings != before
}

fn store_profile(profiles: &mut Vec<WorkspaceLayout>, profile: WorkspaceLayout) -> bool {
    if let Some(old) = profiles.iter_mut().find(|p| p.name == profile.name) {
        *old = profile;
    } else if profiles.len() < 32 {
        profiles.push(profile);
    } else {
        return false;
    }
    true
}

/// Snap the dragged bottom/right edges without delaying or smoothing pointer input.
pub fn resize(ctx: &Context, panel: &str, rect: Rect, size: Vec2, min: Vec2) -> Vec2 {
    let config = ctx
        .data(|d| d.get_temp::<WorkspaceSettings>(Id::new("paintfe_workspace_settings")))
        .unwrap_or_default();
    if !config.snapping || ctx.input(|i| i.modifiers.alt) {
        return size.max(min);
    }
    let viewport = ctx.content_rect();
    let snapshot = ctx
        .data(|d| d.get_temp::<Snapshot>(Id::new("paintfe_workspace_snapshot")))
        .unwrap_or_default();
    let mut xs = vec![viewport.right() - 12.0];
    let mut ys = vec![viewport.bottom() - 12.0];
    for p in snapshot
        .panels
        .values()
        .filter(|p| p.panel != panel && p.visible)
    {
        xs.extend([
            viewport.left() + p.position[0] - config.panel_gap,
            viewport.left() + p.position[0] + p.size[0],
        ]);
        ys.extend([
            viewport.top() + p.position[1] - config.panel_gap,
            viewport.top() + p.position[1] + p.size[1],
        ]);
    }
    let x = closest(rect.left() + size.x, &xs, config.snap_distance, None)
        .filter(|x| *x - rect.left() >= min.x);
    let y = closest(rect.top() + size.y, &ys, config.snap_distance, None)
        .filter(|y| *y - rect.top() >= min.y);
    if config.guides {
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            Id::new("panel_alignment_guides"),
        ));
        let stroke = egui::Stroke::new(1.0, guide_color(ctx));
        if let Some(x) = x {
            painter.line_segment(
                [
                    Pos2::new(x, viewport.top()),
                    Pos2::new(x, viewport.bottom()),
                ],
                stroke,
            );
        }
        if let Some(y) = y {
            painter.line_segment(
                [
                    Pos2::new(viewport.left(), y),
                    Pos2::new(viewport.right(), y),
                ],
                stroke,
            );
        }
    }
    Vec2::new(
        x.map_or(size.x, |x| x - rect.left()),
        y.map_or(size.y, |y| y - rect.top()),
    )
    .max(min)
}

#[cfg(test)]
mod polish_tests {
    use super::*;
    fn layout() -> WorkspaceLayout {
        WorkspaceLayout {
            version: 1,
            name: " Test ".into(),
            viewport: [1000.0, 800.0],
            panels: vec![PanelPlacement {
                panel: "Layers".into(),
                position: [750.0, 140.0],
                size: [220.0, 200.0],
                visible: true,
                right_anchored: true,
                expanded: None,
                section_mask: None,
            }],
        }
    }
    #[test]
    fn validates_names_versions_duplicates_and_sizes() {
        let mut p = layout();
        p.validate().unwrap();
        assert_eq!(p.name, "Test");
        p.panels.push(p.panels[0].clone());
        assert!(p.validate().is_err());
        let mut p = layout();
        p.panels[0].size[0] = f32::NAN;
        assert!(p.validate().is_err());
        let mut p = layout();
        p.version = 2;
        assert!(p.validate().is_err());
    }
    #[test]
    fn right_anchor_and_tiny_viewports_keep_headers_reachable() {
        let p = layout();
        let pos = p.panel_position(
            &p.panels[0],
            Rect::from_min_size(Pos2::ZERO, egui::vec2(1400.0, 900.0)),
        );
        assert_eq!(pos, Pos2::new(1150.0, 140.0));
        let tiny = Rect::from_min_size(Pos2::ZERO, egui::vec2(100.0, 50.0));
        let pos = reachable(Pos2::new(-500.0, 4000.0), egui::vec2(300.0, 1000.0), tiny);
        assert!(pos.y >= 0.0 && pos.y <= 22.0);
    }
    #[test]
    fn snap_hysteresis_and_capture_keep_outer_size_and_sections() {
        assert_eq!(closest(107.0, &[100.0], 8.0, None), Some(100.0));
        assert_eq!(closest(111.0, &[100.0], 8.0, Some(100.0)), Some(100.0));
        assert_eq!(closest(114.0, &[100.0], 8.0, Some(100.0)), None);
        let ctx = Context::default();
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 800.0))),
                ..Default::default()
            },
            |_| {
                begin_frame(&ctx, &WorkspaceSettings::default(), &[("Colors", true)]);
                register(
                    &ctx,
                    "Colors",
                    Rect::from_min_size(Pos2::new(20.0, 130.0), egui::vec2(240.0, 310.0)),
                );
                color_sections(&ctx, true, 3);
                let p = capture(&ctx, "Paint".into()).unwrap();
                assert_eq!(p.panels[0].size, [240.0, 310.0]);
                assert_eq!(p.panels[0].expanded, Some(true));
                assert_eq!(p.panels[0].section_mask, Some(3));
                begin_frame(&ctx, &WorkspaceSettings::default(), &[("Colors", false)]);
                assert!(!capture(&ctx, "Paint".into()).unwrap().panels[0].visible);
            },
        );
    }
    #[test]
    fn resize_snaps_edges_and_alt_bypasses() {
        let ctx = Context::default();
        let viewport = Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 800.0));
        for alt in [false, true] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(viewport),
                    modifiers: egui::Modifiers {
                        alt,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                |_| {
                    begin_frame(&ctx, &WorkspaceSettings::default(), &[]);
                    let rect =
                        Rect::from_min_size(Pos2::new(100.0, 100.0), egui::vec2(200.0, 200.0));
                    let size = resize(
                        &ctx,
                        "Palette",
                        rect,
                        egui::vec2(883.0, 683.0),
                        egui::vec2(80.0, 60.0),
                    );
                    assert_eq!(
                        size,
                        if alt {
                            egui::vec2(883.0, 683.0)
                        } else {
                            egui::vec2(888.0, 688.0)
                        }
                    );
                },
            );
        }
    }
    #[test]
    fn profile_replacement_and_limit() {
        let mut profiles = vec![];
        for n in 0..32 {
            let mut p = layout();
            p.name = n.to_string();
            assert!(store_profile(&mut profiles, p));
        }
        assert!(!store_profile(&mut profiles, layout()));
        let mut p = layout();
        p.name = "0".into();
        p.panels[0].visible = false;
        assert!(store_profile(&mut profiles, p));
        assert_eq!(profiles.len(), 32);
        assert!(!profiles[0].panels[0].visible);
    }
}
