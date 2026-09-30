// ============================================================================
// ACTION DIALOG WIDGETS — shared chrome for the import / oversized-paste /
// unsaved-changes dialogs (compact card, icon tiles, animated rows).
// ============================================================================

use crate::assets::{Assets, Icon};

/// One row in an action-list dialog ("Open", "Add as Layer", …).
pub struct DialogAction {
    pub icon: Icon,
    pub title: String,
    pub desc: String,
    pub recommended: bool,
}

impl DialogAction {
    pub fn new(icon: Icon, title: impl Into<String>, desc: impl Into<String>) -> Self {
        Self {
            icon,
            title: title.into(),
            desc: desc.into(),
            recommended: false,
        }
    }

    pub fn recommended(mut self) -> Self {
        self.recommended = true;
        self
    }
}

/// One button in a confirm dialog footer ("Save", "Don't Save", …).
pub struct ConfirmButton {
    pub label: String,
    pub icon: Option<Icon>,
    pub primary: bool,
}

impl ConfirmButton {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            icon: None,
            primary: false,
        }
    }

    pub fn icon(mut self, icon: Icon) -> Self {
        self.icon = Some(icon);
        self
    }

    pub fn primary(mut self) -> Self {
        self.primary = true;
        self
    }
}

/// Card metrics: compact, consistent with the app's widget radius scale.
const CARD_WIDTH: f32 = 380.0;
const ROW_HEIGHT: f32 = 52.0;
const TILE: f32 = 30.0;
const ROW_RADIUS: f32 = 6.0;

/// Colors for the dialog card, derived from the current visuals.
struct DialogTint {
    fill: Color32,
    stroke: Color32,
    text: Color32,
    muted: Color32,
    accent: Color32,
    accent_faint: Color32,
    /// Readable accent for text/icons (the raw accent is too light on light
    /// themes to be legible on tinted pills).
    accent_text: Color32,
}

impl DialogTint {
    fn from_ui(ui: &egui::Ui) -> Self {
        let v = ui.visuals();
        let accent = v.selection.bg_fill;
        // The saturated accent (selection stroke) carries far more contrast
        // than the pale fill tint — use it for text, pills and borders.
        let line = v.selection.stroke.color;
        let accent_text = if v.dark_mode {
            line
        } else {
            Color32::from_rgba_premultiplied(
                (line.r() as f32 * 0.72) as u8,
                (line.g() as f32 * 0.72) as u8,
                (line.b() as f32 * 0.72) as u8,
                255,
            )
        };
        Self {
            fill: v.window_fill,
            stroke: v.widgets.noninteractive.bg_stroke.color,
            text: v.text_color(),
            muted: v.widgets.noninteractive.fg_stroke.color,
            accent,
            accent_faint: accent.gamma_multiply(0.16),
            accent_text,
        }
    }
}

/// Open animation for dialog content (fade-in over ~120 ms).
fn dialog_fade(ctx: &egui::Context, salt: &str) -> f32 {
    ctx.animate_value_with_time(egui::Id::new(("dialog_fade", salt)), 1.0_f32, 0.12)
}

/// Scale a `Color32`'s intensity (used for fade-in of dialog content).
fn faded(c: Color32, t: f32) -> Color32 {
    c.gamma_multiply(t)
}

/// Header row: tinted icon tile + bold title + close ✕.
/// Returns true when the close button is clicked.
pub fn dialog_card_header(ui: &mut egui::Ui, assets: &Assets, icon: Icon, title: &str) -> bool {
    let tint = DialogTint::from_ui(ui);
    let mut closed = false;
    ui.horizontal(|ui| {
        icon_tile(ui, assets, icon, tint.accent_text, tint.accent_faint);
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(title)
                .strong()
                .size(15.0)
                .color(tint.text),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let resp = close_button(ui, tint.muted);
            if resp.clicked() {
                closed = true;
            }
        });
    });
    ui.add_space(4.0);
    ui.separator();
    ui.add_space(4.0);
    closed
}

/// Rounded icon tile (accent-tinted) used in headers and rows.
fn icon_tile(ui: &mut egui::Ui, assets: &Assets, icon: Icon, icon_color: Color32, tint: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(TILE, TILE), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same(6), tint);
    if let Some(tex) = assets.get_texture(icon) {
        let inner = rect.shrink(7.0);
        ui.painter().image(
            tex.id(),
            inner,
            egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            icon_color,
        );
    }
}

/// Small ✕ close button — bare glyph, subtle hover plate (concept style).
fn close_button(ui: &mut egui::Ui, muted: Color32) -> egui::Response {
    let size = egui::vec2(20.0, 20.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let t = ui
        .ctx()
        .animate_bool(response.id, response.hovered() || response.is_pointer_button_down_on());
    if t > 0.01 {
        ui.painter()
            .rect_filled(rect, 5.0, muted.gamma_multiply(0.12 * t));
    }
    let stroke = Stroke::new(1.5, muted);
    let c = rect.center();
    let r = 4.5;
    ui.painter()
        .line_segment([c - egui::vec2(r, r), c + egui::vec2(r, r)], stroke);
    ui.painter()
        .line_segment([c + egui::vec2(r, -r), c - egui::vec2(r, -r)], stroke);
    response
}

/// Style A — question + vertical action rows.
///
/// Returns the index of the clicked action. `caption` (optional) is a small
/// muted line under the question, e.g. "photo.png · 2 more waiting".
pub fn action_list(
    ui: &mut egui::Ui,
    assets: &Assets,
    question: &str,
    caption: Option<&str>,
    actions: &[DialogAction],
) -> Option<usize> {
    let tint = DialogTint::from_ui(ui);
    let fade = dialog_fade(ui.ctx(), question);

    ui.label(
        egui::RichText::new(question)
            .size(12.5)
            .color(faded(tint.text, fade)),
    );
    if let Some(caption) = caption {
        ui.add_space(1.0);
        ui.label(
            egui::RichText::new(caption)
                .small()
                .color(faded(tint.muted, fade)),
        );
    }
    ui.add_space(8.0);

    let mut chosen = None;
    for (idx, action) in actions.iter().enumerate() {
        if action_row(ui, assets, idx, action, &tint, fade) {
            chosen = Some(idx);
        }
        if idx + 1 < actions.len() {
            ui.add_space(5.0);
        }
    }
    chosen
}

/// One animated action row (hover tint + chevron nudge, press feedback).
fn action_row(
    ui: &mut egui::Ui,
    assets: &Assets,
    _idx: usize,
    action: &DialogAction,
    tint: &DialogTint,
    fade: f32,
) -> bool {
    let width = ui.available_width().min(CARD_WIDTH);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, ROW_HEIGHT), egui::Sense::click());
    let id = response.id;

    let hover = ui.ctx().animate_bool(id.with("hover"), response.hovered());
    let press = ui
        .ctx()
        .animate_bool(id.with("press"), response.is_pointer_button_down_on());

    // Background: subtle hover tint; the recommended row is tinted from the start.
    let base_tint = if action.recommended { 0.85 } else { 0.0 };
    let tint_amt = (base_tint + (1.0 - base_tint) * hover).clamp(0.0, 1.0) - press * 0.15;
    let fill = Color32::from_rgba_premultiplied(
        tint.accent.r(),
        tint.accent.g(),
        tint.accent.b(),
        (tint_amt * 26.0) as u8,
    );
    ui.painter()
        .rect_filled(rect, ROW_RADIUS, fill);

    // Border: accent for the recommended row (brightens on hover), hairline otherwise.
    let border = if action.recommended {
        Stroke::new(1.2, tint.accent_text.gamma_multiply(0.75 + 0.25 * hover))
    } else {
        Stroke::new(1.0, tint.stroke.gamma_multiply(0.55 + 0.45 * hover))
    };
    ui.painter()
        .rect_stroke(rect, ROW_RADIUS, border, egui::StrokeKind::Middle);

    // Icon tile.
    let tile = egui::Rect::from_min_size(
        rect.left_center() + egui::vec2(10.0, -TILE / 2.0),
        egui::vec2(TILE, TILE),
    );
    ui.painter().rect_filled(
        tile,
        6.0,
        if action.recommended {
            tint.accent_faint
        } else {
            tint.stroke.gamma_multiply(0.10)
        },
    );
    if let Some(tex) = assets.get_texture(action.icon) {
        ui.painter().image(
            tex.id(),
            tile.shrink(7.0),
            egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            if action.recommended {
                tint.accent_text
            } else {
                tint.text
            }
                .gamma_multiply(fade),
        );
    }

    // Title + description.
    let text_x = tile.max.x + 10.0;
    let title_pos = egui::pos2(text_x, rect.top() + 11.0);
    ui.painter().text(
        title_pos,
        egui::Align2::LEFT_TOP,
        &action.title,
        egui::FontId::proportional(13.5),
        tint.text.gamma_multiply(fade),
    );
    ui.painter().text(
        egui::pos2(text_x, rect.top() + 28.0),
        egui::Align2::LEFT_TOP,
        &action.desc,
        egui::FontId::proportional(10.5),
        tint.muted.gamma_multiply(fade),
    );

    // Chevron always sits at the far right; the "Recommended" pill sits left of it.
    let chevron = egui::pos2(rect.max.x - 12.0 + hover * 3.0, rect.center().y);
    if action.recommended {
        let pill_w = 74.0;
        let pill = egui::Rect::from_min_size(
            egui::pos2(chevron.x - 20.0 - pill_w, rect.center().y - 8.0),
            egui::vec2(pill_w, 16.0),
        );
        ui.painter()
            .rect_filled(pill, 5.0, tint.accent.gamma_multiply(0.35));
        ui.painter().text(
            pill.center(),
            egui::Align2::CENTER_CENTER,
            "Recommended",
            egui::FontId::proportional(10.0),
            tint.accent_text,
        );
    }
    ui.painter().text(
        chevron,
        egui::Align2::RIGHT_CENTER,
        "\u{203A}",
        egui::FontId::proportional(16.0),
        tint.muted.gamma_multiply(0.7 + 0.3 * hover),
    );

    response.clicked()
}

/// Style B — message lines + footer button row.
///
/// Returns the index of the clicked button.
pub fn confirm_row(
    ui: &mut egui::Ui,
    assets: &Assets,
    buttons: &[ConfirmButton],
) -> Option<usize> {
    let tint = DialogTint::from_ui(ui);
    ui.add_space(8.0);
    let mut chosen = None;
    ui.horizontal(|ui| {
        for (idx, button) in buttons.iter().enumerate() {
            if confirm_button(ui, assets, button, &tint) {
                chosen = Some(idx);
            }
            if idx + 1 < buttons.len() {
                ui.add_space(7.0);
            }
        }
    });
    chosen
}

/// One animated footer button (primary = filled accent, optional icon).
fn confirm_button(
    ui: &mut egui::Ui,
    assets: &Assets,
    button: &ConfirmButton,
    tint: &DialogTint,
) -> bool {
    let height = 28.0;
    let pad_x = 14.0;
    let icon_w = if button.icon.is_some() { 20.0 } else { 0.0 };
    let galley = ui.painter().layout_no_wrap(
        button.label.clone(),
        egui::FontId::proportional(12.5),
        tint.text,
    );
    let width = (galley.size().x + pad_x * 2.0 + icon_w).max(84.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
    let id = response.id;

    let hover = ui.ctx().animate_bool(id.with("hover"), response.hovered());
    let press = ui
        .ctx()
        .animate_bool(id.with("press"), response.is_pointer_button_down_on());

    let (fill, fg) = if button.primary {
        (
            tint.accent.gamma_multiply(0.88 + 0.12 * hover - 0.10 * press),
            Color32::WHITE,
        )
    } else {
        (
            tint.stroke.gamma_multiply(0.14 + 0.12 * hover - 0.08 * press),
            tint.text,
        )
    };
    ui.painter().rect_filled(rect, 5.0, fill);
    if !button.primary {
        ui.painter().rect_stroke(
            rect,
            5.0,
            Stroke::new(1.0, tint.stroke.gamma_multiply(0.8)),
            egui::StrokeKind::Middle,
        );
    }

    // Icon (e.g. the white save glyph on the accent button) + label.
    let mut x = rect.min.x + pad_x;
    if let Some(icon) = button.icon {
        if let Some(tex) = assets.get_texture(icon) {
            let tile = egui::Rect::from_min_size(
                egui::pos2(x, rect.center().y - 7.0),
                egui::vec2(14.0, 14.0),
            );
            ui.painter().image(
                tex.id(),
                tile,
                egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                fg,
            );
        }
        x += icon_w;
    }
    ui.painter().text(
        egui::pos2(x, rect.center().y),
        egui::Align2::LEFT_CENTER,
        &button.label,
        egui::FontId::proportional(12.5),
        fg,
    );

    response.clicked()
}
