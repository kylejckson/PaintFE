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

/// Card metrics shared by the dialog styles.
const CARD_WIDTH: f32 = 420.0;
const ROW_HEIGHT: f32 = 62.0;
const TILE: f32 = 36.0;

/// Colors for the dialog card, derived from the current visuals.
struct DialogTint {
    fill: Color32,
    stroke: Color32,
    text: Color32,
    muted: Color32,
    accent: Color32,
    accent_faint: Color32,
}

impl DialogTint {
    fn from_ui(ui: &egui::Ui) -> Self {
        let v = ui.visuals();
        Self {
            fill: v.window_fill,
            stroke: v.widgets.noninteractive.bg_stroke.color,
            text: v.text_color(),
            muted: v.widgets.noninteractive.fg_stroke.color,
            accent: v.selection.bg_fill,
            accent_faint: v.selection.bg_fill.gamma_multiply(0.12),
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
        icon_tile(ui, assets, icon, tint.accent, tint.accent_faint);
        ui.add_space(10.0);
        ui.label(
            egui::RichText::new(title)
                .strong()
                .size(17.0)
                .color(tint.text),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let resp = close_button(ui, tint.muted);
            if resp.clicked() {
                closed = true;
            }
        });
    });
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(6.0);
    closed
}

/// Rounded icon tile (accent-tinted) used in headers and rows.
fn icon_tile(ui: &mut egui::Ui, assets: &Assets, icon: Icon, accent: Color32, tint: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(TILE, TILE), egui::Sense::hover());
    ui.painter().rect_filled(rect, egui::CornerRadius::same(8), tint);
    if let Some(tex) = assets.get_texture(icon) {
        let inner = rect.shrink(8.0);
        ui.painter().image(
            tex.id(),
            inner,
            egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            accent,
        );
    }
}

/// Small ✕ close button with hover feedback.
fn close_button(ui: &mut egui::Ui, muted: Color32) -> egui::Response {
    let size = egui::vec2(24.0, 24.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let t = ui
        .ctx()
        .animate_bool(response.id, response.hovered() || response.is_pointer_button_down_on());
    let bg = muted.gamma_multiply(0.12 + 0.12 * t);
    ui.painter().rect_filled(rect, 6.0, bg);
    let stroke = Stroke::new(1.6, muted);
    let c = rect.center();
    let r = 5.0;
    ui.painter()
        .line_segment([c - egui::vec2(r, r), c + egui::vec2(r, r)], stroke);
    ui.painter()
        .line_segment([c + egui::vec2(r, -r), c - egui::vec2(r, -r)], stroke);
    response
}

/// Style A — question + vertical action rows.
///
/// Returns the index of the clicked action. `caption` (optional) is a small
/// muted line under the question, e.g. "File 2 of 3 · name.png".
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
            .size(13.5)
            .color(faded(tint.text, fade)),
    );
    if let Some(caption) = caption {
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(caption)
                .small()
                .color(faded(tint.muted, fade)),
        );
    }
    ui.add_space(10.0);

    let mut chosen = None;
    for (idx, action) in actions.iter().enumerate() {
        if action_row(ui, assets, idx, action, &tint, fade) {
            chosen = Some(idx);
        }
        if idx + 1 < actions.len() {
            ui.add_space(6.0);
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
        (tint_amt * 28.0) as u8,
    );
    ui.painter().rect_filled(rect, 10.0, fill);

    // Border: accent for the recommended row (brightens on hover), hairline otherwise.
    let border = if action.recommended {
        Stroke::new(1.5, tint.accent.gamma_multiply(0.65 + 0.35 * hover))
    } else {
        Stroke::new(1.0, tint.stroke.gamma_multiply(0.6 + 0.4 * hover))
    };
    ui.painter().rect_stroke(rect, 10.0, border, egui::StrokeKind::Middle);

    // Icon tile.
    let tile = egui::Rect::from_min_size(
        rect.left_center() + egui::vec2(12.0, -TILE / 2.0),
        egui::vec2(TILE, TILE),
    );
    ui.painter().rect_filled(
        tile,
        8.0,
        if action.recommended {
            tint.accent_faint
        } else {
            tint.stroke.gamma_multiply(0.12)
        },
    );
    if let Some(tex) = assets.get_texture(action.icon) {
        ui.painter().image(
            tex.id(),
            tile.shrink(8.0),
            egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            if action.recommended {
                tint.accent
            } else {
                tint.text
            }
                .gamma_multiply(fade),
        );
    }

    // Title + description.
    let text_x = tile.max.x + 12.0;
    let title_pos = egui::pos2(text_x, rect.top() + 15.0);
    ui.painter().text(
        title_pos,
        egui::Align2::LEFT_TOP,
        &action.title,
        egui::FontId::proportional(15.0),
        tint.text.gamma_multiply(fade),
    );
    ui.painter().text(
        egui::pos2(text_x, rect.top() + 35.0),
        egui::Align2::LEFT_TOP,
        &action.desc,
        egui::FontId::proportional(11.5),
        tint.muted.gamma_multiply(fade),
    );

    // "Recommended" pill (before the chevron).
    let mut chevron_x = rect.max.x - 16.0;
    if action.recommended {
        let galley = ui.painter().layout_no_wrap(
            "Recommended".to_string(),
            egui::FontId::proportional(10.5),
            tint.accent,
        );
        let pill = egui::Rect::from_min_size(
            egui::pos2(chevron_x - galley.size().x - 16.0, rect.center().y - 9.0),
            egui::vec2(galley.size().x + 16.0, 18.0),
        );
        ui.painter().rect_filled(pill, 9.0, tint.accent_faint);
        ui.painter().galley(
            pill.left_center() + egui::vec2(8.0, -galley.size().y / 2.0),
            galley,
            faded(tint.accent, fade),
        );
        chevron_x = pill.min.x - 10.0;
    }

    // Chevron, nudged right on hover.
    let chev = egui::pos2(chevron_x + hover * 3.0, rect.center().y);
    ui.painter().text(
        chev,
        egui::Align2::RIGHT_CENTER,
        "\u{203A}",
        egui::FontId::proportional(22.0),
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
    ui.add_space(10.0);
    let mut chosen = None;
    ui.horizontal(|ui| {
        for (idx, button) in buttons.iter().enumerate() {
            if confirm_button(ui, assets, button, &tint) {
                chosen = Some(idx);
            }
            if idx + 1 < buttons.len() {
                ui.add_space(8.0);
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
    let height = 34.0;
    let pad_x = 18.0;
    let icon_w = if button.icon.is_some() { 22.0 } else { 0.0 };
    let galley = ui.painter().layout_no_wrap(
        button.label.clone(),
        egui::FontId::proportional(13.5),
        tint.text,
    );
    let width = (galley.size().x + pad_x * 2.0 + icon_w).max(96.0);
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
            tint.stroke.gamma_multiply(0.16 + 0.12 * hover - 0.08 * press),
            tint.text,
        )
    };
    ui.painter().rect_filled(rect, 8.0, fill);
    if !button.primary {
        ui.painter().rect_stroke(
            rect,
            8.0,
            Stroke::new(1.0, tint.stroke.gamma_multiply(0.8)),
            egui::StrokeKind::Middle,
        );
    }

    // Icon (e.g. the white save glyph on the accent button) + label.
    let mut x = rect.min.x + pad_x;
    if let Some(icon) = button.icon {
        if let Some(tex) = assets.get_texture(icon) {
            let tile = egui::Rect::from_min_size(
                egui::pos2(x, rect.center().y - 8.0),
                egui::vec2(16.0, 16.0),
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
        egui::FontId::proportional(13.5),
        fg,
    );

    response.clicked()
}
