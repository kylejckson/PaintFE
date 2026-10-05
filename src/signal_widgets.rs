//! Custom widget library implementing the Signal Grid design language.
//!
//! These widgets replace stock egui components with custom-painted equivalents
//! that match the website's visual language: badges, button variants, gradient
//! dividers, pill tab bars, card frames, and glow effects.

use eframe::egui::{self, Color32, CornerRadius, Rect, Response, Sense, Stroke, Ui, Vec2};
use eframe::epaint::Shadow;

use crate::signal_draw;
use crate::theme::Theme;

// ============================================================================
// SignalBadge — monospace uppercase tag with colored border
// ============================================================================

/// A small capsule label in monospace uppercase with a colored border and
/// semi-transparent fill. Inspired by the website's `.badge` class.
///
/// # Example
/// ```ignore
/// SignalBadge::new("TOOLS", theme.accent3).show(ui, theme);
/// ```
pub struct SignalBadge<'a> {
    text: &'a str,
    color: Color32,
}

impl<'a> SignalBadge<'a> {
    pub fn new(text: &'a str, color: Color32) -> Self {
        Self { text, color }
    }

    /// Render the badge into the UI and return the response.
    pub fn show(self, ui: &mut Ui, theme: &Theme) -> Response {
        let is_light = matches!(theme.mode, crate::theme::ThemeMode::Light);
        // In light mode, darken badge text significantly for readable contrast
        let text_color = if is_light {
            Color32::from_rgb(
                (self.color.r() as u16 / 3) as u8,
                (self.color.g() as u16 / 3) as u8,
                (self.color.b() as u16 / 3) as u8,
            )
        } else {
            self.color
        };
        let font = egui::FontId::monospace(Theme::FONT_LABEL);
        let text_upper = self.text.to_uppercase();
        let galley = ui.painter().layout_no_wrap(text_upper, font, text_color);

        let padding = Vec2::new(10.0, 3.0);
        let desired = galley.size() + padding * 2.0;
        let (rect, response) = ui.allocate_exact_size(desired, Sense::hover());

        if ui.is_rect_visible(rect) {
            let (fill_alpha, stroke_alpha) = if is_light {
                (60u8, 180u8) // strong fill + border in light mode for contrast
            } else {
                (20u8, 64u8) // original dark mode values
            };
            let fill = Color32::from_rgba_unmultiplied(
                self.color.r(),
                self.color.g(),
                self.color.b(),
                fill_alpha,
            );
            let stroke_color = Color32::from_rgba_unmultiplied(
                self.color.r(),
                self.color.g(),
                self.color.b(),
                stroke_alpha,
            );

            ui.painter().rect(
                rect,
                CornerRadius::same(theme.badge_rounding as u8),
                fill,
                Stroke::new(1.0, stroke_color),
                egui::StrokeKind::Middle,
            );

            let text_pos = rect.min + padding;
            ui.painter().galley(
                egui::pos2(text_pos.x, text_pos.y),
                galley,
                egui::Color32::TRANSPARENT,
            );
        }

        response
    }
}

// ============================================================================
// SignalButton — 3 style variants (Primary, Ghost, OutlineAccent)
// ============================================================================

/// Button style variant for `SignalButton`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SignalButtonStyle {
    /// Filled accent background, white text, glow on hover.
    Primary,
    /// Transparent with 1px border, subtle hover fill.
    Ghost,
    /// Accent3 (green) border and text, green tint on hover.
    OutlineAccent,
}

/// A styled button matching the Signal Grid design language.
///
/// # Example
/// ```ignore
/// if SignalButton::new("Apply").primary().show(ui, theme).clicked() { ... }
/// if SignalButton::new("Cancel").ghost().show(ui, theme).clicked() { ... }
/// ```
pub struct SignalButton<'a> {
    text: &'a str,
    style: SignalButtonStyle,
}

impl<'a> SignalButton<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            style: SignalButtonStyle::Ghost,
        }
    }

    pub fn primary(mut self) -> Self {
        self.style = SignalButtonStyle::Primary;
        self
    }

    pub fn ghost(mut self) -> Self {
        self.style = SignalButtonStyle::Ghost;
        self
    }

    pub fn outline_accent(mut self) -> Self {
        self.style = SignalButtonStyle::OutlineAccent;
        self
    }

    pub fn style(mut self, style: SignalButtonStyle) -> Self {
        self.style = style;
        self
    }

    /// Render the button and return the response.
    pub fn show(self, ui: &mut Ui, theme: &Theme) -> Response {
        let font = egui::FontId::proportional(Theme::FONT_BODY * theme.polish.text_scale);
        let text_galley = ui
            .painter()
            .layout_no_wrap(self.text.to_string(), font, Color32::WHITE);

        let padding = Vec2::new(12.0, 5.0) * theme.polish.spacing_scale;
        let desired = text_galley.size() + padding * 2.0;
        let (rect, response) = ui.allocate_exact_size(desired, Sense::click());

        response.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), self.text)
        });
        if ui.is_rect_visible(rect) {
            let hovered = response.hovered();
            let hover = crate::ui::polish::hover(
                ui.ctx(),
                response.id.with("button_hover"),
                hovered && ui.is_enabled(),
            );
            let active = response.is_pointer_button_down_on();

            match self.style {
                SignalButtonStyle::Primary => {
                    self.paint_primary(ui, rect, theme, hover, active);
                }
                SignalButtonStyle::Ghost => {
                    self.paint_ghost(ui, rect, theme, hover, active);
                }
                SignalButtonStyle::OutlineAccent => {
                    self.paint_outline_accent(ui, rect, theme, hover, active);
                }
            }

            // Draw text centered
            let text_color = match self.style {
                SignalButtonStyle::Primary => Color32::WHITE,
                SignalButtonStyle::Ghost => {
                    if hovered {
                        theme.text_color
                    } else {
                        theme.text_muted
                    }
                }
                SignalButtonStyle::OutlineAccent => {
                    if hovered {
                        Color32::WHITE
                    } else {
                        theme.accent3
                    }
                }
            };

            let font = egui::FontId::proportional(Theme::FONT_BODY * theme.polish.text_scale);
            let galley = ui
                .painter()
                .layout_no_wrap(self.text.to_string(), font, text_color);
            let text_pos = rect.center() - galley.size() / 2.0;
            ui.painter().galley(
                egui::pos2(text_pos.x, text_pos.y),
                galley,
                egui::Color32::TRANSPARENT,
            );
        }

        if !ui.is_enabled() {
            ui.painter().rect_filled(
                rect,
                theme.widget_rounding,
                theme.panel_bg.gamma_multiply(0.5),
            );
        }
        crate::ui::polish::focus(ui, &response, theme.widget_rounding);
        response
    }

    fn paint_primary(&self, ui: &Ui, rect: Rect, theme: &Theme, hover: f32, active: bool) {
        let fill = if active {
            darken(theme.accent, 15)
        } else {
            Theme::lerp_color(theme.accent, lighten(theme.accent, 20), hover)
        };
        ui.painter().rect_filled(rect, theme.widget_rounding, fill);
    }

    fn paint_ghost(&self, ui: &Ui, rect: Rect, theme: &Theme, hover: f32, active: bool) {
        let fill = if active {
            theme.bg3
        } else {
            theme.bg2.gamma_multiply(hover)
        };
        let stroke = Theme::lerp_color(theme.border_color, theme.border_lit, hover);
        ui.painter().rect(
            rect,
            theme.widget_rounding,
            fill,
            Stroke::new(theme.polish.border_width, stroke),
            egui::StrokeKind::Inside,
        );
    }

    fn paint_outline_accent(&self, ui: &Ui, rect: Rect, theme: &Theme, hover: f32, active: bool) {
        let strength = if active { 0.16 } else { hover * 0.10 };
        ui.painter().rect(
            rect,
            theme.widget_rounding,
            theme.accent3.gamma_multiply(strength),
            Stroke::new(theme.polish.border_width, theme.accent3),
            egui::StrokeKind::Inside,
        );
    }
}

// ============================================================================
// GradientDivider — gradient-fade separator
// ============================================================================

/// Draw a solid divider line under a panel header, spanning the full width.
pub fn gradient_divider(ui: &mut Ui, theme: &Theme) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 1.0), Sense::hover());
    if ui.is_rect_visible(rect) {
        let y = rect.center().y;
        ui.painter().line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            Stroke::new(1.0, theme.separator_color),
        );
    }
}

/// Tool shelf tag badge — a small monospace uppercase label with a colored border,
/// matching the website's `.section-tag` / `.badge` pattern.
///
/// Example: `[BRUSH]` in accent color with rounded border and tinted background.
pub fn tool_shelf_tag(ui: &mut Ui, label: &str, color: Color32, theme: &Theme) {
    let fill = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 18);
    let stroke_color = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 80);

    let text = egui::RichText::new(label)
        .font(egui::FontId::monospace(10.0))
        .color(color)
        .strong();

    egui::Frame::NONE
        .fill(fill)
        .corner_radius(egui::CornerRadius::same(theme.badge_rounding as u8))
        .stroke(Stroke::new(1.0, stroke_color))
        .inner_margin(egui::Margin::symmetric(8, 3))
        .show(ui, |ui| {
            ui.set_height(14.0); // Fixed inner height so badge doesn't shift between tools
            ui.label(text);
        });
}

// ============================================================================
// PillTabBar — pill-container tab strip
// ============================================================================

/// A single tab entry for `PillTabBar`.
pub struct PillTab {
    pub label: String,
    pub closable: bool,
}

impl PillTab {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            closable: false,
        }
    }

    pub fn closable(mut self) -> Self {
        self.closable = true;
        self
    }
}

/// Result of rendering a `PillTabBar`.
pub struct PillTabBarResponse {
    /// Index of the newly selected tab (if changed), or current active.
    pub active: usize,
    /// Index of the tab whose close button was clicked (if any).
    pub closed: Option<usize>,
}

/// A pill-shaped tab bar container matching the website's `.tab-bar`.
///
/// # Example
/// ```ignore
/// let tabs = vec![PillTab::new("Canvas 1").closable(), PillTab::new("Canvas 2")];
/// let resp = PillTabBar::new(&tabs, active_tab).show(ui, theme);
/// active_tab = resp.active;
/// if let Some(closed) = resp.closed { ... }
/// ```
pub struct PillTabBar<'a> {
    tabs: &'a [PillTab],
    active: usize,
}

impl<'a> PillTabBar<'a> {
    pub fn new(tabs: &'a [PillTab], active: usize) -> Self {
        Self { tabs, active }
    }

    pub fn show(self, ui: &mut Ui, theme: &Theme) -> PillTabBarResponse {
        let mut new_active = self.active;
        let mut closed = None;

        // Outer pill container
        let container_padding = 4.0;
        let tab_h = 28.0;
        let container_h = tab_h + container_padding * 2.0;

        let available_w = ui.available_width();
        let (container_rect, _) =
            ui.allocate_exact_size(Vec2::new(available_w, container_h), Sense::hover());

        if ui.is_rect_visible(container_rect) {
            // Draw pill container background
            signal_draw::draw_pill_container(ui.painter(), container_rect, theme);
        }

        // Lay out tabs inside the container
        let inner_rect = container_rect.shrink(container_padding);
        let mut child_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(inner_rect)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );

        for (i, tab) in self.tabs.iter().enumerate() {
            let is_active = i == self.active;
            let tab_resp = self.paint_tab(&mut child_ui, theme, tab, is_active);

            if tab_resp.clicked {
                new_active = i;
            }
            if tab_resp.close_clicked {
                closed = Some(i);
            }
        }

        PillTabBarResponse {
            active: new_active,
            closed,
        }
    }

    fn paint_tab(&self, ui: &mut Ui, theme: &Theme, tab: &PillTab, is_active: bool) -> TabResponse {
        let font = egui::FontId::proportional(Theme::FONT_BODY * theme.polish.text_scale);
        let text_color = if is_active {
            theme.text_color
        } else {
            theme.text_muted
        };

        let galley = ui
            .painter()
            .layout_no_wrap(tab.label.clone(), font.clone(), text_color);

        let text_width = galley.size().x;
        let close_width = if tab.closable { 18.0 } else { 0.0 };
        let h_pad = 14.0;
        let tab_width = text_width + close_width + h_pad * 2.0;
        let tab_height = 28.0;

        let (tab_rect, response) =
            ui.allocate_exact_size(Vec2::new(tab_width, tab_height), Sense::click());

        let mut close_clicked = false;

        if ui.is_rect_visible(tab_rect) {
            let hovered = response.hovered();

            let tab_cr = CornerRadius::same(theme.tab_rounding as u8);
            response.widget_info(|| {
                egui::WidgetInfo::selected(
                    egui::WidgetType::Button,
                    ui.is_enabled(),
                    is_active,
                    &tab.label,
                )
            });
            let selected = crate::ui::polish::animate(
                ui.ctx(),
                response.id.with("selected"),
                if is_active { 1.0 } else { 0.0 },
                crate::ui::polish::MotionKind::Selection,
            );
            let hover = crate::ui::polish::hover(ui.ctx(), response.id.with("hover"), hovered);
            ui.painter().rect_filled(
                tab_rect,
                tab_cr,
                theme.bg3.gamma_multiply((selected + hover * 0.3).min(1.0)),
            );
            crate::ui::polish::focus(ui, &response, tab_cr);

            // Tab label
            let text_pos = egui::pos2(
                tab_rect.left() + h_pad,
                tab_rect.center().y - galley.size().y / 2.0,
            );
            ui.painter()
                .galley(text_pos, galley, egui::Color32::TRANSPARENT);

            // Close button
            if tab.closable {
                let close_rect = Rect::from_min_size(
                    egui::pos2(tab_rect.right() - h_pad - 12.0, tab_rect.center().y - 6.0),
                    Vec2::splat(12.0),
                );
                let close_response =
                    ui.interact(close_rect, response.id.with("close"), Sense::click());

                let close_color = if close_response.hovered() {
                    theme.accent
                } else if is_active {
                    theme.text_muted
                } else {
                    theme.text_faint
                };

                close_response.widget_info(|| {
                    egui::WidgetInfo::labeled(
                        egui::WidgetType::Button,
                        ui.is_enabled(),
                        format!("Close {}", tab.label),
                    )
                });
                crate::ui::polish::control(
                    ui,
                    &close_response,
                    false,
                    theme.widget_rounding.min(6.0),
                );
                paint_close_cross(ui, close_rect, close_color);

                if close_response.clicked() {
                    close_clicked = true;
                }
            }
        }

        TabResponse {
            clicked: response.clicked(),
            close_clicked,
        }
    }
}

struct TabResponse {
    clicked: bool,
    close_clicked: bool,
}

// ============================================================================
// CardFrame — panel wrapper with hover border light-up
// ============================================================================

/// Create a card-style frame matching the website's card components.
///
/// - `panel_bg` fill, `12px` rounding, `1px` border, `12px` inner margin.
/// - Hover detection and border light-up handled via `show()`.
///
/// # Example
/// ```ignore
/// card_frame(theme).show(ui, |ui| {
///     ui.label("Card content");
/// });
/// ```
pub fn card_frame(theme: &Theme) -> egui::Frame {
    egui::Frame::NONE
        .fill(theme.panel_bg)
        .corner_radius(CornerRadius::same(12))
        .stroke(Stroke::new(1.0, theme.border_color))
        .shadow(Shadow {
            offset: [0, 0],
            blur: 6,
            spread: 0,
            color: Color32::from_black_alpha(20),
        })
        .inner_margin(egui::Margin::same(12))
}

/// Show a card frame with hover border light-up animation.
///
/// Returns the inner `Response` from the content closure.
pub fn card_frame_interactive<R>(
    ui: &mut Ui,
    id: egui::Id,
    theme: &Theme,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> egui::InnerResponse<R> {
    let hover_t = crate::ui::polish::panel_hover(ui.ctx(), id.with("card_hover"));

    let border = lerp_color(theme.border_color, theme.border_lit, hover_t);

    let frame = egui::Frame::NONE
        .fill(theme.panel_bg)
        .corner_radius(theme.widget_cr(12))
        .stroke(Stroke::new(1.0, border))
        .shadow(Shadow {
            offset: [0, 0],
            blur: 6,
            spread: 0,
            color: Color32::from_black_alpha(20),
        })
        .inner_margin(egui::Margin::same(12));

    let resp = frame.show(ui, add_contents);

    // Update hover animation state for next frame
    let hovered = ui.rect_contains_pointer(resp.response.rect);
    crate::ui::polish::remember_hover(ui.ctx(), id.with("card_hover"), hovered);

    resp
}

// ============================================================================
// Panel header — title + optional badge + close button + gradient divider
// ============================================================================

/// Draw a panel header with badge, close button, and divider.
///
/// When a badge is provided, it is shown as the sole identifier (no duplicate title).
/// When no badge is given, the title string is displayed as a heading instead.
///
/// The header row spans the exact panel width so the close button sits in the
/// corner and the divider below runs corner to corner — this matters on
/// fixed-size panels (Colors, Palette) where content-based sizing would leave
/// both of them short.
///
/// Returns `true` if the close button was clicked.
/// Preserve resize gestures even when press, move and release arrive in one frame.
pub(crate) fn floating_resize(
    ctx: &egui::Context,
    response: &egui::Response,
    id: &'static str,
    min: Vec2,
) -> Option<Vec2> {
    let panel = id;
    let id = egui::Id::new(("floating_resize_capture", id));
    let mut capture = ctx.data_mut(|d| d.get_temp::<(egui::Pos2, Vec2)>(id));
    let grip = egui::Rect::from_min_max(response.rect.max - Vec2::splat(18.0), response.rect.max);
    let mut size = None;
    let events = ctx.input(|i| i.events.clone());
    for event in events {
        match event {
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                ..
            } if response.enabled()
                && grip.contains(pos)
                && ctx.layer_id_at(pos) == Some(response.layer_id) =>
            {
                capture = Some((pos, response.rect.size()));
            }
            egui::Event::PointerMoved(pos) if capture.is_some() => {
                let (start, original) = capture.unwrap();
                size = Some((original + (pos - start)).max(min));
            }
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                ..
            } if capture.is_some() => {
                let (start, original) = capture.take().unwrap();
                size = Some((original + (pos - start)).max(min));
            }
            egui::Event::PointerGone => capture = None,
            _ => {}
        }
    }
    ctx.data_mut(|d| {
        if let Some(c) = capture {
            d.insert_temp(id, c);
        } else {
            d.remove::<(egui::Pos2, Vec2)>(id);
        }
    });
    size.map(|size| crate::ui::workspace::resize(ctx, panel, response.rect, size, min))
}

pub fn panel_header(
    ui: &mut Ui,
    theme: &Theme,
    title: &str,
    badge: Option<(&str, Color32)>,
) -> bool {
    let mut close_clicked = false;
    // Capture the full available width BEFORE entering the horizontal layout.
    let header_width = ui.available_width();
    ui.horizontal(|ui| {
        // Exact width: the close button must land on the panel's right edge.
        ui.set_width(header_width);

        let label = badge.map_or(title, |(label, _)| label);
        ui.label(
            egui::RichText::new(label)
                .font(egui::FontId::new(
                    11.0 * theme.polish.text_scale,
                    egui::FontFamily::Name("WidgetTitle".into()),
                ))
                .strong()
                .color(theme.text_color),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            close_clicked = close_button(ui, theme).clicked();
        });
    });
    if matches!(
        title,
        "Tools" | "Layers" | "Colors" | "Palette" | "History" | "ScriptEditor"
    ) {
        let rect = egui::Rect::from_min_size(
            ui.min_rect().min,
            egui::vec2((header_width - 24.0).max(0.0), 18.0),
        );
        ui.interact(rect, ui.layer_id().id.with("header_drag"), Sense::drag())
            .on_hover_cursor(egui::CursorIcon::Grab);
        let gesture_id = egui::Id::new(("floating_header_pointer", title));
        let mut last = ui.ctx().data_mut(|d| d.get_temp::<egui::Pos2>(gesture_id));
        let mut delta = Vec2::ZERO;
        // Process the press position before later motion, including fast gestures batched into one frame.
        let events = ui.input(|input| input.events.clone());
        for event in &events {
            match event {
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    ..
                } if ui.is_enabled()
                    && rect.intersect(ui.clip_rect()).contains(*pos)
                    && ui
                        .ctx()
                        .layer_id_at(*pos)
                        .is_none_or(|layer| layer == ui.layer_id()) =>
                {
                    last = Some(*pos)
                }
                egui::Event::PointerMoved(pos) => {
                    if let Some(previous) = last {
                        delta += *pos - previous;
                        last = Some(*pos);
                    }
                }
                egui::Event::PointerButton {
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    ..
                }
                | egui::Event::PointerGone => last = None,
                _ => {}
            }
        }
        ui.ctx().data_mut(|d| {
            if let Some(pos) = last {
                d.insert_temp(gesture_id, pos);
            } else {
                d.remove::<egui::Pos2>(gesture_id);
            }
            if delta != Vec2::ZERO {
                d.insert_temp(egui::Id::new(("floating_header_drag", title)), delta);
            }
        });
        if delta != Vec2::ZERO {
            ui.ctx().request_repaint();
        }
    }
    ui.add_space(4.0);
    close_clicked
}

// ============================================================================
// Section header — bold label + gradient divider
// ============================================================================

/// Draw a section header: bold label text followed by a gradient divider.
///
/// Used in panels and dialogs where the plan calls for `section_tag + divider`.
pub fn section_header(ui: &mut Ui, theme: &Theme, label: &str) {
    ui.add_space(Theme::SPACE_SM);
    ui.label(
        egui::RichText::new(label)
            .strong()
            .size(Theme::FONT_HEADING),
    );
    gradient_divider(ui, theme);
    ui.add_space(Theme::SPACE_XS);
}

/// Draw a section header with a colored badge tag above the label.
///
/// The badge shows a monospace uppercase tag (e.g. "TOOLS", "LAYERS")
/// above the section title.
pub fn section_header_with_badge(
    ui: &mut Ui,
    theme: &Theme,
    badge: &str,
    badge_color: Color32,
    label: &str,
) {
    ui.add_space(Theme::SPACE_SM);
    SignalBadge::new(badge, badge_color).show(ui, theme);
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new(label)
            .strong()
            .size(Theme::FONT_HEADING),
    );
    gradient_divider(ui, theme);
    ui.add_space(Theme::SPACE_XS);
}

// ============================================================================
// Helpers
// ============================================================================

/// Lighten a color by adding `amount` to each RGB channel.
fn lighten(c: Color32, amount: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(
        c.r().saturating_add(amount),
        c.g().saturating_add(amount),
        c.b().saturating_add(amount),
        c.a(),
    )
}

/// Darken a color by subtracting `amount` from each RGB channel.
fn darken(c: Color32, amount: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(
        c.r().saturating_sub(amount),
        c.g().saturating_sub(amount),
        c.b().saturating_sub(amount),
        c.a(),
    )
}

/// Linearly interpolate between two colors by factor `t` (0.0 = a, 1.0 = b).
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    Color32::from_rgba_premultiplied(
        (a.r() as f32 * inv + b.r() as f32 * t) as u8,
        (a.g() as f32 * inv + b.g() as f32 * t) as u8,
        (a.b() as f32 * inv + b.b() as f32 * t) as u8,
        (a.a() as f32 * inv + b.a() as f32 * t) as u8,
    )
}

/// Shared font-independent close mark for panels and document tabs.
pub fn paint_close_cross(ui: &Ui, rect: Rect, color: Color32) {
    let custom = ui
        .ctx()
        .data(|d| {
            d.get_temp::<Option<egui::TextureHandle>>(egui::Id::new("paintfe_close_override"))
        })
        .flatten();
    if let Some(texture) = custom {
        ui.painter().image(
            texture.id(),
            Rect::from_center_size(rect.center(), Vec2::splat(10.0)),
            Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        return;
    }
    let center = rect.center();
    let d = 3.5;
    let stroke = egui::Stroke::new(1.35, color);
    for (a, b) in [
        (Vec2::new(-d, -d), Vec2::new(d, d)),
        (Vec2::new(-d, d), Vec2::new(d, -d)),
    ] {
        ui.painter().line_segment([center + a, center + b], stroke);
        ui.painter().circle_filled(center + a, 0.675, color);
        ui.painter().circle_filled(center + b, 0.675, color);
    }
}

pub fn close_button(ui: &mut Ui, theme: &Theme) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(18.0), Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), "Close")
    });
    crate::ui::polish::control(ui, &response, false, theme.widget_rounding.min(6.0));
    let hover = crate::ui::polish::hover(
        ui.ctx(),
        response.id.with("close_color"),
        response.hovered(),
    );
    paint_close_cross(
        ui,
        rect,
        Theme::lerp_color(theme.text_muted, theme.accent, hover),
    );
    response.on_hover_text("Close")
}

#[cfg(test)]
mod refinement_tests {
    use super::*;
    #[test]
    fn fast_resize_releases_capture_and_respects_minimum() {
        let ctx = egui::Context::default();
        let mut rect = Rect::NOTHING;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            rect = ui.allocate_exact_size(Vec2::splat(100.0), Sense::hover()).0;
        });
        let start = rect.max - Vec2::splat(5.0);
        let end = start + egui::vec2(80.0, 40.0);
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        let mut result = None;
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![
                    button(start, true),
                    egui::Event::PointerMoved(end),
                    button(end, false),
                ],
                ..Default::default()
            },
            |ui| {
                let response = ui.allocate_exact_size(Vec2::splat(100.0), Sense::hover()).1;
                result = floating_resize(&ctx, &response, "test", Vec2::splat(50.0));
            },
        );
        assert_eq!(result, Some(egui::vec2(180.0, 140.0)));
        let _ = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::PointerMoved(start)],
                ..Default::default()
            },
            |ui| {
                let response = ui.allocate_exact_size(Vec2::splat(100.0), Sense::hover()).1;
                result = floating_resize(&ctx, &response, "test", Vec2::splat(50.0));
            },
        );
        assert_eq!(result, None);
    }
}
