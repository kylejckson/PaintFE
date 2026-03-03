//! Custom widget library implementing the Signal Grid design language.
//!
//! These widgets replace stock egui components with custom-painted equivalents
//! that match the website's visual language: badges, button variants, gradient
//! dividers, pill tab bars, card frames, and glow effects.

use eframe::egui::{self, Color32, Rect, Response, Rounding, Sense, Stroke, Ui, Vec2};
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
        // In light mode, darken badge text for better contrast
        let text_color = if is_light {
            // Shift toward a darker version of the badge color for readability
            Color32::from_rgb(
                (self.color.r() as u16 * 3 / 5) as u8,
                (self.color.g() as u16 * 3 / 5) as u8,
                (self.color.b() as u16 * 3 / 5) as u8,
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
                (30u8, 90u8) // stronger fill + border in light mode for contrast
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
                Rounding::same(4.0),
                fill,
                Stroke::new(1.0, stroke_color),
            );

            let text_pos = rect.min + padding;
            ui.painter()
                .galley(egui::pos2(text_pos.x, text_pos.y), galley);
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
        let font = egui::FontId::proportional(Theme::FONT_BODY);
        let text_galley =
            ui.painter()
                .layout_no_wrap(self.text.to_string(), font, Color32::WHITE);

        let padding = Vec2::new(16.0, 6.0);
        let desired = text_galley.size() + padding * 2.0;
        let (rect, response) = ui.allocate_exact_size(desired, Sense::click());

        if ui.is_rect_visible(rect) {
            let hovered = response.hovered();
            let active = response.is_pointer_button_down_on();

            match self.style {
                SignalButtonStyle::Primary => {
                    self.paint_primary(ui, rect, theme, hovered, active);
                }
                SignalButtonStyle::Ghost => {
                    self.paint_ghost(ui, rect, theme, hovered, active);
                }
                SignalButtonStyle::OutlineAccent => {
                    self.paint_outline_accent(ui, rect, theme, hovered, active);
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

            let font = egui::FontId::proportional(Theme::FONT_BODY);
            let galley =
                ui.painter()
                    .layout_no_wrap(self.text.to_string(), font, text_color);
            let text_pos = rect.center() - galley.size() / 2.0;
            ui.painter()
                .galley(egui::pos2(text_pos.x, text_pos.y), galley);
        }

        response
    }

    fn paint_primary(
        &self,
        ui: &Ui,
        rect: Rect,
        theme: &Theme,
        hovered: bool,
        active: bool,
    ) {
        let fill = if active {
            darken(theme.accent, 15)
        } else if hovered {
            lighten(theme.accent, 20)
        } else {
            theme.accent
        };

        // Glow behind on hover
        if hovered {
            signal_draw::draw_glow_rect(ui.painter(), rect, theme.glow_accent, 8.0, 8.0);
        }

        ui.painter()
            .rect_filled(rect, Rounding::same(8.0), fill);
    }

    fn paint_ghost(&self, ui: &Ui, rect: Rect, theme: &Theme, hovered: bool, active: bool) {
        let fill = if active {
            theme.bg3
        } else if hovered {
            theme.bg2
        } else {
            Color32::TRANSPARENT
        };

        let stroke_color = if hovered {
            theme.border_lit
        } else {
            theme.border_color
        };

        ui.painter().rect(
            rect,
            Rounding::same(6.0),
            fill,
            Stroke::new(1.0, stroke_color),
        );
    }

    fn paint_outline_accent(
        &self,
        ui: &Ui,
        rect: Rect,
        theme: &Theme,
        hovered: bool,
        active: bool,
    ) {
        let fill = if active {
            Color32::from_rgba_unmultiplied(
                theme.accent3.r(),
                theme.accent3.g(),
                theme.accent3.b(),
                40,
            )
        } else if hovered {
            Color32::from_rgba_unmultiplied(
                theme.accent3.r(),
                theme.accent3.g(),
                theme.accent3.b(),
                25,
            )
        } else {
            Color32::TRANSPARENT
        };

        // Glow behind on hover
        if hovered {
            signal_draw::draw_glow_rect(ui.painter(), rect, theme.glow_accent3, 6.0, 6.0);
        }

        ui.painter().rect(
            rect,
            Rounding::same(6.0),
            fill,
            Stroke::new(1.0, theme.accent3),
        );
    }
}

// ============================================================================
// GradientDivider — gradient-fade separator
// ============================================================================

/// Draw a gradient-fade divider in place of `ui.separator()`.
///
/// Allocates 1px of vertical space and draws a horizontal line that fades
/// from transparent at the edges to `theme.separator_color` in the center.
pub fn gradient_divider(ui: &mut Ui, theme: &Theme) {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 1.0), Sense::hover());
    if ui.is_rect_visible(rect) {
        signal_draw::draw_gradient_divider(ui.painter(), rect, theme.separator_color);
    }
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
        let mut child_ui = ui.child_ui(inner_rect, egui::Layout::left_to_right(egui::Align::Center));

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

    fn paint_tab(
        &self,
        ui: &mut Ui,
        theme: &Theme,
        tab: &PillTab,
        is_active: bool,
    ) -> TabResponse {
        let font = egui::FontId::proportional(Theme::FONT_BODY);
        let text_color = if is_active {
            theme.text_color
        } else {
            theme.text_muted
        };

        let galley =
            ui.painter()
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

            // Tab background
            if is_active {
                ui.painter().rect(
                    tab_rect,
                    Rounding::same(7.0),
                    theme.bg3,
                    Stroke::NONE,
                );
                // Subtle inset shadow for active tab
                let shadow_rect = Rect::from_min_size(
                    tab_rect.min,
                    Vec2::new(tab_rect.width(), 2.0),
                );
                ui.painter().rect_filled(
                    shadow_rect,
                    Rounding::same(1.0),
                    Color32::from_black_alpha(20),
                );
            } else if hovered {
                ui.painter().rect_filled(
                    tab_rect,
                    Rounding::same(7.0),
                    Color32::from_rgba_unmultiplied(
                        theme.bg3.r(),
                        theme.bg3.g(),
                        theme.bg3.b(),
                        80,
                    ),
                );
            }

            // Tab label
            let text_pos = egui::pos2(
                tab_rect.left() + h_pad,
                tab_rect.center().y - galley.size().y / 2.0,
            );
            ui.painter().galley(text_pos, galley);

            // Close button
            if tab.closable {
                let close_rect = Rect::from_min_size(
                    egui::pos2(tab_rect.right() - h_pad - 12.0, tab_rect.center().y - 6.0),
                    Vec2::splat(12.0),
                );
                let close_response = ui.interact(close_rect, response.id.with("close"), Sense::click());

                let close_color = if close_response.hovered() {
                    theme.accent
                } else if is_active {
                    theme.text_muted
                } else {
                    theme.text_faint
                };

                // Draw × symbol
                let close_font = egui::FontId::proportional(10.0);
                let close_galley = ui.painter().layout_no_wrap(
                    "×".to_string(),
                    close_font,
                    close_color,
                );
                let close_text_pos = close_rect.center() - close_galley.size() / 2.0;
                ui.painter().galley(
                    egui::pos2(close_text_pos.x, close_text_pos.y),
                    close_galley,
                );

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
    egui::Frame::none()
        .fill(theme.panel_bg)
        .rounding(Rounding::same(12.0))
        .stroke(Stroke::new(1.0, theme.border_color))
        .shadow(Shadow {
            extrusion: 6.0,
            color: Color32::from_black_alpha(20),
        })
        .inner_margin(egui::Margin::same(12.0))
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
    let hover_t = ui
        .ctx()
        .animate_bool(id.with("card_hover"), false);

    let border = lerp_color(theme.border_color, theme.border_lit, hover_t);

    let frame = egui::Frame::none()
        .fill(theme.panel_bg)
        .rounding(Rounding::same(12.0))
        .stroke(Stroke::new(1.0, border))
        .shadow(Shadow {
            extrusion: 6.0,
            color: Color32::from_black_alpha(20),
        })
        .inner_margin(egui::Margin::same(12.0));

    let resp = frame.show(ui, add_contents);

    // Update hover animation state for next frame
    let hovered = ui.rect_contains_pointer(resp.response.rect);
    ui.ctx()
        .animate_bool(id.with("card_hover"), hovered);

    resp
}

// ============================================================================
// Panel header — title + optional badge + close button + gradient divider
// ============================================================================

/// Draw a panel header with badge, close button, and gradient divider.
///
/// When a badge is provided, it is shown as the sole identifier (no duplicate title).
/// When no badge is given, the title string is displayed as a heading instead.
///
/// Returns `true` if the close button was clicked.
pub fn panel_header(ui: &mut Ui, theme: &Theme, title: &str, badge: Option<(&str, Color32)>) -> bool {
    let mut close_clicked = false;
    ui.horizontal(|ui| {
        // Fill the parent's width so the close button aligns to the right edge.
        // Cap at 400 to avoid bloating auto-sized windows on the first frame.
        let w = ui.available_width().min(400.0);
        ui.set_min_width(w);

        if let Some((badge_text, badge_color)) = badge {
            SignalBadge::new(badge_text, badge_color).show(ui, theme);
        } else {
            // No badge — show the title text as a heading
            ui.label(
                egui::RichText::new(title)
                    .size(Theme::FONT_HEADING)
                    .color(theme.text_color),
            );
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Ghost-style close button
            let (close_rect, close_resp) =
                ui.allocate_exact_size(Vec2::splat(18.0), Sense::click());
            if ui.is_rect_visible(close_rect) {
                let hovered = close_resp.hovered();
                if hovered {
                    ui.painter().rect_filled(
                        close_rect,
                        Rounding::same(4.0),
                        theme.bg3,
                    );
                }
                let color = if hovered { theme.accent } else { theme.text_muted };
                let font = egui::FontId::proportional(13.0);
                let galley =
                    ui.painter()
                        .layout_no_wrap("×".to_string(), font, color);
                let pos = close_rect.center() - galley.size() / 2.0;
                ui.painter().galley(egui::pos2(pos.x, pos.y), galley);
            }
            if close_resp.clicked() {
                close_clicked = true;
            }
        });
    });
    gradient_divider(ui, theme);
    ui.add_space(2.0);
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
pub fn section_header_with_badge(ui: &mut Ui, theme: &Theme, badge: &str, badge_color: Color32, label: &str) {
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
    Color32::from_rgba_unmultiplied(
        (a.r() as f32 * inv + b.r() as f32 * t) as u8,
        (a.g() as f32 * inv + b.g() as f32 * t) as u8,
        (a.b() as f32 * inv + b.b() as f32 * t) as u8,
        (a.a() as f32 * inv + b.a() as f32 * t) as u8,
    )
}
