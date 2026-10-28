use super::history_tests::{app, context};
use super::*;

fn frame(
    app: &mut PaintFEApp,
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
) {
    let _ = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1200., 900.),
            )),
            events,
            modifiers,
            ..Default::default()
        },
        |ui| {
            app.update_runtime_input(ctx);
            app.show_runtime_dialogs_menu(ctx, ui);
        },
    );
}
fn click(app: &mut PaintFEApp, ctx: &egui::Context, rect: egui::Rect) {
    for pressed in [true, false] {
        frame(
            app,
            ctx,
            vec![
                egui::Event::PointerMoved(rect.center()),
                egui::Event::PointerButton {
                    pos: rect.center(),
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            egui::Modifiers::NONE,
        );
    }
}
fn rect(ctx: &egui::Context, key: &str) -> egui::Rect {
    ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::new(key)))
        .unwrap_or_else(|| panic!("Missing production control: {key}"))
}
#[test]
fn toolbar_menu_and_shortcut_fit_the_same_document_in_both_themes() {
    for theme in [egui::Theme::Light, egui::Theme::Dark] {
        let mut app = app();
        let ctx = context();
        ctx.set_theme(theme);
        app.projects[0].canvas_state.width = 4000;
        app.projects[0].canvas_state.height = 3000;
        app.canvas.last_canvas_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000., 600.),
        ));
        for _ in 0..3 {
            frame(&mut app, &ctx, vec![], egui::Modifiers::NONE);
        }
        let fit = rect(&ctx, "toolbar_fit_test_rect");
        click(&mut app, &ctx, fit);
        let expected = app.canvas.view_state();
        assert!((expected.0 - 0.192).abs() < 0.00001);
        assert_eq!(expected.1, egui::Vec2::ZERO);
        app.canvas.set_view_state(2., egui::vec2(200., 100.));
        let menu = rect(&ctx, "view_menu_test_rect");
        click(&mut app, &ctx, menu);
        frame(&mut app, &ctx, vec![], egui::Modifiers::NONE);
        let fit = rect(&ctx, "menu_fit_test_rect");
        click(&mut app, &ctx, fit);
        assert_eq!(app.canvas.view_state(), expected);
        app.canvas.set_view_state(3., egui::vec2(-100., 20.));
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::Key {
                key: egui::Key::Num0,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::CTRL,
            }],
            egui::Modifiers::CTRL,
        );
        assert_eq!(app.canvas.view_state(), expected);
        frame(
            &mut app,
            &ctx,
            vec![egui::Event::Key {
                key: egui::Key::Num1,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::CTRL,
            }],
            egui::Modifiers::CTRL,
        );
        assert_eq!(app.canvas.view_state(), (1., egui::Vec2::ZERO));
    }
}
