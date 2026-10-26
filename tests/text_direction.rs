//! Production text layout, including logical model/cursor and shared selection paint.
use egui::{
    Color32, Context, FontDefinitions, FontFamily, FontId,
    text::{CCursor, LayoutJob},
    text_selection::{CCursorRange, visuals::paint_text_selection},
};
use std::sync::Arc;
fn context() -> Context {
    let ctx = Context::default();
    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "arabic".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/NotoSansArabic.ttf")).into(),
    );
    fonts
        .families
        .get_mut(&FontFamily::Proportional)
        .unwrap()
        .insert(0, "arabic".into());
    ctx.set_fonts(fonts);
    ctx
}
fn layout(ctx: &Context, text: &str, width: f32, size: f32, elide: bool) -> Arc<egui::Galley> {
    let mut galley = None;
    let _ = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900.0, 700.0),
            )),
            ..Default::default()
        },
        |ui| {
            let mut job = LayoutJob::simple(
                text.into(),
                FontId::proportional(size),
                Color32::WHITE,
                width,
            );
            if elide {
                job.wrap.max_rows = 1;
                job.wrap.break_anywhere = true;
            }
            galley = Some(ui.painter().layout_job(job));
        },
    );
    galley.unwrap()
}
#[test]
fn arabic_words_follow_rtl_order_while_source_stays_logical() {
    let ctx = context();
    let text = "تغيير حجم لوحة الرسم";
    let galley = layout(&ctx, text, 800.0, 20.0, false);
    assert_eq!(galley.text(), text);
    assert_eq!(galley.rows[0].text(), text);
    let row = &galley.rows[0];
    let mut index = 0;
    let mut centres = Vec::new();
    for word in text.split(' ') {
        let count = word.chars().count();
        centres.push(
            row.glyphs[index..index + count]
                .iter()
                .map(|g| g.pos.x)
                .sum::<f32>()
                / count as f32,
        );
        index += count + 1;
    }
    assert!(
        centres.windows(2).all(|w| w[0] > w[1]),
        "Arabic words: {centres:?}"
    );
}
#[test]
fn mixed_numbers_filenames_wrapping_and_marks_keep_their_logical_text() {
    let ctx = context();
    for text in [
        "حفظ image-12.png بنسبة 75%",
        "تَغْيير حَجم لوحة الرَّسم",
        "لا لاختبار (RGB 255) ١٢٣",
        "English ثم العربية\nسطر جديد 2026",
    ] {
        for width in [95.0, 160.0, 800.0] {
            for size in [12.0, 24.0] {
                let galley = layout(&ctx, text, width, size, false);
                let mut rebuilt = String::new();
                for row in &galley.rows {
                    rebuilt.push_str(&row.text());
                    if row.ends_with_newline {
                        rebuilt.push('\n');
                    }
                }
                assert_eq!(rebuilt, text, "width={width} size={size}");
                assert!(
                    galley
                        .rows
                        .iter()
                        .flat_map(|r| &r.glyphs)
                        .all(|g| g.pos.x.is_finite())
                );
            }
        }
    }
}
#[test]
fn arabic_caret_hit_testing_arrows_and_selection_share_logical_offsets() {
    let ctx = context();
    let mut galley = layout(&ctx, "مرحبا", 800.0, 24.0, false);
    for i in 0..=5 {
        let cursor = CCursor::new(i);
        let rect = galley.pos_from_cursor(cursor);
        let hit = galley.cursor_from_pos(rect.center().to_vec2());
        assert_eq!(hit.index.0, i, "caret {i}");
    }
    assert_eq!(
        galley.cursor_left_one_character(&CCursor::new(0)).index.0,
        1
    );
    assert_eq!(
        galley.cursor_right_one_character(&CCursor::new(1)).index.0,
        0
    );
    let before = galley.rows[0].visuals.mesh.indices.len();
    let range = CCursorRange::two(CCursor::new(1), CCursor::new(4));
    paint_text_selection(&mut galley, &egui::Visuals::dark(), &range, None);
    assert!(galley.rows[0].visuals.mesh.indices.len() > before);
    assert_eq!(galley.text(), "مرحبا");
}
#[test]
fn elision_mixed_scripts_and_ltr_layouts_stay_valid_in_both_themes() {
    let ctx = context();
    for dark in [false, true] {
        ctx.set_visuals(if dark {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });
        for text in [
            "العربية الطويلة image.png 2026",
            "العربية (الملف) 123",
            "Ordinary English 123",
            "Русский текст 123",
        ] {
            let galley = layout(&ctx, text, 50.0, 18.0, true);
            assert!(galley.elided);
            assert_eq!(galley.text(), text);
            for row in &galley.rows {
                assert!(
                    row.visuals
                        .mesh
                        .vertices
                        .iter()
                        .all(|v| v.pos.x.is_finite() && v.pos.y.is_finite())
                );
            }
        }
    }
}
#[test]
fn mixed_latin_filename_and_numbers_keep_ltr_positions_and_all_arabic_ink() {
    let ctx = context();
    let text = "تغيير image-12.png بنسبة 75%";
    let galley = layout(&ctx, text, 800.0, 24.0, false);
    let row = &galley.rows[0];
    for substring in ["image-12.png", "75"] {
        let byte = text.find(substring).unwrap();
        let start = text[..byte].chars().count();
        let count = substring.chars().count();
        assert!(
            row.glyphs[start..start + count]
                .windows(2)
                .all(|w| w[0].pos.x <= w[1].pos.x),
            "{substring}"
        );
    }
    assert!(row.glyphs.iter().any(|g| g.extra_glyphs.is_some()));
    let ink: usize = row
        .glyphs
        .iter()
        .map(|g| {
            usize::from(!g.uv_rect.is_nothing())
                + g.extra_glyphs.as_ref().map_or(0, |extra| extra.len())
        })
        .sum();
    assert_eq!(row.visuals.glyph_vertex_range.len(), ink * 4);
}
#[test]
fn arrow_navigation_stops_at_visual_edges_and_crosses_wrapped_rows() {
    let ctx = context();
    let galley = layout(&ctx, "مرحبا", 800.0, 24.0, false);
    assert_eq!(
        galley.cursor_right_one_character(&CCursor::new(0)).index.0,
        0
    );
    assert_eq!(
        galley.cursor_left_one_character(&CCursor::new(5)).index.0,
        5
    );
    let galley = layout(&ctx, "مرحبا بكم في البرنامج", 65.0, 24.0, false);
    assert!(galley.rows.len() > 1);
    let end = galley.rows[0].char_count_excluding_newline().0;
    let next = galley.cursor_left_one_character(&CCursor::new(end));
    assert_eq!(galley.layout_from_cursor(next).row, 1);
}
#[test]
fn copying_from_the_shared_editor_keeps_logical_arabic_and_filename_order() {
    let ctx = context();
    let expected = "حفظ image-12.png بنسبة 75%";
    let mut text = expected.to_owned();
    let id = egui::Id::new("arabic_copy_editor");
    let _ = ctx.run_ui(Default::default(), |ui| {
        let output = egui::TextEdit::singleline(&mut text).id(id).show(ui);
        output.response.request_focus();
        let mut state = output.state;
        state.cursor.set_char_range(Some(CCursorRange::two(
            CCursor::new(0),
            CCursor::new(text.chars().count()),
        )));
        state.store(&ctx, id);
    });
    let output = ctx.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Copy],
            ..Default::default()
        },
        |ui| {
            egui::TextEdit::singleline(&mut text).id(id).show(ui);
        },
    );
    assert!(
        output.platform_output.commands.iter().any(
            |command| matches!(command,egui::OutputCommand::CopyText(value) if value==expected)
        )
    );
    assert_eq!(text, expected);
}

#[test]
fn rtl_rich_text_background_and_underline_cover_the_visual_row() {
    let ctx = context();
    let mut galley = None;
    let _ = ctx.run_ui(Default::default(), |ui| {
        let mut job = LayoutJob::simple(
            "مرحبا بالعالم".into(),
            FontId::proportional(24.0),
            Color32::WHITE,
            800.0,
        );
        job.sections[0].format.background = Color32::RED;
        job.sections[0].format.underline = egui::Stroke::new(1.0, Color32::GREEN);
        galley = Some(ui.painter().layout_job(job));
    });
    let galley = galley.unwrap();
    let row = &galley.rows[0];
    let mesh = &row.visuals.mesh;
    let red: Vec<_> = mesh
        .vertices
        .iter()
        .filter(|v| v.color == Color32::RED)
        .collect();
    assert!(!red.is_empty());
    let min = red.iter().map(|v| v.pos.x).fold(f32::INFINITY, f32::min);
    let max = red
        .iter()
        .map(|v| v.pos.x)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(min <= row.rect().left() + 1.0);
    assert!(max >= row.rect().right() - 1.0);
    assert!(mesh.vertices.iter().any(|v| v.color == Color32::GREEN));
}
