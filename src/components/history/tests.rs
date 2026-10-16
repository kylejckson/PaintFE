use super::*;

fn contains_texture(shape: &egui::epaint::Shape, id: egui::TextureId) -> bool {
    match shape {
        egui::epaint::Shape::Mesh(mesh) => mesh.texture_id == id,
        egui::epaint::Shape::Vec(shapes) => shapes.iter().any(|s| contains_texture(s, id)),
        _ => false,
    }
}

#[test]
fn every_history_action_has_a_reviewed_icon_and_draws_its_real_texture() {
    let expectations = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/history-icons.tsv"
    ));
    let expected: std::collections::BTreeMap<_, _> = expectations
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split_once('\t').unwrap())
        .collect();
    assert_eq!(expected.len(), HistoryAction::ALL.len());
    let ctx = egui::Context::default();
    let mut assets = Assets::new();
    assets.init(&ctx);
    for style in [
        crate::config::icon_packs::BundledIconStyle::Classic,
        crate::config::icon_packs::BundledIconStyle::Luminous,
    ] {
        assets.set_bundled_icon_style(style).unwrap();
        for dark in [false, true] {
            assets.update_theme(&ctx, !dark);
            assets.update_theme(&ctx, dark);
            let theme = if dark {
                crate::theme::Theme::dark()
            } else {
                crate::theme::Theme::light()
            };
            for &action in HistoryAction::ALL {
                let name = format!("{action:?}");
                assert_eq!(
                    expected[name.as_str()],
                    format!("{:?}", action.icon()),
                    "{name}"
                );
                let texture = assets
                    .get_texture(action.icon())
                    .expect("registered History texture")
                    .id();
                let mut state = CanvasState::new(4, 4);
                state.layers[0].pixels.put_pixel(0, 0, Rgba([0, 0, 0, 0]));
                let mut command = SingleLayerSnapshotCommand::new(
                    action.label("任意 / Gradient / Selection"),
                    &state,
                );
                state.layers[0].pixels.put_pixel(0, 0, Rgba([1, 2, 3, 255]));
                command.set_after(&state);
                let mut history = HistoryManager::new(10);
                history.push(Box::new(command));
                let entry = history.presentation_history().remove(0);
                assert_eq!(entry.action, action);
                assert_eq!(entry.icon, action.icon());
                let output = ctx.run_ui(egui::RawInput::default(), |ui| {
                    HistoryPanel::card(ui, &assets, &theme, &entry, false, None, 64., 1.);
                });
                assert!(
                    output
                        .shapes
                        .iter()
                        .any(|s| contains_texture(&s.shape, texture)),
                    "{name}, dark={dark}"
                );
                history.undo(&mut state);
                assert_eq!(state.layers[0].pixels.get_pixel(0, 0)[3], 0);
                history.redo(&mut state);
                assert_eq!(state.layers[0].pixels.get_pixel(0, 0).0, [1, 2, 3, 255]);
                assert_eq!(history.presentation_history()[0].action, action);
                eprintln!("HISTORY_ICON\t{name}\t{:?}\tdark={dark}", entry.icon);
            }
        }
    }
}

#[test]
fn custom_pack_uses_the_same_history_texture_and_falls_back_for_other_actions() {
    let dir = std::env::temp_dir().join(format!("paintfe-history-icons-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&dir).unwrap();
    image::RgbaImage::from_pixel(2, 2, Rgba([17, 234, 19, 255]))
        .save(dir.join("tool_magic_wand.png"))
        .unwrap();
    let ctx = egui::Context::default();
    let mut assets = Assets::new();
    assets.init(&ctx);
    assets.load_icon_pack(&dir).unwrap();
    for dark in [false, true] {
        assets.update_theme(&ctx, !dark);
        assets.update_theme(&ctx, dark);
        assert!(assets.icon_source(Icon::MagicWand).from_pack());
        assert!(!assets.icon_source(Icon::MeshWarp).from_pack());
        let entry = HistoryPresentation {
            action: HistoryAction::MagicWand,
            icon: Icon::MagicWand,
            title: "Magic Wand".into(),
            detail: None,
        };
        let texture = assets.get_texture(entry.icon).unwrap().id();
        let output = ctx.run_ui(egui::RawInput::default(), |ui| {
            HistoryPanel::card(
                ui,
                &assets,
                &crate::theme::Theme::light(),
                &entry,
                false,
                None,
                64.,
                1.,
            );
        });
        assert!(
            output
                .shapes
                .iter()
                .any(|s| contains_texture(&s.shape, texture))
        );
        for &action in HistoryAction::ALL {
            assert!(assets.has_texture(action.icon()), "{action:?}");
        }
    }
    assert!(dir.starts_with(std::env::temp_dir()));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn layer_operation_identity_ignores_names_and_tracks_direction_and_visibility() {
    for (operation, expected) in [
        (
            LayerOperation::Add {
                index: 0,
                name: "Gradient Selection Text".into(),
                width: 1,
                height: 1,
                folder_id: None,
            },
            HistoryAction::AddLayer,
        ),
        (
            LayerOperation::Move {
                from_index: 0,
                to_index: 1,
            },
            HistoryAction::MoveLayer,
        ),
        (
            LayerOperation::Move {
                from_index: 1,
                to_index: 0,
            },
            HistoryAction::MoveLayerDown,
        ),
        (
            LayerOperation::Opacity {
                index: 0,
                old_opacity: 1.,
                new_opacity: 0.5,
            },
            HistoryAction::LayerOpacity,
        ),
        (
            LayerOperation::Visibility {
                index: 0,
                was_visible: true,
            },
            HistoryAction::HideLayer,
        ),
        (
            LayerOperation::Visibility {
                index: 0,
                was_visible: false,
            },
            HistoryAction::ShowLayer,
        ),
        (
            LayerOperation::Rename {
                index: 0,
                old_name: "Brush".into(),
                new_name: "Magic Wand".into(),
            },
            HistoryAction::RenameLayer,
        ),
    ] {
        let command = LayerOpCommand::new(operation);
        assert_eq!(command.action(), expected);
        assert_eq!(command.presentation().unwrap().icon, expected.icon());
    }
}

#[test]
fn missing_texture_still_paints_a_visible_history_fallback() {
    let ctx = egui::Context::default();
    let assets = Assets::new();
    let entry = HistoryPresentation {
        action: HistoryAction::MagicWand,
        icon: Icon::MagicWand,
        title: "Selection".into(),
        detail: None,
    };
    let output = ctx.run_ui(egui::RawInput::default(), |ui| {
        HistoryPanel::card(
            ui,
            &assets,
            &crate::theme::Theme::light(),
            &entry,
            false,
            None,
            64.,
            1.,
        );
    });
    assert!(
        output
            .shapes
            .iter()
            .any(|s| matches!(s.shape, egui::epaint::Shape::Text(_)))
    );
}
