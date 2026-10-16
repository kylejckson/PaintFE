use super::*;

fn case(
    expected: HistoryAction,
    operation: impl FnOnce(&mut LayersPanel, &mut CanvasState, &mut HistoryManager, u64),
) {
    let mut panel = LayersPanel::default();
    let mut state = CanvasState::new(8, 8);
    state.layers.push(Layer::new(
        "Gradient Selection".into(),
        8,
        8,
        image::Rgba([17, 23, 41, 255]),
    ));
    let mut history = HistoryManager::new(100);
    panel.add_layer_folder(&mut state, &mut history);
    let folder = state.layer_folders[0].id;
    let count = history.undo_count();
    operation(&mut panel, &mut state, &mut history, folder);
    assert!(history.undo_count() > count, "no entry for {expected:?}");
    let entry = &history.presentation_history()[0];
    assert_eq!(entry.action, expected);
    assert_eq!(entry.icon, expected.icon());
    history.undo(&mut state);
    history.redo(&mut state);
    assert_eq!(history.presentation_history()[0].action, expected);
}

#[test]
fn layer_and_folder_producers_emit_their_own_history_identity() {
    case(HistoryAction::AddLayer, |p, s, h, _| p.add_new_layer(s, h));
    case(HistoryAction::AddTextLayer, |p, s, h, _| {
        p.add_new_text_layer(s, h)
    });
    case(HistoryAction::AddAdjustmentLayer, |p, s, h, _| {
        p.add_adjustment_layer(s, h)
    });
    case(HistoryAction::AddFolder, |p, s, h, _| {
        p.add_layer_folder(s, h)
    });
    case(HistoryAction::AddFolder, |p, s, h, f| {
        p.add_layer_folder_above(f, s, h)
    });
    case(HistoryAction::MoveFolder, |p, s, h, f| {
        p.move_folder_block(f, 1, s, h)
    });
    case(HistoryAction::MoveToFolder, |p, s, h, f| {
        p.set_layer_folder(0, Some(f), s, h)
    });
    case(HistoryAction::RenameFolder, |p, s, h, f| {
        p.rename_layer_folder(f, "Clone Gradient Text".into(), s, h)
    });
    case(HistoryAction::HideFolder, |p, s, h, f| {
        p.toggle_layer_folder_visibility(f, s, h)
    });
    case(HistoryAction::FolderVisibility, |p, s, h, f| {
        s.layer_folder_mut(f).unwrap().visible = false;
        p.toggle_layer_folder_visibility(f, s, h)
    });
    case(HistoryAction::DeleteFolder, |p, s, h, f| {
        p.delete_layer_folder(f, s, h)
    });
    case(HistoryAction::DeleteLayer, |p, s, h, _| {
        p.delete_layer(1, s, h)
    });
    case(HistoryAction::DuplicateLayer, |p, s, h, _| {
        p.duplicate_layer(1, s, h)
    });
    // The UI moves a layer up by increasing its index, and down by decreasing it.
    case(HistoryAction::MoveLayer, |p, s, h, _| {
        p.move_layer(0, 1, s, h)
    });
    case(HistoryAction::MoveLayerDown, |p, s, h, _| {
        p.move_layer(1, 0, s, h)
    });
    case(HistoryAction::MoveLayer, |p, s, h, _| {
        p.move_layer_group(vec![0], 2, None, s, h);
    });
    case(HistoryAction::MoveLayerDown, |p, s, h, _| {
        p.move_layer_group(vec![1], 0, None, s, h);
    });
    case(HistoryAction::MoveFolderDown, |p, s, h, f| {
        p.set_layer_folder(1, Some(f), s, h);
        p.move_folder_block(f, 0, s, h);
    });
    case(HistoryAction::MergeDown, |p, s, h, _| p.merge_down(1, s, h));
    case(HistoryAction::Flatten, |p, s, h, _| p.flatten_image(s, h));
}
