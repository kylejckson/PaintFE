# Save Safety

Save, Save As, Save All, and saving before closing share the implementation in `src/app/save_flow.rs`. Format warnings appear before writing a file, with choices to save a PaintFE project, continue saving the image, or cancel. Layered documents initially suggest PFE in Save As; the user can still choose an image format.

PFE keeps the project's editable structure. Image exports preserve the open layers and keep a layered document marked dirty so exporting cannot silently replace its editable project save. Exporting an existing PFE document also keeps its original PFE save target. A close or exit action may continue after an explicitly approved image export, but only after the write succeeds for the same document revision. It is still the user's choice to close without an editable copy.

Documents containing layer masks use PFE4, which stores mask pixels and the enabled state. Older PaintFE versions cannot open these files. Documents without masks keep using the existing PFE versions, and existing project files still open.

Save results use UUIDs instead of tab positions. An edit made while a normal save is running remains unsaved. Save All, tab close, and exit process documents sequentially; canceling a warning or picker stops that sequence. Save failures show an error and keep the document open. Saves are written to a unique temporary file beside the target and flushed before replacement, so encoding failures do not truncate the original. This is not a guarantee against power loss or filesystem failure.

The new safety text has English fallback entries in every language catalog. Native translations of these new messages can be reviewed separately.

## Manual Checks

1. Open a disposable PNG, add a second layer, and draw on it. Press Ctrl+S. Confirm the warning appears before the PNG changes. Cancel and verify that the canvas and unsaved indicator remain.
2. Repeat and choose Save PaintFE project. Save to a new PFE, reopen it, and confirm the layers remain separately editable.
3. Export that PFE to PNG using Save As. Choose Save image anyway. Confirm the open layers remain and Ctrl+S still targets the PFE.
4. Open the PNG again, add a layer, close the tab, and choose Save. Confirm the loss warning appears and the tab closes only after a successful write. Cancel either dialog and confirm the tab remains open.
5. With two dirty documents, use Save All and then test exiting with Save. Confirm each document is processed, and canceling one prompt keeps the app open. Try an unwritable destination and confirm the error prevents closing.
6. Try a single editable text layer, transparency saved to JPEG, and a high-depth image saved to an 8-bit format. Confirm the relevant warnings. Repeat the dialog checks in light and dark mode using Tab, Enter, and Escape.

Focused regressions: `cargo test -j1 --lib save_safety_tests -- --test-threads=1`.
