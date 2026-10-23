# Save safety

Save, Save As, Save All and saving before closing share the implementation in
`src/app/save_flow.rs`. Format warnings appear before writing a file, with choices
to save a PaintFE project, continue saving the image, or cancel. Layered documents
initially suggest PFE in Save As; the user can still choose an image format.

PFE retains the project's supported editable structure. Image exports preserve
the open layers and keep a layered document dirty so exporting cannot silently
replace its editable project save. Exporting an existing PFE document also keeps
its original PFE save target. A close/exit operation may proceed after an explicitly
approved image export, but only once the write succeeds for the same document
revision. It is still the user's choice to close without an editable copy.

Documents containing layer masks use PFE4, which stores mask pixels and the
enabled state. Older PaintFE versions cannot open these files. Documents without
masks keep using the existing PFE versions, and existing project files still open.

Save results use UUIDs instead of tab positions. An edit made while a normal save
runs remains unsaved. Save All, tab close and exit process documents sequentially;
cancelling a warning or picker stops that sequence. Save failures show an error
and keep the document open. Saves are written to a unique temporary file beside
the target and flushed before replacement, so encoding failures do not truncate
the original. This is not a guarantee against power loss or filesystem failure.

The new safety text has English fallback entries in every language catalogue.
Native translations of these new messages can be reviewed separately.

## Manual checks

1. Open a disposable PNG, add a second layer and draw on it. Press Ctrl+S.
   Confirm the warning appears before the PNG changes. Cancel and verify the
   canvas and unsaved indicator remain.
2. Repeat and choose Save PaintFE project. Save to a new PFE, reopen it and
   confirm the layers are separately editable.
3. Export that PFE to PNG using Save As. Choose Save image anyway. Confirm the
   open layers remain and Ctrl+S still targets the PFE.
4. Open the PNG again, add a layer, close the tab and choose Save. Confirm the
   loss warning appears and the tab closes only after a successful write.
   Cancel either dialog and confirm the tab remains open.
5. With two dirty documents, use Save All and then test exiting with Save.
   Confirm each document is processed and cancelling one prompt keeps the app
   open. Try an unwritable destination and confirm the error prevents closing.
6. Try a single editable text layer, transparency saved to JPEG and a high-depth
   image saved to an 8-bit format. Confirm relevant warnings. Repeat the dialog
   checks in light and dark mode using Tab, Enter and Escape.

Focused regressions: `cargo test -j1 --lib save_safety_tests -- --test-threads=1`.
