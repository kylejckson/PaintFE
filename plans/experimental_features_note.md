# Experimental Feature Branch Notes

## New

- Added experimental high-depth pixel payload support for RGBA UI16, RGBA FP16, and RGBA FP32 project data.
- Added HDR metadata and tone-mapped preview helpers for high-dynamic-range image data.
- Added editable high-depth paths so brush edits update dirty regions in deep pixel buffers instead of only preserving them.
- Added adjustment layers with exposure, brightness/contrast, invert, and channel mixer adjustment data.
- Added channel extraction and alpha replacement tools.
- Added layer folders with collapse, visibility, rename, drag/reorder, multi-layer selection, folder colors, selectable contents, and PFE persistence.
- Added folder color palette customization in Preferences.
- Added PFE v3 round-trip support for folders, adjustment layers, high-depth payloads, HDR metadata, and source metadata.
- Added import/export coverage for 16-bit PNG/TIFF and FP32 HDR TIFF.
- Added experimental feature tests covering high-depth editing, HDR export, adjustment layers, metadata, and PFE round trips.

## Changed

- Composite/export paths now respect folder visibility and adjustment layers.
- Layer panel add/reorder behavior now inserts relative to the selected layer or selected folder.
- Folder row icons use the existing asset icon inversion path for readable contrast over folder colors.
- Layer and folder rename fields now keep keyboard shortcuts local to text editing, including Ctrl+A and Enter/Escape handling.
- Mirror mode changes now invalidate preview state and force redraw to avoid stale tool preview buffers.
- Layers panel window size now persists across restarts.

## Fixed

- Fixed startup/open crash caused by mismatched GPU texture upload dimensions when opening images directly from file manager or CLI on Linux/KDE.
- Fixed high-depth images being only preserved: edits now sync back into 16-bit/HDR deep buffers for dirty regions or full-layer operations.
- Fixed adjustment layers being visually ineffective in common composites.
- Fixed metadata preservation for supported PNG text/source metadata paths, including Stable Diffusion-style text chunks.
- Fixed folder visibility not affecting composite/export output.
- Fixed layer-folder reorder edge cases for dragging folders and layers in/out of folders.
- Fixed layer/folder rename keyboard shortcuts leaking through to canvas selection.
- Fixed folder icon contrast over configured folder colors.
- Fixed layers panel resized height not persisting across restarts.
- Fixed mirror-mode toggling while editing from reusing stale preview state.
