# Changelog

All notable changes to PaintFE are documented here. Dates are in YYYY-MM-DD format.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.17] - 2026-05-20

### Added
- Color to Alpha -- Added a new Color > Color to Alpha dialog for removing a chosen color from the active layer or selection while deriving transparency from mixed pixels.
- Color sampling -- Added target-color controls with primary-color defaults, manual color editing, and coordinate-based image sampling inside the Color to Alpha dialog.

### Fixed
- Live preview history -- System dialog live previews now update the visible layer without adding undo history entries; only applying the dialog records the final edit.

## [1.2.16] - 2026-05-18

### Fixed
- Preferences without a project -- Edit > Preferences can now be opened before any file/project is open.
- Windows live resize -- Reduced dropped frames during corner-drag resizing by deferring full canvas GPU recomposite/readback work until the live resize finishes.
- Hardware preferences -- Removed the unused GPU acceleration placeholder toggle while keeping preferred adapter selection.

## [1.2.15] - 2026-05-11

### Changed
- Window maximize responsiveness -- Reduced maximize/resize stutter by throttling window-state persistence and keeping GPU compositing on the async path during repaint-heavy updates.

### Fixed
- Drag-and-drop duplicate opens -- Dropping an image into PaintFE no longer opens the same file twice in separate projects.

## [1.2.14] - 2026-05-01

### Added
- Custom SVG shapes -- Added a new Shapes picker workflow for importing, previewing, and persisting custom SVG-based shapes across restarts.
- Move Selection handles -- Added resize handles to the Move Selection tool so selection bounds can be transformed directly.
- Interface customization -- Added preferences for tool button corner radius and selection stripe color/alpha.

### Changed
- Floating panel anchoring -- Improved Color and History panel anchoring, persistence, and fullscreen/resize behavior.
- Shape editing workflow -- Tightened shape snapping, minimum sizing, polygon stretching, inward outline growth, and overall transform behavior.
- Paste overlay presentation -- Improved pixel-grid visibility and transform interaction behavior while editing pasted content.
- Save As shortcut routing -- Ctrl+Shift+S now resolves correctly to Save As.

### Fixed
- Adjustment dialog crash -- Opening Vibrance, Threshold, Posterize, Color Balance, Gradient Map, and Black & White no longer panics.
- Oversized image rejection -- Images above 25,000 px in width or height are now rejected before opening.
- Paste transform history -- Paste overlay movement checkpoints now round-trip correctly through undo/redo.
- Paste Commit & Select -- Commit & Select now targets the pasted area instead of selecting the canvas.
- Hue/Saturation layout bounce -- Removed the dialog layout jitter around the numeric controls.
- Keybind unbind persistence -- Explicitly unbound shortcuts now stay unbound across restarts.
- Mirror centering -- Mirror placement now reflects around the correct pixel-centered axis.
- Shift resize anchoring -- Shift-constrained pasted-image resizing now holds the opposite edge/corner correctly.
- Project tab hitboxes -- Tab body selection and close-button behavior now work together reliably.
- Drop Shadow and Outline -- Improved white shadow fidelity and sped up heavy outline/shadow operations.
- Shape correctness -- Corrected heart alignment and multiple pentagon/hexagon/octagon stretch/render issues.

## [1.2.13] - 2026-04-26

### Added
- Custom brush tips workflow -- Added a dedicated Add Brush Tip dialog so PNG brush tips can be imported, categorized, saved across restarts, and managed directly from the brush tip picker.

### Changed
- Shape fill mode controls -- Replaced the Outline/Filled/Both text dropdown with icon-based controls for faster switching in the shape tool context bar.

### Fixed
- Text layer editing regressions -- Restored reliable text layer selections and property updates so width, height, bold styling, and related text editing controls behave correctly again.

## [1.2.11] - 2026-04-25

### Bug Fixes
- Fixed Ctrl+Shift+Z keybind -- Redo shortcut now works reliably.
- Fixed paste freeze on oversized images -- Pasting an image larger than the canvas (via Ctrl+V or Edit > Paste) no longer hangs the application.
- Fixed project tab hitboxes -- Clickable area now spans the entire tab widget (excluding the close X button), not just the text label.
- Fixed unbound key persistence -- Keys set to "unbound" in Preferences > Keybinds now survive restarts.
- Fixed Wayland drawing tablet input -- Cursor icon, keyboard shortcuts, and cursor position tracking now work correctly when using a tablet on Wayland (workaround for missing zwp_tablet_v2 protocol support in winit).

### New Features
- Paste preview overlay - Pasted images now show a preview with transform handles before committing. Right-click for context menu options.
- Optimized paste previews in Overwrite mode -- Faster rendering when pasting with overwrite blending.
- Layer Align moved to Transform menu -- Right-click a layer > Transform > Align... (moved from the Canvas menu, sits just above Rotate/Scale).
- Alpha settings in Interface preferences -- New alpha-related display options in the Interface settings tab.
- Del key cancels paste -- Press Delete to dismiss an active paste overlay.
- Control corner rounding for Badges and Project Tabs -- New preference options to customize corner radii.

### Improvements
- Enhanced logging - PaintFE now logs more detailed diagnostic information for troubleshooting.
- Text field capture guard - Canvas shortcuts no longer fire while typing in dialog text fields (font search, file name, settings values).

## [1.2.10] - 2026-04-24

### Added
- **Paste alpha preservation option**: Added a configurable option to preserve alpha when pasting content.
- **Line tool tracker**: Added a dedicated tracker for line-tool interactions.
- **Length/position info line**: Added a new on-canvas info line that reports measurement and position details.

### Changed
- **Major modular refactor**: Broke up large modules into smaller, more maintainable components.
- **Settings/UI configurability**: Expanded and refined settings options for improved customization.
- **Off-canvas tool behavior**: Improved brush/tool behavior when interacting near or beyond canvas bounds.
- **Fill tool rewrite**: Reworked the fill pipeline for more accurate region detection and application.

### Fixed
- **Selection persistence**: Selection state now remains consistent and reliable across editing flows.
- **Panel sizing and option polish**: Fixed panel sizing issues and related option behavior inconsistencies.
- **Shape correctness fixes**: Addressed shape rendering/behavior issues for more predictable results.

---

## [1.2.9] - 2026-04-23

### Added
- **Multi-file open dialog**: The native Open flow can now select multiple files at once and opens each file into its own project tab.
- **Fill reseed regression coverage**: Added a focused test for the Fill tool's commit-and-reseed-on-next-press behavior.

### Changed
- **Fill tool interaction**: The next Fill click now commits the current preview and immediately starts previewing the newly clicked region in the same interaction.
- **History and settings window sizing**: Relaxed fixed-size limits so the History panel and Settings window can be resized more comfortably.
- **Selection-aware tool behavior**: Tool operations now honor active selections more consistently, producing tighter, more predictable edits inside isolated regions.
- **Keyboard shortcut expressiveness**: Keybinding capture and display are clearer, with broader support for character input and more readable shortcut definitions.
- **Outline rendering quality**: Outline-based effects now render with improved edge fidelity, including anti-aliased output for smoother contours.
- **Shape rasterization crispness**: Shape output has been refined for cleaner, sharper geometry with more deliberate edge handling.

### Fixed
- **Tab close button hover flicker**: Non-active tabs now keep a stable close button instead of flickering when the pointer moves onto the button.
- **Selection history fidelity**: Selection changes now round-trip through undo/redo more reliably, preserving editing context across history operations.
- **Text layer deletion cleanup**: The Text tool now clears its active-layer editing state correctly when the currently edited layer is deleted.

---

## [1.2.8] - 2026-04-22

### Added
- **Line anti-aliasing control**: Added anti-aliasing support for line rendering with a dedicated toggle for sharper or smoother edges as needed.
- **Selection alpha copy option**: Added a setting to control transparent cutout behavior when copying selections, preserving expected alpha semantics for compositing workflows.
- **Improved copy/paste transform workflow**: Added support for copying transformed pasted content and re-pasting it at the original placement.

### Changed
- **Fill precision for pixel art**: Improved Fill bucket precision for low-resolution and narrow features, with better behavior on 1px structures.
- **Color widget interaction**: Updated color selection controls to respond reliably on click/press interactions.
- **Customization and control polish**: Refined input handling, slider behavior, and preference control consistency across settings and tool dialogs.
- **Shape rendering rework**: Reworked Heart, Trapezoid, and Cross shape implementations for more stable and predictable output.
- **Text tool font support**: Expanded text font handling and settings behavior, including stronger cross-platform support on Linux and macOS.

### Fixed
- **Wayland startup noise**: Suppressed non-critical Wayland/Linux runtime noise for cleaner startup diagnostics.
- **Async cancellation robustness**: Improved cancellation and lifecycle handling for async actions and preview jobs.
- **egui color picker issues**: Resolved color picker regressions in settings and related UI flows.
- **Layer panel text interaction**: Removed unintended text selection behavior in the layers widget.
- **Window state handling**: Fixed fake fullscreen/maximized persistence and restoration behavior.
- **Selection commit behavior**: Ensured Ctrl+A and related selection actions properly commit active transform/preview states.
- **Layer deletion selection state**: Selection is now cleared correctly when deleting the active layer containing it.
- **Liquify history tracking**: Fixed Liquify commits not being captured correctly in undo/redo history.
- **Paste lifecycle isolation**: Fixed image paste overlays persisting across new files/projects.
- **Selection-aware transforms**: Flip/rotate operations now correctly apply within active selections.
- **Pencil tool stability**: Fixed edge-case Pencil behavior and improved reliability.

---

## [1.2.7] - 2026-04-22

### Changed
- **Magic Wand accuracy update**: Reworked selection behavior for better precision and predictability, including improved click-mode semantics and whole-canvas color matching with Ctrl+Shift.
- **Magic Wand additive workflow**: Subsequent additive clicks now behave as independent selection origins for tolerance-driven updates, improving disconnected region targeting.
- **Magic Wand hover targeting**: Added hovered pixel indicator to make seed-pixel selection clearer before clicking.

### Fixed
- **Magic Wand mode behavior**: Normal mode now correctly replaces prior selections instead of unintentionally accumulating regions.
- **Magic Wand selection finalization**: Improved session finalization and history consistency for wand-driven selection edits.

---

## [1.2.0] - 2026-04-20

### Added
- **Canvas Align dialog**: Added Canvas > Align with a 3x3 anchor grid and live preview to align the active raster layer using non-transparent bounds.
- **Canvas Border filter**: Added Filter > Stylize > Canvas Border with width control and live preview, applying an inward border from canvas edges using the primary color.
- **Color Palette widget workflow**: Added palette-focused UI workflow updates to improve color organization and palette usability.
- **New icon registrations/placeholders**: Added menu icon placeholders and wiring for new actions (Align and Canvas Border).
- **Regression coverage**: Added transform/filter tests for layer alignment and canvas border behavior.

### Changed
- **Version bump**: Updated release version from 1.1.13 to 1.2.0.
- **egui stack update**: Updated egui/eframe stack with major UX/platform improvements, including better Linux drawing tablet behavior, improved font rendering quality, and cleaner overall UI presentation.
- **Dialog consistency pass**: Refined dialog controls and layout behavior across operation/effect dialogs for more consistent interaction.
- **Menu/dialog routing expansion**: Extended app menu and dialog routing for newly introduced actions and workflows.
- **Assets/localization expansion**: Extended icon loading maps and English localization strings for new and updated UI features.

### Fixed
- **Stability and regressions**: Cleaned and revalidated build/test/clippy issues encountered during this release cycle.

---

## [1.1.13] - 2026-04-16

### Added
- **Arrow slider thumbs**: All sliders in the Color panel (RGB and HSV channels)
  and every dialog slider now use Paint.NET-style arrowhead thumbs — a solid black
  body with a white outline — instead of the default round handle, making the
  current value position crisp and immediately visible.
- **Step increment/decrement controls in Color panel**: Each RGB and HSV channel
  slider now has flanking − and + buttons alongside a draggable value field,
  allowing single-unit nudging of channel values without grabbing the slider.

### Fixed
- **Text tool typing regression**: Typed characters were silently swallowed when
  any DragValue in the text tool options bar (letter spacing, scale, etc.) or a
  floating color panel slider retained keyboard focus. The canvas now pre-claims
  focus at the start of each frame whenever the text tool is actively editing,
  preventing options-bar and panel widgets from intercepting keystrokes.

---

## [1.1.12] - 2026-04-15

### Added
- **Ctrl+LMB adds to selection**: Holding Ctrl while starting a drag with the
  Rectangle Select, Ellipse Select, or Lasso tool now enters Add mode, matching
  the behaviour of Shift (Union) and Alt (Subtract).
- **Step arrows and reset buttons in Settings**: Every slider and number input in
  the Settings window now has ◀/▶ buttons to increment or decrement by one step,
  and a ↺ reset button (dimmed when the value already equals the default) to
  restore the factory default without reopening the window.
- **Keyboard shortcuts for Color, Filter, and Generate menus**: All 51 menu
  actions across the Color, Filter, and Generate menus are now rebindable via
  Settings > Keybindings. Instant Color adjustments (Auto Levels, Desaturate,
  Invert Colors, Invert Alpha, Sepia Tone) apply immediately; all dialog-based
  actions open their respective dialogs. Remove Background additionally requires
  ONNX Runtime to be configured.

---

## [1.1.11] - 2026-04-14

### Added
- **Interactive save preview**: The Save As dialog preview panel now supports zoom
  and pan. Scroll to zoom, drag to pan, double-click to reset to fit. NEAREST
  filtering activates automatically when zoom exceeds 2× for pixel-crisp display.
  Scrollbar indicators and a zoom strip (−, %, +, Fit) are shown below the panel.

### Fixed
- **Minimum selection drag threshold**: Accidental single-pixel drags no longer
  create a selection; a minimum drag distance is now required.
  *(Reported by wigiliuszek-byte — closes #31)*
- **Paste overlay pixel alignment**: Pasted content could land on sub-pixel
  positions, causing edge fringing. The overlay center is now snapped to whole
  pixels on every move, arrow-key nudge, and Tab-center.
  *(Reported by wigiliuszek-byte — closes #32)*
- **Shift key conflict in selection mode**: Holding Shift while using a selection
  tool no longer inadvertently triggered the fill shortcut.
  *(Reported by wigiliuszek-byte — closes #33)*
- **Magic Wand selection behaviour**: Corrected edge cases where the Magic Wand
  tool produced unexpected or empty selections.
  *(Reported by wigiliuszek-byte — closes #34)*
- **Ctrl++ zoom shortcut not working**: The zoom-in keybind now correctly responds
  to Ctrl++ (which sends Shift on physical keyboards) in addition to Ctrl+=.
  *(Reported by wigiliuszek-byte — closes #35)*
- **Selection changes not tracked in undo/redo**: Selecting, deselecting, and
  Ctrl+A are now tracked with `SelectionCommand`; the selection mask is also
  captured and restored in canvas snapshots, so cut/paste no longer discards the
  selection on undo.
  *(Reported by wigiliuszek-byte — closes #36)*
- **Shift+Fill not working globally**: Shift+Fill with the Fill tool now correctly
  applies a global flood fill regardless of selection state.
  *(Reported by wigiliuszek-byte — closes #37)*
- **Drawing tablet pressure not recognized on Wayland**: Tablet input events on
  Wayland are now handled via the correct libinput path, restoring pressure
  sensitivity for stylus users.
  *(Reported by Yasumora — closes #39)*

---

## [1.1.10] - 2026-03-25

### Fixed
- **Clippy**: Resolved CI build failures — collapsed nested `if let`/`if` in drag-and-drop handler; removed redundant always-false branches in keyboard shortcut helper.

---

## [1.1.9] - 2026-03-25

### Added
- **Hold Shift + drag to resize brush**: While any brush-based tool is active, holding Shift and dragging left/right now resizes the brush size interactively.
- **Drag and drop support**: Files can now be opened by dragging and dropping them onto the canvas.
- **Horizontal/vertical letter width modifiers for text layers**: Text layers now support per-axis character width scaling, enabling condensed or expanded type styles.

### Fixed
- **Text Layer rasterization**: Resolved issues with incorrect rendering when rasterizing vector text layers to pixel data.

### Improved
- **Text Layer drag performance**: Dragging/moving text at higher resolutions is significantly faster.

---

## [1.1.7] - 2026-03-20

### Added
- **Configurable startup canvas**: Settings > General now has a "Startup Canvas" section. The default canvas size (width × height) is configurable. A toggle lets users disable the startup canvas entirely so the app opens to an empty workspace — useful for users who always open existing files.
- **Empty app state**: Closing the last project tab no longer auto-creates a replacement blank canvas. The app can now be fully empty; all tools and menus (Edit, Canvas, Color, Filter, Generate) are grayed out until a project is opened via File > New or File > Open.
- **Integration test suite (260 tests)**: 14 test files covering visual filters, color adjustments, blend modes, transforms, shapes, tool strokes, layer ops, selection, text layers, scripting API, IO roundtrips, GPU pipelines, inpainting, and Catmull-Rom/affine math. Golden image reference system with per-channel tolerance support.

### Fixed
- Potential panic when using the Move Selection tool with no projects open.
- Pre-existing clippy warnings in scripting host API (`map_or(true, …)` → `is_none_or`).

---

## [1.1.6] - 2026-03-16

### Added
- **GPU-accelerated Fill and Magic Wand tools**: Distance maps for flood fill and Magic Wand selection are now computed on the GPU using a dedicated compute pipeline (wave-function BFS on GPU), giving near-instant response on large canvases. GPU fill preview dirty regions are also rendered on GPU. CPU fallback retained for non-GPU environments.

### Fixed
- **Rectangle outline sharp corners**: Plain (non-rounded) rectangles drawn in Outline or Both (fill + outline) mode now produce sharp axis-aligned corners instead of the previous rounded artefact caused by the SDF-band approach. Rounded Rectangle still uses smooth corners as expected.
- **Windows error sound on every keypress**: A hook inside the Windows message loop was manually calling `DispatchMessageW` for `WM_KEYDOWN` events and marking them consumed, which broke winit's normal input routing and triggered `MessageBeep(0)` on every key press across the entire app (including resize dialogs). The intercept has been removed; only harmless control-character `WM_CHAR` suppression remains.

---

## [1.1.5] - 2026-03-16

### Added
- **Wrap preview (seamless tiling preview)**: New toolbar toggle (next to mirror mode) renders ghost copies of the canvas in all 8 surrounding positions so pixel artists can see how a texture tiles seamlessly. Live-updated during drawing. Off by default.

### Improved
- **New File dialog**: Width field is focused and selectable on open; aspect ratio lock is now a checkbox and defaults to on; width/height fields accept math expressions (`800/2`, `1920+100`, `512*3`, etc.); values round to integers on commit; expressions commit on Enter, Tab, or focus loss.
- **Resize Image dialog**: Same math-capable text fields as New File — expressions, integer rounding, commit on Enter/Tab/focus loss, width focused first. Aspect ratio lock converted from a selectable label to a proper checkbox (default on).
- **Resize Canvas dialog**: Same math-capable text fields and focus behavior as New File and Resize Image. Aspect ratio lock checkbox added (default on).

---

## [1.1.4] - 2026-03-15

### Improved
- **Static screen-space checkerboard**: Transparency checkerboard is now a static screen-anchored pattern (Paint.NET / Photoshop style) rendered as a single textured quad — O(1) cost at any zoom or canvas size, eliminates the previous per-cell rect tessellation that caused panning lag at 4K.
- **Eraser checkerboard alignment**: Eraser preview no longer bakes a canvas-resolution checkerboard into its texture (which caused moiré and misalignment at non-100% zooms). Instead, the screen-resolution checkerboard is drawn as an underlay, giving a seamless pattern at any zoom.
- **Release binary optimization**: Added `strip = true`, `lto = true`, and `codegen-units = 1` to the release profile for smaller binaries and reduced false-positive AV heuristic triggers.

### Fixed
- **Gradient tool crash on 4K images**: Fixed crash caused by `from_raw_rgba` receiving full canvas dimensions with a downscaled buffer, and CPU fallback using a different downscale factor than the GPU path.
- **Panning lag at intermediate zoom levels on 4K**: Root cause was checkerboard tessellation generating 41K rects at ~31% zoom. Resolved by the new texture-based approach (1 quad).
- **History panel scrollbar position**: Scrollbar no longer appears in the middle of the panel; uses `auto_shrink(false)` to fill available width without a feedback loop.

---

## [1.1.3] - 2026-03-09

### Improved
- **Magic Wand tool — monotonic selection with Dijkstra distance map**: The selection no longer shrinks unexpectedly or jumps when dragging the tolerance slider. A minimax (bottleneck) Dijkstra distance map is computed once on click; tolerance changes re-threshold the map instantly (O(n) parallel scan) with no re-flood-fill. Higher tolerance always adds pixels, never removes them. Computation is now async (runs on a background rayon thread so the UI stays responsive on large canvases). Threshold scan is parallelized and writes directly to raw buffers for minimum latency on tolerance drag. Anti-alias edge softening toggle added to the context bar.
- **Color picker secondary-swatch targeting**: Right-clicking with the color picker now directly sets the secondary color (previously always set primary).
- **Tab key swaps primary/secondary colors**: Pressing Tab while any non-text tool is active swaps the foreground and background colors.
- **Text gradient fill effect**: Text layers support a gradient fill effect with configurable start/end colors, angle, scale, offset, and tiling.

### Fixed
- **Off-canvas text clipping**: Text that extends outside the canvas bounds is no longer clipped at the canvas edge.

---

## [1.1.2] - 2026-03-06

### Fixed
- **History panel width growing**: The history panel no longer expands indefinitely when long action names are added — content is now capped to the panel width instead of ratcheting the minimum upward.
- **Font size cap removed**: The text tool font size is no longer capped at 500px and can now be set to any value.

---

## [1.1.1] - 2026-03-09

### Fixed
- **Text layer font switching**: Changing font family or weight now updates the canvas instantly (previously required clicking away and back).
- **Decorative/bulky font clipping**: Fonts with large ascenders, descenders, or wide strokes (e.g. Impact, Showcard Gothic) are no longer clipped or squished — rasterizer now uses actual outline bounds with scaled padding.
- **Text move handle off-canvas**: The drag handle for repositioning text blocks can now be grabbed even when it extends outside the visible canvas area.
- **GPU blur shader crash on DX12**: Fixed a crash on launch caused by the Gaussian blur compute shader using `workgroupBarrier()` inside divergent control flow, which the FXC compiler rejects.

### Added
- **Resize cursors on text bounding boxes**: Hovering over text box resize handles now shows the appropriate system resize cursor (↔, ↗↙, ↘↖) instead of the default pointer.

---

## [1.1.0] - 2026-03-06

### Added
- **Visual overhaul ("Signal Grid" design language)**: Complete UI redesign to match the PaintFE website aesthetic. Blue-tinted neutral backgrounds, orange primary accent, multi-tier depth system, rounded corners, subtle glow effects, and updated typography throughout.
  - **Project tabs**: Drag-to-reorder tab bar showing canvas dimensions (e.g. "800x600") on the active tab. Unsaved projects display a colored dot indicator. Close button appears on hover. Horizontal scrolling when many projects are open, with a `+` button for new projects.
  - **Floating tool shelf**: Redesigned compact vertical tool strip with frameless 26px icon buttons arranged in a 3-column grid. Tool groups separated by visual dividers. Active tool has an accent-colored glow background.
  - **Theme management**: Full theme customization with 13 built-in presets (Blue, Orange, Purple, Red, Green, Lime, Nebula, Ember, Sakura, Glacier, Midnight, Signal, Custom). Advanced settings for surface colors, accent colors (normal/faint/strong), glow intensity, shadow strength, and UI density. Export and import themes as `.paintfe-theme` files.
  - **Merged brush size selector**: Combined DragValue, preset dropdown, and +/- stepper buttons into a single bordered control. Preset dropdown provides quick access to common sizes (5, 10, 20, 30, 50, 75, 100px). Used consistently across Brush, Eraser, Text, Content Aware, and other sizing tools.
  - **Signal Grid canvas background**: Subtle 40px grid texture behind the canvas area matching the website design. Color-adaptive (blue-tinted gray in dark mode, dark blue-black in light mode). Auto-hides when grid cells would be smaller than 5px on screen. Toggleable in View settings.
  - **Color widget**: Compressed and reorganized layout with compact and expandable modes. Compact mode is width-constrained; expanded mode adds HSL sliders and a color preview column. Consistent 4px spacing throughout.
  - **Updated icons and fonts**: Refreshed toolbar icons and adjusted fonts/spacing to align with the Signal Grid design language.
- **Text layers**: Non-destructive editable text layers created via Canvas > New Text Layer or the Layers Panel right-click menu.
  - Rich text formatting with multiple font families, weights, sizes, italic, underline, strikethrough, letter spacing, and per-run coloring. Bold (Ctrl+B), Italic (Ctrl+I), Underline (Ctrl+U) keyboard shortcuts.
  - Multi-block editing: click empty areas to create new text blocks within a layer, Tab to cycle between blocks, per-block delete via the X button.
  - Word wrapping with resizable text boxes (drag side handles to set max width), rotation via the top handle, and repositionable blocks.
  - Text effects and warps accessible from the Layer Settings dialog (gear icon on text layer rows, or right-click > "Text Effects..." / "Text Warp..."). Effects include outline (inside/center/outside positions), drop shadow (with blur and spread), inner shadow, and texture fill (tiled from imported images). Geometric warps include arc (bend + distortion), circular (radius + start angle), and envelope (top/bottom deformation curves).
  - PFE V2 project file format automatically stores both vector text data and pre-rasterized pixels, so V1 readers can still display text layers (without editability). Auto-versioning selects V2 only when text layers are present.
  - Lightweight undo: TextLayerEditCommand stores only vector data (~1-50 KB per edit) instead of pixel snapshots.
  - Rasterize on demand via Layers Panel right-click, or automatically when painting on a text layer or exporting.
- **Arrow line endings on the Line tool**: Triangular arrowheads with configurable placement (start, end, or both ends of the line). Selectable via dropdowns in the Line tool context bar. (Closes #17, suggested by @zero54git)
- **Layer search/filter**: Search bar appears when more than 2 layers exist, providing real-time case-insensitive filtering with a clear button.
- **macOS builds**: Apple Silicon (ARM64) and Intel (x86_64) `.dmg` builds are now produced automatically by the release workflow, with macOS Clippy checks added to CI. (Closes #18, contributed by @fewtarius)
- **Pen/tablet pressure sensitivity**: Brush size and opacity respond to stylus pressure on supported devices, with configurable minimum thresholds for both. (Closes #18, contributed by @fewtarius)

### Fixed
- **Wayland window icon**: Set `app_id` on the Wayland viewport so compositors resolve the PaintFE icon from the desktop entry instead of showing the default egui icon.
- **macOS keybinding display**: Keyboard shortcuts now show Cmd/Option symbols on macOS instead of Ctrl/Alt. macOS icon loading corrected. (Closes #18, contributed by @fewtarius)
- Layer search field no longer captures single-key tool keybinds (e.g. pressing "B" in the search box no longer switches to the Brush tool).

### Changed
- New application icon and logo, including an MSIX-compliant safe-zone icon for the Microsoft Store.

### Security
- Updated `codeql-action` to v3.28.13 via Dependabot. (Closes #16)

### Contributors
- @fewtarius: macOS support, pen/tablet pressure, icon fix (#18)
- @zero54git: Arrow line endings feature request (#17)
- @dependabot: CI dependency update (#16)

### Known Issues
- On Windows, system error sounds may occasionally play when pressing keys. A relaunch typically resolves it. Under investigation; not observed in the latest build.

---

## [1.0.12] - 2026-03-03

### Added
- Async resize: Resize Image and Resize Canvas now run in a background thread with a loading spinner, keeping the UI responsive during large canvas operations.
- Exit dialog Save As flow: clicking "Save" in the exit confirmation dialog now opens sequential Save As dialogs for each unsaved untitled project before exiting.
- Auto-focus: Resize Image and Resize Canvas dialogs now auto-focus the width field when opened via keyboard shortcut.
- Reusable `open_save_as_for_project(idx)` helper, eliminating duplicated Save As setup code across handle_save and close-tab flows.

### Changed
- Exit dialog redesigned with three centered, same-size buttons: Save, Exit Without (red), and Cancel.
- Close-tab unsaved changes dialog buttons now use uniform sizing.
- Layers panel context menu reorganized: added separator between layer management and property toggle groups.

---

## [1.0.11] - 2026-02-28

### Added
- Single-instance IPC via Windows named pipe: right-click "Open with PaintFE" sends the file to the running instance instead of launching a duplicate. New instances forward paths and focus the existing window.
- Positional file argument support: `paintfe.exe photo.png` opens the image directly (file association / drag-onto-exe).

### Fixed
- Selection overlay (rect, ellipse, crosshatch, border segments) now pixel-snaps when zoomed in, making pixel art selection precise.
- Selection tools (rect, ellipse, lasso) and gradient handles can now be dragged outside the canvas area. Dragging from outside to the other side selects the full canvas edge-to-edge.

### Changed
- Windows binaries statically link vcruntime via `.cargo/config.toml` (`crt-static`), eliminating the Visual C++ Redistributable dependency.

---

## [1.0.10] - 2026-02-25

### Fixed
- AppImage: window/taskbar icon now appears correctly on Wayland (GNOME, KDE). On first launch the icon is installed to `~/.local/share/icons/hicolor/256x256/apps/` so Wayland compositors can resolve it by app-id.

### Security
- Added Dependabot configuration for weekly Cargo and GitHub Actions dependency updates.
- Fixed `release.yml` workflow token permissions to default `read-all` (the `release` job already scoped `contents: write` to only the upload step).
- All 8 `cargo audit` unmaintained-crate warnings are now acknowledged in `audit.toml` with context; they are transitive dependencies locked by `eframe`/`wgpu`/`rfd` with no known exploits.

---

## [1.0.9] - 2026-02-25

### Added
- `SECURITY.md` with private vulnerability reporting process, coordinated disclosure timeline, and user-facing security guidance.
- Automated SHA-256 checksum file (`checksums-SHA256.txt`) published alongside binaries on every GitHub Release.
- OpenSSF Scorecard GitHub Action (`scorecard.yml`) for automated security posture scoring.

### Fixed
- Resolved all 556 Rust 1.93 Clippy lints (`needless_range_loop`, `manual_clamp`, `match_like_matches_macro`, `ptr_arg`, `field_reassign_with_default`, `wrong_self_convention`, `if_same_then_else`, `doc_lazy_continuation`, `needless_update`, `struct_update_has_no_effect`) to restore a clean CI gate.
- Corrected `dtolnay/rust-toolchain` action SHA in `ci.yml`.

---

## [1.0.8] - 2026-02-21

### Fixed
- MSIX manifest: corrected `DefaultTile` schema, removed invalid `LockScreen` element, fixed version regex for `MinVersion` to avoid corrupting the XML declaration during CI packaging.

---

## [1.0.7] - 2026-02-21

### Fixed
- MSIX manifest: replaced invalid `MaxVersion` attribute with `MaxVersionTested` to pass Store validation.

---

## [1.0.6] - 2026-02-20

### Fixed
- MSIX build CI step: switched manifest patching to use `-creplace` to prevent the BOM being reintroduced and corrupting the XML declaration.

---

## [1.0.5] - 2026-02-20

### Fixed
- MSIX manifest BOM: changed CI step to read raw bytes, strip the UTF-8 BOM explicitly, then write back to guarantee a clean manifest regardless of PowerShell pipeline encoding.

---

## [1.0.4] - 2026-02-19

### Fixed
- MSIX manifest BOM: switched from `Get-Content` pipeline to `ReadAllText` + `TrimStart` to reliably strip the BOM before packaging.

---

## [1.0.3] - 2026-02-19

### Fixed
- AppImage build: corrected icon path in the `.desktop` file so the launcher icon resolves correctly.
- MSIX build: resolved UTF-8 BOM issue in manifest and added publisher identity fields required for Store submission.

---

## [1.0.1] - 2026-02-18

### Added
- AppImage packaging (`packaging/appimage/`) with build script and desktop integration files.
- MSIX packaging (`packaging/msix/`) for Microsoft Store submission, including icon asset generation and CI workflow step.
- GitHub Actions release workflow building Windows binary, Linux binary, and AppImage on tag push.
- Flatpak manifest (`packaging/flatpak/`) for distribution via Flathub.

### Fixed
- AppImage `AppRun` and build script execute permissions set correctly in the repository.

---

## [1.0.0] - 2026-02-17

Initial public release.

### Features
- 23 tools: Brush, Pencil, Eraser, Line, Fill, Gradient, Rect/Ellipse/Lasso/Magic Wand Select, Move Pixels, Move Selection, Clone Stamp, Content-Aware Fill, Color Remover, Liquify, Mesh Warp, Perspective Crop, Color Picker, Text, Zoom, Pan, Shapes.
- 25 blend modes.
- wgpu GPU compositing pipeline (DX12 on Windows, Vulkan on Linux).
- GPU compute paths for Gaussian Blur, HSL, and Median filter.
- Dirty-rect partial GPU readback with bytemuck zero-copy Color32 casting.
- Copy-on-Write tile system (`TiledImage`) using `Arc<RgbaImage>` chunks for fast undo.
- Tiered undo history: `PixelPatch` for strokes, `SingleLayerSnapshotCommand` for filter commits, full `SnapshotCommand` for canvas-wide ops.
- Rhai scripting engine (sandboxed, 50M op limit, no filesystem or network access from scripts).
- CLI batch mode with glob input, script execution, and multi-format output.
- Local AI background removal via dynamically loaded ONNX Runtime (BiRefNet, U2-Net, IS-Net auto-detected).
- Animated GIF and APNG import and export (layers as frames).
- RAW camera file support via `rawloader` and `imagepipe` (CR2, CR3, NEF, ARW, DNG, ORF, RW2, SRW, PEF, RAF).
- 15 built-in UI languages.
- MIT licensed, single portable binary, no installer.
