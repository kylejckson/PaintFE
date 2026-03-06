# Changelog

All notable changes to PaintFE are documented here. Dates are in YYYY-MM-DD format.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
