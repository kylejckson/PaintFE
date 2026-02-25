# Changelog

All notable changes to PaintFE are documented here. Dates are in YYYY-MM-DD format.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
