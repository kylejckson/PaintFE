# Changelog

All notable changes to PaintFE are documented here. Dates are in YYYY-MM-DD format.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.4.23] - 2026-10-31

- Expand image export settings across supported formats.
- Add lossless AVIF options through the optional codec pack.
- Simplify HEIF handling and refresh the documentation.

### Export improvements

- Add JPEG sampling, PNG interlacing, WebP presets and more palette and dithering controls.
- Expand GIF transparency, BMP/TGA colour modes and TIFF colour and precision options.
- Add AVIF encoding speed, lossless output and lossless alpha options in the updated optional pack.
- Offer additional HEIC sampling and lossless controls when supported by the selected codec.

### Codec and documentation improvements

- Replace the remaining Heifer container dependency with an independent HEIF metadata reader. Related to issue 196.
- Update README and codec documentation to reflect current features and optional packs. Related to issue 222.

Export improvements relate to issue 221.

## [1.4.22] - 2026-10-29

- Improve menu interactions and Save All handling.
- Expand Save As with clearer previews and more PNG export options.
- Improve how configuration files are handled in Windows recent documents.

### Interface and saving

- Make menu click areas more consistent and reduce unwanted dismissal around spacing and unavailable actions. Include untitled documents without edits in Save All. Related to issue 219.
- Make Save As resizable in both directions, with a stacked layout in narrower windows and accessible Save and Cancel buttons. Related to issue 221.
- Update previews in the background, move their status below the image, and show output precision and thumbnail size estimates more clearly. Related to issue 221.
- Remember export settings by format after saving, with a Reset this format button. Related to issue 221.

### Export options

- Add PNG colour modes, 8-bit and 16-bit output, compression presets, and indexed palettes with dithering and partial transparency. Keep warnings for choices that reduce colours or precision. Related to issue 221.
- Improve consistency between preview and saved output, including GIF dithering and TIFF compression settings.

### Windows and codecs

- Keep model, runtime and other configuration-resource selections out of Windows recent documents. Add cleanup for currently configured model, runtime and HEIC library entries under Preferences > AI. Related to issue 220.
- Remove the retired Heifer decoder and encoder experiment. Related to issue 196.

## [1.4.21] - 2026-10-28

- Stop large image resizes promptly when Cancel is pressed.
- Make Fit consistent across the toolbar, menu and keyboard shortcut.
- Explain HEIC limits and library setup more clearly.

### Fixes

- Run native image resizing in a separate process using the same executable, so Cancel can stop a running resize without applying a partial result. Related to issue 207.
- Make toolbar Fit use the same action as the View menu and Ctrl+0, rather than resetting to 100%. Keep a small margin around the fitted image and base the zoom-out limit on the canvas and window size. Related to issue 211.
- Show the image dimensions and PaintFE's 64-million-pixel HEIC processing limit before saving. Disable HEIC saving when the image exceeds that limit. Related to issue 196.
- Accept both .heic and .heif filenames from one HEIC / HEIF save entry. Related to issue 196.
- Add a lowercase paintfe command to Linux packages while keeping PaintFE available, and use it in the desktop entry. Improve package building from Windows checkouts and isolate the RPM build database. Related to issue 187.

### Interface and documentation

- Add HEIC library setup instructions and upstream links in Preferences > Optional features, alongside the existing library picker and support check. Related to issue 196.
- Explain Linux runtime and plugin dependencies, including why some managed DLLs are needed on Linux. Related to issue 187.

## [1.4.20] - 2026-10-27

- Prevent crashes during oversized resizes and WebP saves.
- Improve large-image fitting, zooming and frame-rate limits.
- Add optional HEIC library selection and simplify the CJK download.

### Fixes

- Check resize dimensions before allocating memory and reject unsupported sizes with an error. Related to issue 212.
- Prevent cancelled or outdated processing results from replacing a changed document. Related to issue 212.
- Handle WebP's 16,383-pixel dimension limit without crashing or replacing existing files. Related to issue 207.
- Fit large images when opened, allow zooming below 10%, and keep small images at no more than 100% initially. Related to issue 211.
- Save all seven Debug Info options and move them into Preferences > Interface. Related to issue 209.
- Stop small timing delays accumulating in the frame-rate limiter. Related to issue 203.

### Features and interface

- Add processing cancellation and block conflicting editor actions. CPU resizing may finish its current step before stopping. Related to issue 212.
- Add optional libheif DLL selection and a HEIC support check in Preferences > Optional Features. System codecs remain the default. Related to issue 196.
- Use one universal CJK font pack across supported platforms. Existing font packs remain compatible. Related to issue 210.

## [1.4.19] - 2026-10-26

- Fix the Linux build failure that blocked release downloads.
- Improve colour precision in transparent HEIF images on macOS.
- Make Rust toolchain setup consistent across build workflows.

## [1.4.18] - 2026-10-26

- Open and edit large photos at their original resolution.
- Improve Arabic text direction across the interface.
- Preserve duplicate document names and make language settings easier to find.

### Fixes

- Respect the graphics device's limits and use smaller display tiles for large canvases, keeping the original pixels and dimensions intact. Related to issue 207.
- Give the bounded Windows HEIF decoder enough temporary memory to open the reported 51 MP photo. Related to issue 207.
- Show an error dialog with the log location when an unexpected interface error stops the app.
- Correct Arabic interface word order, including cursor movement, selection and copying with mixed filenames and numbers. Related to discussion 194.
- Name duplicated documents `photo (copy).jpg` and keep that name when changing the export format. Duplicates remain separate from the original save target. Related to issue 198.
- Keep JPEG XL in the Open filters regardless of the export pack, and apply installed fonts immediately with refreshed settings layouts. Related to issue 195.

### Interface

- Move language and label customisation into Preferences > Interface, with a clearer Customize interface labels button. Related to issue 199.
- Keep Reset all visible in the label editor and remove redundant label-key tooltips. Changes still take effect after Apply. Related to issue 199.

## [1.4.17] - 2026-10-24

- Use installed operating-system codecs for HEIC and HEIF images.
- Line up artwork with canvas rulers and document guides.
- Choose a preferred GPU and automatic or custom frame-rate limits.

### Guides and interface

- Add document guides, optional rulers, exact positioning, locking and undo. Guides are saved in PFE5 projects and excluded from image exports. Related to issue 206.
- Keep rulers against the canvas edges as you pan and zoom, with right-aligned vertical labels and intermediate ticks. Rulers are off by default.
- Add ruler font, size, color and opacity settings in Preferences. Compact vertical labels default to 80% text size, and ruler backgrounds default to 75% opacity. Both can be adjusted independently.
- Add optional guide snapping for the edges and centres of pasted content and moved selections.
- Keep line previews and their undo history attached to the original layer and mask target when switching or reordering layers.
- Remove repetitive Preferences and Recent Files tooltips when their text is already fully visible. Related to issue 204.
- Add automatic monitor-refresh detection and a manual frame-rate limit, preserving mouse and pen samples. Stationary paste previews can go idle. Related to issue 203.
- Apply the preferred GPU at startup and show the actual adapter, pending changes and fallback status. Related to issue 202.

### HEIC support

- Remove the bundled HEVC decoder and retire the HEIC export download. Windows uses WIC, macOS ImageIO, and Linux installed libheif. JPEG XL and AVIF are unchanged.
- Show separate HEIC opening and saving availability in Preferences, with guidance when a required system codec is missing.
- Keep codec work in a bounded, cancellable process using the same executable. Save safety and original-file protection remain in place.
- Report unsupported transparency or precision instead of silently losing it. Related to issue 196.

## [1.4.16] - 2026-10-24

- Warn before saving a document to a format that loses editable information.
- Offer PaintFE project saving to preserve layers, text and masks.
- Wait for successful saves before closing documents or exiting.

### Save safety

- Save, Save As, Save All and save-on-close now share the same loss checks. Related to issue 205.
- Image exports keep the layered document open and retain an existing PaintFE project save target.
- Layer masks and their enabled state are now stored in PFE projects. Masked projects require this version or newer to reopen.
- Failed writes leave the destination intact and keep the document open. Background save results track the original document and do not mark newer edits as saved.

Related to issue 205. The issue remains open for users to test and confirm.

## [1.4.15] - 2026-10-23

- Open HEIC/HEIF images and save them with an optional HEIC export pack.
- Open and save AVIF images with an optional codec pack.
- Customize interface labels with a searchable editor for each language.

### Release build fixes

- Detect Visual Studio 2022 or 2026 when building the HEIC export helper, with a compatible CMake version and separate compiler caches.
- Set up and verify a software Vulkan driver for Linux CI so the toolbar-save test runs without a physical GPU.
- Include the changes prepared for v1.4.14, whose release was blocked by these build and CI setup errors.

Issues remain open for users to test and confirm.

## [1.4.14] - 2026-10-22

- Open HEIC/HEIF images and save them with an optional HEIC export pack.
- Open and save AVIF images with an optional codec pack.
- Customize interface labels with a searchable editor for each language.

### Images and saving

- Corrected HEIF grid transparency using a local Heifer decoder patch.
- Added separate offline HEIC and AVIF helpers, keeping their encoders out of the main app.
- Toolbar Save now saves directly to the current file, matching Ctrl+S and File > Save.
- HEIC saves keep the original format and destination. Missing export packs offer installation without switching to PNG.

### Interface and installation

- Restyled Document Info, added the supplied icons, and made the label editor movable, resizable and bounded to the app window.
- Matched Preferences section headings to the other dialogs. Related to #197.
- Added Info and Duplicate above Print in the File menu. Related to #198.
- Added DEB/RPM packages, signed Linux repositories and the community Arabic translation with an offline font fallback.

### Release checks

- Release publishing verifies all sixteen signed optional packs and their public downloads.
- HEIC colour export is lossy 8-bit, including at quality 100; alpha is lossless.
- Linux/macOS codec execution and Linux package installation still need platform testing. Arabic needs native-speaker review; full RTL layout is not implemented.

Issues remain open for users to test and confirm.

## [1.4.13] - 2026-10-22

- Fixed the RPM signing error that blocked the previous release.
- Reduced storage used by temporary Actions build downloads.
- Kept Microsoft Store submission downloads available for fourteen days.

### Release maintenance

- Fixed RPM signing by exporting the public key in the format required by Ubuntu's RPM tools, then checking the package has a signature.
- Added a disposable-key CI test for RPM signing, incorrect passwords and damaged packages.
- Reduced temporary release artifact retention to one day. Store submission packages remain available for fourteen days.
- Made development portable builds manual-only, with downloads kept for three days.
- Added a read-only cleanup report for older Actions artifacts. Existing artifacts and published releases have not been deleted.

Issues remain open for users to test and confirm.

## [1.4.12] - 2026-10-21

- Fixed the Linux release build and macOS CI checks.
- Updated ZIP handling and a concurrency dependency.
- Added broader optional-pack checks across Windows, Linux and macOS.

### Fixed

- Release builds now install Python 3.12 explicitly, avoiding the missing TOML parser on Linux.
- Codec tests use their platform's generated pack metadata instead of looking for Windows files on macOS.
- Fixed a Windows-only mutable variable triggering macOS Clippy.

### Maintenance

- Updated ZIP to 9.0.5 and event-listener to 5.4.2.
- Updated deprecated Actions runtimes and added Linux CI checks.
- Added signed-pack tests for symlinks and archive permissions.

Issues remain open for users to test and confirm.
Further testing is continuing.

## [1.4.11] - 2026-10-21

- Reduced startup delays and pauses caused by settings saves.
- Made the CJK font and JPEG XL encoder optional downloads.
- Reduced the Windows executable from 61.89 MiB to 43.29 MiB.

### Improved

- Moved settings writes off the UI thread while keeping automatic saving, error reporting and safe shutdown.
- Moved AI runtime availability checks into the background.
- Deferred GPU pipeline setup and reused the last working Windows graphics backend to reduce startup work.
- Cached colour-slider gradients and checkerboards to reduce drawing work without changing their appearance or controls.

### Added

- Added Preferences > Optional features for downloading, importing and removing the CJK font and JPEG XL export packs.
- Added signed pack verification and a warning before the one-time online download. Installed features work offline.
- Added font installation prompts when a selected CJK language needs coverage unavailable from system fonts.
- Added an installation offer when saving JPEG XL without its encoder. Opening JPEG XL files remains available in the core application.

### Validation

- Full test suite: 655 passed, none failed, 10 intentionally ignored.
- App and encoder-helper Clippy checks passed.
- Formatting and shared UI control checks passed.

Existing issues remain open for users to test and confirm.
Further testing is continuing.

## [1.4.10] - 2026-10-19

- Added JPEG XL opening and saving, with lossless and lossy options.
- Made numeric controls more consistent and added double-click slider resets.
- Improved Preferences scrolling and stopped panel cursors appearing through menus.

### Fixed

- Replaced separate numeric editors, sliders, colour-channel fields and step buttons with shared controls across dialogs, Preferences and tool settings. Typing, dragging, keyboard adjustments and resets now follow the same rules, while colour sliders keep their triangular handles.
- Reduced work while scrolling the Interface page by reusing offscreen section measurements and colour-wheel geometry. Unchanged AI paths are no longer checked on disk every frame, and scrolling over numeric fields no longer changes their values on this page.
- Panel resize grips and the Tools heading now respect menus and windows above them, preventing resize or drag cursors from showing through a menu.
- Corrected a browser-only compile error in the translated print-window title.

### Added

- Added JPEG XL still-image import and desktop export, including lossless and lossy modes, encoding effort, background previews and CLI conversion. The encoder is built into the application and needs no extra codec DLL. Lossless tests cover exact 8-bit and 16-bit RGBA samples, including transparent pixels.
- Added double-click reset on slider tracks where a meaningful default is available. Numeric fields keep double-click text editing, optional theme overrides return to their inherited values, and the gesture can be disabled in Preferences > Interface > Motion and interaction details.

### Maintenance

- Documented the shared control system and added an automated audit to help keep future controls consistent.
- Updated the pinned CI actions from pull requests 175, 176 and 177, and added JPEG XL codec checks to CI.

Related issue: 185. The issue remains open for users to test and confirm. Further testing is continuing.

## [1.4.9] - 2026-10-17

- Fixed text editing, crop undo redraws and layer settings getting out of sync.
- Improved resize dialogs, Save As controls and pasted-image handling.
- Added Preferences search and appearance settings for image information.

### Fixed

- Text editing now keeps clipboard, selection and delete shortcuts with the text editor, preventing them from changing canvas pixels underneath.
- Undoing and redoing a crop now refreshes the GPU layer contents, preventing stale pixels from appearing.
- Layer settings now follow the active layer and document instead of showing values from a previous selection.
- Pasted content stays editable while using navigation tools and commits when switching to a painting tool.
- Selection dimensions include single-pixel selections and remain visible after releasing the mouse.
- Document Info now shows readable Windows paths without the extended-path prefix. Recent Files entries are left-aligned.
- Resize dialogs select the initial width properly, use consistent drag adjustments and keep the canvas anchor preview within its bounds.
- Dialog resets restore the original resize dimensions and colour-sampling position. Duplicate reset controls are hidden where a shared header reset is available.
- Save As and Preferences now move only by their headings, preventing numeric controls from dragging the whole window.
- Fixed the Preferences search row expanding vertically and pushing the settings out of view.

### Added

- Added Preferences search using English and translated setting names, with navigation to matching sections.
- Added font size, opacity, text colour and background colour controls for the bottom-right image information panel, with theme defaults and a reset button.
- Added direct percentage editing for Save As preview zoom and consistent numeric controls for export quality.

Related issues: 168, 185, 186, 188, 189, 190, 191, 192 and 193. Issues remain open for users to test and confirm. Further testing is continuing.

## [1.4.8] - 2026-10-16

- Reworked Tools resizing with compact layouts, smoother movement and tool visibility controls.
- Corrected History icons and improved keyboard navigation in unsaved-change dialogs.
- Added paste placement preferences and made Preferences save changes more consistently.

### Fixed

- History icons now follow the action that created an undo entry instead of its displayed name or the currently selected tool. This covers tools, effects, filters, layer operations and selection commands, including Color Range, Invert Selection and Deselect. Applying Color Range with unchanged controls also records the selection for Undo.
- Replaced rigid Tools layouts with arrangements that fit their contents. Single rows and columns no longer leave large empty areas, the header stays aligned at narrow widths, and the colour controls keep their own divider and spacing. Shapes uses its normal two-slot button when it fits and a compact icon when only one slot is available.
- Unsaved-change dialogs now support Tab, Shift+Tab and arrow-key navigation. Enter activates the focused choice, and the dialogs can be moved by their heading.
- Recent Files now sizes its submenu to the paths being shown, capped at 700 pixels or half the application width.
- Preferences now apply and save changes consistently. Repeated writes are combined, files are replaced atomically, and save failures are shown. Invalid AI paths show a warning while preserving the previous valid configuration.
- Disabled document-only controls when no image is open, while keeping actions such as New, Open and Preferences available.
- Fixed fast paste drags losing their starting position or passing through to the brush underneath.

### Added

- Added a Tools visibility checklist with icons, names and a Show all button. Hidden tools remain available through their shortcuts.
- Added optional resize animation for Tools, following the existing motion and reduced-motion settings.
- Added preferences for pasting at the copied position and keeping pasted content in view. Editable pasted content outside the canvas is shown faded; committing still clips it to the canvas.
- Added optional Ctrl+Insert, Shift+Insert and Shift+Delete clipboard shortcuts.
- Made Enter-to-deselect optional and off by default, without changing Enter confirmation for active tools and dialogs.
- Added colour and opacity controls for the docked tool-options separator, with follow-theme and reset options.
- Added built-in and light/dark icon-pack support for the compact Shapes button.

Related issues: 166, 168, 169, 171, 173, 178, 179, 180, 181 and 182. Issues remain open for users to test and confirm the changes. Further testing is continuing.

## [1.4.7] - 2026-10-15

- Fixed filters ignoring selections, Pixel Drag direction and incomplete canvas borders.
- Added a resizable Tools panel and more control over the tool options bar.
- Fixed panel positions, file-picker folders and editing behind open dialogs.

### Fixed

- Kept Layers and History in their saved positions when starting with display scaling.
- Separated remembered folders for images, themes, scripts and AI files.
- Cleaned up Recent Files paths and reordered the File menu.
- Corrected History icons and Auto Levels behaviour.
- Blocked document changes behind adjustment and Save As dialogs while keeping canvas wheel zoom available.
- Made Glitch, Ink, Oil Painting and Color Filter respect selections. Fixed Pixel Drag direction and Canvas Border drawing on all four sides.
- Removed the duplicate Straighten Canvas tooltip and centred its button in the toolbar.
- Recognised AppImage portable settings directories beside the original AppImage.

### Added

- Made Tools resizable in portrait and landscape layouts, with an option to hide its colour controls.
- Added floating and docked tool options, background opacity, spacing, padding and corner radius controls.
- Added optional Escape deselection with Undo support, while preserving text-field and tool cancellation behaviour.

Related issues: 155, 165, 166, 167, 168, 169, 170, 171, 172, 173 and 174. These remain open for users to confirm the changes.

## [1.4.6] - 2026-10-14

- Fixed background-removal edge contrast and faint transparency in solid areas.
- Added editable shortcuts for clearing and filling selected pixels.
- Simplified background-removal model settings, keeping automatic detection as the default.

### Fixed

- Restored BiRefNet and U²-Net edge contrast while keeping the IS-Net fixes. Corrected alpha rounding and kept the threshold at half opacity. Closes #162.
- Stopped saved model-profile overrides from carrying into the next background-removal dialog. Manual overrides are tucked into a collapsible section.

### Added

- Added Clear Selected Pixels and Fill Selected Pixels to Preferences > Keybinds. Delete and Backspace remain the defaults, and both can be changed. Closes #164.

### Build

- Returned Microsoft Store packaging to manual submission. The release workflow provides a downloadable MSIX without uploading to Partner Center or requesting deployment approval.

## [1.4.5] - 2026-10-13

- Show clear errors when an image cannot be opened.
- Improve IS-Net background removal and remember dialog settings.
- Complete translations and add five languages with separate Portuguese variants.

### Fixed

- Report unsupported and corrupted image imports before asking how to open them. Failed imports leave documents unchanged. Addresses #161.
- Add model-specific preprocessing and mask conversion for IS-Net exports, with automatic detection and manual profile selection. Addresses #162.
- Reuse supported dialog settings during a session, with optional persistence between launches and a reset control. Addresses #163.
- Translate remaining controls and keep long panel titles clear of close buttons.

### Added

- Add Korean, Indonesian, Vietnamese, Thai and Hungarian translations, and separate Brazilian and European Portuguese. Existing Portuguese preferences keep the Brazilian wording.
- Complete all 25 language catalogs and add offline CJK and Thai font fallbacks.

## [1.4.4] - 2026-10-12

- Save As now uses the remembered export format for opened images too.
- Made the Colors panel more compact at smaller sizes.
- Fixed Save As folders after drag-and-drop and Windows command-line text.

### Fixed

- Kept the original folder when opening an image by drag-and-drop, so Save As starts in the right place. Closes #159.
- Fixed garbled non-English text in the Windows command line and kept redirected output and errors working. Closes #160.
- Applied the remembered export format to existing documents while keeping their name and folder.
- Shrunk Colors controls to fit smaller panels, moved hex editing and copying into a popup when needed, and removed scrolling from the basic view. Small panel sizes are remembered after restarting.

## [1.4.3] - 2026-10-12

- Fixed Contours preview overload when dragging sliders quickly.
- Made panel resizing steadier, with smaller Colors layouts and subtler resize grips.
- Tidied tab scrolling and improved crash logs.

### Fixed

- Limited queued effect previews and stopped outdated Contours calculations. Cancel and disabling live preview restore the original image. Closes #156.
- Fixed jumping resize gestures for Colors, Layers, History and Palette. Panels keep their top-left corner in place and stop at the screen edge. Closes #157.
- Made the Colors ring and triangle shrink with the panel and limited oversized layouts. Follow-up to #152.
- Moved tab scroll arrows to opposite ends of the tab strip. They appear only when tabs are hidden in that direction, with no button background and an accent color on hover. Follow-up to #149.
- Stopped repeated dialog-opening log entries, used local timestamps and added panic backtraces and filter errors to the log file.

### Added

- Added resize-grip color and opacity controls in Preferences > Interface. Grips now use two stripes at half the previous opacity by default.

Further testing is continuing.

## [1.4.2] - 2026-10-10

- Fixed keyboard input, dialog controls and Save As previews.
- Made document tabs easier to manage, with recent files, scrolling and more right-click actions.
- Added a resizable Colors panel, tooltip timing and script cropping with batch export.

### Fixed

- Reworked Windows keyboard handling to prevent missed keys and stale modifiers after switching apps. Addresses #138; the reported long-held Alt+Tab case still needs confirmation.
- Improved Tab and Shift+Tab navigation, selected Resize field contents on focus, and added keyboard selection in its preset menu. Closes #143.
- Kept Recover Transparency within the available screen space, with scrolling and visible action buttons. Closes #144.
- Added Enter and Escape handling to dialogs without interfering with open dropdowns. Closes #145.
- Restored Backspace input in editable controls and added draggable Resize values. Closes #146.
- Fixed Save As movement, resizing, closing, preview fitting and wheel zoom. It now remembers the export format and uses the current file's folder first. Closes #147.
- Remembered user-profile fallback for unwritable portable folders and made both storage choices selectable in Preferences. Addresses #142.
- Resolved AppImage storage against the original image rather than its read-only mount. Addresses #155; native Linux verification is pending.

### Added

- Added scrolling document tabs, shorter labels and Ctrl+Tab / Ctrl+Shift+Tab switching. Closes #149.
- Added Recent Files, Close, tab preview thumbnails and right-click actions for saving, duplicating, viewing information and opening the containing folder. Closes #150.
- Added a tooltip delay setting, including instant tooltips, and explanations for Motion settings. Closes #151.
- Made the Colors panel resizable and remembered its size. Closes #152.
- Added right-click Color Picker support for the secondary color. Closes #153.
- Added `crop_canvas(...)` and documented CLI crop/save and batch processing, with protection against accidental overwrites. Closes #154.
- Added macOS app-bundle and DMG packaging, optional signing/notarization, and release-generated Homebrew cask files. Addresses #148; signed distribution and native installation testing are pending.

---

## [1.4.1] - 2026-10-08

- Fixed text previews and missed Ctrl+Z presses.
- Added optional middle-click tab closing and fixed Magic Wand preview updates.
- Windows and Linux builds now use portable storage by default, with new AI memory settings.

### Fixed

- Fixed text previews showing only the first letter or leaving old text behind. #137.
- Fixed missed Ctrl+Z presses after using tools and controls, including quick key taps. #138.
- Fixed Magic Wand previews not updating when changing tolerance or anti-aliasing. #140.
- Improved background-removal memory cleanup and error handling. Failed jobs leave the document unchanged. #141.
- Kept background filter results attached to the correct document when tabs are moved or closed. #139.

### Added

- Added middle-click tab closing, enabled by default and configurable in Preferences. Unsaved documents still prompt before closing. #139.
- Added AI graph optimization and CPU memory arena settings. The memory arena is off by default. #141.
- Made Windows EXE and Linux builds portable by default. Preferences can switch to profile storage after restart, with data migration and a fallback for unwritable folders. #142.

---

## [1.4.0] - 2026-10-05

- A fresh look with Luminous icons and redesigned Color, Palette, and History panels. Classic icons are still available.
- Make the workspace yours with panel snapping, saved layouts, configurable motion, and sharper numeric controls.
- Enjoy a faster, more responsive app, corrected brush hardness, and a clearer pixel-art workspace.

### Added

- Added bundled Luminous and Classic icon styles, custom folder or ZIP icon packs, and icon template export.
- Added compact Color and Advanced modes, collapsible HSV/RGB/HSL controls, and a resizable Palette with recent colors and favorites.
- Added a History timeline with tool icons, compact action details, and subtle entry animations.
- Added optional panel edge and alignment snapping, faint guides, named workspace layouts, and layout reset.
- Added configurable motion, reduced-motion support, keyboard focus styling, fine numeric adjustment, and consistent resets.
- Added stroke stabilization, Normal/Uniform/Build Up brush modes, separate opacity controls, and a Pixel Art preset.
- Added stepped zoom, improved Fit to Window, and a rebindable 100% zoom action.
- Added paste crop handles, Commit & Crop, select-after-paste, and layer content bounds selection. Closes #131.
- Added before/after comparison by toggling live effect previews. Closes #128.

### Changed

- Refreshed the application logo, window and taskbar icons, Store assets, and README screenshots.
- Standardized panel titles, close buttons, spacing, focus feedback, and dialog styling across light and dark themes.
- Made the floating tool shelf span the available width and simplified icon pack preferences.
- Improved soft brush coverage when strokes cross themselves, while preserving soft edges and avoiding stationary buildup in Normal mode.
- Made hovering, clicking, opening menus, and painting more responsive.
- Reduced brush work, icon rendering overhead, repeated UI layout work, and unnecessary settings writes.
- Added low-latency presentation settings and customizable pixel-grid and selection-outline appearance.
- Made the pixel grid clearer and smoother, with subtler lines and configurable opacity and dashes.
- Improved pixel-art editing with raw Pencil input, stepped zoom, clearer cursor feedback, and the Pixel Art preset.
- Limited unsaved-file lists to four names plus the remaining count.

### Fixed

- Fixed the longstanding brush hardness falloff: hardness now controls the opaque inner radius, with soft edges fading to zero instead of leaving hard rims.
- Fixed stale canvas images after switching or closing documents, including returning to an existing blank tab. Closes #123.
- Fixed layer thumbnails retaining images from another project with matching layer counts.
- Avoided repeated viewport title and theme commands while idle. Addresses #134; Linux verification is still pending.
- Fixed dropped images opening before the import choice was confirmed.
- Fixed palette clicks ignoring the selected Primary or Secondary color slot.
- Fixed Color panel positioning, panel header alignment, and hex-field spacing.
- Preserved custom icon colors in menus, Preferences, toolbar controls, layer actions, and cursors.
- Fixed Preferences wheel input reaching the canvas and improved stroke, gradient, and cursor feedback.
- Corrected the Flatpak application icon source path.

## [1.3.11] - 2026-08-10

- Recover transparency from artwork flattened over solid or lightly noisy backgrounds, including enclosed background islands.
- Resize pixel art with palette-aware, structure-preserving, or topology-preserving retargeting.
- Made Select All look like a normal selection again, without whitening the canvas.

### Added

- Added Recover Transparency with automatic edge sampling, configurable connectivity and edge recovery, interior-island detection, manual remove/protect points, selection support, and alpha/reconstruction diagnostic previews.
- Added Pixel Art Retarget to Resize Image with palette-aware, structure-preserving, and topology-preserving modes.
- Added controls for palette preservation, silhouette and thin-feature protection, connectivity, detail and topology priority, phase optimization, selection protection, and comparison previews.

### Changed

- Updated the pinned GitHub Actions .NET setup and Rust toolchain actions.
- Preserved Paint.NET compatibility-host binaries during AUR packaging so their published SHA-256 sidecars remain valid.
- Bundled matching Compose data with the AppImage and selected it through `XLOCALEDIR` for reliable keyboard input across host systems.

### Fixed

- Fixed Color Picker clicks being ignored unless the pointer moved while held.
- Fixed Ctrl+A rendering the canvas white instead of showing the normal selection treatment.
- Fixed Save As previews stretching the outermost image pixels across letterboxed areas. Closes #120.
- Fixed AppImage keyboard Compose data mismatches that could trigger `dead_hamza` and host-parser errors. Closes #118.

## [1.3.10] - 2026-07-30

- Turn images into seamless game-ready textures with live tiled previewing and organic edge repair.
- Made AI background removal work reliably on Linux, including systems using versioned ONNX Runtime libraries.
- Made AppImage startup quieter on both Wayland and X11.

### Added

- Added Filter > Stylize > Make Seamless Texture for active-layer tile repair.
- Added horizontal and vertical controls, blend width, repair strength, linear or smooth fading, plus seeded organic dents.
- Added a regression test that verifies opposing texture edges match after two-axis organic repair.

### Changed

- The ONNX Runtime picker can now select package-managed versioned shared libraries such as `libonnxruntime.so.1.23.2`.

### Fixed

- Fixed Linux ONNX model loading by passing UTF-8 C paths on Linux and macOS while retaining UTF-16 paths on Windows. Closes #119.
- Fixed validation for versioned Linux ONNX Runtime shared libraries. Closes #119.
- Suppressed the harmless `dead_hamza` xkbcommon Compose warning during AppImage startup on both Wayland and X11, while preserving user-set diagnostic log levels.

## [1.3.9] - 2026-07-22

- Added an interactive Straighten mode for quickly correcting photo rotation.
- Fixed stale canvas ghosts when creating or switching documents after copying pixels.
- Made pasted images easier to find and align precisely.

### Added
- Added Straighten to the floating tool shelf.
- Drag directly on the image to set a rotation angle, or enter an exact angle manually.
- Added nearest-neighbour and bilinear interpolation options for Straighten.
- Added transparent corner previewing, Shift whole-degree snapping, Enter to apply, and Escape to cancel.
- Added Shift-drag snapping for pasted images to canvas edges and centre guides.

### Changed
- Same-size clipboard images now paste at the document origin.
- Other pasted images prefer the current cursor position and otherwise remain within the target canvas.

### Fixed
- Fixed stale GPU composite textures showing a scaled ghost of another document after creating or switching projects. Closes #114.
- Fixed copied regions pasting at an unusable source-image position in newly created documents. Closes #114.
- Added arbitrary whole-canvas rotation that preserves canvas dimensions and fills revealed pixels with transparency. Closes #108.

## [1.3.8] - 2026-07-17

- Made canvas rendering significantly faster and lighter on large 4K and 8K projects.
- Improved Fill Bucket, blur effects, filters, and adjustments to reduce stalls and memory use.
- Reduced GPU memory pressure while preserving support for large images and multi-layer projects.

### Changed
- Kept normal canvas compositing on the GPU instead of reading completed frames back into system memory and uploading them again.
- Combined normal layer compositing into a single GPU render pass and reused uniform buffers between frames.
- Removed unnecessary per-layer mipmap generation, reducing layer texture memory use by roughly 25 percent.
- Limited retained temporary GPU textures to 128 MB and reduced duplicate textures kept for each size.
- Moved connected Fill Bucket processing to an asynchronous path so large fills no longer monopolize the interface.
- Improved Box Blur with a sliding-window implementation whose processing cost no longer grows with the blur radius.
- Improved Bokeh Blur by reusing row sums instead of sampling the complete disc kernel for every pixel.
- Reduced temporary memory use during Gaussian Blur.
- Applied common pixel adjustments directly to populated image tiles instead of flattening and rebuilding the entire layer.

### Fixed
- Fixed the mouse cursor disappearing over Layer Options, including adjustment layer options.
- Fixed the same cursor and canvas input leak in the brush-tip context menu.

## [1.3.7] - 2026-07-16

- Made selections, clipboard operations, pasted-image transforms, and text editing significantly smoother on large canvases.
- Improved transform controls with off-canvas interaction, adaptive high-contrast outlines, and correct resize/grab cursors.
- Added rebindable aspect-preserving selection resizing and Enter-to-confirm layer renaming.

### Added
- Added aspect-preserving resizing in selection edit mode. Hold Alt while resizing to preserve the selection's aspect ratio at the moment the resize begins.
- Added a rebindable "Selection Preserve Aspect Modifier" setting, with Alt as the default and Ctrl available as an alternative.
- Added native modifier-state handling to make held shortcuts more reliable during pointer drags.

### Changed
- Optimized Select All on large canvases by representing a full-canvas selection without allocating and processing a full-resolution selection mask.
- Reduced stalls during large image copy, cut, and paste operations.
- Improved pasted-image movement and resizing by caching expensive compositing work during interactive transforms.
- Improved text-box and text-layer responsiveness on large canvases by reducing expensive per-frame rasterization and preview work.
- Moved the text tool's drag handle farther left to separate it from the resize area.
- Updated transform overlays to remain interactive outside the canvas unless covered by a floating panel, widget, or upper application UI.
- Added adaptive light-and-dark transform outlines so text and image controls remain visible over bright, dark, or mixed images.

### Fixed
- Fixed missing resize and grab cursors on pasted images, text boxes, handles, and anchor controls.
- Fixed image, shape, selection, and text handles becoming inaccessible when positioned outside the canvas.
- Fixed selection aspect locking not reliably detecting held modifier keys.
- Fixed Enter not confirming a layer name while renaming it.
- Fixed strict Clippy failures affecting general and macOS CI jobs.
- Fixed additional platform-specific Clippy warnings in WebAssembly builds.

## [1.3.6] - 2026-07-11

### Added
- Added locked aspect ratios for Rectangle Select. Enter ratios such as `3:4` or `16:9`, or use `image` to match the current canvas ratio. Closes #102.

## [1.3.5] - 2026-07-10

### Changed
- Upgraded egui/eframe to 0.35.0 and migrated PaintFE's root app layout to the new `eframe::App::ui` flow. Fixes #91.

### Fixed
- Fixed canvas input handling after the egui 0.35 migration so the canvas no longer self-blocks tool clicks.
- Fixed floating panel resize cursors and prevented resize drags from leaking through to canvas tools.
- Fixed off-canvas line and shape edit handles so placed lines/shapes can be adjusted when handles sit outside the image bounds.

## [1.3.4] - 2026-07-10

### Fixed
- Fixed canvas tools painting or transforming through floating panel resize drags. Fixes #92.

## [1.3.3] - 2026-07-10

### Added
- Added Ctrl-click solo toggling from layer visibility icons.
- Added selectable origin controls for Bulge/Pinch and Twist.
- Added resize cursor feedback for Move Selection and placed shape handles.

### Changed
- Ctrl+D now commits active paste/tool edits before clearing the selection.
- Drop Shadow radius previews now update on slider release or manual preview instead of continuously while dragging.
- Placed shapes now move and resize smoothly without per-frame pixel snapping.

### Fixed
- Fixed Pinch distortion behavior.
- Fixed selection-aware effect dialogs so effects respect the active selection mask.
- Fixed stale empty text previews after clearing or moving empty text.

## [1.3.2] - 2026-07-08

### Added
- Added experimental Paint.NET `.pdn` import support.
- Added experimental Paint.NET legacy effect/plugin DLL support with explicit trust confirmation, exact-file hash checks, and host sidecar validation.
- Added a Windows portable release workflow and pinned/locked release tooling.

### Changed
- Improved brush hardness so hardness controls material alpha falloff while preserving anti-aliased edges.
- Preserved Move Selection interpolation and anti-aliasing settings between uses.

### Fixed
- Fixed brush hardness behavior at small brush sizes.
- Fixed canvas tools drawing or acting behind floating/top UI controls.
- Fixed Move Pixels undo so extraction and commit undo as one operation.
- Fixed layer/settings UI persistence behavior.
- Fixed transient console window flashes during `.pdn` import on Windows.

### Security
- Hardened Paint.NET plugin loading so untrusted or swapped DLLs require confirmation before use.
- Restricted release and host restore behavior with pinned toolchain and NuGet lockfiles.

## [1.3.1] - 2026-06-10

### Added
- Added WebP lossless export support, with static WebP now defaulting to lossless output.
- Added animated WebP import/export support.
- Added per-layer animated WebP frame compression metadata, allowing each exported frame to be saved as lossless or lossy.
- Added a WebP frame compression option in Layer Options, persisted in PFE files and preserved when duplicating layers.
- Added `--webp-lossy` CLI export option for lossy WebP output.

### Changed
- Save As WebP now exposes a lossless toggle and hides the quality slider when lossless mode is active.
- Projects can now have zero layers; deleting the final layer leaves an empty gray workspace instead of forcing one layer to remain.
- Layer actions that require an active layer are disabled when the layer stack is empty.

### Fixed
- Fixed brush, pencil, and eraser release flicker by flushing the committed stroke region into the visible composite before async GPU readback catches up.
- Fixed add-layer and undo/redo paths so they behave correctly after the final layer has been deleted.

## [1.3.0] - 2026-06-10

### Added
- Added experimental RGBA UI16, RGBA FP16, and RGBA FP32 image payload support.
- Added HDR metadata handling and tone-mapped preview support.
- Added editable high-depth image paths so brush edits update 16-bit/HDR buffers.
- Added adjustment layers with exposure, brightness/contrast, invert, and channel mixer support.
- Added channel extraction and alpha replacement tools.
- Added layer folders with collapse, visibility, renaming, drag/reorder, multi-select movement, folder colors, select-contents, and project persistence.
- Added Preferences controls for customizing the layer folder color palette.
- Added PFE v3 persistence for folders, adjustment layers, high-depth payloads, HDR metadata, and source metadata.
- Added 16-bit PNG/TIFF and FP32 HDR TIFF import/export coverage.

### Changed
- Composite and export paths now respect folder visibility and adjustment layers.
- Add Layer now inserts relative to the selected layer or selected folder.
- Layer/folder rename fields now keep Ctrl+A, Enter, and Escape behavior local to text editing.
- Folder icons now use the existing icon asset inversion path for contrast over folder colors.
- Mirror mode changes now clear preview state before redrawing.

### Fixed
- Fixed Linux/KDE crash when opening an image directly from file manager or CLI due to mismatched GPU texture upload dimensions.
- Fixed Layers panel resized height not persisting across restarts.
- Fixed stale preview state when toggling mirror modes mid-edit.

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
