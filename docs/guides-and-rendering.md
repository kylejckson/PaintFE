# Guides and Rendering Controls

## Guides (Issue 206)

View > Guides controls rulers, guide visibility, locking, and snapping. Rulers are off by default. When enabled, they sit just outside the image canvas on the top and left edges, following zoom and pan and clipping at the workspace edge. Drag from one of the rulers onto the image to create a guide. Use Move Pixels, Move Selection, or Pan to move an existing guide. Drag outside the image or back onto a ruler to remove it; Escape cancels. A completed drag is one undo action. Holding Alt while placing a guide keeps fractional pixel positions.

Preferences > Interface > Ruler appearance controls the font (interface or monospace), size, text/tick/background colors, overall opacity, and background opacity (75% by default). Text and tick opacity are independent of background opacity. Colors inherit the current theme until overridden. Settings save automatically and move with exported themes. Reset restores inheritance and the default appearance. Vertical labels share a right edge beside the ticks; the ruler grows to fit larger text and longer numbers. Compact vertical text is enabled by default at 80% of the horizontal size (minimum 6 pt), with an adjustable scale and an off switch. Intermediate ticks are enabled by default and can be disabled; their spacing follows zoom without drawing subpixel ticks.

A new guide accepts a pixel or percentage position through the shared numeric editor. Guides belong to the document, not to a layer. They never enter image composites or PNG/JPEG exports. Save as PFE to keep them. Projects containing guides use PFE5 and require this version or newer; the existing PFE0-PFE4 loaders remain available. PFE5 includes the existing mask-preserving PFE4 document.

Snapping initially covers translated paste bounds and moved selection bounds, using edges and center points with a six-screen-point tolerance. It is off by default. Freehand drawing is unchanged. Shape-handle snapping can use the same helper in a later extension.

Guides scale with image resize, translate and clip with anchored canvas resize and crop, and follow whole-canvas flips and quarter-turn rotations. Rotating only a selected region leaves document guides in place. Arbitrary-angle artwork rotation or straightening keeps guides aligned to the document axes.

Line previews capture a session layer identity and mask/pixel target. Commits and history use that original target after layer selection or reordering. A removed source cancels the pending preview instead of painting another layer. Interacting with Layers or switching documents commits an editable line first. Session identities are not added to existing on-disk layer formats.

## Tooltips (Issue 204)

Preferences navigation and Recent Files show full-text tooltips only when the visible text is truncated. Explanatory tooltips are unaffected.

## Rendering (Issue 203)

Preferences > Hardware > Rendering defaults to detecting the current monitor's refresh rate, with 60 FPS as the fallback. Disable detection to choose 15-360 FPS. The native event-loop wrapper delays RedrawRequested events until their deadline; it forwards input events unchanged. Mouse/pen samples continue to reach the existing sub-frame stroke collector. No timer is started just to maintain a minimum FPS. Window resize and platform lifecycle work may repaint immediately. Monitor information is cached briefly and invalidated after monitor movement or DPI changes. Presentation and low-latency options remain independent of the cap. A stationary paste preview no longer requests continuous repaint.

Use opt-in performance diagnostics for comparisons. The continuous-drive mode is an intentional load test, not an idle-use measurement. Debug-build timings and shared-PC GPU utilization must not be presented as release performance.

## GPU Selection (Issue 202)

Preferences > Hardware selects the adapter used at the next restart. Existing saved GPU names remain accepted; new selections include vendor/device identity. Selection checks surface compatibility, and the normal canvas shares eframe's selected device. The status shows the actual adapter and backend rather than the saved preference. A missing device falls back to a compatible adapter and is reported. A startup failure can retry once using automatic selection. The picker reuses eframe's startup adapter inventory instead of creating a second graphics instance when Hardware opens. Restart after connecting a new GPU.

A machine with two usable adapters is required to verify physical switching. Desktop composition may still run on a different GPU from PaintFE's workloads.

Desktop regression: `cargo test -j1 --test frame_pacing -- --ignored --test-threads=1 --nocapture`. It launches two disposable, self-closing review windows and verifies real frame intervals plus clean shutdown. Run it only with a usable desktop and no other PaintFE instance.
