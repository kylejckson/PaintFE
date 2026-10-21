# Performance fixes and review results, 21 October 2026

Implemented locally for review. Version remains 1.4.10. No commit, push, release,
optional-pack implementation or dependency change was made.

## Measured results

Three alternating before/after pairs for each scenario, using optimized builds,
the same copied settings and 1280 x 900 windows on this Windows PC. The final
12 desktop captures had 18.2-25.9% overall CPU activity. Other work continued;
these are comparable warm runs, not an idle-machine or cold-boot guarantee.
The preserved baseline includes the pre-existing local Arabic/font changes,
so this is not an exact comparison against the published release executable.

Startup here means elapsed time inside the app to its first UI callback.
It excludes earlier OS executable loading and does not measure first presented
frame, monitor refresh rate or first usable interaction.

| Measurement | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Editor first UI, median | 688.48 ms | 382.41 ms | 306.07 ms earlier (44.5%) |
| Interface preferences first UI, median | 615.89 ms | 341.06 ms | 274.83 ms earlier (44.6%) |
| Settings-save UI time, largest matched sample | 651.42 ms | 0.1193 ms | File I/O moved off UI thread |
| Peak process working set, median of matched captures | 223.46 MiB | 161.61 MiB | About 62 MiB lower |
| Wide colour-slider stress case, 24 controls | 16,680 shapes | 240 shapes | 98.6% fewer shapes |
| Same stress case, median CPU tessellation | 3.70 ms | 0.13 ms | Headless paint benchmark |

Editor startup samples were 875.04 / 686.65 / 688.48 ms before and
381.89 / 382.41 / 390.13 ms after. Interface samples were
682.10 / 610.39 / 615.89 ms before and 360.15 / 341.06 / 324.00 ms after.

Steady UI work was broadly unchanged: editor medians were about 0.55 ms before
and 0.54 ms after; Interface about 0.70 ms before and 0.69 ms after. This does
not support replacing the shared widget architecture. The gains primarily come
from startup and removing intermittent blocking work.

Actual background settings writes still took roughly 31-148 ms in the matched
captures. They no longer occupied the UI save scope. Warm background pipeline
preparation took about 8-12 ms; configured ONNX availability probes about 3 ms.
That work was deferred, not eliminated.

The gradient probe covered all 32 light/dark, 100%/125% scale, narrow/wide,
empty/numeric/standard/gradient configurations. All eight gradient cases used
240 shapes, compared with 4,080 or 16,680 before. The table uses the light,
100%, 600-point case. This is a stress benchmark, not normal colour-panel FPS;
the after probe also ran with background activity and is not a perfectly idle
CPU comparison. Geometry counts do not depend on CPU load.

The first 16 captures were retained as supplementary data. Fourteen exceeded
the pre-established 30% CPU-load threshold. None are used in the matched
desktop headline results. The heavier-load traces still demonstrate the
settings-save issue, but were not substituted for the matched comparisons.

## What changed

- [Native settings persistence](../src/config/persistence/native.rs) now uses one
  debounced writer. It releases the queue mutex before file creation, fsync and
  atomic replacement. Duplicate saves coalesce; failures remain visible and the
  same value can be retried. Normal shutdown, explicit CLI exit and storage
  migration wait for pending writes. Ordinary UI frames do not wait for disk.
  Browser local storage retains synchronous behaviour.
- [ONNX availability](../src/ops/ai/availability.rs) now checks configured paths
  on a worker after an initial repaint delay. Only one availability job runs at
  a time; newer paths supersede stale results. The Preferences Test button also
  probes on a worker. Inference and model processing were not changed.
- [Graphics startup](../src/services/graphics.rs) prefers a cached working
  Windows backend, starting with Vulkan, and preserves explicit WGPU_BACKEND
  overrides. A failure before app construction restarts once using automatic
  selection. The replacement waits for the parent's IPC server to exit and
  preserves command-line file arguments. Other platforms keep their automatic
  backend policy. The small graphics_backend.cfg file is advisory local state.
- [GPU pipelines](../src/gpu/pipeline.rs) initialize on first use. Native UI
  renderers prepare them after a short delay on one owned worker. First use can
  claim a queued job; a job already compiling may still need to finish. Renderer
  teardown cancels queued work and joins its worker before dropping resources.
  Headless renderers do not start background compilation. All 11 pipeline types
  completed preparation in the native captures.
- [Colour tracks](../src/ui/controls/gradient.rs) now use cached gradient and
  theme-specific checkerboard textures instead of one shape per pixel. Samples
  are checked each frame, so changing linked colours updates the texture.
  Width/DPI changes invalidate its resolution. Triangle handles and the shared
  numeric, pointer and keyboard behaviour remain in the same components.

The executable grew from 64,525,312 to
64,893,440 bytes, about 0.35 MiB. This pass improves
runtime behaviour; it does not reduce binary size or remove JPEG XL/fonts.
Optional packs remain a separate future decision. The core remains one
executable and PaintFE gained no network requests.

## Validation and corrections during implementation

- 66 distinct focused and integration tests passed in sequential batches:
  44 unit/UI checks, 12 GPU pipeline checks, 7 JPEG XL checks and 3 storage
  restart checks. One existing manual CPU-profile test was intentionally ignored.
- Two additional complete GPU runs passed, 11 tests each, after the initial
  native crash was addressed. A new first-use colour/median pixel test increased
  the final GPU target to 12 tests.
- The new ownership test first caught incorrect mutable lazy initialization;
  that was corrected. The first broad GPU batch then crashed in native code
  after eight checks. Preparation was restricted to UI renderers and given an
  explicit cancel/join lifetime. Full GPU runs then passed. No native stack
  trace was obtained, so the exact driver-internal cause is not established.
- A Preferences persistence check initially hit a temporary-directory access
  error. It passed using isolated workspace storage. User settings were not used
  as test storage. The original debug settings SHA-256 remains unchanged.
- Optimized build passed with no compiler warnings; cargo fmt --check passed.
  Shared-control boundary self-tests passed with zero violations. Debug builds
  emitted environmental incremental-cache hard-link/copy warnings. Existing
  CMake generator caches required matching the generator for each build profile.
- Live light-theme checks: wheel scrolling over a slider left opacity at 90;
  scrollbar dragging reached advanced sections; alpha dragging changed 255 to
  127 and End restored 255; changing saturation updated the linked value track.
- Live dark-theme inspection confirmed the track/checkerboard rendering. A brush
  mark appeared and Ctrl+Z restored the blank canvas. Some earlier helper drags
  did not produce the intended action, so they were not counted as passes.
- The actual Preferences Test button reported ONNX Runtime v1.24.1 loaded.
  Automatic review initially blocked that exact DLL execution; the check
  proceeded only after the user explicitly approved its configured path.
- Injected preferred-backend failure produced no app in the parent, exactly one
  recovered app using Vulkan, and the window title PaintFE - graphics recovery
  test.png. This verified preservation of an image argument containing spaces.
  Early timed exit also completed successfully. The short trace lost its final
  UI record, so it is not used as a startup measurement.

This performance pass did not rerun Clippy or the complete app test suite.
Windows was tested at runtime; Linux, macOS and browser runtime validation remain.

## Limits and remaining investigation

Some early callback gaps around 57-80 ms remain in both builds, even when UI
work itself is short. These traces cannot attribute those gaps to scheduling,
driver work, presentation or other system activity. No displayed-FPS or external
GPU/OS stack trace was captured. Large-image editing was not benchmarked.

Opt-in profiling remains available for review and comparison. Normal review
launches have it disabled. Temporary diagnostics can be removed after review.
See [capture instructions](performance-profiling.md) and the
[original diagnostic report](performance-investigation-2026-10-21.md).

Raw captures, process metadata, comparison.json, validation.json and build/test
logs are under target/performance-after-20261021/. Source hashes and executable
hashes are recorded there. The preserved baseline captures remain under
target/performance-investigation-20261021/. These generated files are ignored
by Git; this report preserves the important numbers outside target/.

## Manual review

An untimed, profiling-disabled copy is open from
[target/performance-after-20261021/app/PaintFEProfile.exe](../target/performance-after-20261021/app/PaintFEProfile.exe).
It uses separate portable settings. Normal settings and documents were not
changed, and confirmation before exiting unsaved work is enabled in this copy.

1. Compare launching it with your existing build. Expect the initial window
   sooner; judge first-use tool responsiveness as well as window appearance.
2. In Preferences > Interface, scroll with the wheel over sliders, then drag the
   scrollbar. Verify values stay unchanged until edited deliberately.
3. Change a preference, close and reopen the app, and confirm it persists.
4. In Colors, check alpha and expanded HSV/RGB tracks in light and dark themes.
   Change linked channels; test dragging, End/Home, typing, Tab and Escape.
5. Immediately after a fresh launch, try Gradient, Fill, Liquify and a filter.
   Then draw and undo. Report any first-use pause or unexpected cursor/focus issue.
