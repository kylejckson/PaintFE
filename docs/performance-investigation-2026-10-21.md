# Performance investigation, 21 October 2026

Status: original diagnostic baseline. Approved fixes have since been implemented
locally; see [measured results and validation](performance-results-2026-10-21.md).
No commit, push, version bump or optional-pack implementation was performed.

## Main finding

The shared control architecture is not the dominant measured startup or steady
Preferences cost on this PC. There are confirmed UI-thread settings-save stalls,
substantial graphics initialization costs, and a potentially expensive synchronous
ONNX Runtime probe. Colour-gradient painting has a separate scaling problem.

## Measurements

An optimized local working-tree build was used, including the pre-existing local
Arabic/font changes. This is not an exact rebuild of the published release.
Compilation and captures were sequential; no compiler ran during measurement.

- 22 successful native capture processes, 100,105 recorded UI passes.
- 32 control configurations: light/dark, 100%/125% scale, narrow/wide layouts,
  empty layout/numeric/standard/gradient controls, 24 controls per configuration.
  Each configuration measured 80 warmed passes after 20 warmup passes.
- Four font-registration variants, eight repetitions each.
- Live wheel scrolling, scrollbar dragging, brush drawing, toolbar zoom, panning
  and Ctrl+Z were observed in disposable review documents.
- Native runs used a separate portable folder and a fresh settings file or a
  copy of the existing debug configuration. The original settings SHA-256 stayed
  `295edcde26dd2769519cde5f3c906da1c9abb654835e01207c52c07ee49b5d1e`.
- The machine reports 16 logical CPUs. Overall system load during captures was
  approximately 3.5-16.3%. Continuous captures deliberately request uncapped
  repainting; their CPU usage is not representative of normal idle use.
- Two later backend follow-up captures encountered 77-80% overall system load.
  They are retained in `excluded.json` and excluded from the 22-capture analysis
  because their conditions were no longer comparable. No further runs were made.

Raw traces, process summaries, drivers, microbenchmarks and `summary.json` are in
`target/performance-investigation-20261021/`. They are local diagnostic artifacts.
See `performance-profiling.md` for the capture/analyzer interface.

## Confirmed findings and proposed fixes

### 1. Settings persistence can stall the UI

Native traces measured individual `AppSettings::save` calls taking **37.34 ms**,
**52.40 ms** and **173.55 ms**. The last produced a **174.23 ms** app UI pass.
The 52 ms case occurred with Preferences open; the 174 ms case occurred in the
editor after changing tools. These are not slow slider-drawing measurements.

`src/config/persistence.rs` performs file creation, writes, `sync_all` and rename
synchronously while holding the shared saves mutex. `flush` also clears the
deferred flag. Window/tool persistence later in the update can therefore write
immediately, even though Preferences previously entered deferred mode.

A separate supporting OS I/O probe wrote a disposable 7,376-byte file eight times.
Its median write time was 0.31 ms, median fsync 37.34 ms, and median complete
replacement 38.42 ms. This supports disk flushing as a plausible contributor;
it does not independently split the Rust save span into individual system calls.

**Proposal:** one settings-writer worker owns debouncing and atomic replacement.
The UI queues the newest immutable snapshot and never waits for disk I/O or a
mutex held during I/O. Retain deduplication, error reporting, last-revision wins,
storage-migration ordering and a shutdown drain. Increasing the debounce alone
does not remove the stall when a write eventually runs.

**Validation after approval:** inject slow writes; verify UI callbacks remain
responsive, newest settings win, failures preserve the previous complete file,
and shutdown/restart/migration retain the expected settings.

### 2. Window/graphics startup is the largest ordinary startup stage

Settled automatic-backend runs reached the first app UI callback at approximately
**619-625 ms**. About **560-566 ms** was before app construction, inside the
window/graphics framework initialization stage.

Backend-specific experiments, changing only the child process's supported
`WGPU_BACKEND` environment setting, found:

| Configuration | Framework initialization | Canvas GPU pipelines | First UI callback |
| --- | ---: | ---: | ---: |
| Automatic, settled repeats | about 560 ms | about 9.5 ms | about 620 ms |
| Vulkan, settled repeat | 282 ms | 9.5 ms | 340 ms |
| DirectX 12, two repeat runs | 467-500 ms | 1,291-1,293 ms | 1,809-1,841 ms |

Initial automatic/Vulkan pipeline creation also took approximately 262-272 ms
before later runs fell to about 9.5 ms. This is consistent with caching, but no
driver-cache purge or reboot was performed. These are not controlled cold boots.

The framework enables multiple backends and enumerates available adapters before
device creation. The experiment demonstrates a backend-dependent cost; it does
not identify every internal driver/window call or prove identical adapter
selection between all configurations.

`src/gpu/renderer.rs::from_context` additionally constructs the compositor and
eleven compute/tool pipelines eagerly, including effects not needed to display
the initial canvas. It already shares eframe's device; there is no demonstrated
second-device initialization problem in this path.

**Proposal:** investigate backend initialization more narrowly, then implement a
preferred-backend path with a verified compatibility/fallback design. Do not
force Vulkan universally. Defer uncommon processing pipelines and prepare them
outside interaction-critical work where the backend permits it. Measure first
use as well as startup so the pause is not simply moved to the first tool use.

**Validation after approval:** automatic/explicit backend comparisons, hardware
and software fallback, unavailable/failed adapters, repeated startup and first
use of every deferred pipeline, plus representative Windows/Linux/macOS checks.

### 3. AI availability probing blocks startup and path changes

The first capture using the copied configured runtime spent **311.9 ms** in the
startup ONNX probe. A later copied-settings run took **2.6 ms**. The first value
must not be presented as an unavoidable cost on every launch.

`src/app/bootstrap.rs` synchronously calls `probe_onnx_runtime`, which validates
the file, loads the native library and queries its API/version. The modal update
path repeats this synchronously when configured paths change.

**Proposal:** show the editor first and probe on a worker, or defer probing until
AI use. Represent checking/available/unavailable separately, preserve validation
and minimum-version checks, and discard results for superseded paths.

**Validation after approval:** delayed, missing and incompatible runtimes; rapid
path changes; correct AI availability and errors; no UI stall during probing.

### 4. Colour-gradient painting scales with width

At light theme/100% scale with 24 controls and a 600-point available width:

| Workload | Drawing shapes | Median UI construction | Median tessellation |
| --- | ---: | ---: | ---: |
| Standard sliders | 288 | 0.33 ms | 0.14 ms |
| Colour gradients | 16,680 | 1.14 ms | 3.70 ms |

`src/ui/controls/gradient.rs` paints individual vertical strips across each track
and separate alpha-checker squares. This is a genuine width-dependent rendering
cost. The synthetic 24-wide-gradient workload is much heavier than the normal
colour panel; it is not proof that the normal panel causes the reported lag.

**Proposal:** retain the triangle-handle appearance and shared input code, but
use a bounded mesh or cached gradient texture and a shared checker texture.
Cache invalidation must account for colour parameters, theme, width and DPI.
The current arbitrary sample closure needs a reliable cache key or a bounded
sampling representation; it must not silently reuse stale gradients.

**Validation after approval:** pixel/alpha comparisons, light/dark and DPI scales,
wide/narrow panels, changing linked colour channels, plus pointer, keyboard and
numeric-editor interactions for both approved styles.

## What did not reproduce as a major bottleneck

- Settled normal editor UI passes were approximately **0.27 ms median**.
- At the top of Interface, advanced settings/copy configuration gave roughly
  **0.43-0.53 ms median**, with p95 about **0.48-0.61 ms**.
- During live scrolling/dragging, Interface median was **0.68-0.72 ms**, p95 about
  **0.95-1.09 ms**. The exceptional long passes were dominated by settings saves.
- Section culling worked: typical views rendered two or three sections and
  skipped six or seven. Input sometimes rebuilt all nine sections; those layout
  passes did not explain the large measured stalls.
- Normal idle runs settled without continuous repainting. Motion-off and
  motion-on settled Interface costs were similar; no broad animation bottleneck
  was established.
- The system CJK read took approximately 6-8 ms in warm runs and 27 ms initially.
  Font registration with the system CJK font cost about 4.2 ms in the isolated
  probe. Font enumeration was about 1.7 ms, translation initialization 11 ms,
  and built-in assets about 19-20 ms. These are secondary startup costs here.
- Unused native JPEG XL encoding is not invoked by the inspected startup path.
  A build excluding JPEG XL was not measured, so loader/AV/binary-size effects
  cannot be ruled out by this investigation.

## Limitations and next decision

Timings are inclusive and overlapping. UI callbacks are not displayed FPS or
GPU-completion timings. Native event-driven gaps can include ordinary idle time
between separate inputs. First UI callback is not first visible/usable frame.
Results are from one Windows PC and small/default documents, not large-image or
all-hardware guarantees. Every possible expanded Preferences combination was
not exercised; a numeric edit and a section-expansion attempt were not verified
because helper clicks did not reliably reach those controls.

The trace writer is bounded and asynchronous; the final buffered records can be
lost at process exit. An external WPA exporter could not initialize within the
restricted environment, so no external stack-sampling or GPU presentation trace
was analysed. No ETW recording was started or left running.

The optimized build completed without compiler warnings. Shared-control boundary
self-tests passed with zero violations. Analyzer fixtures checked startup/frame
separation, incomplete final records and multiple-pass accounting. This was a
diagnostic pass, not a rerun of the full app test suite.

Recommended order: asynchronous settings persistence, nonblocking ONNX checks,
graphics/backend and deferred-pipeline work, then gradient painting. Font/package
size work can follow. Keep the shared controls rather than replace them based on
the current measurements. Temporary diagnostics remain opt-in for before/after
comparison and can be removed after approved fixes are confirmed.
