# Local performance investigation

Profiling is opt-in and records timings, counts and fixed scenario labels, never
document pixels, text, filenames or settings contents. Nothing is uploaded.
Unset `PAINTFE_UI_PROFILE` to disable it. Debug review copies also recognise the
existing `ui-profile.enabled` marker beside their executable.

## Capture

Use an isolated portable copy of an optimized executable. Do not profile while
compiling or running tests. Keep window size, document and settings comparable.

- `PAINTFE_UI_PROFILE`: absolute output JSONL path; enables the bounded background
  writer (20,000 records maximum, nonblocking queue).
- `PAINTFE_UI_PROFILE_DRIVE`: when present, requests continuous repaint for the
  first 20 seconds, plus active input. This is a load test, not normal idle use.
- `PAINTFE_PROFILE_SCENARIO`: `interface` or `general` opens that Preferences page
  for a diagnostic run. Only effective while profiling is enabled.
- `PAINTFE_PROFILE_EXIT_AFTER_MS`: optional automatic close time measured from
  tracer initialization. Use only in a disposable review copy with no documents.

Without continuous drive, leave the editor idle to check its natural repaint
frequency. Do not interpret idle gaps as stuttering. Separate first-launch runs
from warm repeats; freshly copying an executable is not a true cold-cache test.

## Analyse

`python tools/analyze_ui_profile.py trace.jsonl --output report.json`

The analyzer supports multiple traces. Stage timings are inclusive: numeric
editors are inside sliders, Preferences is inside menus/dialogs, and sections
are inside Preferences. Do not add their timings together.

Startup spans end before the event loop takes over, at app construction, and
around settings, fonts, assets, runtime probes and canvas setup. First UI timing
is not a measured first-presented or first-usable timestamp. `previous_cpu_ms`
belongs to the previous callback and includes framework/rendering CPU work;
frame gaps include scheduling and presentation waits. Neither is GPU timing.

`examples/performance_probe.rs` compares production controls in light/dark themes,
100%/125% scale and narrow/wide layouts, and measures font registration separately.
It is a headless CPU microbenchmark, not proof of displayed frame rates. Build
with `cargo build --release --example performance_probe -j 1`, then run the built
executable separately after compilation finishes.

Keep raw traces and before/after summaries together. Remove temporary diagnostic
instrumentation only after approved fixes have been measured and confirmed.

`PAINTFE_PROFILE_FAIL_PREFERRED=1` simulates a Windows preferred-backend
selection failure only when a profiling scenario is enabled. The replacement
process uses automatic selection once and preserves command-line arguments.
When capturing this recovery case, wait for the replacement process as well as
the parent; the child writes a separate `.recovery.jsonl` trace. Do not use a real document.

Native UI renderers prepare compute pipelines on one owned worker after an
initial delay. Renderer teardown cancels queued jobs and joins that worker.
Headless renderers create pipelines on first use without starting background
compilation. Record both first UI and background preparation: moving work later
does not mean its CPU cost disappeared. Settings write spans also run in the
background and must not be counted as UI-thread time.
