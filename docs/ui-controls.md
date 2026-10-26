# Shared UI controls

All scalar numeric inputs use `crate::ui::controls`. Extend this implementation
instead of adding another editor, slider handler, stepper or expression parser.
Rust composition replaces a base-class hierarchy: presets configure the same
component, and specialised controls compose the same interaction primitives.

## Supported components

| Component | Use |
| --- | --- |
| `Numeric::dimension` | Pixel dimensions, arithmetic expressions, integer precision, confirm typed input, +/- buttons |
| `Numeric::percentage` | Percentages from 0 to 100; configure a different range when necessary |
| `Numeric::angle` | Angles from -180 to 180 degrees |
| `Numeric::new` | Explicitly configured numeric quantities, including integers |
| `Slider` | Standard settings/dialog slider, using the shared Numeric editor |
| `GradientSlider` | Colour gradients and alpha checkerboard with a triangular handle |
| `color_edit_button_srgba/srgb/rgb` | Colour popups with shared HSV/alpha/channel editors, gamma/linear channels and hex copy |
| `stepper` / `reset` | Shared buttons for composites such as size/preset dropdowns |
| `ValueRow` | Stable labelled-row composition; exposes its label response for Preferences search |

`Slider` deliberately disables egui's built-in number editor. Showing a native
slider value would introduce a second input implementation. Its optional value
is our Numeric component, including custom formatters/parsers (for example a
0..1 model displayed as 0..100%).

Native egui colour-picker popups also contain private DragValue fields. Use our
colour buttons instead. Their saturation/value square is a specialised 2D editor;
the scalar controls use GradientSlider/Numeric. Opaque, normal alpha and additive
colours are supported. Linear RGB inputs are not quantised by merely rendering a
button; model values update only after an edit.

## Identity and interaction contract

- Every component has a call-site identity. In loops, dynamic lists and reusable
  row helpers, also set `.id_salt(stable_model_key)`. Do not use translated labels
  or array positions for reorderable data. Keep the surrounding UI scope stable.
- Dragging previews values live. Numeric text is live by default; dimension text
  commits with Enter, Tab or loss of focus. `.update_while_editing(false)` selects
  that policy explicitly. Escape restores the value at the start of editing.
- Numeric wheel adjustment follows Preferences. An event is consumed once, so it
  cannot change the value and scroll the container. Disabled controls do neither.
- The Preferences Interface page reserves the wheel for scrolling. Use
  `controls::without_wheel_adjustment` for such containers: it scopes the policy
  to child controls without changing the saved global preference. Dragging,
  typing, keyboard stepping and buttons remain available.
- Shift uses the configured fine-adjustment factor; integer quantities still
  change by at least one. Slider arrows step; Home/End select their limits.
- Native slider accessibility actions remain native. Custom colour tracks expose
  slider metadata and keyboard focus. Preserve these when extending a control.
- Linked values are updated by the model. The editor discards stale edit state
  when a linked field changes externally.
- Use `.focus_on_open(first_frame)` when a dialog initially focuses a field.
  The shared component handles registration and selecting the formatted value.
- Integer quantities use integer-backed widget values, even when a legacy effect
  model stores them as floats. Convert at the model boundary; decimal formatting
  alone does not make a floating-point editor an integer control.
- Use `.steppers(amount)` on Numeric, `.steppers(true)` on Slider, and
  `.reset_to(default)` for resettable values. Supply the actual model default;
  do not independently paint or implement these buttons in screens.
- Slider tracks double-click to reset when a semantic default is supplied.
  `.double_click_reset_to(default)` adds the gesture without another reset
  button when a dialog already has its header reset. The number editor retains
  double-click text selection. Never derive a reset default from zero or the
  range minimum. A neutral default outside the pointer range uses edit-only
  clamping. Optional theme overrides use `.on_double_click_reset` to clear the
  override and inherit the theme. Colour channels have no implied reset colour;
  opacity may reset to fully opaque. Preferences → Interface → Motion and
  interaction details can disable the gesture.
- Step buttons do not add extra Tab stops to a spinner. Its editor handles
  keyboard stepping and accessibility increment/decrement actions. Reset remains
  a separate keyboard-accessible action.
- Genuine strings such as search queries, file paths, hex colours and W:H aspect
  ratio expressions remain text editors. A colour wheel, gradient-stop editor,
  curves editor or canvas transform is a specialised 2D/multi-value editor, not a
  scalar slider. Reuse the shared primitives where applicable; document any new
  numeric exception here and give it independent interaction tests.

## Examples and validation

Run `cargo run --example control_gallery -j1` for both styles, presets, theme/text
scale variations and real New Image, resize and effect dialogs. It has no settings
persistence and does not modify user documents.

Run `python tools/check_ui_controls.py --self-test --report target/ui-controls.json`.
The CI boundary check rejects raw editors/sliders (including imported aliases),
dimension TextEdit bypasses and use of the old Numeric implementation path.
It skips strings/comments and inventories references in source, tests and examples.
It cannot detect every possible hand-painted implementation; review still checks
that custom controls compose the approved primitives.

`src/ui/controls/tests.rs` exercises both styles in a plain panel, labelled grid and
popup-style frame. `src/ui/numeric.rs` tests expressions, typing/cancel, drag,
clamping and wheel policy. Production resize, Preferences and Save As tests verify
linked dimensions and that numeric dragging does not move its window. Visual
review complements value/focus assertions; screenshots alone do not prove input
correctness. Run tests sequentially in small batches.

When changing the shared implementation, test its interaction contract and two
or three real callers, including light/dark themes and a larger text scale. When
adding a screen, use a preset or existing component, add an integration check for
its model-specific behaviour, and run the boundary audit. Do not bless visual
changes by regenerating baselines without reviewing them.

Interface sections use `panels/settings_sections.rs` to retain variable heights
offscreen. Keep explicit section identities, focus/drag ownership, search reveal
and measurement invalidation when changing this layout. Profile the full app
and tessellation as well as isolated layout; a small SettingsWindow CPU number
does not establish smooth GPU presentation. Debug review copies can opt into a
local frame trace with an adjacent `ui-profile.enabled` file. Analyse it using
`python tools/analyze_ui_profile.py path/to/ui-profile.jsonl`; idle gaps are
excluded, and the report does not claim to measure GPU completion.

Colour tracks cache sampled textures per stable widget identity instead of
creating a shape for every pixel. Re-sample caller closures each frame and
upload only changed pixels; do not cache by closure identity. Keep checkerboard
textures theme-specific and track resolution aware of DPI. The paint tests
check bounded geometry and invalidation; shared interaction tests still apply.

## Text direction

Shared widgets use the patched egui/epaint layout pipeline for RTL UI text. Keep
values and clipboard text in logical order. Do not add per-widget reversal or
custom numeric text editors. See `docs/rtl-text.md` and the vendored patch notes
when upgrading the renderer or editing selection/caret behaviour.
