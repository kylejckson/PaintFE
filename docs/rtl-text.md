# Arabic text direction

PaintFE keeps catalogues, label overrides, filenames and copied strings in logical
Unicode order. Text must never be reversed or converted to presentation forms in
the translation lookup or individual screens.

The shared egui/epaint 0.35 text pipeline includes the focused patch documented in
`vendor/epaint/PAINTFE-PATCHES.md`. It adds paragraph/line bidi ordering, preserves
Arabic shaping clusters, and maps editing and selection to logical text. It uses
the already bundled Noto Sans Arabic font and existing HarfRust rasterization.
No optional download or network access is required.

This corrects text direction; it does not mirror the tools panel, canvas, numeric
fields or the whole application. Document text layers use a separate engine and
are not changed by this UI fix. Run `tests/text_direction.rs` when updating the
text pipeline, then inspect menus, Preferences and label editors in both themes
and at large text sizes. Include Arabic with English filenames, numbers, percent
signs, punctuation, marks, ligatures, wrapped lines and copied selections. Have an
Arabic speaker review the build before claiming complete native-language parity.
