# PaintFE text-layout patch

Base: upstream egui/epaint 0.35.0, copied from the published crates. Both original
MIT and Apache-2.0 license texts are preserved. Only egui text-selection painting
and epaint's text layout, glyph clusters, caret mapping and row metadata change.
No widgets, numeric controls or application action identities are reimplemented.

- Shape runs with explicit direction; preserve logical source order for wrapping.
- Represent a logical character cluster separately from its additional ink glyphs.
- Apply Unicode bidi levels to shaped clusters after line breaking, before drawing.
- Keep logical cursor indexes and original clipboard text; map hit-testing and
  visual arrow navigation through the row's caret positions.
- Paint selected mixed-direction ranges as separate visual spans.
- Keep the existing fast path for text with no RTL characters.

`unicode-bidi` 0.3.18 supplies UAX #9 levels. It does not shape or rasterize text;
HarfRust and the existing font atlas continue doing that work.

Regression coverage: `tests/text_direction.rs` plus the normal shared-control,
label-editor, dialog and locale tests. Include Arabic marks/ligatures, filenames,
numbers, wrapping/elision, selection and copy when upgrading egui. Native-speaker
review is still required for language quality. This patch does not mirror the
application layout or replace the document text tool's separate layout engine.

Upgrade both vendored crates together with eframe. Rebase these changes on the
new upstream text implementation and run the regression suite; do not silently
remove either `[patch.crates-io]` entry or add display-order transformations to
`i18n::translate`. Keep saved strings in logical Unicode order.
