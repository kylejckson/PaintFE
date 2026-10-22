Local copy of the published Heifer 0.2.0 crate (MIT OR Apache-2.0).
The upstream source is https://github.com/HarpeLm/Heifer.

PaintFE maintains these changes locally:

- Attach auxiliary alpha in the common item path, before the parent image's
  geometric transforms. This covers coded, grid and overlay parents.
- Decode derived alpha items through the same bounded item path, including
  grid alpha, with nesting limits and bit-depth scaling.
- Require alpha width **and** height to match the decoded parent; mismatches
  return an error rather than indexing a short plane.
- Limit default decoded images to 64 million pixels and compressed input to
  256 MiB before a decoder hook reads it into memory.
- Normalize the package manifest for vendoring; codec subcrates remain pinned
  to published 0.2.0 versions. Preserve both upstream license files.

Regression inputs and independently decoded reference pixels live in
`tests/fixtures/heif/`; `tests/heif.rs` exercises PaintFE's real I/O hooks.
This patch does not add HEIC encoding. No upstream contribution is performed.
