# Large-image rendering and HEIF regression checks

Related issue: [207](https://github.com/kylejckson/PaintFE/issues/207).

The egui renderer and PaintFE compositor share a wgpu device. The adapter's
maximum texture size can exceed the limit requested for that device. Request the
adapter's supported dimension at creation, then validate allocations against the
actual device limits. Never treat adapter capabilities as granted device limits.

Full-canvas GPU allocations also have a 64 MiB per-texture budget and a 256 MiB
estimated visible-layer/scratch budget. Larger documents use committed CPU
composites in viewport tiles, with display-only downsampling when zoomed out.
Tiles are at most 1024 pixels per side and never exceed the device limit. Original
document dimensions, layers and export pixels are unchanged. Paste, selection and
straighten display previews are bounded separately; live preview caches remain
reusable between frames. Large-image editing can be slower than the GPU path.

The public `sample1.heif` linked from
[FileSamples](https://filesamples.com/formats/heif) contains 8736x5856 pixels.
An isolated Windows WIC probe reproduced an additional failure: a 2 GiB worker
limit returned `0x8007000E`, while the same decode with a 3 GiB limit succeeded.
The Windows worker now has a 3 GiB upper bound, matching Linux. This does not
reserve that memory. Existing input/output sizes, pixel-count validation and the
parent timeout still apply. macOS retains its separate resident-memory policy.

Keep downloaded examples outside tracked project files. Run the normal
`large_images` integration target and `canvas::large_image::tests` for texture
guards, bounded previews, tile invalidation, panning, CPU adjustments and preview
separation. To test the external 51 MP example through the real codec worker:

```powershell
$env:PAINTFE_TEST_ISSUE207_HEIF = 'D:\path\to\sample1.heif'
cargo test -j1 --test heif issue_207_large_heif_decodes_in_the_bounded_worker -- --ignored --exact --test-threads=1
```

This check is opt-in because it needs a locally installed system codec, an
external sample and substantially more memory than the small regression tests.
Run it separately, not alongside other heavy jobs.

For GUI regression checks, a debug build accepts
`PAINTFE_REVIEW_TEXTURE_LIMIT=8192`. Open the original HEIF, inspect its full
dimensions, zoom and pan, draw a stroke, undo it, and check saving to a separate
file. Test a normal small image afterward. The override is absent from release
builds. The fatal-error dialog can be checked only in a disposable process with
no user document open; it must show the log path and failure details.
