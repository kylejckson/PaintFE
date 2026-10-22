# HEIC and AVIF experiment

Local experiment on 2026-10-22, Windows x86_64, Intel Core i9-9900KF
(16 logical processors), Rust 1.96.1. These measurements describe the initial isolated experiment. PaintFE now bundles
a locally patched Heifer decoder and supports a signed optional AVIF pack.
See [implementation details](heif-avif.md). The separate [probe package](../tools/codec_probe/README.md) has its own
manifest and lockfile. Production codec dependencies are unchanged.

## Method and reproducibility

- Release builds use stripping, LTO, one codegen unit and PaintFE's static Windows
  CRT configuration. Helper sizes include the command runtime; they are not a
  measurement of the incremental cost of bundling codecs into PaintFE.
- Decode timings are medians of five in-memory decodes, including conversion to
  RGBA8. Encoding uses three runs. File I/O, process startup, checksums and the
  Python reference comparisons are outside those timers. These are local
  measurements, not end-to-end PaintFE import/export timings.
- Heifer 0.2.0 is compared with pillow-heif 1.8.1 / libheif 1.23.6 / libde265
  1.1.3. AVIF output is checked through Pillow 13.0.0's libavif integration, as
  well as image's dav1d decoder. Pillow may share the underlying AV1 decoder;
  this is an additional integration check, not independent AV1 conformance.
- Colour and alpha differences are sampled at roughly 100,000 pixels; RGB under
  fully transparent pixels is ignored. Higher-bit-depth HEIC is converted to
  RGBA8 by both paths, so small differences can reflect conversion/rounding.
  This does not test preservation of HDR or original 10/12-bit samples.
- Native dav1d 1.5.0 is statically built with assembly enabled and both 8-bit and
  high-bit-depth decoding. This matches the Rust binding's fallback source tag.
  A production pack should explicitly pin and validate a current patched
  version instead of inheriting that fallback tag.
- The minimal AVIF encoder has assembly/threading disabled. The optimized build
  enables `image/nasm` and `image/rayon` and requests four encoding threads.
- HEIC samples come from [pillow-heif fixtures at a pinned commit](https://github.com/bigcat88/pillow_heif/tree/b16be1196dfa465a342d68894696e685ca3655cb/tests/images)
  and its benchmark photo. They remain under ignored `target/codec-fixtures/`;
  photos are not redistributed with the experiment source.

## Measurements

Full measurements, checksums and fixture identities are saved in
[the Windows result data](../tools/codec_probe/results/windows-x86_64.json).

| Standalone executable | Size |
| --- | ---: |
| Common probe baseline | 216.5 KiB |
| Heifer decoder + encoder primitive | 583.5 KiB |
| Minimal AVIF encoder | 2.09 MiB |
| Optimized AVIF encoder | 3.23 MiB |
| Optimized AVIF encoder + dav1d decoder | 5.28 MiB |

Heifer opened all 15 nominally valid HEIC/HEIF fixtures. The three deliberately
damaged/empty fixtures returned errors without a panic. The truncated fixture
was rejected even though libheif recovered a partial 16x16 image.

| HEIC photo | Heifer decode + RGBA8 | Reference decode + RGBA8 |
| --- | ---: | ---: |
| 4032x3024 pug | 97 ms | 182 ms |
| 3024x4032 arrow | 127 ms | 211 ms |
| 8000x6000 photo | 471 ms | 742 ms |

Opening successfully did **not** imply correct transparency: both
`LA_8__29x100.heif` and `RGBA_8__29x100.heif` returned opaque pixels, while the
reference had alpha values from 4 to 251. Both primary images are `grid` items;
the 128x128 `hvc1` counterparts preserve alpha. Inspection shows auxiliary alpha
attachment in `decode_hvc1`, but no corresponding attachment for the derived
grid in `decode_item_with`. This strongly indicates a grid-alpha integration
gap; a patch would still need independent regression validation.

The 8-bit photos otherwise closely matched the reference. Higher-bit-depth
fixtures differed by up to three RGB levels after conversion to 8-bit. These
measurements do not establish which conversion is preferable.

Separate one-shot process instrumentation measured peak working sets of about
194 MiB for the 12 MP pug and 747 MiB for the 48 MP photo. Heifer materializes
16-bit RGBA samples even for 8-bit input, and the probe also materializes RGBA8.
The instrumented runs are not the timing runs above. Production integration
needs a deliberate memory limit and buffer-lifetime policy.

AVIF quality 80, speed 8:

| Input | Minimal encode | Optimized encode | Decode | AVIF bytes |
| --- | ---: | ---: | ---: | ---: |
| 512x512 artwork with alpha | 972 ms | 122 ms | 8 ms | 101,739 |
| 1440x1080 photo | 3,925 ms | 454 ms | 18 ms | 214,764 |

All six AVIF encoding configurations produced readable files. The optimized
and minimal builds produced identical encoded bytes for the tested settings.
image/dav1d and Pillow/libavif outputs differed by at most one RGB level and
matched alpha exactly. Invalid AVIF input returned an error without a panic.

At speed 4, optimized photo encoding took 2,287 ms for 205,873 bytes: substantially
slower than speed 8 for a modest size saving on this sample. Quality 100 still
was not pixel-exact, including alpha differences of up to two levels on the
artwork. Do not label this export path lossless or use it as the document format.

## Original proposal (followed by local implementation)

1. Keep the experiment separate from the app until the grid-alpha issue is fixed
   and verified, or evaluate a different HEIC backend as a fallback. Do not
   silently import an opaque version of a transparent document.
2. Heifer is small enough to consider bundling after correctness work. Measure
   actual PaintFE release growth before deciding; the standalone size does not
   prove the same incremental cost. An optional helper also fits the existing
   download model and releases codec memory when it exits.
3. An optional AVIF import/export helper looks practical at roughly 5.3 MiB on
   Windows. Keep assembly optimizations and threading enabled; start with speed
   8, run exports in the background and support cancellation. Further compare
   colour-model and alpha-quality choices on artwork before choosing defaults.
4. Keep HEIC export separate: Heifer cannot provide it yet. A future optional
   libheif + Kvazaar helper is the working candidate to test. Validate its
   platform builds, output quality and distribution requirements first.
5. Before shipping any helper, cover all four supported desktop targets, strict
   bounded raw-pixel IPC, signed packs, colour profiles and resource limits.
   Downloads remain explicit; subsequent editing/encoding stays offline.

## Heifer encoder status

The published `heifer-hevc-enc` 0.2.0 and upstream Heifer commit
`5220e04b51280ae26db33187795b432fc5563754` expose a CABAC arithmetic encoder and
an error enum. They do **not** expose an image encoder or HEIF container writer.
The compiled primitive smoke test produces three bytes, not a decodable HEVC
picture or HEIC file. A full save/round-trip test is therefore unavailable;
implementing the missing codec would exceed an integration experiment.

See [Heifer's published roadmap](https://docs.rs/crate/heifer/0.2.0) and
[libheif's working HEIC encoder backends](https://github.com/strukturag/libheif#supported-features).

## Scope of validation

These are Windows measurements. Linux, Intel macOS and Apple Silicon builds,
packaging, representative real-user files, ICC handling, cancellation, resource
limits and a larger malformed-input corpus still need validation before shipping.
The initial experiment added no production formats. The implementation follow-up
adds HEIC hooks and AVIF pack registration without automatic network activity.

The UI changes made alongside this experiment reuse the existing document
duplicate action and information window, and share dialog section typography
with Preferences. Existing duplicate-document independence and Preferences
search-navigation regression tests passed. Cargo check, formatting and locale
checks passed. Cargo normalized the pre-existing root lockfile's package version
from 1.4.12 to the already-declared 1.4.13; production dependency versions did not
change during that initial experiment. The follow-up now adds vendored Heifer
and moves AVIF codecs into the optional helper. PR #200 was merged separately
after its CI checks passed.
