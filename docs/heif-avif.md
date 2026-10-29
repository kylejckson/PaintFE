# HEIC/HEIF and AVIF

## HEIC/HEIF: installed system codecs

PaintFE no longer compiles the Heifer image decoder or its HEVC implementation. The old
HEIC export pack is no longer offered, installed or used. Existing pack files
are left on disk; PaintFE does not delete user data. No HEVC codec is downloaded
by PaintFE. JPEG XL and AVIF keep their existing implementations.

- Windows uses WIC and the installed HEIF Image Extensions / HEVC Video
  Extensions. Install missing extensions from Microsoft Store and restart PaintFE.
- macOS uses Apple's ImageIO and CoreGraphics frameworks.
- Linux loads the installed `libheif.so.1`. Its HEVC decoder/encoder must also
  be installed through the distribution. Available packages and licence terms
  depend on the distribution; PaintFE does not install them.
- The browser build does not contain an HEVC decoder. HEIC requires the desktop
  app and a supported system codec.

Preferences > Optional features includes a **HEIC / HEIF (system codecs)** section
with separate Open and Save availability. Availability is cached for the session;
restart after installing or changing system codecs. Presence does not guarantee
support for every HEIF variant. Errors explain when the installed codec cannot
handle an operation. No capability check uses the network.

The bridge exchanges 8-bit or 16-bit RGBA still images. Higher-depth inputs require
a native path exposing that precision; otherwise import is rejected rather than
silently reduced to 8-bit. Supported 16-bit buffers use the editor's existing
deep-pixel import path. HDR gain maps, ICC/EXIF metadata and image sequences are
not preserved by this interface. HEIC export currently takes 8-bit RGBA.
Export is normally lossy even at quality 100. Use PNG/TIFF/PFE for appropriate
preservation workflows. Existing HEIC save targets are retained; saves still pass
through the shared format-loss checks and atomic replacement path.

### Windows validation and limitations

Locally checked with HEIF Image Extension 1.2.57.0 and HEVC Video Extension
2.4.109.0. The source-transform alpha plane must be requested separately: a
normal WIC RGB-to-RGBA conversion synthesized opaque alpha. The new bridge reads
that plane and matches the three 8-bit reference alpha fixtures exactly.

This Windows codec does not expose the required precision for the supplied
10/12-bit fixtures. They are rejected rather than imported as 8-bit or opaque
copies. Its HEIC encoder does not accept
transparent input; PaintFE reports that limitation and leaves the destination
untouched. Opaque 29x100, 32x32 and 128x128 export round trips are covered.

Native Windows chroma reconstruction differs from libheif on the synthetic RGB
grid: measured RGB mean absolute difference 8.04/255, maximum 44/255; alpha is
exact. Tests use both mean and peak bounds for this native path, rather than
claiming pixel-identical RGB output between independent codecs.

A wider Windows corpus check covered fourteen nominally valid images: ten
opened with reference-matching dimensions, including rotated and spatial photos;
four high-depth variants were unsupported. The 48-MP sample was skipped to limit
PC memory use. Corrupt, empty and truncated samples were rejected by the container
preflight instead of relying on the native codec's partial-image recovery.

Linux was tested locally under Ubuntu 26.04.1 in WSL2 with system libheif 1.21.2
and its libde265 decoder and x265 encoder plugins. All seven Rust import/export
tests passed, including the same-binary worker, malformed input, alpha, 10/12-bit
precision, file loading and export round trips. The three 8-bit reference fixtures
matched exactly; 10/12-bit comparisons converted to 8-bit had maximum RGB error
3/255 and alpha error 1/255, while decoded buffers retained 16-bit precision.
On this Ubuntu installation the codec plugins needed to be installed separately:
`sudo apt install libheif1 libheif-plugin-libde265 libheif-plugin-x265`.
Other distributions may use different package names. macOS still needs native
testing. No claim of full HEIF conformance is made.

A disposable-profile Linux GUI smoke check also launched and exited cleanly
through WSLg/X11 using Mesa software rendering. The X11 keyboard library
`libxkbcommon-x11-0` was required on this installation. This is a startup check,
not validation of a physical Linux GPU or desktop performance.

### Isolation and distribution

Codec operations run in a private mode of the same PaintFE executable, before GUI
or settings startup. There is no extra codec executable to install. The bridge
uses only platform APIs/system libraries, with bounded stdin/stdout, serialized
jobs, cancellation, a 120-second timeout, 64-million-pixel / 256-MiB input limits,
and worker memory limits. Decoded RGBA and encoded output lengths are validated.
Process separation contains crashes but is not a security sandbox.

The small `heifer-isobmff` dependency reads container metadata only. It has no
HEVC decoding/encoding implementation. It validates primary and auxiliary-alpha
dimensions, identifies source precision and bounds derived-item traversal before
the system codec runs. The HEVC decoder/encoder crates are absent from the app's
dependency graph. No manual byte-pattern heuristic substitutes for item associations.

Transparent exports are either preserved by the installed encoder (checked by
decoding the result) or rejected. Unsupported transparency is never flattened
silently. These measures avoid distributing an HEVC implementation; they do not
constitute a patent licence or legal clearance for every use or jurisdiction.

The former [Heifer grid-alpha patch](https://github.com/kylejckson/PaintFE/blob/feda9d2411ebcd9d578cf9879fff2e1cdcd8137a/vendor/heifer/PAINTFE-PATCHES.md) remains in Git history
for an upstream report. Its decoder and encoder experiment have been removed;
only the bounded `heifer-isobmff` container parser remains in production. Nothing has been posted upstream on the user's behalf.

## AVIF optional pack

In **Preferences > Optional features**, install **AVIF import and export**, or
import its signed `.pfepack` file offline. Opening an AVIF without the pack offers
installation; reopen the image after installing. Save As also offers the pack
and keeps the dialog open until an encoder is available.

Downloads occur only after user confirmation, from the matching GitHub release.
Opening and saving remain offline afterward. The helper is signed and verified
like the JPEG XL pack, including platform identity, protocol, payload names and
checksums. The core does not compile the AV1 encoder/decoder. The initial Windows
pack is approximately 2.15 MiB compressed, including license notices.

AVIF export is a lossy **8-bit still image**, with the existing quality control
and encoder speed 8 using four threads and CPU optimizations. Quality 100 is not
lossless. Use PFE/PNG/JPEG XL when exact pixels are needed. Import supports the
native decoder's supported high-bit-depth images. ICC/HDR metadata are not
preserved by this helper protocol. Animated AVIF is not imported as animation.

Expensive main open/save operations run in the existing background I/O flow;
preview jobs are coalesced separately. A processing dialog can cancel active
AVIF jobs. Helpers have input/output and dimension limits, a 120-second timeout,
and process memory controls. macOS monitors resident memory instead of applying
a small address-space limit to its large shared-cache mappings. Process isolation
contains codec crashes; it is not a security sandbox.

CLI example, after installing the pack into the same profile:

```sh
PaintFE --input input.png --output output.avif --format avif --quality 80
```

## Maintainer builds

No new account or secret is required. Existing `OPTIONAL_PACK_SIGNING_KEY`
release setup continues to sign the CJK, JPEG XL and AVIF packs. From v1.4.17,
publication verifies twelve assets across Windows x86_64, Linux
x86_64, Apple Silicon and Intel macOS.

```sh
python tools/build_optional_packs.py --release
```

The system bridge is built by Cargo using the platform C++ compiler. On macOS
this includes Objective-C++ and the system frameworks. Linux requires no libheif
headers or HEVC implementation at build time; the codec library is loaded only
when requested. The old HEIC source-build tool is historical and is not invoked
by the optional-pack/release pipeline. Older signed releases retain their original
sixteen-pack staging rules for repair; no existing release assets are removed.

Linux CI installs Mesa's software Vulkan driver and runs
`tools/check_ci_vulkan.py` before compiling the codec/UI tests. The preflight
selects Lavapipe, checks it with `vulkaninfo --summary`, then exports the driver
and private runtime directory to later steps. The real toolbar-save regression
stays enabled and needs no physical GPU or desktop session.

The AVIF helper builds separately from `optional/avif-codec/Cargo.toml`.
`tools/build_avif_native.py` downloads a SHA-256-pinned dav1d 1.5.4 source archive
and uses isolated pinned Meson/Ninja tools. Windows uses a verified portable
NASM download when needed and a static CRT. Unix build hosts need their C toolchain
and NASM for x86 assembly. Intel macOS cross-builds explicitly configure the
native decoder's target architecture.

Third-party license texts are collected into the signed AVIF pack. Rust dependency
audits do not replace reviewing native codec security updates. Local builds use
a disposable signing key unless explicitly configured otherwise: keep their
core executable and pack files together. Do not upload local-key packs to an
official release or replace an existing release's signed assets.

See [the original experiments](codec-experiments.md) for measured performance,
resource use and the independent-reference transparency failures that motivated
the patch. Windows system-codec execution and Ubuntu WSL2 runtime checks are
locally tested; CI checks still need confirmation on an approved push. macOS
still needs testing on a Mac.
