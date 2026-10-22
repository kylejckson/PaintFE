# HEIC/HEIF and AVIF

## HEIC/HEIF

HEVC-based HEIC/HEIF still-image opening is bundled. No download or external
system codec is needed. `vendor/heifer/` contains Heifer 0.2.0 with PaintFE's
local grid-alpha fix and upstream licenses. The patch stays in this repository;
`vendor/heifer/PAINTFE-PATCHES.md` documents its scope.

The final Windows corpus run opened all 15 valid samples and rejected three
malformed inputs. Both formerly broken grid-alpha samples now match reference
alpha exactly; 10/12-bit alpha differs by at most one RGBA8 level after conversion.
Measured results are in
[`windows-heifer-patched.json`](../tools/codec_probe/results/windows-heifer-patched.json).

The import hooks preserve supported 8/10/12-bit data through PaintFE's existing
high-depth import path. Container orientation/crop/mirror are applied by the
decoder. A 64-million-pixel limit and 256-MiB compressed-input limit bound imports.
Very large files can still require substantial memory. This does not implement
HDR gain-map import or ICC colour management beyond existing application paths.

An opened HEIC document retains its path and HEIC format. Save writes back to
that file when the optional export pack is installed; otherwise it offers the
pack without opening a file browser or changing the format. Use Save As to
choose a different destination or format. This is image import, not full support
for HEIF video sequences, depth maps or every auxiliary image type.

## HEIC export optional pack

Install **HEIC export** in Preferences > Optional features, or import its signed
`.pfepack` offline. Opening HEIC remains built in. Save As > HEIC offers the pack
when it is missing. Export is a flattened, lossy 8-bit still image with alpha;
quality 100 does not mean lossless. PFE retains editable layers.

The separate helper contains statically linked libheif 1.23.6 and Kvazaar 2.3.2,
with a fast preset, four worker threads and lossless auxiliary alpha. No HEVC encoder library is linked
into PaintFE. Saving and preview encoding use the existing asynchronous I/O,
signed-pack verification, bounded stdin/stdout protocol, cancellation, timeout
and process-memory limits. HEIC previews reopen the result with bundled Heifer.
No external process is used to decode ordinary HEIC imports.

The Windows signed pack is approximately **3.54 MiB**, including corresponding
source. Tests cover tiny/odd sizes, transparency, malformed requests, cancelled
jobs, CLI output and preserving existing files when a pack is missing. Independent-decoder samples from 1x1 to 512x512 retained alpha exactly.
The helper explicitly forces lossless alpha, including complex masks, while
colour remains lossy. These
checks are not full codec conformance or colour-managed/HDR validation.

The helper is GPL-3.0-or-later, libheif is LGPL-3.0-or-later, and Kvazaar is
BSD-3-Clause. The pack includes LICENSE.txt, BUILDING.txt and SOURCES.tar.gz with
complete sources and the build script, allowing users to rebuild and modify the
separate program offline. The signed manifest checks all four payload files;
other packs retain their existing two-file whitelist. Official signatures are
not bypassed for replacement executables. Maintainers can rebuild and sign a
local PaintFE/helper pair using the existing local pack-build workflow.

All four desktop release targets are configured. Windows native runtime is
locally tested; Linux/macOS native builds and execution remain pending CI.

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
release setup signs AVIF and HEIC too. The build tool now produces four packs per
platform, and publication verifies sixteen assets across Windows x86_64, Linux
x86_64, Apple Silicon and Intel macOS.

```sh
python tools/build_optional_packs.py --release
```

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
the patch. Windows execution is locally tested. Unix runtime checks are configured
in CI and remain unverified until an approved push runs them.
