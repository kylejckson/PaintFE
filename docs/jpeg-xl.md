# JPEG XL

PaintFE opens JPEG XL (`.jxl`) still images. Desktop builds also save them, using
an optional libjxl export helper. Install it once in Preferences > Optional
features, or import a downloaded pack. Installed encoding works offline and
needs no extra DLL or account.

In Save As, choose **JPEG XL**. **Lossless** preserves pixels, including RGB under
fully transparent pixels. Turning it off enables quality from 1 to 100. Quality
100 is a lossy setting; use the Lossless checkbox for exact pixels. Alpha remains
lossless in either mode. **Encoding effort** ranges from 1 to 9, default 5. Higher
effort uses more time to improve compression, rather than increasing quality.
The preview runs in the background; its size is an estimate from a thumbnail.

The supported export path covers 8-bit RGBA and the app's supported 16-bit RGBA
export preparation. Existing limitations of deep-layer compositing still apply.
Floating-point/HDR export, animation, JPEG bitstream reconstruction and complete
metadata round-tripping are not included. Exports are tagged as sRGB. Imported
metadata is subject to the same application limitations as other raster imports;
do not use this workflow to archive ICC/EXIF/XMP or HDR metadata losslessly.

```
PaintFE -i picture.png -o picture.jxl
PaintFE -i picture.png -o picture.jxl --jxl-lossy --quality 90 --jxl-effort 5
PaintFE -i picture.jxl -o picture.png
```

The Rust decoder registers both JPEG XL signatures and the extension with the
shared image loader, covering file, layer and byte-based imports. Browser builds
use this decoder; native encoding is excluded from their format picker.

## Building

Building the optional encoder requires CMake, a C++ compiler and libclang for
generated bindings. These are no longer required by the core JPEG XL import path.
Windows uses Visual Studio C++ tools and LLVM; if needed, set `LIBCLANG_PATH` to
the LLVM `bin` directory and put CMake on PATH. On Linux install `cmake` and
`libclang-dev` alongside the normal application build dependencies. macOS uses
Xcode's compiler and libclang with CMake. No system libjxl is required.

`vendor/jxl-sys` contains the pinned permissive binding and libjxl 0.12 sources.
Its documented build patch honours Cargo's job count and Windows `crt-static`.
Use `python tools/build_optional_packs.py --release` to build the helper and
core with limited compilation concurrency. Encoder calls use one
thread; saves use the app's existing background jobs. Third-party notices are
embedded under **Preferences → General → About → JPEG XL · libjxl / jxl-rs**.

The regression suite uses the independent Rust decoder to check native encoder
output, exact 8/16-bit samples, transparent pixels, malformed data, allocation
limits, file handlers, previews and CLI conversion.

## Optional desktop export

JPEG XL decoding remains built into PaintFE. Encoding now uses a signed optional
helper pack, installed in Preferences > Optional features. It works offline after
installation and does not upload images. Packs can also be imported from a file.
The CLI uses the installed pack too and reports how to install it when absent.
See [Optional features](optional-features.md) for installation and release setup.
