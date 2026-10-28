# Optional HEIC libraries on Windows

PaintFE normally uses Windows HEIF and HEVC extensions. It can also use a
libheif DLL that you supply. PaintFE does not bundle or download HEVC codecs.

## Getting and selecting a library

1. Obtain a maintained Windows build of [libheif](https://github.com/strukturag/libheif)
   from a publisher you trust, or use a compatible build you already have.
2. Extract the entire library distribution into a folder you control. A NuGet
   `.nupkg` is a ZIP archive; Windows users can open a copy renamed to `.zip`.
   Look for the native DLL in `runtimes/win-x64/native` when that layout is used.
3. Keep dependencies and codec plugins beside the library as supplied. Do not
   copy just one DLL out of a larger distribution, or mix versions.
4. Choose the library in Preferences, then check HEIC support. No restart is
   required. A decoder-only library can open HEIC without being able to save it.

The reported `LibHeif.Native.win-x64` 1.15.1 NuGet package is a third-party build
last updated in 2023. NuGet hosting is not a Microsoft endorsement of its
contents or a codec licence. Upstream has published security fixes since that
version. PaintFE therefore does not offer automatic installation of that old
package. Refer to the upstream [security advisories](https://github.com/strukturag/libheif/security/advisories)
when choosing a maintained build. Windows system extensions remain the default.

Open Preferences > Optional Features > Optional HEIC library. Choose your
`libheif.dll` or `heif.dll`, then click **Check HEIC support**. PaintFE shows
whether the selected library can open and save HEIC images. Library loading and
image processing happen in the bounded codec worker, not the UI process.

Keep the library's matching dependencies and codec plugins together in their
original folder. The DLL must match PaintFE's architecture and export the
libheif 1.x API (version 1.13 or newer). Decoder-only distributions can open
images without supporting exports. A filename by itself does not establish
compatibility or trust. Select only a library you trust.

The selected folder and its `plugins` subdirectory are supplied as plugin search
locations to the worker. PaintFE does not search other applications' folders or
load random DLLs placed beside the editor. A failed load reports an error rather
than falling back silently. **Use system codec** clears the override and restores
the Windows backend. The choice is saved locally, including for CLI operations.

To check the setup, open a HEIC with transparency, inspect its alpha, then save a
disposable copy if export support is available. Compare dimensions and colour
with an independent decoder. Quality 100 does not guarantee lossless HEIC output.

# Processing safety

A processing dialog blocks conflicting editing while a job is running. Cancel
requests stop cooperative operations and discard cancelled results. ONNX
inference uses Runtime's terminate flag; codec subprocesses can be killed.
Native image resizing uses a disposable process launched from the same PaintFE
executable. Cancel terminates that worker, discards its result and removes its
temporary pixel files. Preparation and result loading check cancellation too.
Other CPU effects may still stop cooperatively between processing steps. Never
forcibly terminate a native in-process worker thread.

Background results identify their original document and revision. Closed,
modified or cancelled targets are not replaced by late results. Failed saves
keep the original destination and remove their temporary file where possible.

The current image storage limit is 256 million pixels, with the existing
25,000-pixel opening dimension limit. A 20,000 by 20,000 resize exceeds the pixel
budget and is rejected before allocation. WebP export has a separate limit of
16,383 pixels per dimension; use another format for larger documents.

HEIC has a separate PaintFE processing budget of **64 million pixels**, for both
opening and saving. This is a PaintFE safety limit, not a statement of the HEIC
format's maximum dimensions. Installed codecs may impose additional restrictions
within the PaintFE budget. For example, 8000 by 8000 is within that budget;
17472 by 11712 is about 204.6 million pixels and exceeds it. Save As displays
the current image dimensions and budget before encoding. Use PNG, TIFF, JPEG XL
or PFE for larger images. No automatic resize or replacement of the original
file is performed. HEIC and HEIF filename extensions use the same HEVC encoder;
the combined save entry accepts `.heic` and `.heif`.

# Universal font pack

New releases publish one signed CJK font pack for all supported platforms.
Executable codec packs remain platform-specific. Existing platform-specific
font packs remain accepted, and historical release assets are preserved.
Release staging switches to nine packs starting with version 1.4.20. The working
version is intentionally unchanged until a release is approved.
