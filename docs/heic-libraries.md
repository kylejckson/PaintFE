# Optional HEIC libraries on Windows

PaintFE normally uses Windows HEIF and HEVC extensions. It can also use a
libheif DLL that you supply. PaintFE does not bundle or download HEVC codecs.

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
A CPU resize may need to finish its current resampling step before releasing its
working memory. Never forcibly terminate a native in-process worker thread.

Background results identify their original document and revision. Closed,
modified or cancelled targets are not replaced by late results. Failed saves
keep the original destination and remove their temporary file where possible.

The current image storage limit is 256 million pixels, with the existing
25,000-pixel opening dimension limit. A 20,000 by 20,000 resize exceeds the pixel
budget and is rejected before allocation. WebP export has a separate limit of
16,383 pixels per dimension; use another format for larger documents.

# Universal font pack

New releases publish one signed CJK font pack for all supported platforms.
Executable codec packs remain platform-specific. Existing platform-specific
font packs remain accepted, and historical release assets are preserved.
Release staging switches to nine packs starting with version 1.4.20. The working
version is intentionally unchanged until a release is approved.
