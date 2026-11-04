# Save As and Export Settings

Save As uses a preview plus a separate settings panel. In narrow windows, they stack vertically; Save and Cancel stay below the scrolling area. The preview can be panned and zoomed. Its size estimate comes from a thumbnail, so it is not a promise of the final file size. Preview failures are shown below the image, and the encoder error appears on hover. A previous preview can remain visible while a new one is still being calculated.

Settings are remembered by format after a successful save. Changing formats restores that format's saved settings. "Reset this format" restores the defaults. Normal Save uses the current document's saved format and export settings. PFE remains the editable document format. When exporting to another format, the usual warnings still apply for lost layers, guides, animation, transparency, or precision.

## PNG

Automatic mode keeps color and transparency and preserves 16-bit integer sources at 16 bits per channel. Opaque images can use RGB without an alpha channel. It does not automatically quantize the image or convert it to grayscale. Floating-point sources follow the existing display conversion and become 8-bit PNG unless 16-bit is selected explicitly; this does not preserve the original float range.

Explicit choices include RGB/RGBA, grayscale with or without alpha, 8/16-bit channels, and an indexed palette with 2 to 256 colors. Indexed palettes can preserve partial transparency through PNG's tRNS chunk. Palette reduction and grayscale conversion are intentional lossy choices, and the save dialog warns about that.

Fast, Balanced, and Smaller select real encoder compression presets. They change file size and encoding speed, not pixel quality. Dithering changes palette mapping for indexed PNG and for both static and animated GIF. GIF keeps its binary transparency limit, unlike indexed PNG.

Animated PNG currently uses RGBA8 frames. Its still-image depth and color controls are hidden, and the dialog makes that limitation clear. Still PNG can use a real seven-pass Adam7 interlacing at 8 or 16 bits, including indexed output. Interlaced scanlines are streamed instead of allocating another full image. Existing PNG metadata behavior is unchanged.

## Additional Format Controls

- **JPEG** offers 4:4:4, 4:4:0, 4:2:2, and 4:2:0 sampling. Full-resolution color is the default. Quality and sampling apply to both the preview and the final output.
- **Desktop WebP** offers Default, Photo, Picture, Drawing, Icon, and Text presets for still lossy output. Lossless output preserves RGBA bytes, including hidden RGB beneath transparent pixels. Animated WebP keeps its existing encoding path.
- **Indexed PNG/BMP and GIF** offer median-cut or NeuQuant palette creation and adjustable error-diffusion strength. A strength of zero disables error diffusion. GIF also offers a transparency threshold: alpha below the threshold becomes transparent. Its binary transparency limitation remains explicit.
- **BMP** supports RGB, RGBA, grayscale, and indexed output. **TGA** supports RGB, RGBA, and grayscale, with optional RLE. Their channels remain 8-bit.
- **TIFF** supports explicit RGB/RGBA/grayscale and 8-bit, 16-bit, or 32-bit float channels. Automatic mode keeps the original source precision. Explicit conversion still follows the normal precision, transparency, and grayscale warnings.
- **AVIF** exposes speed 1 (slow) to 10 (fast), default 8. The updated optional helper adds a statically linked libaom backend for exact lossless 8-bit RGBA and lossless alpha. Both use 4:4:4 color. Exact requests are decoded and compared before the result is accepted. The core contains neither the AV1 encoder nor decoder, and the helper needs no separate codec DLL. Older installed packs keep basic lossy export; lossless options require an updated pack from Preferences. Quality 100 on the original lossy backend must not be called lossless.
- **HEIC** shows only the sampling modes reported by the selected system/user codec. Exact lossless output requires explicit codec support and is decoded and compared before the result is accepted. WIC/ImageIO do not advertise these advanced controls. Unsupported saved selections can be cleared or reset.

Color-profile conversion, arbitrary metadata writing, preserving source AVIF tiling, and configurable AVIF chroma and premultiplied-alpha encoding are not provided by these controls. The dialog must not imply they are implemented.

## Implementation and Checks

`io::export::ExportOptions` is the shared snapshot used by the dialog, save action, and file handler. The atomic save flow and background preview use the same encoder entry point on desktop. Preview requests include source generation, format settings, and animation frame. They are debounced and only one encoder is active per dialog. A stale or failed result cannot be labeled as a current successful preview. The web preview keeps its memory-only fallback. Animation playback uses source-frame thumbnails; pausing generates the selected frame's encoded preview. The dialog does not estimate full animation sizes.

`tests/export_advanced.rs` independently decodes output and checks JPEG sampling, Adam7 passes, BMP/TGA headers, TIFF precision, palette controls, and WebP presets. `tests/export_settings.rs` checks actual headers, decoded pixels, palette alpha, dithering, and preview/final consistency. Dialog tests cover stale jobs, compact layout, larger text, and control drag ownership. Save-safety tests cover warnings and clean untitled documents in Save All.

## Windows Recent Files

Document pickers keep Windows recent-file registration. Model, runtime, plugin, palette, and other configuration-resource pickers opt out through the shared file-dialog service. The focused rfd patch preserves the other native picker flags. CI rejects direct picker construction outside that service.

Preferences > AI includes a manual cleanup button for the currently configured model, ONNX runtime, and HEIC library. It removes those exact resource entries from PaintFE's current Windows Jump List. It does not clear image history or remove pins. Older resource paths can be removed manually from the taskbar; PaintFE does not erase Windows-wide recent history.
