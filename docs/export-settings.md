# Save As and export settings

Save As uses a preview and a separately scrolling settings panel. In narrow
windows they stack vertically; Save and Cancel stay below the scrolling area.
The preview can be panned and zoomed. Its size estimate comes from a thumbnail,
so it is not a promise of the final file size. Preview failures are shown below
the image, with the encoder error on hover. A previous preview can remain visible
while a new one is being calculated.

Settings are remembered by format after a successful save. Changing formats
restores that format's settings. Reset this format restores its defaults. Normal
Save uses the current document's saved format and export settings. PFE remains
the editable document format; exporting to another format retains the shared
warnings for losing layers, guides, animation, transparency or precision.

## PNG

Automatic preserves colour and transparency and keeps 16-bit integer sources at
16 bits per channel. Opaque images can use RGB without an alpha channel. It does
not automatically quantize an image or convert it to grayscale. Floating-point
sources follow the existing display conversion and produce 8-bit PNG unless
16-bit is selected explicitly; this does not preserve their original float range.

Explicit choices include RGB/RGBA, grayscale with or without alpha, 8/16-bit
channels, and an indexed palette with 2 to 256 colours. Indexed palettes can
preserve partial transparency through PNG's tRNS chunk. Palette reduction and
grayscale conversion are intentional lossy choices, with save warnings.

Fast, Balanced and Smaller file select real encoder compression presets. They
change size and encoding speed, not pixel quality. Dithering changes palette
mapping for indexed PNG and both static and animated GIF. GIF retains its
binary transparency limit, unlike indexed PNG.

Animated PNG currently uses RGBA8 frames. Its still-image depth and colour
controls are hidden, and this limitation is stated in the dialog. Still PNG can
use real seven-pass Adam7 interlacing at 8 or 16 bits, including indexed output.
Interlaced scanlines are streamed rather than allocating another full image.
Existing PNG metadata behaviour is unchanged.

## Additional format controls

- JPEG offers 4:4:4, 4:4:0, 4:2:2 and 4:2:0 sampling. Full-resolution colour is
  the default. Quality and sampling apply to both preview and final output.
- Desktop WebP offers Default, Photo, Picture, Drawing, Icon and Text presets for
  still lossy output. Lossless output preserves RGBA bytes, including hidden RGB
  beneath transparent pixels. Animated WebP keeps its existing encoding path.
- Indexed PNG/BMP and GIF offer median-cut or NeuQuant palette creation and
  adjustable error-diffusion strength. Strength zero disables error diffusion.
  GIF additionally offers a transparency threshold: alpha below the threshold
  becomes transparent. Its binary transparency limitation remains explicit.
- BMP supports RGB, RGBA, grayscale and indexed output. TGA supports RGB, RGBA
  and grayscale, with optional RLE. Their channels remain eight bits.
- TIFF supports explicit RGB/RGBA/grayscale and 8-bit, 16-bit or 32-bit float
  channels. Automatic retains the original source precision. Explicit conversion
  retains the normal precision, transparency and grayscale warnings.
- AVIF exposes speed 1 (slow) to 10 (fast), default 8. The updated optional
  helper adds a statically linked libaom backend for exact lossless 8-bit RGBA
  and lossless alpha. Both use 4:4:4 colour. Exact requests are decoded and
  compared before accepting the result. The core contains neither AV1 encoder
  nor decoder, and the helper needs no separate codec DLL. Older installed packs
  keep basic lossy export; lossless options need an updated pack from Preferences.
  Quality 100 on the original lossy backend must not be called lossless.
- HEIC shows only sampling modes reported by the selected system/user codec.
  Exact lossless output requires explicit codec support and is decoded and
  compared before accepting the result. WIC/ImageIO do not advertise these
  advanced controls. Unsupported saved selections can be cleared or reset.

Colour-profile conversion, arbitrary metadata writing, preserving source AVIF
tiling and configurable AVIF chroma/premultiplied-alpha encoding are not supplied
by these controls. The dialog must not imply they are implemented.

## Implementation and checks

`io::export::ExportOptions` is the shared snapshot held by the dialog, save
action and file handler. The atomic save flow and background preview use the
same encoder entry point on desktop. Preview requests include source generation,
format settings and animation frame, are debounced, and have at most one active
encoder per dialog. A stale or failed result cannot be labelled as a current
successful preview. The web preview retains its memory-only fallback.
Animation playback uses source-frame thumbnails; pausing generates the selected
frame's encoded preview. The dialog does not estimate complete animation sizes.

`tests/export_advanced.rs` independently decodes output and checks JPEG sampling,
Adam7 passes, BMP/TGA headers, TIFF precision, palette controls and WebP presets.
`tests/export_settings.rs` checks actual headers, decoded pixels, palette alpha,
dithering and preview/final consistency. Dialog tests cover stale jobs, compact
layout, larger text and control drag ownership. Save-safety tests cover warnings
and clean untitled documents in Save All.

## Windows recent files

Document pickers retain Windows recent-file registration. Model, runtime,
plugin, palette and other configuration-resource pickers opt out through the
shared file-dialog service. The focused rfd patch preserves the other native
picker flags. CI rejects direct picker construction outside that service.

Preferences > AI includes a manual cleanup button for the currently configured
model, ONNX runtime and HEIC library. It removes those exact resource entries
from PaintFE's current Windows Jump List. It does not clear image history or
remove pins. Older resource paths can be removed manually from the taskbar;
PaintFE does not erase Windows-wide recent history.
