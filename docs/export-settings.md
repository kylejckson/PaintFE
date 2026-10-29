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
controls are hidden, and this limitation is stated in the dialog. Interlacing,
new metadata-writing controls and additional JPEG/HEIC tuning are outside this
pass. Existing PNG metadata behaviour is unchanged.

## Implementation and checks

`io::export::ExportOptions` is the shared snapshot held by the dialog, save
action and file handler. The atomic save flow and background preview use the
same encoder entry point on desktop. Preview requests include source generation,
format settings and animation frame, are debounced, and have at most one active
encoder per dialog. A stale or failed result cannot be labelled as a current
successful preview. The web preview retains its memory-only fallback.
Animation playback uses source-frame thumbnails; pausing generates the selected
frame's encoded preview. The dialog does not estimate complete animation sizes.

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
