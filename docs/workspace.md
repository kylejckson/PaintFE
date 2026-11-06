# Workspace and shortcuts

Choose tools in the Tools panel and adjust their options in the bar above the
image. Use layers to separate parts of a document, Color to choose primary and
secondary colours, and Palette to reuse swatches. Zoom inspects detail; Pan moves
the view without changing image pixels. Hold Space for temporary panning.

Open Edit > Preferences to adjust the interface and editing behaviour. Changes
are saved automatically. Preferences > Keybinds lists actions and their shortcuts.
Menu shortcut hints reflect the current bindings.

## Panels and themes

Preferences > Interface controls theme colours, text, panel layout and motion.
Resize panels to fit the workspace, change the visible tools, or save a named
workspace layout. Reset controls restore defaults or inherited theme values.

## Tool options bar appearance

Preferences > Interface contains the floating/docked layout and appearance controls.
Follow theme applies to the docked bar's separator colour. Turning it off starts
with the current theme colour; choose a different colour to see a change.
Separator opacity fades that line. Background opacity fades the bar over the
window background. Corner radius rounds the bar's outer corners, with square
corners as the default for a docked bar. Reset buttons restore the defaults or
resume following the theme for inherited values.

Theme background alpha and Background opacity multiply. A docked bar blends over
the window background, while its reserved height keeps the canvas in place when
tools or horizontal overflow change. Reset in Tool options restores that section.

## Pen and tablet input

Preferences > Pen & Tablet contains the tablet-pressure switch, a live pressure
readout and a scratch area for testing strokes. Pressure size and flow
responses remain in Brush settings. The test area never changes a document or
adds history entries; Clear test clears only those temporary test marks.

On Windows, PaintFE receives pen pressure through Windows Ink/system pointer
input. Enable Windows Ink in the tablet driver's application settings when
needed. PaintFE does not change driver settings or offer a WinTab backend.
A mouse uses full pressure; a mouse-only test cannot confirm tablet-driver
compatibility. Preferences search accepts Pen, Tablet, Windows Ink and pressure.

## Compact brush controls

Opacity and Flow are hidden from the tool options bar by default. Open Brush settings using its brush-controls icon to adjust them at any time. A small accent dot marks active non-default brush settings; the tooltip also shows current Opacity and Flow. These controls work with a mouse as well as a pen.

Preferences > Editing > Brush controls > Show advanced brush controls puts both controls back in the bar. Eraser shows Flow only, since its strength comes from colour alpha. Changing visibility preserves the current values. Tablet pressure controls in Brush settings appear when pressure input is enabled in Preferences > Pen & Tablet. Tool options retain their reserved height when controls are shown or hidden.

Show recommended controls for paintbrush presets is enabled by default. It brings strength controls into the bar for the paintbrush presets, where paint loading and gradual buildup are useful. Disable it to keep those controls inside Brush settings too. See [Brushes](brushes.md) for the browser, tips and paintbrush controls.
