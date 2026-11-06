# Brushes

The Tip picker contains 55 bundled choices: Circle, 44 image stamps, two generated
basic shapes and eight paintbrush presets. A category sidebar separates navigation from the brushes.
Narrow windows use a single category selector instead. Search checks all
categories, including translated paintbrush names. Clearing search restores the
category you were browsing. Categories retain their scroll position while open.
New, at the right of the heading, opens the custom tip dialog.

Paintbrush cards show sample strokes made by the production renderer. Image
stamps use compact footprints. The menu beside New switches compact previews and
can reset the current paintbrush. Right-click a brush to add or remove a
favourite. Favourites and Recent provide quick access, and browsing category,
favourites, recent choices and the view preference are saved across restarts.
Tab moves between controls; arrow keys navigate brush cards and Enter selects.
Ctrl+F focuses search. Escape closes the picker.

Preferences > Interface > Advanced customization > Brush browser controls preview
size, spacing and optional background, selection, hover and corner-radius
overrides. Colours, fonts and geometry otherwise inherit the theme. Theme export
includes appearance settings, without exporting personal favourites or recents.

## Paintbrushes

Round Paint, Flat Paint, Filbert and Dense Fan provide fuller paint coverage.
Bristle Wash applies a deliberately translucent coat. Dry Bristle, Dry Fan and
Rake create connected, separated strands. Oval Brush and Flat Brush retain their
solid footprints. Old Round Bristle and Fan Bristle identifiers remain compatible
with saved settings, while their displayed names describe their dry character.
These are lightweight procedural brushes, not a wet-paint or pigment simulation.

Open Brush settings using the brush-controls icon in the tool options bar:

- Follow stroke direction turns the tip along a stroke. Direction smoothing
  softens changes by distance travelled, without moving the pointer position.
- Roundness and Bristle density change the generated footprint. Spread changes
  how widely the bristles contact the image.
- Paint load changes the amount of paint deposited. Loaded presets start strong;
  the wash starts translucent. Texture and Bristle thickness control strand
  detail. Stiffness changes direction response and bending through turns.
- Paint depletion optionally reduces loading with distance travelled. It starts
  disabled, resets for each stroke and does not deposit more paint while still.
- Pressure spread and the pressure curve respond to pen force. Enable pressure
  in Preferences > Pen & Tablet; a mouse uses full pressure.
- Speed response optionally narrows contact at faster drawing speeds and bends
  bristles through turns. It estimates movement from input, not physical tilt.

PaintFE currently receives pressure, not hardware pen tilt or twist. Direction
and bending are calculated from the stroke. Mirror arms track direction
independently. The preview outline follows the brush footprint.

Selecting a paintbrush starts with suitable flow, spacing and shape defaults.
Edits are remembered separately for each paintbrush and saved across restarts.
This includes scatter, colour jitter, antialiasing and pressure limits. Colour,
size and the colour blend mode remain shared when changing presets.
Reset current paintbrush restores its defaults. Returning to an ordinary tip
restores the earlier ordinary brush settings within that session. Size and colour
stay as chosen. That temporary ordinary-brush backup is not saved across restarts;
leaving a restored paintbrush after restart uses ordinary defaults.

## Strength

Colour alpha sets the paint's transparency. Brush Opacity adds a ceiling for the
stroke; Flow controls how much each dab deposits. Low Flow can build up as a stroke
crosses itself. Brush mode also affects accumulation. Eraser strength uses colour
alpha and Flow, without the separate painting Opacity control.

Opacity and Flow normally live in Brush settings. Preferences > Editing > Brush
controls can show them in the options bar. Recommended controls appear for
bristle presets by default; that behaviour has its own switch. Hiding controls
does not reset their values. A small accent dot indicates active brush settings.

## Custom stamps

Image tips use grayscale coverage: white paints, black leaves pixels untouched,
and gray gives partial coverage. Disabling antialiasing removes resampling
smoothing, while retaining the material's gray coverage. Non-square stamps are
centred without stretching their aspect ratio.

Custom tips can be removed from their context menu. Bundled tips cannot be
deleted. A custom tip with an existing bundled name temporarily overrides it;
deleting that custom tip restores the bundled version. Custom image stamps do
not inherit a generated paintbrush's engine just because their names match.

## Tool-specific settings

Brush, Pencil and Eraser remember their own settings, including their selected
size, tip, blend mode and painting parameters. Switching tools restores those
settings, and they are saved across restarts. Brush and Eraser paintbrush presets
are independent too. The active colour still comes from the shared Colours panel.

Pencil always draws single-pixel, aliased marks. Its options bar offers blend mode,
without size, hardness, antialiasing, Opacity, Flow or brush dynamics controls.
Use colour alpha to change Pencil transparency. Enabling tablet controls or advanced
brush controls does not add these ignored settings to Pencil.

Changing tools during a paint gesture waits for release, so the original tool
finishes one stroke and one undo entry. Undo during a gesture completes and undoes
that stroke, then waits for release before accepting another paint gesture.
Selection, clipboard and save boundaries also finish pending paint first.
