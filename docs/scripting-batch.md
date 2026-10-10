# Crop and batch processing

`crop_canvas(x, y, width, height)` crops all layers to a rectangle in pixels.
Coordinates start at the top left. Width and height must be positive and the
entire rectangle must be inside the canvas; invalid rectangles fail the script.
Layer masks, the selection, and editable text positions follow the crop. The
editor records the whole script as one undo step.

Save this as `crop.rhai`:

```rhai
crop_canvas(100, 50, 400, 300);
print_line(`${width()} x ${height()}`);
```

For one image with a new name:

```powershell
PaintFE.exe --input original.png --script crop.rhai --output cropped.png
```

For a batch (quote globs so PaintFE expands them):

```powershell
PaintFE.exe --input "photos/*.jpg" --script crop.rhai --output-dir processed --format png
```

Linux/macOS use the same options. Input discovery and saving happen in the host;
Rhai scripts have no filesystem or network access. Existing output files are
protected unless `--overwrite` is explicitly supplied. Inputs can never be
overwritten and duplicate output names abort the batch before any file is written.
Use separate output directories if two input directories contain the same stem.
Raster outputs use the selected layer or flattened composite as configured;
use `--format pfe` to preserve editable layers.
