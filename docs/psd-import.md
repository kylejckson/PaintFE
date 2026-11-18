# Photoshop documents

The desktop development build opens ordinary 8-bit RGB `.psd` files through
File > Open or drag and drop. PaintFE brings in raster layers with their names,
stack order, visible canvas offsets, opacity, visibility and supported blend
modes. It also preserves simple raster masks with a white default outside the
mask rectangle. Pixels outside the document canvas are clipped.

Groups, editable Photoshop text, unsupported blend modes, effects, adjustment
layers, clipping masks and complex masks may require the embedded composite.
PaintFE tells you when it opens that composite as a single layer. If the
composite is missing, save a compatible copy in Photoshop or export a flat
image there first. PSD import interprets RGB samples in the normal PaintFE
working space. Full Photoshop colour-managed rendering, CMYK, layered
high-bit-depth documents and PSB are outside this first pass.

Choose **Photoshop document** in File > Save As to write an 8-bit RGB PSD.
Export preserves raster layers, names, visibility, opacity, supported blend
modes and simple masks, and writes a merged preview. Opened PSDs use Save As
for their first save so you can choose a new path. A PSD saved by PaintFE can
then be updated with Save.

PaintFE warns before exporting data PSD cannot preserve. Text becomes raster
and layer folders are omitted. An adjustment layer or unsupported blend mode
requires a flattened copy to retain the rendered appearance. Use PFE to keep
the original editable structure, guides and higher bit depths.

The desktop CLI can read and write PSD. It rejects PSD exports that would
lose editable data. Import reads uncompressed, PackBits and ZIP channels.
Export writes uncompressed channels and uses a 256 MiB file limit, with a
separate cap on decoded layer memory.
