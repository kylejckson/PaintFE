# Warp and Liquify

## Mesh Warp

Choose a grid size, then use **Edit mesh** to place reference points over the
parts of the image you want to reshape. This leaves the image pixels unchanged.
Reference cells must remain valid rather than crossing over one another.

Switch to **Deform** and drag those points to reshape the image. The preview
updates in the background at a bounded resolution. Use **Apply** or Enter when ready,
or use Escape to cancel the deformation. Final output uses the original image.

## Liquify

Push moves pixels along the stroke. Expand grows an area, Contract shrinks it,
and the twirl modes rotate it locally. Reconstruct restores the undeformed image
under the brush. Size is the diameter of the circular influence shown by the
cursor, and Strength controls how strongly each sample deforms the image.

For an arm, begin with a broad, low-strength Push or Expand stroke. Select a region and choose **Protect selection** to freeze it, then deselect
and deform the surrounding image. A blue overlay shows the protected area.
**Clear protection** removes that mask. A normal selection also limits which
pixels can change. These tools
reshape existing pixels; they do not generate missing anatomy or backgrounds.

**Apply** or Enter applies the Liquify session. **Cancel** or Escape cancels it. The optional adjustment
history setting in Preferences > Editing makes individual strokes undoable
before the session is applied. Undo after applying restores the committed edit.

Adjustment history is bounded to 128 MiB. If one stroke exceeds the limit,
the toolbar asks you to apply or cancel before using Undo.

Reset in Mesh Warp removes deformation while keeping your reference anchors.
Reset in Liquify clears displacement while retaining the protection mask. With
adjustment Undo enabled, Reset can be undone before applying the session.
