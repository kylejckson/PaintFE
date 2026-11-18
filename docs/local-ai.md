# Local background removal

Open Help > AI setup, or Edit > Preferences > AI. Configure an ONNX Runtime
library for your operating system and an appropriate BiRefNet, U2-Net, IS-Net or RMBG 2.0
model. Preferences includes links and support checks. Read the model's licence
and use a runtime compatible with it.

Model and runtime downloads use your browser. PaintFE does not automatically
download models or send images to an external service. After configuration,
background removal runs offline.

Open an image and choose Filter > Remove Background. If the runtime is not
available, that menu entry opens setup rather than leaving an unexplained
disabled command. A configured model is also required for inference.

Large models may need substantial memory even for a small image because their
input resolution is fixed. Begin with a small test image and a suitable model.
PaintFE's local image processing is separate from the AI tools used during its
development; there is no built-in general-purpose LLM conversation feature.

## RMBG 2.0 and Refine edges

In a development build with RMBG 2.0 support, select your local ONNX file in
Preferences > AI. The profile expects a 1024 × 1024 `pixel_values` input and an
`alphas` output in probability space. **Automatic** recognises that tensor
signature even if the filename is generic. The model selector in Remove
Background also offers **RMBG 2.0**.

Check the [original model card](https://huggingface.co/briaai/RMBG-2.0) for
licensing. The [community ONNX export](https://huggingface.co/baby2008/RMBG-2.0-ONNX/tree/main/onnx)
stores half-precision weights in a separate 513 MB file and runs them as FP32.
The model is user-supplied. File size by itself does not predict inference time
or peak memory.

**Refine edges** is off by default. It starts with the whole-image mask, then
uses up to six overlapping crops around uncertain edges. The crop masks fade
into the original estimate and leave confident foreground and background
alone. It is intended for larger images up to 16 million pixels. Small images
and masks without uncertain edges stay on the normal path. Refined masks can
look different on some subjects. Compare with the regular result if the crops
lose context, and allow more time for the extra inference passes.
