# Local background removal

Open Help > AI setup, or Edit > Preferences > AI. Configure an ONNX Runtime
library for your operating system and an appropriate BiRefNet, U2-Net or IS-Net
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
