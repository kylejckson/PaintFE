These tiny ONNX graphs return a fixed 2x2 mask through an Identity node.
They declare float32, float16, and uint8 outputs; the float16 graph also has
a float16 input. The mask values are 0, 0.25, 0.75 and 1 (byte equivalent
0, 64, 191, 255). They use ONNX IR 8 and opset 13, with no external data.

They exercise PaintFE's native tensor conversion and ONNX C API calls without
loading a large neural network. Set PAINTFE_ONNX_RUNTIME to a local runtime
library and run the ignored `native_runtime_handles` test in onnx_models.
