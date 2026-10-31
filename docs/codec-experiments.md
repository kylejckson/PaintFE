# Codec investigation

Production HEIC processing uses installed system or user-provided codecs. The
former decoder experiment, its metadata dependency, benchmark output and
standalone HEVC helper/build machinery have been retired. Historical measurements
remain in Git history and are not current codec recommendations.

The independent AVIF probe remains under `tools/codec_probe/`. Current application
behaviour and limitations are documented in [HEIC and AVIF](heif-avif.md) and
[export settings](export-settings.md). Codec selection must be validated using
decoded pixels, alpha, precision, malformed files and memory limits; file size or
a thumbnail alone does not prove equivalent output.
