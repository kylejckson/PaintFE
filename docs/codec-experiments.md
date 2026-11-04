# Codec Investigation

Production HEIC processing uses codecs installed on the system or provided by the user. The earlier decoder experiment, its metadata dependency, benchmark results, and the separate HEVC helper/build setup have been retired. Historical measurements are still in Git history, but they are not current codec recommendations.

The separate AVIF probe still exists under `tools/codec_probe/`. Current app behavior and limits are documented in [HEIC and AVIF](heif-avif.md) and [export settings](export-settings.md). Codec choice must be checked against decoded pixels, alpha handling, precision, malformed files, and memory limits. File size or a thumbnail alone does not prove equivalent output.

## Reproducible libaom source acquisition

The optional AVIF build pins libaom 3.15.1 to commit
`44d0a57786f432d933ff64b653347c66f4d0fa1d`. Gitiles can produce gzip/tar
archives with different metadata for identical source contents. The build
therefore verifies a pinned canonical digest of paths, file modes and SHA-256
file contents, while rejecting unsafe archive entries and checking cached source
contents before reuse. It never accepts an arbitrary downloaded checksum.
