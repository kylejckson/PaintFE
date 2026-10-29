# Codec experiment harness

This separate Cargo package measures candidate codecs independently of PaintFE.
The obsolete Heifer decoder and encoder experiment has been removed. Historical
measurements remain in `results/windows-x86_64.json` and
`results/windows-heifer-patched.json`; they do not describe current production.
Build output, reference libraries and downloaded samples belong in
the repository's ignored `target/codec-*` directories.

Features:

- `avif-encode`: image 0.25.9 / ravif 0.12.0 / rav1e 0.8.1 encoding.
- `avif-fast`: the same encoder with assembly optimizations and threading.
- `avif-decode`: image 0.25.9 / dav1d decoding; requires native libdav1d.
- No features: baseline runtime for standalone size comparisons.

Build each variant sequentially, copying the executable before the next build:

```powershell
cargo build --release --locked --manifest-path tools/codec_probe/Cargo.toml --target-dir target/codec-probe --features avif-encode -j1
Copy-Item target/codec-probe/release/paintfe-codec-probe.exe target/codec-fixtures/avif-encode.exe
```

The baseline is built without `--features`; the encoder variant uses
`--features avif-encode`; the optimized encoder uses `--features avif-fast`;
the combined AVIF variant uses `--features avif-fast,avif-decode`.
Their matching names in the scratch directory
are `baseline.exe`, `avif-encode.exe`, `avif-fast-encode.exe`, and `avif.exe`.

For Windows, build dav1d 1.5.0 with Meson/Ninja and NASM. Match PaintFE's static
MSVC runtime with `-Db_vscrt=mt`; also pass `-Ddefault_library=static`,
`-Denable_tools=false`, `-Denable_tests=false`, and `--buildtype=release`.
Retain assembly optimizations. Copy the resulting `src/libdav1d.a` to
`src/dav1d.lib`, then set these variables **in the current shell only** before
the AVIF decoder build:

```powershell
$env:SYSTEM_DEPS_DAV1D_NO_PKG_CONFIG = '1'
$env:SYSTEM_DEPS_DAV1D_LIB = 'dav1d'
$env:SYSTEM_DEPS_DAV1D_SEARCH_NATIVE = (Resolve-Path target/codec-fixtures/dav1d-build/src).Path
$env:SYSTEM_DEPS_DAV1D_LINK = 'static'
```

Linux/macOS can provide libdav1d through pkg-config instead. These are build-time
requirements, not a proposed installation procedure for PaintFE users.

Commands supported by the probe:

```text
heif-decode INPUT OUTPUT.rgba [RUNS=5] [THREADS=0] [MAX_PIXELS=32000000]
heif-encoder-primitive
avif-encode INPUT.rgba WIDTH HEIGHT OUTPUT.avif [SPEED=8] [QUALITY=80] [RUNS=3]
avif-decode INPUT OUTPUT.rgba [RUNS=5]
```

Install Pillow and pillow-heif into an isolated Python environment for reference
comparisons. Place HEIC samples under `target/codec-fixtures/samples/`, including
`image_large.heic`, then run:

```powershell
target/codec-venv/Scripts/python.exe tools/codec_probe/run_samples.py target/codec-fixtures
```

The runner records JSON results beside the samples. Timing excludes file I/O,
includes conversion to RGBA8, and uses repeated in-memory inputs. Optimized AVIF
encoding uses four threads; the minimal variant has threading disabled.
The sample runner permits up to 64 million HEIC pixels so the 48 MP photo is
included; this is an experiment setting, not a proposed production limit.
Reference comparisons sample approximately 100,000 pixels; they do not
establish full conformance or colour-managed equivalence. No HEIC export test can
be performed until Heifer exposes an image encoder and HEIF writer.

See [the results and limitations](../../docs/codec-experiments.md). Do not ship
upstream fixture photos without reviewing their redistribution permissions.
