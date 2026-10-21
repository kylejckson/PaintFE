# Optional features

Preferences > Optional features manages the CJK font and JPEG XL export packs.
The core remains one executable. JPEG XL import stays built in. Images and
encoding stay local; only an explicit Download and install action uses a network
connection. There are no automatic pack checks or downloads.

## Installation

- Download and install shows the compressed size and an online-download warning.
- Import downloaded pack accepts the same signed `.pfepack` archive offline.
- Remove deletes only that feature's installed directory. Full font changes apply
  immediately. If removing CJK coverage needed by the current language, the UI
  falls back to English while preserving the saved language preference.
- The language picker retains a tiny renamed font subset. Selecting a CJK
  language checks suitable system fonts first, then offers the full pack if needed.
- Failed downloads leave the current installation intact. Missing JPEG XL export
  offers installation, and export failure does not truncate an existing file.

Packs live in `paintfe_data/packs/protocol-1/` for portable installs, or the
normal PaintFE profile data directory. Storage migration includes packs.
Compatible packs survive app updates. A new protocol or signing-key rotation
may require reinstalling a pack. Legacy per-version packs are not trusted implicitly.

## Trust and download implementation

An Ed25519 public key is embedded at build time. Each archive contains a signed
manifest plus exactly the expected payload and licence file. Schema, protocol,
feature, platform, lengths, SHA-256 digests, archive size and CJK glyph coverage
are checked before an atomic directory replacement. Private signing keys never
enter project files, compiler environments or build logs. Installed payloads are
verified again before loading the font or starting the helper.

Download requests use the OS curl (`System32/curl.exe` or `/usr/bin/curl`), with
personal curl config disabled, HTTPS-only redirects, bounded size and timeouts.
Linux systems without curl can install it through their package manager or use
offline import. Only the fixed PaintFE GitHub Releases asset URL is accepted;
there is no editable download URL, arbitrary executable picker, telemetry or
upload. This keeps a second TLS stack out of the core executable.

The encoder uses a versioned raw-RGBA pipe protocol. It supports 8/16-bit,
straight alpha, quality, lossless and effort options. It receives pixel bytes,
never document paths. Jobs are serialized and time-limited. Child failure is
reported as an export error. Process isolation is not an operating-system sandbox.

## Local review builds

Run `python tools/build_optional_packs.py --release` with Python 3.11 or later and cryptography
installed. It builds the helper, signs both packs with an ephemeral in-memory
review key, then builds the core with that public key and exact download sizes.
Outputs and public build metadata are in
`target/optional-packs/<os>-<arch>/release/`.

Keep that core and those packs together. Rebuilding without its metadata omits
the trust anchor. Development tests should load the public variables from
`build-env.json`, set `PAINTFE_TEST_STORAGE_DIR` to an isolated target directory,
and set `PAINTFE_TEST_JXL_PACK` to the matching pack path. The JPEG XL tests print
an explicit unavailable-encoder notice when no pack is supplied; golden import
and destination-protection tests always run.

CI explicitly installs Python 3.12 on every pack-building runner. Use
`--helper-only --github-env` in codec-test jobs to export the matching public
key, download sizes, signed encoder path and isolated storage directory through
`GITHUB_ENV`. This works on Windows, Linux and macOS, including Intel macOS
cross-compilation. It never exports the private signing seed.

## Official release setup

Before publishing, a maintainer must create one Ed25519 private signing seed
(32 cryptographically random bytes, encoded as 64 hex characters), store it in
GitHub Actions as `OPTIONAL_PACK_SIGNING_KEY`, and retain a secure backup outside
the repository. Do not paste the seed into issues, chat, files or logs. Release
jobs consume this secret only in the packager and compile its public key into
all cores. Keep the same key across releases so installed packs remain usable.
CI/PR builds use ephemeral keys and cannot access the release secret.

The release workflow builds packs for Windows x64, Linux x64, macOS ARM64 and
macOS x64. AppImage, DEB/RPM and MSIX reuse the corresponding core so their trust
anchors match. Both packs are published as GitHub Release assets alongside core
downloads, including licences and release checksums. No assets are published by
local builds. Signed import and networking can be reviewed locally, but an actual
new-release GitHub download needs those assets to be published first.


## Local validation (21 October 2026)

Initial Windows optional-pack review build, before the release-build repairs:
45,397,504 bytes (43.29 MiB), previously
64,893,440 bytes (61.89 MiB). Reduction: 19,495,936 bytes, or 30.04%.
CJK download: 13,588,486 bytes; encoder download: 1,594,733 bytes.
Installing both packs adds their data/helper files to disk; the reduction above
measures the core executable, not a fully populated installation.

After the release-build repairs, the full Windows app suite passed in sequential
batches: 657 tests passed and
none failed. Ten items remain ignored as configured: three manual profiling
tests, three native ONNX tests requiring external fixtures and four documentation
examples. Four additional optimized-build CLI smoke checks passed earlier. These cover
signed/tampered/incompatible packs, interrupted fetch cleanup, 8/16-bit RGBA,
lossy/lossless export and alpha, missing-encoder destination protection, native
file I/O, CLI, Preferences navigation and storage migration/relocation. New
startup tests cover missing CJK with automatic language selection and suitable
system fonts avoiding prompts. Shared controls boundary/self-tests, locale
validation, formatting and workflow structure checks also passed. The Windows
incremental build cache emitted a hard-link/copy warning; no Rust source warning
was reported by these builds. All-targets Clippy passed with warnings denied
after the lint findings were corrected. The encoder-helper Clippy and test
harness also passed. Four packaging-tool regression tests verified real signed
archives and matching CI metadata for all four release platforms, with native
compilation mocked. Workflow structure and actionlint checks passed. The
dependency audit reports no vulnerabilities under the existing audit policy;
eight unmaintained-dependency notices remain visible, with no new exclusions.
Native Linux/macOS execution still needs confirmation in Actions after an
approved push.

Live UI checks covered light/dark modes, font import, Japanese/Korean text,
restart verification, the download warning/error state and a real JPEG XL Save
As export. The export was decoded and checked pixel-for-pixel as an opaque
800x600 white test canvas. The optimized review executable was launched too.
The real unpublished asset request returned HTTP 404 cleanly. Successful GitHub
installation needs release publication; Linux/macOS native checks need CI or
those systems. New optional-feature labels currently have explicit English
fallbacks in non-English catalogues, pending translation review.

### Manual review

The optimized isolated build is
`target/optional-pack-validation/review/PaintFEPacksReview.exe`. Both matching
packs are already installed there. Your normal settings were not used or changed.

1. Open Preferences > Optional features. Check sizes, prompts and both statuses.
2. To test import again, remove a pack and import its `.pfepack` from
   `target/optional-packs/windows-x86_64/release/`.
3. Switch General > Language to Japanese or Korean and inspect text. Suitable
   system fonts can legitimately avoid a font-download prompt.
4. Save an image as JPEG XL, reopen it, then remove the encoder. Opening JXL
   should still work; exporting should offer installation without damaging an
   existing destination.
5. Download will report unavailable until matching pack assets are published.
   Offline Import is the working local installation route for this review.
