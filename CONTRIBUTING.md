# Contributing to PaintFE

Thanks for taking the time. PaintFE is a solo-maintained project and outside contributions mean a lot, whether that is a bug fix, a new locale file, a documentation improvement, or a filter idea.

This document covers how to get a development build running, the conventions used in the codebase, and what a pull request should look like before it is submitted.

---

## Getting started

You will need a recent stable Rust toolchain. The project uses Rust edition 2024, so you need at least Rust 1.85.

```sh
git clone https://github.com/kylejckson/paintfe
cd paintfe
cargo build
cargo run
```

A debug build is fine for development work. For testing anything performance-sensitive (compositing, GPU pipelines, large images), build with `--release`.

**Linux:** You will need a few system packages before the build will succeed:

```sh
sudo apt-get install -y \
  libgtk-3-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
  libxkbcommon-dev \
  libvulkan-dev \
  libwayland-dev \
  libegl1-mesa-dev \
  libssl-dev \
  pkg-config
```

**Windows:** No additional system dependencies. A working DirectX 12 driver is all that is required at runtime.

---

## Code style

The codebase uses standard `rustfmt` formatting. Run `cargo fmt` before committing. There is no custom `.rustfmt.toml`, so the defaults apply.

Clippy is also expected to pass cleanly:

```sh
cargo clippy -- -D warnings
```

If a Clippy lint is a false positive or genuinely not applicable to a specific case, you can suppress it with `#[allow(...)]` on that item -- but add a comment explaining why.

A few conventions used in the codebase:

- Operations that mutate canvas state live in `src/ops/`. Filters go in `filters.rs`, geometric transforms in `transform.rs`, color adjustments in `adjustments.rs`, generation effects in `effects.rs`.
- UI code lives in `src/app.rs` (main loop, menus) and `src/components/` (panels, tools, dialogs).
- GPU pipelines live in `src/gpu/`. If you are adding a GPU compute path, follow the pattern in `compute.rs` and expose it via `GpuRenderer` in `renderer.rs`.
- New tools are added as variants in the `Tool` enum in `src/components/tools.rs`. There is no per-tool trait file; everything goes through `ToolsPanel`.
- Undo: use `PixelPatch` / `BrushCommand` for strokes, `SingleLayerSnapshotCommand` for single-layer effect commits, and `SnapshotCommand` (full canvas) only for canvas-wide operations like resize or flatten.

---

## Commit messages

Keep commit messages short and factual. No need for a strict format, but aim for something that explains what changed and why if it is not obvious.

Good:
```
Add Crystallize filter with configurable cell size
Fix gradient fast path skipping eraser opacity on single-layer canvas
```

Not helpful:
```
fix stuff
update
wip
```

If a commit fixes a reported issue, reference it: `Fix crash when opening zero-byte PNG (#42)`.

---

## Pull requests

Before opening a PR:

1. Run `cargo fmt` so the diff is clean.
2. Run `cargo clippy -- -D warnings` and resolve anything new that your change introduced.
3. Build with `cargo build --release` at least once to confirm there are no release-mode issues.
4. Test the feature or fix manually in the running application.

For anything non-trivial, open an issue first to discuss the approach. This is not a hard rule, but it avoids situations where significant work gets done in a direction that does not fit the project.

PR descriptions should explain what the change does, why it is being made, and any trade-offs or limitations you are aware of.

---

## Adding a translation

Translations are plain text files in `locales/`. Each file is named with a BCP-47 language code (`de.txt`, `ja.txt`, etc.). The format is one `key = value` pair per line. Comments start with `#`.

To add a new language:

1. Copy `locales/en.txt` to a new file named with the correct language code.
2. Translate the values. Do not change the key names.
3. Open a PR with the new file.

If you are updating an existing translation, check whether any keys in `en.txt` are missing from the file you are editing -- new features sometimes add new strings.

---

## Reporting bugs

Open a GitHub issue. Include:

- Your OS and GPU
- PaintFE version (shown in Edit > Preferences > About)
- Steps to reproduce
- The log file from `%APPDATA%\PaintFE\paintfe.log` (Windows) or `~/.local/share/PaintFE/paintfe.log` (Linux) if there is one

If the bug involves a specific file that triggers it, attaching the file to the issue is very helpful.

---

## Security issues

Do not open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the private reporting process.

---

## License

By submitting a pull request, you agree that your contribution will be licensed under the MIT license that covers this project.
