#!/bin/bash
set -euo pipefail
cd "$SRC/PaintFE"
export CARGO_BUILD_JOBS=1
cargo fetch --locked --manifest-path fuzz/Cargo.toml
lock_before="$(sha256sum fuzz/Cargo.lock)"
for target in heif_metadata numeric_expression; do
  cargo fuzz build "$target" -O --debug-assertions --features libfuzzer
  cp "fuzz/target/x86_64-unknown-linux-gnu/release/$target" "$OUT/"
  (cd "fuzz/corpus/$target" && zip -q "$OUT/${target}_seed_corpus.zip" ./*)
done
test "$lock_before" = "$(sha256sum fuzz/Cargo.lock)"
printf '[libfuzzer]\nmax_len = 65536\ntimeout = 5\nrss_limit_mb = 256\n' > "$OUT/heif_metadata.options"
printf '[libfuzzer]\nmax_len = 4096\ntimeout = 5\nrss_limit_mb = 256\n' > "$OUT/numeric_expression.options"
