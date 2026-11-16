# Parser fuzzing

These targets include the production HEIF metadata reader and numeric expression
parser directly. They do not link the GUI, GPU or native image codecs, so a fuzz
build is small. The metadata adapter uses the production pixel-limit file and
the same dimension predicate, replacing only the translated error text.

Run the small deterministic mutation check on any supported Rust host:

```sh
cargo run --locked --manifest-path fuzz/Cargo.toml --bin fuzz-smoke -- 5000
cargo test --locked --manifest-path fuzz/Cargo.toml -- --test-threads=1
```

For coverage-guided fuzzing, use a Linux host with nightly Rust and cargo-fuzz:

```sh
cargo +nightly fuzz run heif_metadata --features libfuzzer -- -max_total_time=60 -max_len=65536 -rss_limit_mb=256
cargo +nightly fuzz run numeric_expression --features libfuzzer -- -max_total_time=60 -max_len=4096 -rss_limit_mb=256
```

ClusterFuzzLite runs the same targets with AddressSanitizer in a digest-pinned
container. Each CI run has a 120-second fuzz budget and parallel fuzzing is off.
The ordinary CI job also runs 5,000 deterministic mutations. No signing keys,
user documents, optional downloads or native runtime libraries are needed.

Keep crashing inputs and fix the production parser first, then add the smallest
reproduction to its regression tests. Refresh the pinned container and action
deliberately. The corpus HEIF sample comes from the existing BSD-3-Clause fixture
collection; its licence and attribution are retained in `corpus/`.
