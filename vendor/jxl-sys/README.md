# jxl-sys

Rust bindings to [libjxl](https://github.com/libjxl/libjxl)

## Building
Before building, you will need to make sure that git submodules are initialized:
```bash
git submodule update --init --recursive
```

You will also need to make sure you have all prerequisites to build `libjxl`.
Refer to `libjxl/BUILDING.md` for detailed instructions.

Then you can build with `cargo`:
```bash
cargo build
```

### LTO

The `lto` feature builds the bundled libjxl static libraries as LLVM
bitcode using Clang and `-flto=full`. It currently supports Linux targets.

The feature does not configure the final link of the consuming Rust binary.
Consumers must also enable Rust linker-plugin LTO and use an LLVM linker that
is compatible with the bitcode emitted by Clang.

Enable the feature on the dependency:

```toml
[dependencies]
jxl-sys = { version = "0.1", features = ["lto"] }
```

Then build the consuming binary with Rust's bundled LLD, replacing
`your-binary` with its Cargo binary target:

```bash
rust_target_dir="$(dirname "$(rustc --print target-libdir)")"
rust_lld="${rust_target_dir}/bin/gcc-ld/ld.lld"

cargo rustc --release --bin your-binary -- \
  -Clinker-plugin-lto \
  -Clinker=clang \
  -Clink-arg="-fuse-ld=${rust_lld}"
```

## License
`jxl-sys` is primarily distributed under the terms of both the MIT license and
the Apache License (Version 2.0).

See LICENSE-APACHE, LICENSE-MIT for details.

Copyright (c) 2025 Zetier
