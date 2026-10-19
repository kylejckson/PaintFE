Source: crates.io jxl-sys 0.1.12+libjxl-0.12.0, MIT OR Apache-2.0.

Only build.rs is modified: respect target crt-static for the portable Windows
binary, and respect Cargo NUM_JOBS instead of using every CPU core. All libjxl
and third-party source and license files are unchanged. No runtime downloads.
