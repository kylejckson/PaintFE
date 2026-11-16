#![no_main]
libfuzzer_sys::fuzz_target!(|bytes: &[u8]| {
    paintfe_fuzz_targets::check_expression(bytes);
});
