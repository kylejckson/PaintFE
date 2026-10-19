#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    rustdoc::bare_urls,
    rustdoc::broken_intra_doc_links
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_version() {
        let enc_ver = unsafe { JxlEncoderVersion() };
        let dec_ver = unsafe { JxlDecoderVersion() };
        let crate_semver =
            semver::Version::parse(env!("CARGO_PKG_VERSION")).expect("parse cargo semver");
        let libjxl_semver_str = &crate_semver.build.as_str()["libjxl-".len()..];
        let libjxl_semver = semver::Version::parse(libjxl_semver_str).expect("parse libjxl semver");
        let packaged_ver =
            libjxl_semver.major * 1000000 + libjxl_semver.minor * 1000 + libjxl_semver.patch;

        assert_eq!(packaged_ver, enc_ver as _);
        assert_eq!(packaged_ver, dec_ver as _);
    }
}
