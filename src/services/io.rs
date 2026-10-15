pub use crate::io::*;

/// Readable Windows paths for labels only. Keep the original path for I/O.
pub fn display_path(path: &std::path::Path) -> String {
    let text = path.to_string_lossy();
    if let Some(unc) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{unc}")
    } else if let Some(drive) = text.strip_prefix(r"\\?\") {
        drive.to_owned()
    } else {
        text.into_owned()
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    #[test]
    fn display_paths_hide_extended_prefixes_without_altering_the_source() {
        for (source, expected) in [
            (r"\\?\C:\Pictures\image.png", r"C:\Pictures\image.png"),
            (
                r"\\?\UNC\server\share\image.png",
                r"\\server\share\image.png",
            ),
            ("/home/user/image.png", "/home/user/image.png"),
        ] {
            let path = std::path::PathBuf::from(source);
            assert_eq!(display_path(&path), expected);
            assert_eq!(path, std::path::PathBuf::from(source));
        }
    }
}
