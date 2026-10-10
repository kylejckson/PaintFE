//! Exercise the real CLI binary, including output protection before batch writes.
use std::path::{Path, PathBuf};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/cli-tests")
            .join(uuid::Uuid::new_v4().to_string());
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
    fn image(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        image::RgbaImage::from_fn(16, 12, |x, y| image::Rgba([x as u8, y as u8, 90, 255]))
            .save(&path)
            .unwrap();
        path
    }
    fn run(&self, args: &[&std::ffi::OsStr]) -> std::process::Output {
        std::process::Command::new(env!("CARGO_BIN_EXE_PaintFE"))
            .args(args)
            .env("PAINTFE_TEST_STORAGE_DIR", self.0.join("profile"))
            .env("RAYON_NUM_THREADS", "2")
            .output()
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn arg(s: &str) -> &std::ffi::OsStr {
    std::ffi::OsStr::new(s)
}

#[test]
fn cli_crops_a_glob_batch_and_protects_existing_outputs() {
    let fixture = Fixture::new();
    let source = fixture.image("in/a.png");
    fixture.image("in/b.png");
    let original = std::fs::read(&source).unwrap();
    let script = fixture.0.join("crop.rhai");
    std::fs::write(&script, "crop_canvas(3, 2, 5, 4);").unwrap();
    let pattern = fixture.0.join("in/*.png");
    let output = fixture.0.join("out");
    let args = [
        arg("--input"),
        pattern.as_os_str(),
        arg("--script"),
        script.as_os_str(),
        arg("--output-dir"),
        output.as_os_str(),
        arg("--format"),
        arg("png"),
    ];
    let result = fixture.run(&args);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for name in ["a.png", "b.png"] {
        let actual = image::open(output.join(name)).unwrap().into_rgba8();
        assert_eq!(actual.dimensions(), (5, 4));
        assert_eq!(actual.get_pixel(0, 0).0, [3, 2, 90, 255]);
    }
    assert!(!fixture.run(&args).status.success());
    let mut overwrite = args.to_vec();
    overwrite.push(arg("--overwrite"));
    assert!(fixture.run(&overwrite).status.success());
    assert_eq!(std::fs::read(source).unwrap(), original);
}

#[test]
fn cli_rejects_same_name_batch_collisions_before_writing_any_output() {
    let fixture = Fixture::new();
    fixture.image("one/same.png");
    fixture.image("two/same.png");
    let pattern = fixture.0.join("*/same.png");
    let output = fixture.0.join("out");
    let result = fixture.run(&[
        arg("--input"),
        pattern.as_os_str(),
        arg("--output-dir"),
        output.as_os_str(),
    ]);
    assert!(!result.status.success());
    assert!(!output.join("same.png").exists());
}

#[test]
fn cli_never_overwrites_input_even_when_overwrite_is_explicit() {
    let fixture = Fixture::new();
    let source = fixture.image("source.png");
    let original = std::fs::read(&source).unwrap();
    let result = fixture.run(&[
        arg("--input"),
        source.as_os_str(),
        arg("--output"),
        source.as_os_str(),
        arg("--overwrite"),
    ]);
    assert!(!result.status.success());
    assert_eq!(std::fs::read(source).unwrap(), original);
    assert!(Path::new(&fixture.0).exists());
}
