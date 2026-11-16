#![cfg(not(target_arch = "wasm32"))]
use paintfe::services::plugins::*;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
const COLOUR: &[u8] = include_bytes!("fixtures/plugin-paintfe.colour-workshop-1.0.0.pfeplugin");
const GLITCH: &[u8] = include_bytes!("fixtures/plugin-paintfe.glitch-workshop-1.0.0.pfeplugin");
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("paintfe-plugin-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn entries(bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
    let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    let mut result = BTreeMap::new();
    for i in 0..z.len() {
        let mut f = z.by_index(i).unwrap();
        let name = f.name().unwrap().into_owned();
        let mut b = Vec::new();
        f.read_to_end(&mut b).unwrap();
        result.insert(name, b);
    }
    result
}
fn archive(files: BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    let mut z = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, b) in files {
        z.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        z.write_all(&b).unwrap();
    }
    z.finish().unwrap().into_inner()
}
fn manifest(bytes: &[u8]) -> Manifest {
    let files = entries(bytes);
    let e: Envelope = serde_json::from_slice(&files["manifest.json"]).unwrap();
    serde_json::from_str(&e.payload).unwrap()
}
#[test]
fn published_pack_signatures_install_and_sources_are_verified() {
    for bytes in [COLOUR, GLITCH] {
        let at = Temp::new();
        let m = install(bytes, &at.0, false, None).unwrap();
        let prefs = Preferences::load(&at.0);
        let all = installed(&at.0, &prefs);
        assert_eq!(all.len(), 1);
        assert!(all[0].signed);
        assert_eq!(m.id, all[0].manifest.id);
        for e in &m.effects {
            assert!(!effect_source(&all[0], e).unwrap().is_empty());
        }
    }
}
#[test]
fn tampered_pack_does_not_replace_installed_version() {
    let at = Temp::new();
    let m = install(COLOUR, &at.0, false, None).unwrap();
    let before = std::fs::read(at.0.join("preferences.json")).unwrap();
    let mut files = entries(COLOUR);
    files.get_mut(&m.effects[0].script).unwrap().push(b' ');
    assert!(install(&archive(files), &at.0, false, None).is_err());
    assert_eq!(
        before,
        std::fs::read(at.0.join("preferences.json")).unwrap()
    );
    assert_eq!(installed(&at.0, &Preferences::load(&at.0)).len(), 1);
}
#[test]
fn unsigned_local_import_is_explicit_and_labelled() {
    let mut files = entries(COLOUR);
    let mut envelope: Envelope = serde_json::from_slice(&files["manifest.json"]).unwrap();
    envelope.signature.clear();
    files.insert(
        "manifest.json".into(),
        serde_json::to_vec(&envelope).unwrap(),
    );
    let b = archive(files);
    let at = Temp::new();
    assert!(install(&b, &at.0, false, None).is_err());
    install(&b, &at.0, true, None).unwrap();
    assert!(!installed(&at.0, &Preferences::load(&at.0))[0].signed);
}
#[test]
fn invalid_signature_and_unexpected_archive_entries_are_rejected() {
    let mut files = entries(COLOUR);
    let mut e: Envelope = serde_json::from_slice(&files["manifest.json"]).unwrap();
    e.signature = "00".repeat(64);
    files.insert("manifest.json".into(), serde_json::to_vec(&e).unwrap());
    let at = Temp::new();
    assert!(install(&archive(files), &at.0, true, None).is_err());
    for name in ["../escape", "/absolute", "CON.txt", "extra.exe"] {
        let mut f = entries(COLOUR);
        f.insert(name.into(), vec![1]);
        assert!(install(&archive(f), &at.0, true, None).is_err());
    }
    assert!(!at.0.join("preferences.json").exists());
}
#[test]
fn modified_installed_source_cannot_run_as_the_signed_original() {
    let at = Temp::new();
    install(COLOUR, &at.0, false, None).unwrap();
    let p = installed(&at.0, &Preferences::load(&at.0)).remove(0);
    let e = &p.manifest.effects[0];
    std::fs::write(p.directory.join(&e.script), "apply_invert();").unwrap();
    assert!(effect_source(&p, e).is_err());
}
#[test]
fn catalogue_and_path_boundaries_are_portable() {
    for p in [
        "../x", "/x", "a\\x", "a//b", "CON.png", "a./b", "NUL", "C:/x",
    ] {
        assert!(!resource_path(p), "{p}");
    }
    for url in [
        "http://github.com/kylejckson/PaintFE_plugins/releases/download/x/y",
        "https://example.com/x",
        "https://github.com/kylejckson/PaintFE_plugins/releases/download/../x",
        "https://github.com/kylejckson/PaintFE_plugins/releases/download/v1/a?secret=1",
    ] {
        assert!(!allowed_url(url), "{url}");
    }
    assert!(allowed_url(
        "https://github.com/kylejckson/PaintFE_plugins/releases/download/v0.1.0/paintfe.colour-1.0.0.pfeplugin"
    ));
}
#[test]
fn future_api_can_be_listed_but_cannot_be_installed() {
    let mut m = manifest(COLOUR);
    m.api = 2;
    assert!(m.validate_listing().is_ok());
    assert!(m.validate().is_err());
}
#[test]
fn invalid_parameter_defaults_and_duplicate_ids_fail_before_execution() {
    let original = manifest(COLOUR);
    for mutation in 0..4 {
        let mut m = original.clone();
        let p = &mut m.effects[0].parameters[0];
        match mutation {
            0 => p.default = serde_json::json!([300, 0, 0, 255]),
            1 => p.step = 0.,
            2 => p.min = 2.,
            _ => p.kind = "native_code".into(),
        };
        assert!(m.validate().is_err());
    }
    let mut m = original;
    let duplicate = m.effects[0].parameters[0].clone();
    m.effects[0].parameters.push(duplicate);
    assert!(m.validate().is_err());
}
fn request(source: &str) -> worker::Request {
    let mut effect = manifest(COLOUR).effects.remove(0);
    effect.parameters.clear();
    worker::Request {
        source: source.into(),
        width: 8,
        height: 6,
        values: BTreeMap::new(),
        seed: 19,
        effect,
        has_mask: false,
    }
}
#[test]
fn actual_worker_preserves_alpha_and_excludes_unselected_pixels() {
    let mut r = request("apply_invert();");
    r.has_mask = true;
    let input: Vec<u8> = (0..48)
        .flat_map(|i| [i, 100, 200, if i % 2 == 0 { 0 } else { 127 }])
        .collect();
    let mask: Vec<u8> = (0..48).map(|i| if i % 3 == 0 { 0 } else { 255 }).collect();
    let out = worker::execute(
        r,
        input.clone(),
        Some(mask.clone()),
        &AtomicBool::new(false),
    )
    .unwrap();
    for (i, (a, b)) in input
        .chunks_exact(4)
        .zip(out.pixels.chunks_exact(4))
        .enumerate()
    {
        assert_eq!(a[3], b[3]);
        if mask[i] == 0 {
            assert_eq!(a, b)
        } else {
            assert_eq!(b, [255 - a[0], 155, 55, a[3]])
        }
    }
}
#[test]
fn actual_worker_cancel_stops_an_infinite_script() {
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    let timer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        stop.store(true, Ordering::Relaxed);
    });
    let start = Instant::now();
    let r = worker::execute(
        request("loop { let x=1+1; }"),
        vec![255; 8 * 6 * 4],
        None,
        &cancel,
    );
    timer.join().unwrap();
    assert!(r.is_err());
    assert!(start.elapsed() < Duration::from_secs(8));
}
#[test]
fn actual_worker_rejects_canvas_resizing_without_returning_an_edit() {
    assert!(
        worker::execute(
            request("resize_canvas(2,2,\"center\");"),
            vec![255; 8 * 6 * 4],
            None,
            &AtomicBool::new(false)
        )
        .is_err()
    );
}
#[test]
fn script_parameters_and_randomness_are_repeatable() {
    let m = manifest(COLOUR);
    let e = m.effects[0].clone();
    let v = e.defaults();
    let source = String::from_utf8(entries(COLOUR)[&e.script].clone()).unwrap();
    let pixels = [100, 120, 140, 180].repeat(8 * 6);
    let a = paintfe::ops::scripting::execute_script_parameters(
        &source,
        pixels.clone(),
        8,
        6,
        None,
        &v,
        Some(19),
    )
    .unwrap()
    .0;
    let b = paintfe::ops::scripting::execute_script_parameters(
        &source,
        pixels.clone(),
        8,
        6,
        None,
        &v,
        Some(19),
    )
    .unwrap()
    .0;
    assert_eq!(a, b);
    assert_ne!(a, pixels);
}
#[test]
fn general_module_imports_are_disabled() {
    let at = Temp::new();
    let path = at.0.join("secret.rhai");
    std::fs::write(&path, "export const secret = 99;").unwrap();
    let source = format!(
        "import {:?} as resource; print_line(resource::secret);",
        path.to_string_lossy().replace('\\', "/")
    );
    assert!(paintfe::ops::scripting::execute_script_sync(&source, vec![0; 4], 1, 1, None).is_err());
}
#[test]
fn console_output_is_bounded() {
    let (_, _, _, lines, _) = paintfe::ops::scripting::execute_script_sync(
        "for i in 0..10000 { print_line(\"message\"); }",
        vec![0; 4],
        1,
        1,
        None,
    )
    .unwrap();
    assert!(lines.len() <= 257);
    assert!(lines.iter().map(String::len).sum::<usize>() <= 65536);
}

#[test]
fn signed_reinstall_repairs_damaged_files_without_losing_preferences() {
    let at = Temp::new();
    let m = install(COLOUR, &at.0, false, None).unwrap();
    let mut prefs = Preferences::load(&at.0);
    prefs.packs.get_mut(&m.id).unwrap().favourite = true;
    prefs.save(&at.0).unwrap();
    let pack = installed(&at.0, &prefs).remove(0);
    std::fs::write(pack.directory.join(&m.effects[0].script), "damaged").unwrap();
    install(COLOUR, &at.0, false, None).unwrap();
    let prefs = Preferences::load(&at.0);
    assert!(prefs.packs[&m.id].favourite);
    assert!(prefs.packs[&m.id].previous.is_empty());
    assert!(effect_source(&installed(&at.0, &prefs)[0], &m.effects[0]).is_ok());
    std::fs::write(pack.directory.join("manifest.json"), "damaged").unwrap();
    install(COLOUR, &at.0, false, None).unwrap();
    assert_eq!(installed(&at.0, &Preferences::load(&at.0)).len(), 1);
}

#[test]
fn unsigned_pack_cannot_take_over_a_signed_identity() {
    let at = Temp::new();
    install(COLOUR, &at.0, false, None).unwrap();
    let mut files = entries(COLOUR);
    let mut e: Envelope = serde_json::from_slice(&files["manifest.json"]).unwrap();
    let mut m: Manifest = serde_json::from_str(&e.payload).unwrap();
    m.version = "9.0.0".into();
    e.payload = serde_json::to_string(&m).unwrap();
    e.signature.clear();
    files.insert("manifest.json".into(), serde_json::to_vec(&e).unwrap());
    assert!(install(&archive(files), &at.0, true, None).is_err());
    assert_eq!(Preferences::load(&at.0).packs[&m.id].current, "1.0.0");
}

#[test]
fn uninstall_removes_managed_files_but_preserves_a_separate_local_copy() {
    let at = Temp::new();
    let signed = install(COLOUR, &at.0, false, None).unwrap();
    let mut files = entries(COLOUR);
    let mut e: Envelope = serde_json::from_slice(&files["manifest.json"]).unwrap();
    let mut local: Manifest = serde_json::from_str(&e.payload).unwrap();
    local.id = "local.copy".into();
    e.payload = serde_json::to_string(&local).unwrap();
    e.signature.clear();
    files.insert("manifest.json".into(), serde_json::to_vec(&e).unwrap());
    install(&archive(files), &at.0, true, None).unwrap();
    uninstall(&signed.id, &at.0).unwrap();
    assert!(!at.0.join("installed").join(&signed.id).exists());
    let all = installed(&at.0, &Preferences::load(&at.0));
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].manifest.id, "local.copy");
    assert!(uninstall("../outside", &at.0).is_err());
}

#[test]
fn integer_controls_reject_fractional_or_unrepresentable_bounds() {
    for (min, max, step) in [(0.25, 10., 1.), (0., 10., 0.5), (0., 1e100, 1.)] {
        let mut m = manifest(COLOUR);
        let p = &mut m.effects[0].parameters[0];
        p.kind = "integer".into();
        p.default = 1.into();
        p.min = min;
        p.max = max;
        p.step = step;
        assert!(m.validate().is_err());
    }
}

#[test]
fn preview_textures_keep_landscape_portrait_and_square_aspect_ratios() {
    for (w, h, expected) in [
        (480, 270, (480, 270)),
        (1600, 900, (640, 360)),
        (600, 1200, (200, 400)),
        (800, 800, (400, 400)),
    ] {
        let image = image::RgbaImage::new(w, h);
        let result = thumbnail(&image, 640, 400);
        assert_eq!(result.dimensions(), expected);
    }
}

#[test]
fn local_effect_names_and_categories_preserve_signed_source_metadata() {
    let at = Temp::new();
    let original = install(COLOUR, &at.0, false, None).unwrap();
    let mut prefs = Preferences::load(&at.0);
    let settings = prefs.packs.get_mut(&original.id).unwrap();
    settings
        .names
        .insert(original.effects[0].id.clone(), "My duotone".into());
    settings
        .menus
        .insert(original.effects[0].id.clone(), "My effects".into());
    prefs.save(&at.0).unwrap();
    let restored = Preferences::load(&at.0);
    assert_eq!(
        restored.packs[&original.id].names[&original.effects[0].id],
        "My duotone"
    );
    let installed = installed(&at.0, &restored);
    assert!(installed[0].signed);
    assert_eq!(
        installed[0].manifest.effects[0].name,
        original.effects[0].name
    );
    assert_eq!(
        installed[0].manifest.effects[0].category,
        original.effects[0].category
    );
}

#[test]
fn community_icon_archive_has_bounded_expansion_and_safe_paths() {
    let at = Temp::new();
    let path = at.0.join("icons.zip");
    let mut files = BTreeMap::new();
    files.insert("../escape.png".into(), vec![1; 16]);
    std::fs::write(&path, archive(files)).unwrap();
    assert!(validate_icon_archive(&path).is_err());
    let mut files = BTreeMap::new();
    files.insert("icons.png".into(), vec![0; 2 * 1024 * 1024 + 1]);
    std::fs::write(&path, archive(files)).unwrap();
    assert!(validate_icon_archive(&path).is_err());
    let mut files = BTreeMap::new();
    files.insert("not-an-image.png".into(), vec![1; 24]);
    std::fs::write(&path, archive(files)).unwrap();
    assert!(validate_icon_archive(&path).is_err());
    validate_icon_archive(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icon_packs/luminous.zip"),
    )
    .unwrap();
}
