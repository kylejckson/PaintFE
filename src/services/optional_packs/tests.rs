use super::*;
use ed25519_dalek::{Signer, SigningKey};
use zip::write::SimpleFileOptions;
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/optional-pack-tests")
            .join(uuid::Uuid::new_v4().to_string());
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn key() -> SigningKey {
    SigningKey::from_bytes(&[7; 32])
}
fn manifest(pack: Pack, payload: &[u8]) -> Manifest {
    Manifest {
        schema: 1,
        pack,
        app_version: env!("CARGO_PKG_VERSION").into(),
        platform: platform(),
        protocol: 1,
        files: vec![
            PackFile {
                name: pack.payload().into(),
                sha256: format!("{:x}", Sha256::digest(payload)),
                size: payload.len() as u64,
            },
            PackFile {
                name: "LICENSE.txt".into(),
                sha256: format!("{:x}", Sha256::digest(b"license")),
                size: 7,
            },
        ],
    }
}
fn archive(path: &Path, manifest: &Manifest, payload: &[u8], damage_signature: bool) {
    let text = serde_json::to_string(manifest).unwrap();
    let mut signature = key().sign(text.as_bytes()).to_bytes();
    if damage_signature {
        signature[0] ^= 1;
    }
    let envelope = Envelope {
        manifest: text,
        signature: signature.iter().map(|v| format!("{v:02x}")).collect(),
    };
    let mut zip = zip::ZipWriter::new(fs::File::create(path).unwrap());
    for (name, bytes) in [
        ("manifest.json", serde_json::to_vec(&envelope).unwrap()),
        (manifest.pack.payload(), payload.to_vec()),
        ("LICENSE.txt", b"license".to_vec()),
    ] {
        zip.start_file(name, SimpleFileOptions::default()).unwrap();
        zip.write_all(&bytes).unwrap();
    }
    zip.finish().unwrap();
}
#[test]
fn signed_install_detects_tampering_and_preserves_previous_install() {
    let fixture = Fixture::new();
    let path = fixture.0.join("test.pfepack");
    let root = fixture.0.join("packs");
    let manifest = manifest(Pack::Jxl, b"helper");
    archive(&path, &manifest, b"helper", false);
    let public = key().verifying_key().to_bytes();
    install_at(&path, Pack::Jxl, &root, &public).unwrap();
    let payload = installed_at(&root, Pack::Jxl, &public).unwrap();
    assert_eq!(fs::read(&payload).unwrap(), b"helper");
    archive(&path, &manifest, b"damage", false);
    assert!(install_at(&path, Pack::Jxl, &root, &public).is_err());
    assert_eq!(fs::read(&payload).unwrap(), b"helper");
    archive(&path, &manifest, b"helper", true);
    assert!(install_at(&path, Pack::Jxl, &root, &public).is_err());
    fs::write(payload, b"damage").unwrap();
    assert!(installed_at(&root, Pack::Jxl, &public).is_err());
}

fn set_payload_archive_mode(path: &Path, mode: u32) {
    // Only change ZIP metadata. The signed payload and its checksum stay valid.
    let mut bytes = fs::read(path).unwrap();
    let payload = Pack::Jxl.payload().as_bytes();
    let offset = bytes
        .windows(4)
        .enumerate()
        .find_map(|(offset, signature)| {
            (signature == b"PK\x01\x02"
                && bytes.get(offset + 46..offset + 46 + payload.len()) == Some(payload))
            .then_some(offset)
        })
        .expect("payload central directory entry");
    bytes[offset + 5] = 3; // Unix creator system.
    bytes[offset + 38..offset + 42].copy_from_slice(&(mode << 16).to_le_bytes());
    fs::write(path, bytes).unwrap();
}

#[test]
fn signed_payload_cannot_be_installed_as_a_symlink() {
    let fixture = Fixture::new();
    let path = fixture.0.join("symlink.pfepack");
    let root = fixture.0.join("packs");
    archive(&path, &manifest(Pack::Jxl, b"helper"), b"helper", false);
    set_payload_archive_mode(&path, 0o120777);
    let error = install_at(&path, Pack::Jxl, &root, &key().verifying_key().to_bytes()).unwrap_err();
    assert!(error.contains("Invalid pack entry"), "{error}");
    assert!(!root.join("protocol-1").exists());
}

#[test]
fn signed_payload_does_not_preserve_privileged_archive_permissions() {
    let fixture = Fixture::new();
    let path = fixture.0.join("permissions.pfepack");
    let root = fixture.0.join("packs");
    archive(&path, &manifest(Pack::Jxl, b"helper"), b"helper", false);
    set_payload_archive_mode(&path, 0o106755);
    let public = key().verifying_key().to_bytes();
    install_at(&path, Pack::Jxl, &root, &public).unwrap();
    let payload = installed_at(&root, Pack::Jxl, &public).unwrap();
    assert_eq!(fs::read(&payload).unwrap(), b"helper");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(payload).unwrap().permissions().mode() & 0o7777,
            0o700
        );
    }
}
#[test]
fn rejects_wrong_key_platform_feature_protocol_and_paths() {
    let fixture = Fixture::new();
    let path = fixture.0.join("test.pfepack");
    let root = fixture.0.join("packs");
    let public = key().verifying_key().to_bytes();
    for field in 0..6 {
        let mut manifest = manifest(Pack::Jxl, b"helper");
        match field {
            0 => manifest.platform = "other".into(),
            1 => manifest.schema = 9,
            2 => manifest.protocol = 2,
            3 => manifest.files[0].name = "../escape".into(),
            4 => manifest.files[1].name = manifest.files[0].name.clone(),
            _ => {}
        }
        archive(&path, &manifest, b"helper", false);
        let key = if field == 5 { [42; 32] } else { public };
        assert!(
            install_at(&path, Pack::Jxl, &root, &key).is_err(),
            "field {field}"
        );
        assert!(install_at(&path, Pack::Cjk, &root, &public).is_err());
    }
    assert!(!root.exists());
    assert!(!fixture.0.join("escape").exists());
}
#[test]
fn cjk_font_is_validated_and_installable_without_network() {
    let fixture = Fixture::new();
    let path = fixture.0.join("cjk.pfepack");
    let root = fixture.0.join("packs");
    let font = include_bytes!("../../../assets/fonts/NotoSansCJK-Regular.otf");
    archive(&path, &manifest(Pack::Cjk, font), font, false);
    install_at(&path, Pack::Cjk, &root, &key().verifying_key().to_bytes()).unwrap();
    for language in ["ja", "ko", "zh-CN", "zh-TW"] {
        assert!(font_covers(font, language));
    }
    archive(
        &path,
        &manifest(Pack::Cjk, b"not a font"),
        b"not a font",
        false,
    );
    assert!(install_at(&path, Pack::Cjk, &root, &key().verifying_key().to_bytes()).is_err());
}
#[test]
fn rejects_oversized_archive_and_manifest() {
    let fixture = Fixture::new();
    let path = fixture.0.join("large.pfepack");
    fs::File::create(&path)
        .unwrap()
        .set_len(MAX_PACK + 1)
        .unwrap();
    assert!(
        install_at(
            &path,
            Pack::Jxl,
            &fixture.0.join("packs"),
            &key().verifying_key().to_bytes()
        )
        .is_err()
    );
    assert!(
        verify_manifest(
            &vec![0; MAX_MANIFEST as usize + 1],
            Pack::Jxl,
            &key().verifying_key().to_bytes()
        )
        .is_err()
    );
}
#[test]
fn missing_encoder_and_invalid_buffers_fail_without_network_or_output() {
    assert!(encode_jxl(0, 1, &[], 8, 90, Default::default()).is_err());
    assert!(encode_jxl(1, 1, &[0; 3], 8, 90, Default::default()).is_err());
    assert!(encode_jxl(1, 1, &[0; 4], 32, 90, Default::default()).is_err());
    assert!(needs_cjk("ko"));
    assert!(!needs_cjk("en"));
}

#[test]
fn compatible_signed_packs_survive_application_version_changes() {
    let fixture = Fixture::new();
    let path = fixture.0.join("previous.pfepack");
    let mut manifest = manifest(Pack::Jxl, b"helper");
    manifest.app_version = "1.0.0".into();
    archive(&path, &manifest, b"helper", false);
    install_at(
        &path,
        Pack::Jxl,
        &fixture.0.join("packs"),
        &key().verifying_key().to_bytes(),
    )
    .unwrap();
    assert!(
        installed_at(
            &fixture.0.join("packs"),
            Pack::Jxl,
            &key().verifying_key().to_bytes()
        )
        .is_ok()
    );
}

#[test]
fn download_transport_verifies_success_and_cleans_interrupted_or_invalid_files() {
    let fixture = Fixture::new();
    let source = fixture.0.join("source.pfepack");
    let root = fixture.0.join("packs");
    archive(&source, &manifest(Pack::Jxl, b"helper"), b"helper", false);
    let key = key().verifying_key().to_bytes();
    download_with(Pack::Jxl, &root, &key, |path| {
        fs::copy(&source, path).map_err(|e| e.to_string())?;
        Ok(())
    })
    .unwrap();
    let installed = installed_at(&root, Pack::Jxl, &key).unwrap();
    for interrupted in [false, true] {
        assert!(
            download_with(Pack::Jxl, &root, &key, |path| {
                fs::write(path, b"partial download").unwrap();
                if interrupted {
                    Err("Connection interrupted".into())
                } else {
                    Ok(())
                }
            })
            .is_err()
        );
        assert_eq!(fs::read(&installed).unwrap(), b"helper");
        assert!(!fs::read_dir(&root).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".download-")
        }));
    }
    assert!(
        Pack::Jxl
            .url()
            .starts_with("https://github.com/kylejckson/PaintFE/releases/download/")
    );
}
