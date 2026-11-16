//! Render a source pack using the production sandbox, for honest catalogue previews.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 3 && args[1] == "--install" {
        let root = paintfe::services::plugins::root();
        let mut prefs = paintfe::services::plugins::Preferences::load(&root);
        for entry in std::fs::read_dir(&args[2])? {
            let path = entry?.path();
            if path.extension().and_then(|v| v.to_str()) != Some("pfeplugin") {
                continue;
            }
            let bytes = paintfe::services::plugins::read_limited(
                &path,
                paintfe::services::plugins::MAX_PACKAGE,
            )?;
            let m = paintfe::services::plugins::install(&bytes, &root, true, None)?;
            prefs.packs.entry(m.id).or_default().current = m.version;
        }
        prefs.save(&root)?;
        return Ok(());
    }
    if args.len() != 4 {
        return Err("Usage: community_pack_preview SOURCE_PACK INPUT.png OUTPUT.png".into());
    }
    let at = std::path::Path::new(&args[1]);
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(at.join("manifest.json"))?)?;
    if value.get("files").is_none() {
        value["files"] = serde_json::json!([]);
    }
    let manifest: paintfe::services::plugins::Manifest = serde_json::from_value(value)?;
    let effect = &manifest.effects[0];
    let source = std::fs::read_to_string(at.join(&effect.script))?;
    let img = image::open(&args[2])?.to_rgba8();
    let (w, h) = img.dimensions();
    let (pixels, ow, oh, _, ops) = paintfe::ops::scripting::execute_script_parameters(
        &source,
        img.into_raw(),
        w,
        h,
        None,
        &effect.defaults(),
        Some(0),
    )
    .map_err(|e| e.message)?;
    if !ops.is_empty() || ow != w || oh != h {
        return Err("Preview effect changed dimensions".into());
    }
    image::RgbaImage::from_raw(w, h, pixels)
        .ok_or("Invalid preview pixels")?
        .save(&args[3])?;
    Ok(())
}
