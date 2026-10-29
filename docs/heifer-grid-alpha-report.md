# Draft upstream report: grid auxiliary alpha in Heifer 0.2.0

This is a draft for maintainer review. No issue or PR has been submitted.

With the published Heifer 0.2.0 decoder, the synthetic pillow-heif fixtures
`LA_8__29x100.heif` and `RGBA_8__29x100.heif` decode with opaque alpha. The primary
items are derived `grid` images. Their independently decoded reference alpha
ranges from 4 to 251. The 128x128 coded-image alpha counterpart works.

Fixtures and provenance: `tests/fixtures/heif/README.md`, BSD-3-Clause. Original
upstream fixture commit: `b16be1196dfa465a342d68894696e685ca3655cb` in pillow-heif.
No personal photographs are needed to reproduce the problem.

Minimal reproduction with the unpatched dependency:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("HEIF fixture path");
    let decoded = heifer::decode(&std::fs::read(path)?)?;
    let rgba = decoded.to_rgba8();
    println!("min alpha: {:?}", rgba.chunks_exact(4).map(|p| p[3]).min());
    Ok(())
}
```

The local experimental patch moves auxiliary-alpha attachment into the common
item path, before parent transforms, so coded, grid and overlay parents share it.
It also decodes derived auxiliary items through the bounded item path and checks
both alpha dimensions. See the [historical patch](https://github.com/kylejckson/PaintFE/blob/feda9d2411ebcd9d578cf9879fff2e1cdcd8137a/vendor/heifer/PAINTFE-PATCHES.md). The obsolete decoder sources have since been removed.
The independent alpha references matched after that patch.

PaintFE's new production backend uses system codecs instead, so this report
concerns the original published Heifer dependency and the historical local patch.
