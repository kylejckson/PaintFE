//! Bounded disk/decode batches. The UI only uploads one prepared image per frame.
use super::*;
use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;

struct Job {
    generation: u64,
    key: String,
    digest: String,
    paths: Vec<PathBuf>,
    icon: bool,
}
struct Prepared {
    generation: u64,
    key: String,
    image: Option<egui::ColorImage>,
}
#[derive(Default)]
pub(super) struct Artwork {
    generation: u64,
    textures: BTreeMap<String, egui::TextureHandle>,
    used: VecDeque<String>,
    pending: BTreeSet<String>,
    missing: BTreeSet<String>,
    queued: VecDeque<Job>,
    ready: VecDeque<Prepared>,
    active: Option<mpsc::Receiver<Vec<Prepared>>>,
    last_frame: Option<u64>,
}
impl Artwork {
    pub(super) fn is_busy(&self) -> bool {
        self.active.is_some() || !self.ready.is_empty() || !self.queued.is_empty()
    }
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }
    pub(super) fn retry_missing(&mut self) {
        self.missing.clear();
        self.generation = self.generation.wrapping_add(1);
    }
    pub(super) fn request(
        &mut self,
        manifest: &Manifest,
        installed: &[Installed],
        icon: bool,
    ) -> Option<egui::TextureHandle> {
        let path = if icon {
            &manifest.icon
        } else {
            &manifest.screenshot
        };
        self.request_at(manifest, installed, path, icon)
    }
    pub(super) fn request_at(
        &mut self,
        manifest: &Manifest,
        installed: &[Installed],
        path: &str,
        icon: bool,
    ) -> Option<egui::TextureHandle> {
        let file = manifest.files.iter().find(|file| file.path == path)?;
        let key = format!("{}-{icon}", file.sha256);
        if let Some(texture) = self.textures.get(&key) {
            self.used.retain(|old| old != &key);
            self.used.push_back(key);
            return Some(texture.clone());
        }
        if self.pending.contains(&key) || self.missing.contains(&key) || self.queued.len() >= 16 {
            return None;
        }
        let mut paths = Vec::new();
        if let Some(pack) = installed.iter().find(|pack| {
            pack.manifest.id == manifest.id && pack.manifest.version == manifest.version
        }) {
            paths.push(pack.directory.join(path));
        }
        paths.push(p::root().join("artwork").join(&file.sha256));
        self.pending.insert(key.clone());
        self.queued.push_back(Job {
            generation: self.generation,
            key,
            digest: file.sha256.clone(),
            paths,
            icon,
        });
        None
    }
    pub(super) fn poll(&mut self, ctx: &Context) {
        if self.last_frame == Some(ctx.cumulative_frame_nr()) {
            return;
        }
        self.last_frame = Some(ctx.cumulative_frame_nr());
        if let Some(active) = &self.active {
            match active.try_recv() {
                Ok(images) => {
                    self.ready.extend(images);
                    self.active = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending.clear();
                    self.active = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        if let Some(prepared) = self.ready.pop_front() {
            self.pending.remove(&prepared.key);
            if let Some(image) = prepared.image {
                while self.textures.len() >= 48 {
                    if let Some(old) = self.used.pop_front() {
                        self.textures.remove(&old);
                    } else {
                        break;
                    }
                }
                let texture = ctx.load_texture(
                    format!("plugin-art-{}", prepared.key),
                    image,
                    egui::TextureOptions::LINEAR,
                );
                self.used.push_back(prepared.key.clone());
                self.textures.insert(prepared.key, texture);
            } else if prepared.generation == self.generation {
                if self.missing.len() >= 256 {
                    self.missing.clear();
                }
                self.missing.insert(prepared.key);
            }
            ctx.request_repaint();
        }
    }
    pub(super) fn start_batch(&mut self, ctx: &Context) {
        if self.active.is_some() || !self.ready.is_empty() || self.queued.is_empty() {
            return;
        }
        let jobs: Vec<_> = self.queued.drain(..self.queued.len().min(2)).collect();
        let (tx, rx) = mpsc::channel();
        ctx.request_repaint();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let images = jobs.into_iter().map(prepare).collect();
            let _ = tx.send(images);
            ctx.request_repaint();
        });
        self.active = Some(rx);
    }
}
fn prepare(job: Job) -> Prepared {
    let image = job.paths.iter().find_map(|path| {
        let bytes = read_limited(path, 4 * 1024 * 1024).ok()?;
        if hash(&bytes) != job.digest {
            return None;
        }
        let mut reader =
            image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Png);
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(1600);
        limits.max_image_height = Some(1200);
        limits.max_alloc = Some(16 * 1024 * 1024);
        reader.limits(limits);
        let decoded = reader.decode().ok()?.to_rgba8();
        let (width, height) = if job.icon { (128, 128) } else { (640, 400) };
        let image = p::thumbnail(&decoded, width, height);
        Some(egui::ColorImage::from_rgba_unmultiplied(
            [image.width() as usize, image.height() as usize],
            image.as_raw(),
        ))
    });
    Prepared {
        generation: job.generation,
        key: job.key,
        image,
    }
}

pub(super) fn favourite(ctx: &Context, selected: bool) -> egui::TextureHandle {
    let id = egui::Id::new(("community_favourite_icon", selected));
    if let Some(texture) = ctx.data(|data| data.get_temp::<egui::TextureHandle>(id)) {
        return texture;
    }
    let bytes: &[u8] = if selected {
        include_bytes!("../../../../assets/ui/favourite_on.png")
    } else {
        include_bytes!("../../../../assets/ui/favourite_off.png")
    };
    let image = image::load_from_memory(bytes)
        .expect("Bundled favourite icon")
        .to_rgba8();
    let pixels: Vec<_> = image
        .pixels()
        .flat_map(|pixel| [255, 255, 255, pixel[3]])
        .collect();
    let texture = ctx.load_texture(
        format!("community-favourite-{selected}"),
        egui::ColorImage::from_rgba_unmultiplied(
            [image.width() as usize, image.height() as usize],
            &pixels,
        ),
        egui::TextureOptions::LINEAR,
    );
    ctx.data_mut(|data| data.insert_temp(id, texture.clone()));
    texture
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_old_missing_result_cannot_hide_newly_downloaded_artwork() {
        let ctx = Context::default();
        let mut cache = Artwork::default();
        cache.ready.push_back(Prepared {
            generation: 0,
            key: "new-art".into(),
            image: None,
        });
        cache.retry_missing();
        let _ = ctx.run_ui(egui::RawInput::default(), |_| cache.poll(&ctx));
        assert!(cache.missing.is_empty());
        cache.ready.push_back(Prepared {
            generation: cache.generation,
            key: "still-missing".into(),
            image: None,
        });
        let _ = ctx.run_ui(egui::RawInput::default(), |_| cache.poll(&ctx));
        assert!(cache.missing.contains("still-missing"));
    }
    #[test]
    fn prepared_artwork_uploads_at_most_once_per_frame_and_evicts_one_entry() {
        let ctx = Context::default();
        let mut cache = Artwork::default();
        for index in 0..50 {
            cache.ready.push_back(Prepared {
                generation: 0,
                key: index.to_string(),
                image: Some(egui::ColorImage::filled([1, 1], egui::Color32::WHITE)),
            });
        }
        for index in 0..50 {
            let _ = ctx.run_ui(egui::RawInput::default(), |_| {
                cache.poll(&ctx);
                cache.poll(&ctx);
                assert_eq!(cache.ready.len(), 49 - index);
            });
        }
        assert_eq!(cache.textures.len(), 48);
        assert!(!cache.textures.contains_key("0"));
        assert!(cache.textures.contains_key("2"));
    }
    #[test]
    fn requesting_artwork_queues_without_reading_or_decoding_on_ui() {
        let catalogue = catalogue(include_bytes!(
            "../../../../tests/fixtures/plugin-catalogue.json"
        ))
        .unwrap();
        let mut cache = Artwork::default();
        let manifest = &catalogue.packs[0].manifest;
        assert!(cache.request(manifest, &[], false).is_none());
        assert_eq!(cache.queued.len(), 1);
        assert!(cache.textures.is_empty());
        assert!(cache.missing.is_empty());
        assert!(cache.request(manifest, &[], false).is_none());
        assert_eq!(cache.queued.len(), 1);
    }
    #[test]
    fn bundled_favourite_icons_keep_alpha_and_can_be_tinted_in_both_themes() {
        let ctx = Context::default();
        for selected in [false, true] {
            assert_eq!(favourite(&ctx, selected).size(), [32, 32]);
        }
    }
}
