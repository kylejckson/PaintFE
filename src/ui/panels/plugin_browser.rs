//! Native, cached community catalogue. This screen never executes plugin code.
use crate::services::plugins::{self as p, *};
use egui::{Context, RichText, Ui, Vec2};
mod artwork;
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

pub fn tr(key: &str) -> String {
    crate::i18n::translate(&format!("plugins.{key}"))
}
#[derive(Clone)]
pub enum Action {
    Run(Installed, Effect),
    Create,
    Edit(Installed),
    Export(Installed),
    Appearance(Installed),
    RemovedAppearance(Installed),
}
#[derive(Debug)]
enum ResultMessage {
    Refreshed(Catalogue),
    Installed(Manifest),
    Artwork { done: usize, total: usize },
    Complete,
}
struct Task {
    receiver: mpsc::Receiver<Result<ResultMessage, String>>,
    cancel: Arc<AtomicBool>,
}
impl Drop for Task {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
pub struct Browser {
    pub open: bool,
    pub catalogue: Option<Catalogue>,
    pub installed: Vec<Installed>,
    pub prefs: Preferences,
    search: String,
    view: usize,
    sort: usize,
    kind: PackKind,
    category: String,
    details: Option<String>,
    message: String,
    task: Option<Task>,
    action: Option<Action>,
    artwork: artwork::Artwork,
    artwork_progress: Option<(usize, usize)>,
    confirm_remove: Option<String>,
    remove_seen: Option<String>,
    arrivals: BTreeMap<String, f64>,
    filter_state: Option<(String, usize, usize, String, u64, PackKind)>,
    results: Vec<Manifest>,
    categories: Vec<String>,
}
impl Default for Browser {
    fn default() -> Self {
        let at = p::root();
        let prefs = Preferences::load(&at);
        Self {
            open: false,
            catalogue: cached(&at),
            installed: installed(&at, &prefs),
            prefs,
            search: String::new(),
            view: 0,
            sort: 0,
            kind: PackKind::All,
            category: String::new(),
            details: None,
            message: String::new(),
            task: None,
            action: None,
            artwork: artwork::Artwork::default(),
            artwork_progress: None,
            confirm_remove: None,
            remove_seen: None,
            arrivals: BTreeMap::new(),
            filter_state: None,
            results: Vec::new(),
            categories: Vec::new(),
        }
    }
}
pub fn state(ctx: &Context) -> Arc<Mutex<Browser>> {
    let id = egui::Id::new("community_plugin_browser");
    if let Some(s) = ctx.data(|d| d.get_temp::<Arc<Mutex<Browser>>>(id)) {
        return s;
    }
    let s = Arc::new(Mutex::new(Browser::default()));
    ctx.data_mut(|d| d.insert_temp(id, s.clone()));
    s
}
pub fn is_open(ctx: &Context) -> bool {
    ctx.data(|d| d.get_temp::<Arc<Mutex<Browser>>>(egui::Id::new("community_plugin_browser")))
        .is_some_and(|state| state.lock().is_ok_and(|s| s.open))
}
pub fn open(ctx: &Context) {
    if let Ok(mut s) = state(ctx).lock() {
        s.open = true;
    }
}
pub fn reload(ctx: &Context) {
    if let Ok(mut s) = state(ctx).lock() {
        s.reload();
    }
}
impl Browser {
    fn reload(&mut self) {
        self.prefs = Preferences::load(&p::root());
        self.installed = installed(&p::root(), &self.prefs);
        self.artwork.clear();
        self.filter_state = None;
    }
    fn save(&mut self) {
        self.filter_state = None;
        if let Err(e) = self.prefs.save(&p::root()) {
            self.message = e;
        }
    }
    fn refresh(&mut self, ctx: &Context) {
        if self.task.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = (|| {
                let b = fetch(CATALOGUE_URL, MAX_CATALOGUE, &stop)?;
                let c = catalogue(&b)?;
                if cached(&p::root()).is_some_and(|old| old.sequence > c.sequence) {
                    return Err("The catalogue is older than the cached copy.".into());
                }
                atomic_write(&p::root().join("catalogue.json"), &b)?;
                let total = c.packs.len().min(12);
                let _ = tx.send(Ok(ResultMessage::Refreshed(c.clone())));
                ctx.request_repaint();
                // Artwork belongs to this explicit refresh, never to background browsing.
                for (index, listing) in c.packs.iter().take(12).enumerate() {
                    for (url, path) in [
                        (&listing.screenshot_url, &listing.manifest.screenshot),
                        (&listing.icon_url, &listing.manifest.icon),
                    ] {
                        if url.is_empty() || path.is_empty() {
                            continue;
                        }
                        let Some(file) = listing.manifest.files.iter().find(|f| f.path == *path)
                        else {
                            continue;
                        };
                        let dest = p::root().join("artwork").join(&file.sha256);
                        if read_limited(&dest, 4 * 1024 * 1024)
                            .is_ok_and(|b| hash(&b) == file.sha256)
                        {
                            continue;
                        }
                        if let Ok(bytes) = fetch(url, 4 * 1024 * 1024, &stop)
                            && hash(&bytes) == file.sha256
                            && validate_image(&bytes).is_ok()
                        {
                            atomic_write(&dest, &bytes)?;
                        }
                    }
                    if stop.load(Ordering::Relaxed) {
                        return Err(tr("cancelled"));
                    }
                    let _ = tx.send(Ok(ResultMessage::Artwork {
                        done: index + 1,
                        total,
                    }));
                    ctx.request_repaint();
                }
                Ok(ResultMessage::Complete)
            })();
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        self.task = Some(Task {
            receiver: rx,
            cancel,
        });
        self.message.clear();
        self.artwork_progress = None;
    }
    fn download_preview(&mut self, listing: Listing, ctx: &Context) {
        if self.task.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = (|| {
                for (url, path) in [
                    (&listing.screenshot_url, &listing.manifest.screenshot),
                    (&listing.icon_url, &listing.manifest.icon),
                ] {
                    if url.is_empty() || path.is_empty() {
                        continue;
                    }
                    let file = listing
                        .manifest
                        .files
                        .iter()
                        .find(|f| f.path == *path)
                        .ok_or("Missing artwork metadata")?;
                    let bytes = fetch(url, 4 * 1024 * 1024, &stop)?;
                    if hash(&bytes) != file.sha256 {
                        return Err("Preview checksum mismatch".into());
                    }
                    validate_image(&bytes)?;
                    atomic_write(&p::root().join("artwork").join(&file.sha256), &bytes)?;
                }
                let _ = tx.send(Ok(ResultMessage::Artwork { done: 1, total: 1 }));
                Ok(ResultMessage::Complete)
            })();
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        self.task = Some(Task {
            receiver: rx,
            cancel,
        });
    }
    fn install(&mut self, listing: Listing, ctx: &Context) {
        if self.task.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = fetch(&listing.url, MAX_PACKAGE, &stop)
                .and_then(|b| {
                    if stop.load(Ordering::Relaxed) {
                        return Err(tr("cancelled"));
                    }
                    install(&b, &p::root(), false, Some(&listing))
                })
                .map(ResultMessage::Installed);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        self.task = Some(Task {
            receiver: rx,
            cancel,
        });
        self.message.clear();
    }
    fn texture(&mut self, ctx: &Context, m: &Manifest, icon: bool) -> Option<egui::TextureHandle> {
        let _ = ctx;
        self.artwork.request(m, &self.installed, icon)
    }

    fn update_results(&mut self) {
        let key = (
            self.search.clone(),
            self.view,
            self.sort,
            self.category.clone(),
            self.catalogue.as_ref().map(|c| c.sequence).unwrap_or(0),
            self.kind,
        );
        if self.filter_state.as_ref() == Some(&key) {
            return;
        }
        let search = self.search.to_lowercase();
        let mut packs = self.manifests();
        let mut tags: Vec<String> = packs.iter().flat_map(|m| m.tags.clone()).collect();
        tags.sort();
        tags.dedup();
        tags.insert(0, String::new());
        self.categories = tags;
        packs.retain(|m| {
            (self.kind == PackKind::All || self.kind == m.kind)
                && (self.view != 1 || self.installed.iter().any(|p| p.manifest.id == m.id))
                && (self.view != 2 || self.prefs.packs.get(&m.id).is_some_and(|p| p.favourite))
                && (self.category.is_empty() || m.tags.contains(&self.category))
                && format!(
                    "{} {} {} {}",
                    m.name,
                    m.author,
                    m.description,
                    m.tags.join(" ")
                )
                .to_lowercase()
                .contains(&search)
        });
        let scores: BTreeMap<_, _> = self
            .catalogue
            .as_ref()
            .map(|c| {
                c.packs
                    .iter()
                    .map(|l| {
                        (
                            l.manifest.id.as_str(),
                            match self.sort {
                                0 => l.downloads,
                                1 => l.recent_downloads,
                                2 => l.published,
                                _ => l.updated,
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        packs.sort_by(|a, b| {
            scores
                .get(b.id.as_str())
                .copied()
                .unwrap_or(0)
                .cmp(&scores.get(a.id.as_str()).copied().unwrap_or(0))
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
        self.results = packs;
        self.filter_state = Some(key);
    }
    fn listing(&self, id: &str) -> Option<Listing> {
        self.catalogue
            .as_ref()?
            .packs
            .iter()
            .find(|p| p.manifest.id == id)
            .cloned()
    }
    fn manifests(&self) -> Vec<Manifest> {
        let mut all: BTreeMap<String, Manifest> = self
            .catalogue
            .as_ref()
            .map(|c| {
                c.packs
                    .iter()
                    .map(|l| (l.manifest.id.clone(), l.manifest.clone()))
                    .collect()
            })
            .unwrap_or_default();
        for p in &self.installed {
            all.entry(p.manifest.id.clone())
                .or_insert_with(|| p.manifest.clone());
        }
        all.into_values().collect()
    }
}
pub fn menu(ui: &mut Ui, can_run: bool) {
    let ctx = ui.ctx().clone();
    let arc = state(&ctx);
    let Ok(mut s) = arc.lock() else { return };
    s.artwork.poll(&ctx);
    let packs = s.installed.clone();
    let mut shown = false;
    for pack in packs {
        if pack.manifest.kind != PackKind::Effects
            || pack.manifest.effects.iter().all(|e| {
                s.prefs
                    .packs
                    .get(&pack.manifest.id)
                    .is_some_and(|p| p.disabled_effects.contains(&e.id))
            })
        {
            continue;
        }
        if s.prefs
            .packs
            .get(&pack.manifest.id)
            .is_some_and(|p| p.disabled)
        {
            continue;
        }
        let label = s
            .prefs
            .packs
            .get(&pack.manifest.id)
            .map(|p| p.alias.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(&pack.manifest.name)
            .to_string();
        shown = true;
        ui.menu_button(&label, |ui| {
            let mut categories: BTreeMap<String, Vec<Effect>> = BTreeMap::new();
            for e in &pack.manifest.effects {
                if s.prefs
                    .packs
                    .get(&pack.manifest.id)
                    .is_some_and(|p| p.disabled_effects.contains(&e.id))
                {
                    continue;
                }
                let category = s
                    .prefs
                    .packs
                    .get(&pack.manifest.id)
                    .and_then(|p| p.menus.get(&e.id))
                    .filter(|v| !v.trim().is_empty())
                    .unwrap_or(&e.category)
                    .clone();
                categories.entry(category).or_default().push(e.clone());
            }
            let single_category = categories.len() == 1;
            for (category, effects) in categories {
                let contents = |ui: &mut Ui| {
                    for e in effects {
                        let dark = ui.visuals().dark_mode;
                        let path = if dark {
                            &pack.manifest.menu_dark
                        } else {
                            &pack.manifest.menu_light
                        };
                        let texture = s.artwork.request_at(
                            &pack.manifest,
                            std::slice::from_ref(&pack),
                            path,
                            true,
                        );
                        let display_name = s
                            .prefs
                            .packs
                            .get(&pack.manifest.id)
                            .and_then(|p| p.names.get(&e.id))
                            .filter(|name| !name.trim().is_empty())
                            .cloned()
                            .unwrap_or_else(|| e.name.clone());
                        let button = if let Some(t) = texture {
                            egui::Button::image_and_text(
                                egui::Image::new((t.id(), Vec2::splat(16.))),
                                &display_name,
                            )
                        } else {
                            egui::Button::new(&display_name)
                        };
                        let response = ui.add_enabled(can_run, button);
                        if response.clicked() {
                            s.action = Some(Action::Run(pack.clone(), e.clone()));
                            ui.close();
                        }
                        response.context_menu(|ui| {
                            if ui.button(tr("manage")).clicked() {
                                s.details = Some(pack.manifest.id.clone());
                                s.open = true;
                                ui.close();
                            }
                            if ui.button(tr("edit_copy")).clicked() {
                                s.action = Some(Action::Edit(pack.clone()));
                                ui.close();
                            }
                        });
                    }
                };
                if single_category && category == label {
                    contents(ui);
                } else {
                    ui.menu_button(category, contents);
                }
            }
        });
    }
    s.artwork.start_batch(&ctx);
    if shown {
        ui.separator();
    }
}
pub fn frame(ctx: &Context) -> Option<Action> {
    let arc =
        ctx.data(|d| d.get_temp::<Arc<Mutex<Browser>>>(egui::Id::new("community_plugin_browser")))?;
    let Ok(mut s) = arc.lock() else { return None };
    s.artwork.poll(ctx);
    for _ in 0..4 {
        let Some(message) = s.task.as_ref().map(|task| task.receiver.try_recv()) else {
            break;
        };
        match message {
            Ok(Ok(ResultMessage::Refreshed(catalogue))) => {
                s.catalogue = Some(catalogue);
                s.filter_state = None;
                s.artwork.retry_missing();
                s.message = tr("refreshed");
            }
            Ok(Ok(ResultMessage::Artwork { done, total })) => {
                s.artwork.retry_missing();
                s.artwork_progress = Some((done, total));
            }
            Ok(Ok(ResultMessage::Complete)) => {
                s.task = None;
                s.artwork_progress = None;
            }
            Ok(Ok(ResultMessage::Installed(manifest))) => {
                s.task = None;
                s.message = format!("{} {}", manifest.name, tr("installed"));
                s.reload();
            }
            Ok(Err(error)) => {
                s.task = None;
                s.artwork_progress = None;
                s.message = error;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                s.task = None;
                s.message = tr("task_failed");
            }
            Err(mpsc::TryRecvError::Empty) => break,
        }
    }
    if s.task.is_some() {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
    s.artwork.start_batch(ctx);
    if !s.open {
        return s.action.take();
    }
    let mut opened = true;
    let bounds = ctx.content_rect();
    let mut close = false;
    let window = crate::ui::polish::window(ctx, tr("title"))
        .id(egui::Id::new("plugin_browser_window"))
        .heading_drag(
            ctx,
            egui::Id::new("plugin_browser_heading"),
            egui::pos2(
                (bounds.center().x - 530.).max(bounds.left() + 16.),
                (bounds.center().y - 360.).max(bounds.top() + 16.),
            ),
        )
        .title_bar(false)
        .collapsible(false)
        .open(&mut opened)
        .default_pos(egui::pos2(
            (bounds.center().x - 530.).max(bounds.left() + 16.),
            (bounds.center().y - 390.).max(bounds.top() + 16.),
        ))
        .constrain_to(bounds.shrink(12.))
        .default_size([1060., 720.])
        .min_size([440., 340.])
        .max_size([bounds.width() - 40., bounds.height() - 80.])
        .show(ctx, |ui| {
            let colours = crate::ui::dialogs::core::DialogColors::from_ctx(ctx);
            if crate::ui::dialogs::core::paint_dialog_header(ui, &colours, "", &tr("title")) {
                close = true;
            }
            ui.add_space(8.);
            let action_row = ui.horizontal(|ui| {
                if ui.button(tr("import")).clicked()
                    && let Some(path) = crate::services::file_dialogs::dialog(
                        crate::services::file_dialogs::Purpose::Resource,
                        None,
                    )
                    .add_filter("PaintFE plugin", &["pfeplugin"])
                    .pick_file()
                {
                    s.message = match read_limited(&path, MAX_PACKAGE)
                        .and_then(|b| install(&b, &p::root(), true, None))
                    {
                        Ok(m) => format!("{} {}", m.name, tr("installed")),
                        Err(e) => e,
                    };
                    s.reload();
                }
                if ui.button(tr("create")).clicked() {
                    s.action = Some(Action::Create);
                }
                if let Some(c) = &s.catalogue {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(format!("{} {}", tr("updated"), timestamp(c.updated)));
                    });
                }
            });
            #[cfg(test)]
            ctx.data_mut(|d| {
                d.insert_temp(
                    egui::Id::new("plugin_browser_actions"),
                    action_row.response.rect,
                )
            });
            #[cfg(not(test))]
            let _ = action_row;

            ui.separator();
            ui.allocate_ui_with_layout(
                Vec2::new(ui.available_width(), 28.),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    let working = s.task.is_some() || s.artwork.is_busy();
                    let width = (ui.available_width() - 76.).max(20.);
                    ui.allocate_ui_with_layout(
                        Vec2::new(width, 28.),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.set_min_width(width);
                            if working {
                                ui.spinner();
                                ui.label(if let Some((done, total)) = s.artwork_progress {
                                    format!("{} {done}/{total}", tr("working"))
                                } else {
                                    tr("working")
                                });
                            } else if !s.message.is_empty() {
                                ui.add(egui::Label::new(&s.message).truncate());
                            } else if s.catalogue.is_some() {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(tr("offline_notice")).weak(),
                                    )
                                    .truncate(),
                                );
                            }
                        },
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if s.task.is_some() {
                            if ui.button(tr("cancel")).clicked()
                                && let Some(task) = &s.task
                            {
                                task.cancel.store(true, Ordering::Relaxed);
                            }
                        } else if s.catalogue.is_some()
                            && !s.artwork.is_busy()
                            && ui.button(tr("refresh")).clicked()
                        {
                            s.refresh(ctx);
                        }
                    });
                },
            );
            let available = ui.available_width();
            let height =
                (ui.available_height() - 52.).clamp(150., (bounds.height() - 300.).max(150.));
            ui.horizontal_top(|ui| {
                let sidebar = available >= 700.;
                if sidebar {
                    ui.allocate_ui_with_layout(
                        Vec2::new(155., height),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.set_min_width(155.);
                            ui.set_min_height(height);
                            for (i, key) in ["discover", "installed_tab", "favourites"]
                                .iter()
                                .enumerate()
                            {
                                if ui
                                    .selectable_label(
                                        s.view == i,
                                        if i == 1 {
                                            format!("{} ({})", tr(key), s.installed.len())
                                        } else {
                                            tr(key)
                                        },
                                    )
                                    .clicked()
                                {
                                    s.view = i;
                                    s.details = None;
                                }
                            }
                            ui.add_space(18.);
                            ui.strong(tr("categories"));
                            s.update_results();
                            let tags = s.categories.clone();
                            egui::ScrollArea::vertical()
                                .id_salt("plugin_categories")
                                .max_height((height - 150.).max(80.))
                                .show(ui, |ui| {
                                    for tag in tags {
                                        let label = if tag.is_empty() {
                                            tr("all")
                                        } else {
                                            tag.clone()
                                        };
                                        if ui.selectable_label(s.category == tag, label).clicked() {
                                            s.category = tag;
                                            s.details = None;
                                        }
                                    }
                                });
                        },
                    );
                    ui.separator();
                }
                ui.vertical(|ui| {
                    // Use the actual remaining width after the sidebar and
                    // separator. A guessed subtraction grows the window on
                    // every layout pass when spacing differs.
                    ui.set_width(ui.available_width());
                    if !sidebar {
                        ui.horizontal_wrapped(|ui| {
                            for (i, key) in ["discover", "installed_tab", "favourites"]
                                .iter()
                                .enumerate()
                            {
                                let label = if i == 1 {
                                    format!("{} ({})", tr(key), s.installed.len())
                                } else {
                                    tr(key)
                                };
                                ui.selectable_value(&mut s.view, i, label);
                            }
                        });
                    }
                    if !sidebar {
                        s.update_results();
                        egui::ComboBox::from_id_salt("plugin_category_compact")
                            .selected_text(if s.category.is_empty() {
                                tr("all")
                            } else {
                                s.category.clone()
                            })
                            .show_ui(ui, |ui| {
                                for category in s.categories.clone() {
                                    let text = if category.is_empty() {
                                        tr("all")
                                    } else {
                                        category.clone()
                                    };
                                    ui.selectable_value(&mut s.category, category, text);
                                }
                            });
                    }
                    if let Some(id) = s.details.clone() {
                        details(ui, &mut s, &id, ctx, height);
                    } else {
                        grid(ui, &mut s, ctx, height);
                    }
                });
            });
        });
    s.artwork.start_batch(ctx);
    if let Some(window) = window {
        #[cfg(test)]
        ctx.data_mut(|d| {
            d.insert_temp(egui::Id::new("plugin_browser_bounds"), window.response.rect)
        });
        #[cfg(not(test))]
        let _ = window;
    }
    if let Some(id) = s.confirm_remove.clone() {
        let first = s.remove_seen.as_deref() != Some(id.as_str());
        let mut cancel = false;
        let mut accepted = false;
        egui::Modal::new(egui::Id::new("community_plugin_remove")).show(ctx, |ui| {
            ui.set_width((bounds.width() - 80.).clamp(280., 480.));
            let colours = crate::ui::dialogs::core::DialogColors::from_ctx(ctx);
            if first {
                ctx.data_mut(|data| {
                    data.insert_temp(egui::Id::new("dialog_footer_keys"), (false, false))
                });
                ctx.request_repaint();
            }
            cancel |= crate::ui::dialogs::core::paint_dialog_header(
                ui,
                &colours,
                "",
                &tr("confirm_remove"),
            );
            ui.label(
                s.installed
                    .iter()
                    .find(|p| p.manifest.id == id)
                    .map(|p| p.manifest.name.as_str())
                    .unwrap_or(&id),
            );
            ui.label(tr("remove_notice"));
            let (confirm, dismiss) = crate::ui::dialogs::core::dialog_footer_action(
                ui,
                &colours,
                &tr("uninstall"),
                !first,
            );
            accepted = confirm;
            cancel |= dismiss;
        });
        s.remove_seen = Some(id.clone());
        if accepted {
            let removed = s.installed.iter().find(|p| p.manifest.id == id).cloned();
            s.message = match uninstall(&id, &p::root()) {
                Ok(()) => {
                    if let Some(pack) = removed
                        && pack.manifest.kind == PackKind::Icons
                    {
                        s.action = Some(Action::RemovedAppearance(pack));
                    }
                    tr("removed")
                }
                Err(error) => error,
            };
            s.reload();
            cancel = true;
        }
        if cancel {
            s.confirm_remove = None;
            s.remove_seen = None;
        }
    }
    opened &= !close;
    s.open = opened;
    if !opened && let Some(t) = &s.task {
        t.cancel.store(true, Ordering::Relaxed);
    }
    let action = s.action.take();
    if action.is_some() {
        s.open = false;
    }
    action
}
pub fn timestamp(t: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let seconds = now.saturating_sub(t);
    if seconds < 60 {
        tr("just_now")
    } else if seconds < 3600 {
        format!("{} {}", seconds / 60, tr("minutes_ago"))
    } else if seconds < 86400 {
        format!("{} {}", seconds / 3600, tr("hours_ago"))
    } else {
        format!("{} {}", seconds / 86400, tr("days_ago"))
    }
}
fn grid(ui: &mut Ui, s: &mut Browser, ctx: &Context, height: f32) {
    #[cfg(test)]
    ctx.data_mut(|data| {
        data.insert_temp(
            egui::Id::new("addon_card_geometry"),
            Vec::<(egui::Rect, egui::Rect)>::new(),
        )
    });
    ui.horizontal_wrapped(|ui| {
        for (kind, label) in [
            (PackKind::All, "packs.all"),
            (PackKind::Effects, "packs.effects"),
            (PackKind::Theme, "packs.themes"),
            (PackKind::Icons, "packs.icons"),
        ] {
            ui.selectable_value(&mut s.kind, kind, crate::i18n::translate(label));
        }
    });
    let compact = ui.available_width() < 700.;
    ui.horizontal(|ui| {
        if compact {
            egui::ComboBox::from_id_salt("plugin_sort")
                .selected_text(tr(["popular", "trending", "new", "updated_sort"][s.sort]))
                .show_ui(ui, |ui| {
                    for (index, key) in ["popular", "trending", "new", "updated_sort"]
                        .iter()
                        .enumerate()
                    {
                        ui.selectable_value(&mut s.sort, index, tr(key));
                    }
                });
        } else {
            for (index, key) in ["popular", "trending", "new", "updated_sort"]
                .iter()
                .enumerate()
            {
                ui.selectable_value(&mut s.sort, index, tr(key));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut s.search)
                        .hint_text(tr("search"))
                        .desired_width(ui.available_width().clamp(120., 260.)),
                );
            });
        }
    });
    if compact {
        ui.add(
            egui::TextEdit::singleline(&mut s.search)
                .hint_text(tr("search"))
                .desired_width(ui.available_width()),
        );
    }
    ui.add_space(12.);
    s.update_results();
    if s.results.is_empty() || (s.catalogue.is_none() && s.view == 0) {
        if empty_state(ui, s, ctx, height) {
            s.refresh(ctx);
        }
        return;
    }
    let content_width = (ui.available_width()
        - ui.spacing().scroll.bar_width
        - ui.spacing().scroll.bar_inner_margin)
        .max(1.);
    let columns = ((content_width + 12.) / 268.).floor().max(1.) as usize;
    let width = ((content_width - (columns - 1) as f32 * 12.) / columns as f32).max(1.);
    let row_height =
        (width - 26.) * 0.5625 + ui.text_style_height(&egui::TextStyle::Body) * 8. + 90.;
    egui::ScrollArea::vertical()
        .id_salt("plugin_cards")
        .auto_shrink([false, false])
        .max_height((ui.available_height().min(height) - 8.).max(80.))
        .show_rows(
            ui,
            row_height,
            s.results.len().div_ceil(columns),
            |ui, range| {
                for index in range {
                    let row = s.results
                        [index * columns..((index + 1) * columns).min(s.results.len())]
                        .to_vec();
                    ui.horizontal_top(|ui| {
                        ui.spacing_mut().item_spacing.x = 12.;
                        for manifest in &row {
                            let (rect, _) = ui.allocate_exact_size(
                                Vec2::new(width, row_height - 12.),
                                egui::Sense::hover(),
                            );
                            let mut child = ui.new_child(
                                egui::UiBuilder::new()
                                    .id_salt(&manifest.id)
                                    .max_rect(rect)
                                    .layout(egui::Layout::top_down(egui::Align::Min)),
                            );
                            child.set_clip_rect(ui.clip_rect().intersect(rect));
                            let _actual = card(&mut child, s, manifest, ctx, width, row_height);
                            #[cfg(test)]
                            ctx.data_mut(|data| {
                                data.get_temp_mut_or_default::<Vec<(egui::Rect, egui::Rect)>>(
                                    egui::Id::new("addon_card_geometry"),
                                )
                                .push((rect, _actual))
                            });
                        }
                    });
                    ui.add_space(12.);
                }
            },
        );
}
fn empty_state(ui: &mut Ui, s: &mut Browser, ctx: &Context, height: f32) -> bool {
    let unfetched = s.catalogue.is_none() && s.view == 0;
    ui.add_space((height * 0.16).clamp(20., 80.));
    let mut requested = false;
    ui.vertical_centered(|ui| {
        ui.heading(tr(if unfetched { "discover_title" } else { "empty" }));
        ui.add_space(8.);
        ui.label(tr(if unfetched {
            "discover_body"
        } else {
            "empty_body"
        }));
        ui.add_space(16.);
        if unfetched {
            if s.task.is_some() {
                ui.spinner();
                ui.label(tr("working"));
            }
            let colours = crate::ui::dialogs::core::DialogColors::from_ctx(ctx);
            let response = ui.add_enabled(
                s.task.is_none(),
                egui::Button::new(egui::RichText::new(tr("fetch_plugins")).strong().color(
                    crate::ui::dialogs::core::contrast_text_color(colours.accent),
                ))
                .fill(colours.accent)
                .min_size(Vec2::new(200., 42.)),
            );
            #[cfg(test)]
            ctx.data_mut(|d| d.insert_temp(egui::Id::new("plugin_fetch_button"), response.rect));
            if response.clicked() {
                requested = true;
            }
            ui.add_space(10.);
            ui.weak(tr("fetch_notice"));
        } else if (!s.search.is_empty() || !s.category.is_empty())
            && ui.button(tr("clear_filters")).clicked()
        {
            s.search.clear();
            s.category.clear();
        }
    });
    requested
}

fn card(
    ui: &mut Ui,
    s: &mut Browser,
    m: &Manifest,
    ctx: &Context,
    width: f32,
    row_height: f32,
) -> egui::Rect {
    let now = ui.input(|i| i.time);
    let duration = crate::ui::polish::duration(ctx, crate::ui::polish::MotionKind::Rows) as f64;
    let first = *s.arrivals.entry(m.id.clone()).or_insert(now);
    let progress = if duration > 0. {
        ((now - first) / duration).clamp(0., 1.) as f32
    } else {
        1.
    };
    if progress < 1. {
        ctx.request_repaint();
    }
    let response = egui::Frame::group(ui.style())
        .fill(ui.visuals().widgets.inactive.weak_bg_fill)
        .shadow(egui::epaint::Shadow {
            offset: [0, 2],
            blur: ui.visuals().window_shadow.blur.min(8),
            spread: 0,
            color: ui.visuals().window_shadow.color.gamma_multiply(0.35),
        })
        .inner_margin(12.)
        .corner_radius(ui.visuals().window_corner_radius)
        .show(ui, |ui| {
            ui.set_opacity(0.55 + 0.45 * progress);
            ui.set_width((width - 26.).max(120.));
            ui.set_min_height(row_height - 38.);
            let size = Vec2::new(width - 26., (width - 26.) * 0.5625);
            let (image_rect, image) = ui.allocate_exact_size(size, egui::Sense::click());
            ui.painter().rect_filled(
                image_rect,
                ui.visuals().window_corner_radius,
                ui.visuals().faint_bg_color,
            );
            if let Some(texture) = s.texture(ctx, m, false) {
                let texture_size = texture.size_vec2();
                let fit = (size.x / texture_size.x).min(size.y / texture_size.y);
                let ease = 1. - (1. - progress).powi(3);
                let rect = egui::Rect::from_center_size(
                    image_rect.center(),
                    texture_size * fit * (0.96 + 0.04 * ease),
                );
                ui.painter().image(
                    texture.id(),
                    rect,
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                    egui::Color32::WHITE,
                );
            }
            if image.clicked() {
                s.details = Some(m.id.clone());
            }
            ui.add_space(6.);
            ui.horizontal(|ui| {
                if let Some(icon) = s.texture(ctx, m, true) {
                    ui.image((icon.id(), Vec2::splat(24.)));
                }
                let title = ui.add_sized(
                    [
                        (ui.available_width() - 32. - ui.spacing().item_spacing.x).max(40.),
                        32.,
                    ],
                    egui::Button::new(
                        RichText::new(&m.name)
                            .strong()
                            .text_style(egui::TextStyle::Heading),
                    )
                    .frame(false)
                    .truncate(),
                );
                if title.clicked() {
                    s.details = Some(m.id.clone());
                }
                title.on_hover_text(&m.name);
                let mut fav = s.prefs.packs.get(&m.id).is_some_and(|p| p.favourite);
                let icon = artwork::favourite(ctx, fav);
                let tint = if fav {
                    ui.visuals().selection.stroke.color
                } else {
                    ui.visuals().text_color()
                };
                let favourite = ui
                    .add_sized(
                        [32., 32.],
                        egui::Button::image(
                            egui::Image::new((icon.id(), Vec2::splat(20.)))
                                .alt_text(tr("favourite"))
                                .tint(tint),
                        )
                        .selected(fav),
                    )
                    .on_hover_text(tr("favourite"));
                #[cfg(test)]
                ctx.data_mut(|data| {
                    data.insert_temp(egui::Id::new(("addon_favourite", &m.id)), favourite.rect);
                });
                if favourite.clicked() {
                    fav = !fav;
                    s.prefs.packs.entry(m.id.clone()).or_default().favourite = fav;
                    s.save();
                }
            });
            ui.add(
                egui::Label::new(
                    RichText::new(format!("{} · {}", m.author, m.kind.label())).weak(),
                )
                .truncate(),
            );
            let mut description = egui::text::LayoutJob::simple(
                m.description.clone(),
                ui.style().text_styles[&egui::TextStyle::Body].clone(),
                ui.visuals().text_color(),
                ui.available_width(),
            );
            description.wrap.max_rows = 2;
            ui.label(description);
            if let Some(listing) = s.listing(&m.id) {
                ui.weak(if s.sort == 1 {
                    format!(
                        "{} {} · {} {}",
                        listing.recent_downloads,
                        tr("downloads"),
                        listing.window_days,
                        tr(if listing.window_days == 1 {
                            "day"
                        } else {
                            "days"
                        })
                    )
                } else {
                    format!("{} {}", listing.downloads, tr("downloads"))
                });
            }
            if let Some(installed) = s.installed.iter().find(|p| p.manifest.id == m.id).cloned() {
                let label = if s.prefs.packs.get(&m.id).is_some_and(|p| p.disabled) {
                    "disabled"
                } else if version_key(&m.version) > version_key(&installed.manifest.version) {
                    "update_available"
                } else {
                    "installed"
                };
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(tr(label))
                            .strong()
                            .color(ui.visuals().selection.stroke.color),
                    );
                    let mut enabled = !s.prefs.packs.get(&m.id).is_some_and(|p| p.disabled);
                    if m.kind == PackKind::Effects
                        && ui.checkbox(&mut enabled, tr("enabled")).changed()
                    {
                        s.prefs.packs.entry(m.id.clone()).or_default().disabled = !enabled;
                        s.save();
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    if m.kind != PackKind::Effects
                        && ui.button(crate::t!("packs.preview_apply")).clicked()
                    {
                        s.action = Some(Action::Appearance(installed.clone()));
                        s.open = false;
                    }
                    if ui.button(tr("manage")).clicked() {
                        s.details = Some(m.id.clone());
                    }
                    if ui.button(tr("uninstall")).clicked() {
                        s.confirm_remove = Some(m.id.clone());
                    }
                });
            } else if let Some(listing) = s.listing(&m.id)
                && ui
                    .add_enabled(
                        s.task.is_none() && m.api == 1,
                        egui::Button::new(tr("install")),
                    )
                    .on_hover_text(tr("install_notice"))
                    .clicked()
            {
                s.install(listing, ctx);
            }
        })
        .response;
    response.context_menu(|ui| {
        if ui.button(tr("manage")).clicked() {
            s.details = Some(m.id.clone());
            ui.close();
        }
        if m.kind == PackKind::Effects
            && let Some(pack) = s.installed.iter().find(|p| p.manifest.id == m.id).cloned()
            && ui.button(tr("edit_copy")).clicked()
        {
            s.action = Some(Action::Edit(pack));
            ui.close();
        }
    });
    response.rect
}

fn details(ui: &mut Ui, s: &mut Browser, id: &str, ctx: &Context, height: f32) {
    if ui.button(format!("‹ {}", tr("back"))).clicked() {
        s.details = None;
        return;
    }
    let Some(m) = s.manifests().into_iter().find(|m| m.id == id) else {
        return;
    };
    let current = s.installed.iter().find(|p| p.manifest.id == id).cloned();
    egui::ScrollArea::vertical()
        .id_salt(("plugin_details", id))
        .max_height((height - 34.).max(120.))
        .show(ui, |ui| {
            ui.heading(&m.name);
            if m.api != 1 {
                ui.weak(format!("{} {}", tr("requires_api"), m.api));
            }
            ui.weak(format!("{} · {} · {}", m.author, m.version, m.license));
            if let Some(tex) = s.texture(ctx, &m, false) {
                ui.add(
                    egui::Image::new(&tex)
                        .max_width(ui.available_width().min(680.))
                        .maintain_aspect_ratio(true),
                );
            } else if let Some(listing) = s.listing(id)
                && ui
                    .add_enabled(s.task.is_none(), egui::Button::new(tr("load_preview")))
                    .on_hover_text(tr("connect_notice"))
                    .clicked()
            {
                s.download_preview(listing, ctx);
            }
            ui.add_space(8.);
            ui.label(&m.description);
            if let Some(l) = s.listing(id) {
                ui.weak(format!(
                    "{} {} · {} {} ({} {})",
                    l.downloads,
                    tr("downloads"),
                    l.recent_downloads,
                    tr("recent"),
                    l.window_days,
                    tr("days")
                ));
            }
            ui.horizontal_wrapped(|ui| {
                if let Some(l) = s.listing(id) {
                    let newer = current.as_ref().is_none_or(|p| {
                        version_key(&l.manifest.version) > version_key(&p.manifest.version)
                    });
                    if ui
                        .add_enabled(
                            s.task.is_none() && m.api == 1,
                            egui::Button::new(tr(if current.is_none() {
                                "install"
                            } else if newer {
                                "update"
                            } else {
                                "repair"
                            })),
                        )
                        .clicked()
                    {
                        s.install(l, ctx);
                    }
                }
                if let Some(pack) = &current {
                    if m.kind == PackKind::Effects && ui.button(tr("edit_copy")).clicked() {
                        s.action = Some(Action::Edit(pack.clone()));
                    }
                    if ui.button(tr("export")).clicked() {
                        s.action = Some(Action::Export(pack.clone()));
                    }
                }
                ui.hyperlink_to(tr("source"), REPOSITORY);
                ui.hyperlink_to(
                    tr("report"),
                    format!("{REPOSITORY}/issues/new?title=Plugin%20report%3A%20{id}"),
                );
                ui.hyperlink_to(tr("discussion"), format!("{REPOSITORY}/discussions"));
            });
            if let Some(pack) = current {
                if !pack.signed {
                    ui.weak(tr("local_pack"));
                }
                let settings = s.prefs.packs.entry(id.to_string()).or_default();
                let mut enabled = !settings.disabled;
                if m.kind == PackKind::Effects && ui.checkbox(&mut enabled, tr("enabled")).changed()
                {
                    settings.disabled = !enabled;
                }
                ui.horizontal(|ui| {
                    ui.label(tr("local_name"));
                    ui.text_edit_singleline(&mut settings.alias);
                });
                ui.separator();
                ui.strong(tr("included"));
                for effect in &pack.manifest.effects {
                    let mut enabled = !s
                        .prefs
                        .packs
                        .get(id)
                        .unwrap()
                        .disabled_effects
                        .contains(&effect.id);
                    ui.push_id(&effect.id, |ui| {
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut enabled, &effect.name).changed() {
                                let p = s.prefs.packs.get_mut(id).unwrap();
                                if enabled {
                                    p.disabled_effects.remove(&effect.id);
                                } else {
                                    p.disabled_effects.insert(effect.id.clone());
                                }
                            }
                            if ui
                                .add_enabled(enabled, egui::Button::new(tr("run")))
                                .clicked()
                            {
                                s.action = Some(Action::Run(pack.clone(), effect.clone()));
                            }
                        });
                        ui.weak(&effect.description);
                        let mut display_name = s
                            .prefs
                            .packs
                            .get(id)
                            .and_then(|p| p.names.get(&effect.id))
                            .cloned()
                            .unwrap_or_else(|| effect.name.clone());
                        ui.horizontal(|ui| {
                            ui.label(tr("local_name"));
                            if ui
                                .add(egui::TextEdit::singleline(&mut display_name).char_limit(80))
                                .changed()
                            {
                                let names = &mut s.prefs.packs.get_mut(id).unwrap().names;
                                if display_name.trim().is_empty() || display_name == effect.name {
                                    names.remove(&effect.id);
                                } else {
                                    names.insert(effect.id.clone(), display_name.clone());
                                }
                            }
                        });
                        let menu = s
                            .prefs
                            .packs
                            .get_mut(id)
                            .unwrap()
                            .menus
                            .entry(effect.id.clone())
                            .or_insert_with(|| effect.category.clone());
                        ui.horizontal(|ui| {
                            ui.label(tr("category"));
                            ui.text_edit_singleline(menu);
                        });
                    });
                }
                // Persist only after actual changes, not once per immediate-mode frame.
                let before = ctx
                    .data(|d| d.get_temp::<String>(egui::Id::new(("plugin_prefs_snapshot", id))));
                let after = serde_json::to_string(&s.prefs).unwrap_or_default();
                if before.as_ref() != Some(&after) {
                    s.save();
                    ctx.data_mut(|d| {
                        d.insert_temp(egui::Id::new(("plugin_prefs_snapshot", id)), after)
                    });
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button(tr("uninstall")).clicked() {
                        s.confirm_remove = Some(id.to_string());
                    }
                    let previous = s
                        .prefs
                        .packs
                        .get(id)
                        .map(|p| p.previous.clone())
                        .unwrap_or_default();
                    if !previous.is_empty() && ui.button(tr("rollback")).clicked() {
                        let p = s.prefs.packs.get_mut(id).unwrap();
                        std::mem::swap(&mut p.current, &mut p.previous);
                        s.save();
                        s.reload();
                    }
                });
            } else {
                ui.separator();
                ui.strong(tr("included"));
                for e in &m.effects {
                    ui.label(&e.name);
                    ui.weak(&e.description);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn favourite_card_control_accepts_a_primary_click() {
        crate::i18n::init();
        let ctx = render_context();
        let shared = state(&ctx);
        let catalogue = fixture();
        let id = catalogue.packs[0].manifest.id.clone();
        {
            let mut browser = shared.lock().unwrap();
            browser.open = true;
            browser.catalogue = Some(catalogue);
            browser.prefs = Preferences::default();
        }
        let render = |events| {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1200., 900.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    frame(ui.ctx());
                },
            );
        };
        for _ in 0..4 {
            render(vec![]);
        }
        let rect = ctx
            .data(|data| data.get_temp::<egui::Rect>(egui::Id::new(("addon_favourite", &id))))
            .unwrap();
        let point = rect.center();
        render(vec![egui::Event::PointerMoved(point)]);
        for pressed in [true, false] {
            render(vec![egui::Event::PointerButton {
                pos: point,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            }]);
        }
        assert!(
            shared
                .lock()
                .unwrap()
                .prefs
                .packs
                .get(&id)
                .is_some_and(|pack| pack.favourite)
        );
    }
    #[test]
    fn metadata_is_presented_before_artwork_finishes() {
        let ctx = render_context();
        let shared = state(&ctx);
        let (tx, rx) = mpsc::channel();
        {
            let mut browser = shared.lock().unwrap();
            browser.open = false;
            browser.catalogue = None;
            browser.task = Some(Task {
                receiver: rx,
                cancel: Arc::new(AtomicBool::new(false)),
            });
        }
        tx.send(Ok(ResultMessage::Refreshed(fixture()))).unwrap();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            frame(ui.ctx());
        });
        assert!(shared.lock().unwrap().catalogue.is_some());
        assert!(shared.lock().unwrap().task.is_some());
        tx.send(Ok(ResultMessage::Artwork { done: 1, total: 2 }))
            .unwrap();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            frame(ui.ctx());
        });
        assert_eq!(shared.lock().unwrap().artwork_progress, Some((1, 2)));
        tx.send(Ok(ResultMessage::Complete)).unwrap();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            frame(ui.ctx());
        });
        assert!(shared.lock().unwrap().task.is_none());
    }
    #[test]
    fn populated_installed_grid_keeps_cards_and_window_within_requested_width() {
        crate::i18n::init();
        for width in [640., 1060., 1450.] {
            for zoom in [1., 1.35] {
                let ctx = render_context();
                ctx.set_zoom_factor(zoom);
                let shared = state(&ctx);
                let mut catalogue = fixture();
                let template = catalogue.packs[0].clone();
                catalogue.packs = (0..40)
                    .map(|index| {
                        let mut item = template.clone();
                        item.manifest.id = format!("paintfe.layout-{index}");
                        item.manifest.name = format!("A long effect collection name {index}");
                        item
                    })
                    .collect();
                {
                    let mut s = shared.lock().unwrap();
                    s.open = true;
                    s.view = 1;
                    s.installed = catalogue
                        .packs
                        .iter()
                        .map(|listing| Installed {
                            manifest: listing.manifest.clone(),
                            directory: std::path::PathBuf::from("missing-layout-fixture"),
                            signed: true,
                        })
                        .collect();
                    s.catalogue = Some(catalogue);
                }
                let mut initial = None;
                for tick in 0..6 {
                    let _ = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(width, 800.),
                            )),
                            time: Some(tick as f64),
                            ..Default::default()
                        },
                        |ui| {
                            frame(ui.ctx());
                        },
                    );
                    let bounds = ctx
                        .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("plugin_browser_bounds")))
                        .unwrap();
                    if tick == 2 {
                        initial = Some(bounds.width());
                    }
                    if tick > 2 {
                        assert!(
                            bounds.width() <= initial.unwrap() + 1.,
                            "Window grew after rendering cards: {bounds:?}; initial {initial:?}, viewport {width}, zoom {zoom}"
                        );
                    }
                    if tick > 1 {
                        for (slot, actual) in ctx
                            .data(|d| {
                                d.get_temp::<Vec<(egui::Rect, egui::Rect)>>(egui::Id::new(
                                    "addon_card_geometry",
                                ))
                            })
                            .unwrap_or_default()
                        {
                            assert!(
                                actual.width() <= slot.width() + 1.,
                                "Card exceeds its column: {actual:?} vs {slot:?}"
                            );
                            assert!(
                                actual.height() <= slot.height() + 1.,
                                "Card controls would be clipped: {actual:?} vs {slot:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    fn render_context() -> Context {
        let ctx = Context::default();
        let mut fonts = egui::FontDefinitions::default();
        fonts.families.insert(
            egui::FontFamily::Name("WidgetTitle".into()),
            fonts.families[&egui::FontFamily::Proportional].clone(),
        );
        ctx.set_fonts(fonts);
        ctx
    }
    fn fixture() -> Catalogue {
        catalogue(include_bytes!(
            "../../../tests/fixtures/plugin-catalogue.json"
        ))
        .unwrap()
    }
    #[test]
    fn unopened_browser_is_lazy_and_does_not_create_a_registry() {
        let ctx = Context::default();
        assert!(frame(&ctx).is_none());
        assert!(
            ctx.data(
                |d| d.get_temp::<Arc<Mutex<Browser>>>(egui::Id::new("community_plugin_browser"))
            )
            .is_none()
        );
    }
    #[test]
    fn production_browser_renders_all_views_at_small_large_and_scaled_sizes() {
        crate::i18n::init();
        crate::i18n::set_language("en");
        for dark in [false, true] {
            for (w, h, zoom) in [(1060., 720., 1.), (620., 520., 1.), (960., 720., 1.5)] {
                let ctx = Context::default();
                let mut fonts = egui::FontDefinitions::default();
                fonts.families.insert(
                    egui::FontFamily::Name("WidgetTitle".into()),
                    fonts.families[&egui::FontFamily::Proportional].clone(),
                );
                ctx.set_fonts(fonts);
                if dark {
                    ctx.set_visuals(egui::Visuals::dark());
                } else {
                    ctx.set_visuals(egui::Visuals::light());
                }
                ctx.set_zoom_factor(zoom);
                let shared = state(&ctx);
                {
                    let mut s = shared.lock().unwrap();
                    s.open = true;
                    s.catalogue = Some(fixture());
                    s.installed.clear();
                    s.prefs = Preferences::default();
                }
                let mut tick = 0.0;
                for view in 0..3 {
                    for sort in 0..4 {
                        let mut s = shared.lock().unwrap();
                        s.view = view;
                        s.sort = sort;
                        drop(s);
                        for frame_nr in 0..3 {
                            let output = ctx.run_ui(
                                egui::RawInput {
                                    screen_rect: Some(egui::Rect::from_min_size(
                                        egui::Pos2::ZERO,
                                        egui::vec2(w / zoom, h / zoom),
                                    )),
                                    time: Some(tick),
                                    ..Default::default()
                                },
                                |root_ui| {
                                    let _ = frame(root_ui.ctx());
                                },
                            );
                            tick += 1.;
                            if frame_nr == 2 {
                                assert!(!output.shapes.is_empty());
                                let rect = ctx
                                    .data(|d| {
                                        d.get_temp::<egui::Rect>(egui::Id::new(
                                            "plugin_browser_bounds",
                                        ))
                                    })
                                    .unwrap();
                                let actions = ctx
                                    .data(|d| {
                                        d.get_temp::<egui::Rect>(egui::Id::new(
                                            "plugin_browser_actions",
                                        ))
                                    })
                                    .unwrap();
                                assert!(
                                    ctx.content_rect().contains_rect(actions),
                                    "Actions outside viewport: {actions:?}"
                                );
                                assert!(
                                    ctx.content_rect().expand(1.).contains_rect(rect),
                                    "Window escaped viewport: {rect:?}, viewport {:?}",
                                    ctx.content_rect()
                                );
                                assert!(
                                    rect.height() <= ctx.content_rect().height() + 1.,
                                    "Browser exceeded viewport: {rect:?}, {h}/{zoom}"
                                );
                            }
                        }
                        assert!(shared.lock().unwrap().task.is_none());
                    }
                }
            }
        }
    }
    #[test]
    fn empty_discover_requests_fetch_with_one_click_without_a_confirmation() {
        crate::i18n::init();
        let ctx = Context::default();
        let mut browser = Browser {
            catalogue: None,
            installed: vec![],
            ..Default::default()
        };
        let render = |events: Vec<egui::Event>, browser: &mut Browser| {
            let mut requested = false;
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(700., 500.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    requested = empty_state(ui, browser, &ctx, 350.);
                },
            );
            requested
        };
        assert!(!render(vec![], &mut browser));
        let rect = ctx
            .data(|d| d.get_temp::<egui::Rect>(egui::Id::new("plugin_fetch_button")))
            .unwrap();
        let point = rect.center();
        assert!(!render(
            vec![
                egui::Event::PointerMoved(point),
                egui::Event::PointerButton {
                    pos: point,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE
                }
            ],
            &mut browser
        ));
        assert!(render(
            vec![egui::Event::PointerButton {
                pos: point,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE
            }],
            &mut browser
        ));
        assert!(
            browser.task.is_none(),
            "Control test must not start a network request"
        );
    }
    #[test]
    fn large_catalogue_only_animates_visible_cards() {
        crate::i18n::init();
        let ctx = render_context();
        let mut catalogue = fixture();
        let original = catalogue.packs[0].clone();
        catalogue.packs = (0..500)
            .map(|index| {
                let mut listing = original.clone();
                listing.manifest.id = format!("paintfe.test-{index}");
                listing
            })
            .collect();
        let shared = state(&ctx);
        {
            let mut state = shared.lock().unwrap();
            state.open = true;
            state.catalogue = Some(catalogue);
            state.installed.clear();
        }
        for frame_nr in 0..3 {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1100., 800.),
                    )),
                    time: Some(frame_nr as f64),
                    ..Default::default()
                },
                |ui| {
                    frame(ui.ctx());
                },
            );
        }
        let state = shared.lock().unwrap();
        assert_eq!(state.results.len(), 500);
        assert!(
            state.arrivals.len() < 32,
            "Offscreen cards must not be instantiated"
        );
        assert!(state.task.is_none());
    }
    #[test]
    fn removal_confirmation_does_not_accept_the_key_that_opened_it() {
        crate::i18n::init();
        let ctx = render_context();
        let shared = state(&ctx);
        {
            let mut state = shared.lock().unwrap();
            state.open = true;
            state.catalogue = Some(fixture());
            state.confirm_remove = Some("local.test".into());
        }
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1100., 800.),
                )),
                events: vec![egui::Event::Key {
                    key: egui::Key::Enter,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |ui| {
                frame(ui.ctx());
            },
        );
        assert_eq!(
            shared.lock().unwrap().confirm_remove.as_deref(),
            Some("local.test")
        );
        let _ = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1100., 800.),
                )),
                events: vec![egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |ui| {
                frame(ui.ctx());
            },
        );
        assert!(shared.lock().unwrap().confirm_remove.is_none());
    }
    #[test]
    fn popularity_and_trend_data_are_bounded_and_separate() {
        let mut c = fixture();
        c.packs[0].downloads = 100;
        c.packs[0].recent_downloads = 5;
        c.packs[1].downloads = 50;
        c.packs[1].recent_downloads = 20;
        c.validate().unwrap();
        let mut browser = Browser {
            catalogue: Some(c.clone()),
            installed: Vec::new(),
            prefs: Preferences::default(),
            ..Default::default()
        };
        browser.update_results();
        assert_eq!(browser.results[0].id, c.packs[0].manifest.id);
        browser.sort = 1;
        browser.update_results();
        assert_eq!(browser.results[0].id, c.packs[1].manifest.id);
        c.packs[0].recent_downloads = 101;
        assert!(c.validate().is_err());
    }
}
