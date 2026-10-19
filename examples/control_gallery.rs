//! Developer gallery, deliberately using the production components and dialogs.
//! Run: cargo run --example control_gallery -j1
use paintfe::ui::dialogs::core::{
    DialogResult, GaussianBlurDialog, ResizeCanvasDialog, ResizeImageDialog,
};
use paintfe::ui::dialogs::effects::{BokehBlurDialog, InkDialog};

struct Gallery {
    dark: bool,
    scale: f32,
    value: f32,
    dimension: f32,
    angle: f32,
    opacity: f32,
    color: egui::Color32,
    enabled: bool,
    dialog: usize,
    new_image: paintfe::components::dialogs::NewFileDialog,
    resize: ResizeImageDialog,
    canvas: ResizeCanvasDialog,
    gaussian: GaussianBlurDialog,
    bokeh: BokehBlurDialog,
    pencil: InkDialog,
}
impl Default for Gallery {
    fn default() -> Self {
        let canvas = paintfe::canvas::CanvasState::new(800, 600);
        Self {
            dark: false,
            scale: 1.0,
            value: 50.0,
            dimension: 800.0,
            angle: 0.0,
            opacity: 128.0,
            color: egui::Color32::from_rgba_unmultiplied(220, 100, 30, 160),
            enabled: true,
            dialog: 0,
            new_image: Default::default(),
            resize: ResizeImageDialog::new(&canvas),
            canvas: ResizeCanvasDialog::new(&canvas),
            gaussian: GaussianBlurDialog::new(&canvas),
            bokeh: BokehBlurDialog::new(&canvas),
            pencil: InkDialog::new(&canvas),
        }
    }
}
impl eframe::App for Gallery {
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let theme = if self.dark {
            paintfe::theme::Theme::dark()
        } else {
            paintfe::theme::Theme::light()
        };
        theme.apply(&ctx);
        ctx.set_zoom_factor(self.scale);
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading("PaintFE shared controls");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.dark, "Dark theme");
                ui.checkbox(&mut self.enabled, "Controls enabled");
                if ui.button("100% text").clicked() { self.scale = 1.0; }
                if ui.button("125% text").clicked() { self.scale = 1.25; }
            });
            ui.separator();
            ui.add_enabled_ui(self.enabled, |ui| {
                egui::Grid::new("gallery").spacing([16.0, 14.0]).show(ui, |ui| {
                    ui.label("Dimension / expressions");
                    ui.add(paintfe::ui::controls::Numeric::dimension(&mut self.dimension).reset_to(800.0)); ui.end_row();
                    ui.label("Percentage");
                    ui.add(paintfe::ui::controls::Numeric::percentage(&mut self.value).steppers(1.0).reset_to(50.0)); ui.end_row();
                    ui.label("Angle");
                    ui.add(paintfe::ui::controls::Numeric::angle(&mut self.angle).steppers(1.0).reset_to(0.0)); ui.end_row();
                    ui.label("Standard slider");
                    ui.add(paintfe::ui::controls::Slider::new(&mut self.value, 0.0..=100.0).suffix("%").step_by(1.0).steppers(true).reset_to(50.0)); ui.end_row();
                    ui.label("Colour gradient");
                    ui.add(paintfe::ui::controls::GradientSlider::new(&mut self.opacity, 255.0, |t| egui::Color32::from_rgba_unmultiplied(255, 100, 20, (t * 255.0) as u8)).alpha(true).theme(theme.clone())); ui.end_row();
                    ui.label("Colour popup");
                    paintfe::ui::controls::color_edit_button_srgba(ui, &mut self.color, egui::color_picker::Alpha::BlendOrAdditive);
                    ui.end_row();
                });
            });
            ui.separator();
            ui.label("Production dialogs");
            ui.horizontal_wrapped(|ui| {
                for (index, name) in ["New Image", "Resize Image", "Resize Canvas", "Gaussian Blur", "Bokeh Blur", "Ink"].iter().enumerate() {
                    if ui.button(*name).clicked() {
                        self.dialog = index + 1;
                        if self.dialog == 1 { self.new_image.open_dialog(); }
                    }
                }
            });
            ui.label("Check drag, typing, Tab / Shift+Tab, Enter / Escape, wheel and Shift fine adjustment.");
        });
        let done = match self.dialog {
            1 => {
                self.new_image.show(&ctx);
                !self.new_image.open
            }
            2 => !matches!(
                self.resize.show(&ctx),
                DialogResult::Open | DialogResult::Changed
            ),
            3 => !matches!(
                self.canvas.show(&ctx, [1.0; 4]),
                DialogResult::Open | DialogResult::Changed
            ),
            4 => !matches!(
                self.gaussian.show(&ctx),
                DialogResult::Open | DialogResult::Changed
            ),
            5 => !matches!(
                self.bokeh.show(&ctx),
                DialogResult::Open | DialogResult::Changed
            ),
            6 => !matches!(
                self.pencil.show(&ctx),
                DialogResult::Open | DialogResult::Changed
            ),
            _ => false,
        };
        if done {
            self.dialog = 0;
        }
    }
}
fn main() -> eframe::Result {
    paintfe::i18n::init();
    eframe::run_native(
        "PaintFE Control Gallery",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([850.0, 650.0]),
            ..Default::default()
        },
        Box::new(|cc| {
            let mut fonts = egui::FontDefinitions::default();
            for (name, data) in [
                (
                    "gallery_dm",
                    include_bytes!("../assets/fonts/DMSans-Regular.ttf").as_slice(),
                ),
                (
                    "gallery_noto",
                    include_bytes!("../assets/fonts/NotoSans-Regular.ttf").as_slice(),
                ),
            ] {
                fonts
                    .font_data
                    .insert(name.into(), egui::FontData::from_static(data).into());
            }
            let proportional = fonts
                .families
                .get_mut(&egui::FontFamily::Proportional)
                .unwrap();
            proportional.insert(0, "gallery_dm".into());
            proportional.insert(1, "gallery_noto".into());
            fonts.families.insert(
                egui::FontFamily::Name("WidgetTitle".into()),
                fonts.families[&egui::FontFamily::Proportional].clone(),
            );
            cc.egui_ctx.set_fonts(fonts);
            Ok(Box::<Gallery>::default())
        }),
    )
}
