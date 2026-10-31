use super::*;
use crate::io::export::*;
fn choose<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id: &str,
    value: &mut T,
    label: &str,
    items: &[(T, &str)],
) {
    ui.label(crate::i18n::translate_english(label));
    let selected = items
        .iter()
        .find(|(v, _)| v == value)
        .map_or("", |(_, s)| *s);
    egui::ComboBox::from_id_salt(id)
        .width((ui.available_width() - 12.0).max(100.0))
        .wrap_mode(egui::TextWrapMode::Truncate)
        .selected_text(crate::i18n::translate_english(selected))
        .show_ui(ui, |ui| {
            for (v, s) in items {
                ui.selectable_value(value, *v, crate::i18n::translate_english(s));
            }
        });
}
fn palette_options(ui: &mut egui::Ui, algorithm: &mut palette::Algorithm, strength: &mut u8) {
    choose(
        ui,
        "quantizer",
        algorithm,
        "Quantization algorithm",
        &[
            (palette::Algorithm::MedianCut, "Median cut"),
            (palette::Algorithm::NeuQuant, "NeuQuant"),
        ],
    );
    ui.label(crate::i18n::translate_english("Dithering strength"));
    ui.add(
        crate::ui::controls::Slider::new(strength, 0..=100)
            .id_salt("dither_strength")
            .double_click_reset_to(100),
    );
}
impl SaveFileDialog {
    pub(super) fn show_advanced_options(&mut self, ui: &mut egui::Ui) {
        ui.push_id(("export_advanced",self.format.extension()),|ui| {
            let animated=self.animated;
            let o=&mut self.options;
            match o.format {
                SaveFormat::Jpeg=>choose(ui,"jpeg_sampling",&mut o.jpeg_sampling,"Chroma subsampling",&[(JpegSampling::Full,"4:4:4"),(JpegSampling::Vertical,"4:4:0"),(JpegSampling::Horizontal,"4:2:2"),(JpegSampling::Quarter,"4:2:0")]),
                SaveFormat::Webp=>{
                    #[cfg(not(target_arch="wasm32"))]
                    if !animated {choose(ui,"webp_preset",&mut o.webp_preset,"Encoder preset",&[(WebpPreset::Default,"Default"),(WebpPreset::Photo,"Photo"),(WebpPreset::Picture,"Picture"),(WebpPreset::Drawing,"Drawing"),(WebpPreset::Icon,"Icon"),(WebpPreset::Text,"Text")]);}
                    #[cfg(target_arch="wasm32")]
                    {o.webp_lossless=true;ui.weak(crate::i18n::translate_english("Browser WebP export is lossless only."));}
                }
                SaveFormat::Bmp|SaveFormat::Tga|SaveFormat::Tiff=>{
                    let mut items=vec![(RasterMode::Automatic,"Automatic (preserve source)"),(RasterMode::Rgba,"RGBA"),(RasterMode::Rgb,"RGB"),(RasterMode::Gray,"Grayscale")];
                    if o.format==SaveFormat::Bmp {items.push((RasterMode::Indexed,"Indexed palette"));}
                    choose(ui,"raster_mode",&mut o.raster_mode,"Colour mode",&items);
                    if o.format==SaveFormat::Tiff {choose(ui,"tiff_depth",&mut o.raster_depth,"Bit depth per channel",&[(RasterDepth::Automatic,"Automatic (preserve source)"),(RasterDepth::Eight,"8-bit"),(RasterDepth::Sixteen,"16-bit"),(RasterDepth::Float,"32-bit float")]);}
                    if o.format==SaveFormat::Tga {ui.checkbox(&mut o.tga_rle,crate::i18n::translate_english("Run-length compression (RLE)"));}
                    if o.raster_mode==RasterMode::Indexed {palette_options(ui,&mut o.quantizer,&mut o.dither_strength);}
                }
                SaveFormat::Gif=>{
                    palette_options(ui,&mut o.quantizer,&mut o.dither_strength);
                    ui.label(crate::i18n::translate_english("Transparency threshold"));
                    ui.add(crate::ui::controls::Slider::new(&mut o.gif_alpha_threshold,0..=255).id_salt("gif_threshold").double_click_reset_to(128));
                }
                SaveFormat::Png if !animated=>{
                    ui.checkbox(&mut o.png.interlaced,crate::i18n::translate_english("Interlaced (Adam7)"));
                    if o.png.mode==png::Mode::Indexed {palette_options(ui,&mut o.png.quantizer,&mut o.png.dither_strength);}
                }
                SaveFormat::Avif=>{
                    ui.label(crate::i18n::translate_english("Encoding speed (1 slow, 10 fast)"));
                    ui.add(crate::ui::controls::Slider::new(&mut o.avif_speed,1..=10).id_salt("avif_speed").double_click_reset_to(8));
                    ui.checkbox(&mut o.avif_lossless,crate::i18n::translate_english("Exact lossless pixels"));
                    let mut alpha=o.avif_lossless || o.avif_lossless_alpha;
                    ui.add_enabled_ui(!o.avif_lossless,|ui| {if ui.checkbox(&mut alpha,crate::i18n::translate_english("Lossless alpha")).changed(){o.avif_lossless_alpha=alpha;}});
                    ui.weak(crate::i18n::translate_english("AVIF uses full-resolution colour (4:4:4). Advanced options require an updated pack."));
                }
                SaveFormat::Heic=>{
                    let caps=crate::io::heic::display_capabilities();
                    ui.add_enabled_ui(caps.lossless || o.heic_lossless,|ui| {if ui.checkbox(&mut o.heic_lossless,crate::i18n::translate_english("Exact lossless pixels")).changed() && o.heic_lossless {o.heic_chroma=HeicChroma::Automatic;}});
                    let mut items=vec![(HeicChroma::Automatic,"Automatic")];
                    for (v,label,mask) in [(HeicChroma::Full,"4:4:4",8),(HeicChroma::Horizontal,"4:2:2",16),(HeicChroma::Quarter,"4:2:0",32)] {if caps.chroma&mask!=0 || o.heic_chroma==v {items.push((v,label));}}
                    if o.heic_lossless {ui.label(crate::i18n::translate_english("Chroma subsampling"));ui.weak("4:4:4");}else{ui.add_enabled_ui(caps.chroma!=0 || o.heic_chroma!=HeicChroma::Automatic,|ui|choose(ui,"heic_chroma",&mut o.heic_chroma,"Chroma subsampling",&items));}
                    if !caps.lossless || caps.chroma==0 {ui.weak(crate::i18n::translate_english("Available HEIC options depend on the installed encoder. Unsupported options are disabled."));}
                }
                _=>{}
            }
        });
    }
}
