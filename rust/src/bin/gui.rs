use eframe::egui;
use std::path::PathBuf;
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};
use stimmwandler_rs::{engine::Params, stream};

struct App {
    dir: String, models: Vec<String>, model: String,
    ins: Vec<String>, outs: Vec<String>, inp: usize, out: usize,
    pitch: f32, index_rate: f32, protect: f32,
    stop: Option<Arc<AtomicBool>>, status: String,
}

fn scan(dir: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir).into_iter().flatten().flatten()
        .filter_map(|e| e.path().extension().is_some_and(|x| x == "json").then(|| e.path().file_stem().unwrap().to_string_lossy().into_owned())).collect();
    v.sort(); v
}

fn devs(input: bool) -> Vec<String> {
    use cpal::traits::HostTrait;
    let h = cpal::default_host();
    let mut v = vec!["(default)".to_string()];
    let it = if input { h.input_devices().map(|i| i.collect::<Vec<_>>()) } else { h.output_devices().map(|i| i.collect::<Vec<_>>()) };
    v.extend(it.into_iter().flatten().map(|d| stream::dn(&d)));
    v
}

impl App {
    fn new() -> Self {
        Self { dir: "models".into(), models: scan("models"), model: String::new(), ins: devs(true), outs: devs(false), inp: 0, out: 0,
               pitch: 0.0, index_rate: 0.5, protect: 0.33, stop: None, status: "idle".into() }
    }
    fn start(&mut self) {
        let stop = Arc::new(AtomicBool::new(false));
        self.stop = Some(stop.clone());
        let pick = |v: &Vec<String>, i: usize| (i > 0).then(|| v[i].clone());
        let (dir, model, ind, outd) = (PathBuf::from(&self.dir), self.model.clone(), pick(&self.ins, self.inp), pick(&self.outs, self.out));
        let p = Params { pitch: self.pitch, index_rate: self.index_rate, protect: self.protect, sid: 0, rms_mix: 1.0 };
        std::thread::spawn(move || if let Err(e) = stream::run_until(&dir, &model, p, ind, outd, 0.4, 1.0, stop) { eprintln!("stream error: {e:#}"); });
        self.status = "running".into();
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _f: &mut eframe::Frame) {
        ui.heading("Stimmwandler");
        ui.horizontal(|ui| {
            ui.label("Model folder");
            ui.text_edit_singleline(&mut self.dir);
            if ui.button("Browse").clicked() { if let Some(p) = rfd::FileDialog::new().pick_folder() { self.dir = p.to_string_lossy().into(); } }
            if ui.button("Scan").clicked() { self.models = scan(&self.dir); }
        });
        egui::ComboBox::from_label("Model").selected_text(&self.model).show_ui(ui, |ui| for m in &self.models { ui.selectable_value(&mut self.model, m.clone(), m); });
        egui::ComboBox::from_label("Mic in").selected_text(&self.ins[self.inp]).show_ui(ui, |ui| for (i, d) in self.ins.iter().enumerate() { ui.selectable_value(&mut self.inp, i, d); });
        egui::ComboBox::from_label("Out (e.g. virtual cable)").selected_text(&self.outs[self.out]).show_ui(ui, |ui| for (i, d) in self.outs.iter().enumerate() { ui.selectable_value(&mut self.out, i, d); });
        ui.add(egui::Slider::new(&mut self.pitch, -24.0..=24.0).text("Pitch (semitones)"));
        ui.add(egui::Slider::new(&mut self.index_rate, 0.0..=1.0).text("Index rate"));
        ui.add(egui::Slider::new(&mut self.protect, 0.0..=0.5).text("Protect"));
        if self.stop.is_none() {
            if ui.add_enabled(!self.model.is_empty(), egui::Button::new("Start")).clicked() { self.start(); }
        } else if ui.button("Stop").clicked() {
            self.stop.take().unwrap().store(true, Ordering::Relaxed);
            self.status = "idle".into();
        }
        ui.label(&self.status);
    }
}

fn main() -> eframe::Result {
    eframe::run_native("Stimmwandler", Default::default(), Box::new(|_| Ok(Box::new(App::new()))))
}
