use crate::dsp::*;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use std::path::Path;

pub const SR: u32 = 16000;
const HOP: usize = 160;

#[derive(Deserialize)]
struct Meta { sr: u32, dim: usize, onnx: String, hubert: String, rmvpe: String, index: Option<String> }

pub struct Params { pub pitch: f32, pub index_rate: f32, pub protect: f32, pub sid: i64, pub rms_mix: f32 }
impl Default for Params { fn default() -> Self { Self { pitch: 0.0, index_rate: 0.5, protect: 0.33, sid: 0, rms_mix: 0.25 } } }

pub struct Engine {
    pub sr: u32,
    hub: Session,
    rmv: Session,
    syn: Session,
    idx: Option<(Vec<f32>, usize, usize)>,
}

static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();

fn sess(p: &Path) -> Result<Session> {
    if ENV.get().is_none() { let _ = ENV.set(ort::init().build().map_err(|e| anyhow::anyhow!("{e}"))?); }
    Ok(Session::builder(ENV.get().unwrap()).map_err(|e| anyhow::anyhow!("{e}"))?.commit_from_file(p).map_err(|e| anyhow::anyhow!("{} {e}", p.display()))?)
}
macro_rules! ort { ($e:expr) => { $e.map_err(|e| anyhow::anyhow!("{e}"))? }; }

impl Engine {
    /// `dir` holds the files written by `stimmwandler export-onnx`; `name` = model stem.
    pub fn load(dir: &Path, name: &str) -> Result<Self> {
        let m: Meta = serde_json::from_slice(&std::fs::read(dir.join(format!("{name}.json"))).context("meta json")?)?;
        let _ = m.dim;
        Ok(Self {
            sr: m.sr,
            hub: sess(&dir.join(&m.hubert))?,
            rmv: sess(&dir.join(&m.rmvpe))?,
            syn: sess(&dir.join(&m.onnx))?,
            idx: m.index.map(|i| read_npy_f32(&dir.join(i))).transpose()?,
        })
    }

    fn f0(&mut self, a: &[f32]) -> Result<Vec<f32>> {
        let t = Tensor::from_array(Array2::from_shape_vec((1, a.len()), a.to_vec())?)?;
        let out = ort!(self.rmv.run(ort::inputs!["audio" => t]));
        let (sh, d) = ort!(out[0].try_extract_tensor::<f32>());
        let (n, k) = (sh[1] as usize, sh[2] as usize);
        const C0: f64 = 1997.3794084376191;
        let cents = |i: isize| if i < 4 || i >= 364 { f64::NAN } else { 20.0 * (i - 4) as f64 + C0 };
        let _ = cents;
        Ok((0..n).map(|f| {
            let r = &d[f * k..(f + 1) * k];
            let mut c = 0;
            for j in 0..k { if r[j] > r[c] { c = j; } }
            let (mut ps, mut ws) = (0f64, 0f64);
            for j in c as isize - 4..=c as isize + 4 {
                let s = if j < 0 || j >= k as isize { 0.0 } else { r[j as usize] as f64 };
                ps += s * (20.0 * j as f64 + C0);
                ws += s;
            }
            if r[c] <= 0.03 { 0.0 } else { let f = 10.0 * 2f64.powf(ps / ws / 1200.0); if (f - 10.0).abs() < 1e-9 { 0.0 } else { f as f32 } }
        }).collect())
    }

    fn feats(&mut self, a: &[f32]) -> Result<Array2<f32>> {
        let t = Tensor::from_array(Array2::from_shape_vec((1, a.len()), a.to_vec())?)?;
        let out = ort!(self.hub.run(ort::inputs!["audio" => t]));
        let (sh, d) = ort!(out[0].try_extract_tensor::<f32>());
        Ok(Array2::from_shape_vec((sh[1] as usize, sh[2] as usize), d.to_vec())?)
    }

    fn blend(&self, f: &Array2<f32>, rate: f32) -> Array2<f32> {
        let Some((v, n, d)) = &self.idx else { return f.clone() };
        let mut o = f.clone();
        for (t, row) in f.rows().into_iter().enumerate() {
            let mut best: Vec<(f32, usize)> = Vec::with_capacity(9);
            for i in 0..*n {
                let c = &v[i * d..(i + 1) * d];
                let dist: f32 = c.iter().zip(row.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if best.len() < 8 || dist < best[best.len() - 1].0 {
                    best.push((dist, i));
                    best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    best.truncate(8);
                }
            }
            let w: Vec<f32> = best.iter().map(|b| 1.0 / (b.0 * b.0 + 1e-24)).collect();
            let ws: f32 = w.iter().sum();
            for j in 0..*d {
                let g: f32 = best.iter().zip(&w).map(|(b, w)| v[b.1 * d + j] * w).sum::<f32>() / ws;
                o[[t, j]] = g * rate + (1.0 - rate) * row[j];
            }
        }
        o
    }

    fn vc(&mut self, a: &[f32], pc: &[i64], pf: &[f32], p: &Params) -> Result<Vec<f32>> {
        let f0 = self.feats(a)?;
        let fb = if self.idx.is_some() && p.index_rate > 0.0 { self.blend(&f0, p.index_rate) } else { f0.clone() };
        let up = |f: &Array2<f32>| Array2::from_shape_fn((f.nrows() * 2, f.ncols()), |(i, j)| f[[i / 2, j]]);
        let (mut f, f0u) = (up(&fb), up(&f0));
        let n = (a.len() / HOP).min(f.nrows());
        let mut f = f.slice_mut(ndarray::s![..n, ..]).to_owned();
        if p.protect < 0.5 {
            for t in 0..n {
                let m = if pf[t] > 0.0 { 1.0 } else { p.protect };
                for j in 0..f.ncols() { f[[t, j]] = f[[t, j]] * m + f0u[[t, j]] * (1.0 - m); }
            }
        }
        let d = f.ncols();
        let ph = Tensor::from_array(Array3::from_shape_vec((1, n, d), f.into_raw_vec_and_offset().0)?)?;
        let pi = Tensor::from_array(Array2::from_shape_vec((1, n), pc[..n].to_vec())?)?;
        let pff = Tensor::from_array(Array2::from_shape_vec((1, n), pf[..n].to_vec())?)?;
        let sid = Tensor::from_array(Array1::from_vec(vec![p.sid]))?;
        let out = ort!(self.syn.run(ort::inputs!["phone" => ph, "pitch" => pi, "pitchf" => pff, "sid" => sid]));
        Ok(ort!(out[0].try_extract_tensor::<f32>()).1.to_vec())
    }

    /// mono 16 kHz in -> mono audio at `self.sr` out.
    pub fn convert(&mut self, audio: &[f32], p: &Params) -> Result<Vec<f32>> {
        let mut a = audio.to_vec();
        highpass48(&mut a);
        let tp = SR as usize;
        let pad = reflect_pad(&a, tp);
        let pl = pad.len() / HOP;
        let mut f0 = self.f0(&pad)?;
        f0.truncate(pl);
        let voiced: Vec<usize> = (0..f0.len()).filter(|&i| f0[i] > 0.0).collect();
        if !voiced.is_empty() && voiced.len() < f0.len() {
            let orig = f0.clone();
            for i in 0..f0.len() {
                if orig[i] == 0.0 {
                    let k = voiced.partition_point(|&v| v < i);
                    f0[i] = if k == 0 { orig[voiced[0]] } else if k == voiced.len() { orig[voiced[k - 1]] } else {
                        let (l, r) = (voiced[k - 1], voiced[k]);
                        orig[l] + (orig[r] - orig[l]) * (i - l) as f32 / (r - l) as f32
                    };
                }
            }
        }
        let g = 2f32.powf(p.pitch / 12.0);
        f0.iter_mut().for_each(|v| *v *= g);
        let (lo, hi) = (1127.0 * (1.0f32 + 50.0 / 700.0).ln(), 1127.0 * (1.0f32 + 1100.0 / 700.0).ln());
        let pc: Vec<i64> = f0.iter().map(|&f| {
            let mut m = 1127.0 * (1.0 + f / 700.0).ln();
            if m > 0.0 { m = (m - lo) * 254.0 / (hi - lo) + 1.0; }
            m.clamp(1.0, 255.0).round_ties_even() as i64
        }).collect();
        let step = 30 * SR as usize / HOP * HOP;
        let mut cuts = vec![0usize];
        while pad.len() - cuts[cuts.len() - 1] > step + 2 * tp { cuts.push(cuts[cuts.len() - 1] + step); }
        cuts.push(pad.len() - 2 * tp);
        let mut out = vec![];
        let t = (tp as f64 * self.sr as f64 / SR as f64) as usize;
        for w in cuts.windows(2) {
            let (s, e) = (w[0], w[1] + 2 * tp);
            let o = self.vc(&pad[s..e], &pc[s / HOP..e / HOP], &f0[s / HOP..e / HOP], p)?;
            out.extend_from_slice(&o[t..o.len() - t]);
        }
        if p.rms_mix != 1.0 { rms_mix(&a, SR, &mut out, self.sr, p.rms_mix); }
        let m = out.iter().fold(0f32, |a, b| a.max(b.abs())) / 0.99;
        if m > 1.0 { out.iter_mut().for_each(|v| *v /= m); }
        Ok(out)
    }
}

fn rms_curve(x: &[f32], sr: u32, len: usize) -> Vec<f32> {
    let h = (sr / 2) as usize;
    let n = x.len() / h + 1;
    let r: Vec<f32> = (0..n).map(|i| if i * h < x.len() { let s = &x[i * h..((i + 2) * h).min(x.len())]; (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32 + 1e-12).sqrt() } else { 1e-6 }).collect();
    (0..len).map(|k| {
        let t = if len > 1 { k as f32 * (n - 1) as f32 / (len - 1) as f32 } else { 0.0 };
        let (i, f) = (t.floor() as usize, t.fract());
        if i + 1 < n { r[i] * (1.0 - f) + r[i + 1] * f } else { r[n - 1] }
    }).collect()
}

fn rms_mix(src: &[f32], sr1: u32, out: &mut [f32], sr2: u32, rate: f32) {
    let (a, b) = (rms_curve(src, sr1, out.len()), rms_curve(out, sr2, out.len()));
    for i in 0..out.len() { out[i] *= a[i].powf(1.0 - rate) * b[i].max(1e-6).powf(rate - 1.0); }
}
