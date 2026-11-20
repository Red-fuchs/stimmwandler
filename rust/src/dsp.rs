use anyhow::{bail, Result};
use std::path::Path;

const SOS: [[f64; 6]; 3] = [
    [0.9699606451838447, -0.9699606451838447, 0.0, 1.0, -0.9813258904926881, 0.0],
    [1.0, -2.0, 1.0, 1.0, -1.9696106864374547, 0.9699606452559844],
    [1.0, -2.0, 1.0, 1.0, -1.9880652422382208, 0.9884184800471566],
];

fn sos_pass(x: &mut [f32]) {
    for s in SOS {
        let (mut z1, mut z2) = (0f64, 0f64);
        for v in x.iter_mut() {
            let i = *v as f64;
            let o = s[0] * i + z1;
            z1 = s[1] * i - s[4] * o + z2;
            z2 = s[2] * i - s[5] * o;
            *v = o as f32;
        }
    }
}

/// 48 Hz Butterworth high-pass, forward-backward (zero phase, no edge padding).
pub fn highpass48(x: &mut [f32]) {
    sos_pass(x);
    x.reverse();
    sos_pass(x);
    x.reverse();
}

pub fn reflect_pad(x: &[f32], n: usize) -> Vec<f32> {
    let l = x.len();
    let mut o = Vec::with_capacity(l + 2 * n);
    let at = |i: isize| -> f32 {
        let p = 2 * (l as isize - 1);
        let mut j = i.rem_euclid(p.max(1));
        if j >= l as isize { j = p - j; }
        x[j as usize]
    };
    for i in (1..=n as isize).rev() { o.push(at(-i)); }
    o.extend_from_slice(x);
    for i in 0..n as isize { o.push(at(l as isize + i)); }
    o
}

/// Windowed-sinc resampler (Kaiser-less Hann, 16 zero crossings).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() { return x.to_vec(); }
    let r = to as f64 / from as f64;
    let cut = r.min(1.0);
    let half = 16.0 / cut;
    let n = (x.len() as f64 * r).round() as usize;
    (0..n).map(|i| {
        let c = i as f64 / r;
        let (a, b) = ((c - half).ceil().max(0.0) as usize, ((c + half).floor() as usize).min(x.len() - 1));
        let mut acc = 0f64;
        for k in a..=b {
            let d = k as f64 - c;
            let w = 0.5 + 0.5 * (std::f64::consts::PI * d / half).cos();
            let s = if d == 0.0 { 1.0 } else { (std::f64::consts::PI * d * cut).sin() / (std::f64::consts::PI * d * cut) };
            acc += x[k] as f64 * s * w * cut;
        }
        acc as f32
    }).collect()
}

pub fn read_wav_mono(p: &Path) -> Result<(Vec<f32>, u32)> {
    let mut r = hound::WavReader::open(p)?;
    let sp = r.spec();
    let ch = sp.channels as usize;
    let v: Vec<f32> = match sp.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let m = (1i64 << (sp.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|x| x as f32 / m)).collect::<Result<_, _>>()?
        }
    };
    Ok((v.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect(), sp.sample_rate))
}

pub fn write_wav(p: &Path, x: &[f32], sr: u32) -> Result<()> {
    let mut w = hound::WavWriter::create(p, hound::WavSpec { channels: 1, sample_rate: sr, bits_per_sample: 16, sample_format: hound::SampleFormat::Int })?;
    for &s in x { w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?; }
    w.finalize()?;
    Ok(())
}

/// Minimal .npy reader: little-endian f32, C-order, 2-D.
pub fn read_npy_f32(p: &Path) -> Result<(Vec<f32>, usize, usize)> {
    let b = std::fs::read(p)?;
    if &b[..6] != b"\x93NUMPY" { bail!("not npy"); }
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = std::str::from_utf8(&b[off..off + hl])?;
    if !h.contains("<f4") || h.contains("True") { bail!("npy must be float32 C-order: {h}"); }
    let sh = h.split("shape").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).unwrap_or("");
    let d: Vec<usize> = sh.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if d.len() != 2 { bail!("npy must be 2-D"); }
    let v = b[off + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok((v, d[0], d[1]))
}
