use crate::dsp::{resample, Xfade};
use crate::engine::{Engine, Params, SR};
use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::{atomic::{AtomicBool, Ordering}, mpsc, Arc, Mutex};

pub fn dn(d: &cpal::Device) -> String { d.description().map(|x| x.name().to_string()).unwrap_or_default() }

pub fn list_devices() -> Result<()> {
    let h = cpal::default_host();
    for d in h.input_devices()? { println!("in : {}", dn(&d)); }
    for d in h.output_devices()? { println!("out: {}", dn(&d)); }
    Ok(())
}

pub fn find(name: &Option<String>, input: bool) -> Result<cpal::Device> {
    let h = cpal::default_host();
    match name {
        None => (if input { h.default_input_device() } else { h.default_output_device() }).ok_or_else(|| anyhow!("no default device")),
        Some(n) => {
            let mut it = if input { h.input_devices()? } else { h.output_devices()? };
            it.find(|d| dn(d).contains(n.as_str())).ok_or_else(|| anyhow!("device '{n}' not found"))
        }
    }
}

/// Runs until `stop` is set. Block-wise conversion with `context` seconds of history.
pub fn run_until(dir: &Path, model: &str, p: Params, ind: Option<String>, outd: Option<String>, block: f32, context: f32, stop: Arc<AtomicBool>) -> Result<()> {
    let mut eng = Engine::load(dir, model)?;
    let (i, o) = (find(&ind, true)?, find(&outd, false)?);
    let ic = i.default_input_config()?;
    let oc = o.default_output_config()?;
    let (isr, osr, ich, och) = (ic.sample_rate(), oc.sample_rate(), ic.channels() as usize, oc.channels() as usize);
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let q: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
    let qo = q.clone();
    let blk = (block * isr as f32) as usize;
    let mut acc: Vec<f32> = vec![];
    let ins = i.build_input_stream(ic.config(), move |d: &[f32], _| {
        acc.extend(d.chunks(ich).map(|c| c[0]));
        while acc.len() >= blk { let _ = tx.send(acc.drain(..blk).collect()); }
    }, |e| eprintln!("input error: {e}"), None)?;
    let outs = o.build_output_stream(oc.config(), move |d: &mut [f32], _| {
        let mut q = qo.lock().unwrap();
        for f in d.chunks_mut(och) { let v = q.pop_front().unwrap_or(0.0); f.iter_mut().for_each(|s| *s = v); }
    }, |e| eprintln!("output error: {e}"), None)?;
    ins.play()?; outs.play()?;
    let ov = (0.05 * osr as f32) as usize;
    let mut xf = Xfade::new(ov);
    let mut hist = vec![0f32; (context * SR as f32) as usize];
    while !stop.load(Ordering::Relaxed) {
        let Ok(x) = rx.recv_timeout(std::time::Duration::from_millis(100)) else { continue };
        let x = resample(&x, isr, SR);
        let mut seg = hist.clone();
        seg.extend_from_slice(&x);
        hist = seg[seg.len() - hist.len()..].to_vec();
        let y = eng.convert(&seg, &p)?;
        let keep = (x.len() as f64 * eng.sr as f64 / SR as f64) as usize + (0.05 * eng.sr as f32) as usize;
        let y = resample(&y[y.len() - keep.min(y.len())..], eng.sr, osr);
        let y = xf.push(&y);
        let mut q = q.lock().unwrap();
        if q.len() > osr as usize { q.clear(); } // drop backlog instead of growing latency
        q.extend(y);
    }
    Ok(())
}

pub fn run(dir: &Path, model: &str, p: Params, i: Option<String>, o: Option<String>, block: f32, ctx: f32) -> Result<()> {
    eprintln!("streaming; Ctrl+C to stop");
    run_until(dir, model, p, i, o, block, ctx, Arc::new(AtomicBool::new(false)))
}
