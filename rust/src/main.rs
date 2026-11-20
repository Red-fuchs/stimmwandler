use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use stimmwandler_rs::{dsp, engine::{Engine, Params, SR}};

#[derive(Parser)]
#[command(name = "stimmwandler-rs", version, about = "Stimmwandler Rust runtime (ONNX Runtime)")]
struct Cli { #[command(subcommand)] cmd: Cmd }

#[derive(clap::Args, Clone)]
struct Common {
    /// directory written by `stimmwandler export-onnx`
    #[arg(long)] dir: PathBuf,
    /// model stem (file name without extension)
    #[arg(short, long)] model: String,
    #[arg(long, default_value_t = 0.0)] pitch: f32,
    #[arg(long, default_value_t = 0.5)] index_rate: f32,
    #[arg(long, default_value_t = 0.33)] protect: f32,
    #[arg(long, default_value_t = 0.25)] rms_mix: f32,
}

#[derive(Subcommand)]
enum Cmd {
    Convert { #[command(flatten)] c: Common, input: PathBuf, #[arg(short, long)] output: PathBuf },
    Stream { #[command(flatten)] c: Common, #[arg(long)] input_device: Option<String>, #[arg(long)] output_device: Option<String>, #[arg(long, default_value_t = 0.4)] block: f32, #[arg(long, default_value_t = 1.0)] context: f32, #[arg(long)] list_devices: bool },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Convert { c, input, output } => {
            let mut e = Engine::load(&c.dir, &c.model)?;
            let (x, sr) = dsp::read_wav_mono(&input)?;
            let x = dsp::resample(&x, sr, SR);
            let t = std::time::Instant::now();
            let y = e.convert(&x, &Params { pitch: c.pitch, index_rate: c.index_rate, protect: c.protect, sid: 0, rms_mix: c.rms_mix })?;
            dsp::write_wav(&output, &y, e.sr)?;
            println!("{:.1}s audio -> {} ({} Hz) in {:.1}s", x.len() as f32 / SR as f32, output.display(), e.sr, t.elapsed().as_secs_f32());
        }
        Cmd::Stream { c, input_device, output_device, block, context, list_devices } => {
            if list_devices { return stimmwandler_rs::stream::list_devices(); }
            stimmwandler_rs::stream::run(&c.dir, &c.model, Params { pitch: c.pitch, index_rate: c.index_rate, protect: c.protect, sid: 0, rms_mix: c.rms_mix }, input_device, output_device, block, context)?;
        }
    }
    Ok(())
}
