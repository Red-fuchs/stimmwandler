# Stimmwandler

Cross-platform (Windows / Linux / macOS) voice changer based on RVC (Retrieval-based Voice Conversion). MIT licensed.

- **Python side** (`stimmwandler`): PyTorch (CUDA / Apple MPS / CPU) offline conversion, realtime streaming, ONNX export.
- **Rust side** (`stimmwandler-rs`, `stimmwandler-gui`): ONNX Runtime + cpal realtime audio + egui GUI. No Python needed at runtime.

You supply your own trained RVC `.pth` voice model (v1 or v2, 32k/40k/48k, pitch-guided "f0" models for the ONNX/Rust path). No voices are bundled. Only convert voices you have the right to use.

## 1. Install (Python)

Python 3.10+ required.

```bash
git clone <this repo> stimmwandler && cd stimmwandler
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -e .
# NVIDIA GPU (Windows/Linux): install a CUDA build of torch first, see https://pytorch.org/get-started/locally/
```

macOS uses MPS automatically; Windows/Linux use CUDA if available, else CPU. Force with `--device cpu|cuda|mps`.

## 2. Base models

RVC needs two shared base models (HuBERT feature extractor and RMVPE pitch model). Official source: <https://huggingface.co/lj1995/VoiceConversionWebUI>.

```bash
stimmwandler download     # fetches hubert_base/ and rmvpe.pt into ~/.stimmwandler (override: STIMMWANDLER_HOME)
```

Your voice model: put your `.pth` (and optionally the matching `.index`) anywhere.

## 3. Convert a file

```bash
stimmwandler convert in.wav -m voice.pth -o out.wav --pitch 0 --index voice.index --index-rate 0.5
# --pitch  semitones (+12 male->female typical)   --protect 0.33   --rms-mix 0.25   --sid speaker id
```

## 4. Export to ONNX for the Rust app

```bash
stimmwandler export-onnx -m voice.pth --index voice.index --out models/
# writes voice.onnx, voice.json, voice.index.npy, hubert_v{1,2}.onnx, rmvpe.onnx
```

Then without Python:

```bash
stimmwandler-rs convert --dir models -m voice in.wav -o out.wav --pitch 0
stimmwandler-rs stream  --dir models -m voice --input-device "Microphone" --output-device "CABLE Input"
stimmwandler-rs stream --list-devices
stimmwandler-gui         # model picker, mic in / out device, pitch, index rate
```

Prebuilt binaries: GitHub Actions workflow `release.yml` builds Windows, Linux, macOS artifacts. Or build yourself: `cd rust && cargo build --release` (Linux needs `libasound2-dev`).

## 5. kNN-VC (no training, Python only)

Second engine: [kNN-VC](https://github.com/bshall/knn-vc). Replaces each WavLM frame of the input with the mean of its 4 nearest frames from a reference recording of the target voice (about 10 s to several minutes). No `.pth` needed.

```bash
stimmwandler knn in.wav -r target1.wav target2.wav -o out.wav [-k 4] [--device cpu|cuda|mps]
```

First run downloads WavLM-Large (~1.2 GB) and the prematched HiFi-GAN from the upstream GitHub release into `~/.stimmwandler/knnvc`. Output is 16 kHz and RMS-matched to the input (upstream uses LUFS normalisation). No ONNX/Rust path: WavLM-Large plus per-call kNN is heavy for realtime and not exported; kNN-VC is offline/Python-only here.

## 6. Realtime setup (virtual audio cable)

The voice changer writes to an *output device*. Make that a virtual cable, then choose the cable's *input* side as microphone in Discord/OBS/Zoom.

- **Windows**: install [VB-CABLE](https://vb-audio.com/Cable/). Output device = "CABLE Input"; in your chat app choose "CABLE Output" as microphone.
- **macOS**: install [BlackHole 2ch](https://existential.audio/blackhole/) (`brew install blackhole-2ch`). Output = "BlackHole 2ch"; chat app microphone = "BlackHole 2ch". Grant microphone permission to your terminal/app.
- **Linux (PipeWire / PulseAudio)**:
  ```bash
  pactl load-module module-null-sink sink_name=stimmwandler sink_properties=device.description=Stimmwandler
  # output device = "Stimmwandler"; chat app microphone = "Monitor of Stimmwandler"
  ```
- Use headphones to avoid feedback. Latency is roughly `--block` (default 0.4 s) plus 50 ms crossfade overlap plus model time; lower `--block` on a fast GPU/CPU, raise it if audio stutters. `--context` (default 1 s) gives the model history for stable pitch.

Blocks are joined with a 50 ms linear crossfade (Rust and Python) to avoid clicks.

## Status / untested

- Tested on macOS only. Windows and Linux builds, the CI release workflow (shared-library copying for `ort`) and virtual-cable setups are untested.
- Realtime streaming with real audio devices has not been run; the crossfade is verified by unit tests on synthetic audio only.
- Only random-weight dummy models were used for parity tests; no real voice model was evaluated. No-f0 (nono) models work in Python only, not in ONNX/Rust. Rust lacks fcpe/pm pitch extractors; its high-pass differs marginally from scipy `filtfilt`.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/test_xfade.py; (cd rust && cargo test)
```

```bash
python tests/make_dummy.py work/            # random-weight model + synthetic wav (no real voices)
python tests/parity.py work/ ~/.stimmwandler/hubert_base ~/.stimmwandler/rmvpe.pt   # PyTorch vs ONNX
python tests/e2e_compare.py work/ <hubert> <rmvpe>                                   # Python vs Rust CLI
```

## Related projects

- [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - upstream (MIT), model code vendored from here
- [kNN-VC](https://github.com/bshall/knn-vc) - second engine (`stimmwandler knn`), code vendored from here (MIT)
- [Red-fuchs/kitsune](https://github.com/Red-fuchs/kitsune) - local voice assistant; Stimmwandler can serve as its voice-changer stage

## Licenses

MIT. Model code vendored from RVC-Project, kNN-VC, WavLM and HiFi-GAN (all MIT), see `THIRD_PARTY_NOTICES.md`. Base models (HuBERT, RMVPE, WavLM-Large, kNN-VC HiFi-GAN) are downloaded by you from their official source and have their own licenses.
