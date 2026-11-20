"""Python (noise zeroed) vs Rust CLI end-to-end on the same wav. usage: e2e_compare.py WORKDIR HUBERT RMVPE"""
import subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
torch.randn_like = torch.zeros_like
torch.rand = lambda *a, **k: torch.zeros(*a, **k)
from stimmwandler import audio
from stimmwandler.core import SR, Converter

w, hub, rm = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
c = Converter(w / "dummy.pth", hub, rm, "cpu")
x = audio.read_mono(w / "synthetic.wav", SR)
a = c.convert(x, 0, str(w / "dummy.index"), 0.6, 0.33, 0.25)
r = subprocess.run([str(Path(__file__).parent.parent / "rust/target/release/stimmwandler-rs"), "convert", "--dir", str(w / "onnx_det"), "-m", "dummy", "--index-rate", "0.6", "--rms-mix", "0.25", str(w / "synthetic.wav"), "-o", str(w / "out_rs.wav")], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr[-300:])
b, _ = sf.read(w / "out_rs.wav", dtype="float32")
n = min(len(a), len(b)); a, b = a[:n], b[:n]
corr = float(np.corrcoef(a, b)[0, 1])
print(f"len py={len(a)} rs={len(b)} corr={corr:.4f} rms py={np.sqrt((a**2).mean()):.4f} rs={np.sqrt((b**2).mean()):.4f}")
sys.exit(corr < 0.9)
