"""Random-weight RVC v2 (40k, f0) checkpoint + synthetic test wav. Pipeline/parity tests only; output is noise."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from stimmwandler.rvc.models import SynthesizerTrnMs768NSFsid

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
cfg = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5]] * 3, [10, 10, 2, 2], 512, [16, 16, 4, 4], 4, 256, 40000]
net = SynthesizerTrnMs768NSFsid(*cfg, is_half=False)
w = {k: v for k, v in net.state_dict().items() if not k.startswith("enc_q")}
torch.save({"config": cfg, "weight": w, "f0": 1, "version": "v2"}, out / "dummy.pth")
t = np.arange(16000 * 3) / 16000
f = 140 + 20 * np.sin(2 * np.pi * 4 * t)
ph = 2 * np.pi * np.cumsum(f) / 16000
x = sum(np.sin(h * ph) / h for h in range(1, 12)) * 0.2 * (np.sin(np.pi * t / 3) ** 0.5)
sf.write(out / "synthetic.wav", x.astype(np.float32), 16000)
