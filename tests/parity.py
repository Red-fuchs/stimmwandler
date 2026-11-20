"""PyTorch vs ONNX parity. usage: parity.py WORKDIR HUBERT_DIR RMVPE_PT   (WORKDIR has dummy.pth from make_dummy.py)"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from stimmwandler.core import RMVPE, hubert_feats, load_hubert, load_synth

w, hub, rm = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
subprocess.run(["stimmwandler", "export-onnx", "-m", str(w / "dummy.pth"), "--hubert", hub, "--rmvpe", rm, "--out", str(w / "onnx_det"), "--deterministic"], check=True, capture_output=True)
o = w / "onnx_det"
S = lambda n: ort.InferenceSession(str(o / n), providers=["CPUExecutionProvider"])
torch.manual_seed(1)
ok = True


def rep(name, a, b, tol):
    global ok
    e = float(np.abs(a - b).max()); r = e / (float(np.abs(a).max()) + 1e-9)
    print(f"{name}: max abs err {e:.2e} (rel {r:.2e}) tol {tol}")
    ok &= r < tol


for n in (48000, 80000):  # different lengths: dynamic axes check
    x = torch.randn(1, n) * 0.1
    h = load_hubert(hub, "cpu")
    with torch.no_grad():
        a = hubert_feats(h, x, "v2").numpy()
    rep(f"hubert n={n}", a, S("hubert_v2.onnx").run(None, {"audio": x.numpy()})[0], 1e-3)
    r = RMVPE(rm, torch.device("cpu"))
    with torch.no_grad():
        m = r.mel(x)
        k = m.shape[-1]; m = torch.nn.functional.pad(m, (0, 32 * ((k - 1) // 32 + 1) - k))
        a = r.net(m)[:, :k].numpy()
    rep(f"rmvpe n={n}", a, S("rmvpe.onnx").run(None, {"audio": x.numpy()})[0], 1e-3)
net, sr, f0, ver = load_synth(w / "dummy.pth", torch.device("cpu"))
net.remove_weight_norm()
for T in (200, 333):
    ph = torch.randn(1, T, 768); pc = torch.randint(1, 255, (1, T)); pf = torch.rand(1, T) * 300 + 80
    torch.randn_like = torch.zeros_like
    with torch.no_grad():
        a = net.infer(ph, torch.tensor([T]), pc, pf, torch.zeros(1, dtype=torch.long))[0].numpy()
    rep(f"synth T={T}", a, S("dummy.onnx").run(None, {"phone": ph.numpy(), "pitch": pc.numpy(), "pitchf": pf.numpy(), "sid": np.zeros(1, np.int64)})[0], 1e-2)
print("PASS" if ok else "FAIL"); sys.exit(not ok)
