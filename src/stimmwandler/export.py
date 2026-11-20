"""ONNX export for the Rust app: hubert, rmvpe (audio -> salience), synth (per voice)."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .core import HOP, SR, hubert_feats, load_hubert, load_index, load_synth
from .rvc.rmvpe import E2E, MelSpectrogram

OPSET = 17


class HubertWrap(nn.Module):
    def __init__(self, hub, ver):
        super().__init__()
        self.hub, self.ver = hub, ver

    def forward(self, x):
        return hubert_feats(self.hub, x, self.ver)


class RmvpeWrap(nn.Module):
    """audio16k (1,N) -> salience (1,T,360); STFT as strided conv so it exports cleanly."""
    def __init__(self, net, mel):
        super().__init__()
        self.net = net
        self.register_buffer("mb", mel.mel_basis)
        w = torch.hann_window(1024, periodic=True)
        k = torch.fft.rfft(torch.eye(1024), dim=1)  # (1024,513)
        basis = torch.cat([k.real.T, k.imag.T]) * w  # (1026,1024)
        self.register_buffer("basis", basis[:, None, :])

    def forward(self, x):
        x = F.pad(x[:, None], (512, 512), mode="reflect")
        s = F.conv1d(x, self.basis, stride=HOP)
        mag = torch.sqrt(s[:, :513].square() + s[:, 513:].square())
        m = torch.log(torch.clamp(torch.matmul(self.mb, mag), min=1e-5))
        n = m.shape[-1]
        m = F.pad(m, (0, (32 - n % 32) % 32))
        return self.net(m)[:, :n]


class SynthWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, phone, pitch, pitchf, sid):
        ln = torch.full((1,), phone.shape[1], dtype=torch.long)
        return self.net.infer(phone, ln, pitch, pitchf, sid)[0]


def _export(m, args, path, names, out, dyn):
    torch.onnx.export(m, args, str(path), input_names=names, output_names=[out], dynamic_axes=dyn, opset_version=OPSET, dynamo=False)


def export_all(model, hub_dir, rmvpe_pt, out: Path, index=None, deterministic=False):
    if deterministic:  # parity testing only: zero all noise sources
        torch.randn_like = torch.zeros_like
        torch.rand = lambda *a, **k: torch.zeros(*a, **k)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    net, sr, f0, ver = load_synth(model, dev)
    assert f0 == 1, "ONNX export supports pitch-guided (f0) models only"
    net.remove_weight_norm()
    stem = Path(model).stem
    T, d = 200, 256 if ver == "v1" else 768
    _export(SynthWrap(net), (torch.randn(1, T, d), torch.randint(1, 255, (1, T)), torch.rand(1, T) * 300 + 80, torch.zeros(1, dtype=torch.long)),
            out / f"{stem}.onnx", ["phone", "pitch", "pitchf", "sid"], "audio", {"phone": {1: "t"}, "pitch": {1: "t"}, "pitchf": {1: "t"}, "audio": {2: "n"}})
    hp = out / f"hubert_{ver}.onnx"
    if not hp.exists():
        hub = load_hubert(hub_dir, dev)
        _export(HubertWrap(hub, ver), (torch.randn(1, SR * 2),), hp, ["audio"], "feats", {"audio": {1: "n"}, "feats": {1: "t"}})
    rp = out / "rmvpe.onnx"
    if not rp.exists():
        e = E2E(4, 1, (2, 2)); e.load_state_dict(torch.load(rmvpe_pt, map_location="cpu")); e.eval()
        mel = MelSpectrogram(False, 128, SR, 1024, HOP, None, 30, 8000)
        _export(RmvpeWrap(e, mel).eval(), (torch.randn(1, SR * 2),), rp, ["audio"], "salience", {"audio": {1: "n"}, "salience": {1: "t"}})
    if index:
        np.save(out / f"{stem}.index.npy", load_index(index))
    (out / f"{stem}.json").write_text(json.dumps({"version": ver, "sr": sr, "dim": d, "onnx": f"{stem}.onnx", "hubert": hp.name, "rmvpe": rp.name,
                                                  "index": f"{stem}.index.npy" if index else None}, indent=1))
    print("exported to", out)
