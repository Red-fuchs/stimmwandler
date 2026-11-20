"""RVC inference pipeline (PyTorch: CUDA / MPS / CPU). Model code in ./rvc is vendored from RVC-Project (MIT)."""
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn
from transformers import HubertModel

from .rvc import models as M
from .rvc.rmvpe import E2E, MelSpectrogram

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
SR, HOP = 16000, 160
_BH, _AH = signal.butter(5, 48, "high", fs=SR)
_CENTS = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
_CLS = {("v1", 1): M.SynthesizerTrnMs256NSFsid, ("v1", 0): M.SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): M.SynthesizerTrnMs768NSFsid, ("v2", 0): M.SynthesizerTrnMs768NSFsid_nono}


def pick_device(d: str = "auto") -> torch.device:
    if d != "auto":
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HubertFP(HubertModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.final_proj = nn.Linear(cfg.hidden_size, cfg.classifier_proj_size)


def load_hubert(path, dev):
    return HubertFP.from_pretrained(str(path), local_files_only=True).float().to(dev).eval()


def hubert_feats(hub, x, version):
    """v1: layer 9 + final_proj (256-d); v2: last layer (768-d)."""
    out = hub(x, output_hidden_states=version == "v1", return_dict=True)
    return hub.final_proj(out.hidden_states[9]) if version == "v1" else out.last_hidden_state


def load_synth(pth, dev):
    cpt = torch.load(pth, map_location="cpu", weights_only=False)
    cfg = list(cpt["config"])
    cfg[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    f0, ver = cpt.get("f0", 1), cpt.get("version", "v1")
    net = _CLS[(ver, f0)](*cfg, is_half=False)
    del net.enc_q
    net.load_state_dict(cpt["weight"], strict=False)
    return net.float().eval().to(dev), cfg[-1], f0, ver


class RMVPE:
    def __init__(self, path, dev):
        self.dev = dev
        self.mel = MelSpectrogram(False, 128, SR, 1024, HOP, None, 30, 8000).to(dev)
        self.net = E2E(4, 1, (2, 2))
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval().to(dev)

    @torch.no_grad()
    def f0(self, audio: np.ndarray, thred=0.03):
        m = self.mel(torch.from_numpy(audio).float().to(self.dev)[None])
        n = m.shape[-1]
        m = F.pad(m, (0, 32 * ((n - 1) // 32 + 1) - n))
        return decode_rmvpe(self.net(m)[0, :n].cpu().numpy(), thred)


def decode_rmvpe(hid: np.ndarray, thred=0.03) -> np.ndarray:
    c = np.argmax(hid, 1) + 4
    p = np.pad(hid, ((0, 0), (4, 4)))
    idx = c[:, None] + np.arange(-4, 5)
    s, cm = np.take_along_axis(p, idx, 1), _CENTS[idx]
    cents = (s * cm).sum(1) / s.sum(1)
    cents[p.max(1) <= thred] = 0
    f0 = 10 * 2 ** (cents / 1200)
    f0[f0 == 10] = 0
    return f0


def f0_to_coarse(f0):
    lo, hi = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    m = 1127 * np.log(1 + f0 / 700)
    m[m > 0] = (m[m > 0] - lo) * 254 / (hi - lo) + 1
    return np.rint(np.clip(m, 1, 255)).astype(np.int64)


def load_index(path):
    """Returns (N,D) float32 vectors from a faiss .index or a .npy exported by `stimmwandler prep-index`."""
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p).astype(np.float32)
    import faiss
    ix = faiss.read_index(p)
    return ix.reconstruct_n(0, ix.ntotal)


def knn_blend(feats: torch.Tensor, vecs: torch.Tensor, rate: float, k=8):
    d = torch.cdist(feats[0], vecs)  # (T,N)
    sc, ix = d.topk(k, largest=False)
    w = 1 / (sc.square() + 1e-12)
    w = w / w.sum(1, keepdim=True)
    got = (vecs[ix] * w.unsqueeze(-1)).sum(1)
    return (got * rate + (1 - rate) * feats[0])[None]


def rms_mix(src, sr1, out, sr2, rate):
    def rms(x, sr):
        h = sr // 2
        n = max(1, len(x) // h + 1)
        r = np.array([np.sqrt(np.mean(x[i * h:(i + 1) * h + h] ** 2) + 1e-12) if i * h < len(x) else 1e-6 for i in range(n)])
        return np.interp(np.linspace(0, n - 1, len(out)), np.arange(n), r)
    a, b = rms(src, sr1), np.maximum(rms(out, sr2), 1e-6)
    return out * (a ** (1 - rate)) * (b ** (rate - 1))


class Converter:
    def __init__(self, model, hubert_dir, rmvpe_pt, device="auto"):
        self.dev = pick_device(device)
        self.net, self.sr, self.f0_on, self.ver = load_synth(model, self.dev)
        self.hub = load_hubert(hubert_dir, self.dev)
        self.rmvpe = RMVPE(rmvpe_pt, self.dev)

    @torch.no_grad()
    def _vc(self, a16, pitch, pitchf, vecs, index_rate, protect, sid):
        x = torch.from_numpy(a16).float().to(self.dev)[None]
        f = hubert_feats(self.hub, x, self.ver)
        f0_ = f.clone()
        if vecs is not None and index_rate > 0:
            f = knn_blend(f, vecs, index_rate)
        up = lambda t: F.interpolate(t.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        f, f0_ = up(f), up(f0_)
        n = min(len(a16) // HOP, f.shape[1])
        f, pitch, pitchf = f[:, :n], pitch[:, :n], pitchf[:, :n]
        if protect < 0.5:
            m = torch.where(pitchf > 0, 1.0, protect)[..., None]
            f = f * m + f0_[:, :n] * (1 - m)
        ln = torch.tensor([n], device=self.dev)
        sid = torch.tensor([sid], device=self.dev)
        args = (f, ln, pitch, pitchf, sid) if self.f0_on else (f, ln, sid)
        return self.net.infer(*args)[0][0, 0].float().cpu().numpy()

    def convert(self, audio, pitch=0, index=None, index_rate=0.5, protect=0.33, rms_rate=0.25, sid=0, chunk_s=30):
        """audio: mono float32 @16 kHz -> mono float32 @ model sr."""
        vecs = torch.from_numpy(load_index(index)).to(self.dev) if index and index_rate > 0 else None
        audio = signal.filtfilt(_BH, _AH, audio).astype(np.float32)
        tp = SR  # 1 s reflect pad each side
        pad = np.pad(audio, (tp, tp), mode="reflect")
        pl = len(pad) // HOP
        f0 = self.rmvpe.f0(pad)[:pl] if self.f0_on else np.zeros(pl)
        if self.f0_on:
            uv = f0 == 0
            if (~uv).any() and uv.any():
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0 = f0 * 2 ** (pitch / 12)
        pc = torch.from_numpy(f0_to_coarse(f0.copy()))[None].to(self.dev)
        pf = torch.from_numpy(f0.astype(np.float32))[None].to(self.dev)
        cuts, step = [0], chunk_s * SR // HOP * HOP
        while len(pad) - cuts[-1] > step + 2 * tp:
            cuts.append(cuts[-1] + step)
        cuts.append(len(pad) - 2 * tp)
        res, ratio = [], self.sr // SR if self.sr % SR == 0 else self.sr / SR
        for s, e in zip(cuts[:-1], cuts[1:]):
            seg = pad[s:e + 2 * tp]
            fs, fe = s // HOP, (e + 2 * tp) // HOP
            o = self._vc(seg, pc[:, fs:fe], pf[:, fs:fe], vecs, index_rate, protect, sid)
            t = int(tp * self.sr / SR)
            res.append(o[t:len(o) - t])
        out = np.concatenate(res)
        if rms_rate != 1:
            out = rms_mix(audio, SR, out, self.sr, rms_rate)
        return out / max(1.0, np.abs(out).max() / 0.99)
