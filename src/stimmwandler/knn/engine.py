import json
import urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .hifigan import AttrDict, Generator
from .wavlm import WavLM, WavLMConfig

REL = "https://github.com/bshall/knn-vc/releases/download/v0.1/"
FILES = {"wavlm": "WavLM-Large.pt", "hifigan": "prematch_g_02500000.pt"}
SR, HOP, LAYER = 16000, 320, 6


def fetch(home: Path):
    home.mkdir(parents=True, exist_ok=True)
    for f in FILES.values():
        if not (home / f).exists():
            print("downloading", REL + f)
            urllib.request.urlretrieve(REL + f, home / f)


class KnnVC:
    def __init__(self, home: Path, device="cpu"):
        self.dev = torch.device(device)
        ck = torch.load(home / FILES["wavlm"], map_location="cpu", weights_only=False)
        self.wavlm = WavLM(WavLMConfig(ck["cfg"]))
        self.wavlm.load_state_dict(ck["model"])
        self.wavlm.to(self.dev).eval()
        h = AttrDict(json.loads((Path(__file__).parent / "config_v1_wavlm.json").read_text()))
        self.voc = Generator(h)
        self.voc.load_state_dict(torch.load(home / FILES["hifigan"], map_location="cpu")["generator"])
        self.voc.to(self.dev).eval().remove_weight_norm()

    @torch.inference_mode()
    def feats(self, x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x.astype(np.float32))[None].to(self.dev)
        return self.wavlm.extract_features(x, output_layer=LAYER, ret_layer_results=False)[0].squeeze(0)

    def matching_set(self, x: np.ndarray, trim_db=30) -> torch.Tensor:
        import librosa
        return self.feats(librosa.effects.trim(x, top_db=trim_db)[0]).cpu()

    @torch.inference_mode()
    def convert(self, x: np.ndarray, pool: torch.Tensor, k=4, rms=True) -> np.ndarray:
        q, p = self.feats(x), pool.to(self.dev)
        d = 1 - F.normalize(q, dim=-1) @ F.normalize(p, dim=-1).T
        out = p[d.topk(k, largest=False, dim=-1).indices].mean(1)
        y = self.voc(out[None]).squeeze().cpu().numpy()
        if rms:
            y = y * (np.sqrt((x**2).mean()) / (np.sqrt((y**2).mean()) + 1e-9))
        return np.clip(y, -1, 1).astype(np.float32)
