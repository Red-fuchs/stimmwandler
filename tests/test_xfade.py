import numpy as np
from stimmwandler.audio import Xfade


def run(jit, xf, sr=16000, f=440, blk=3200, ov=800):
    w = 2 * np.pi * f / sr; x = Xfade(ov); out = []
    for b in range(1, 9):
        e = (b + 1) * blk
        s = np.sin(w * np.arange(e - blk - ov, e) + (0.6 * (b % 2) if jit else 0)).astype(np.float32)
        out.append(x.push(s) if xf else s[ov:])
    return np.abs(np.diff(np.concatenate(out))).max()


def test_continuous():
    assert run(0, 1) <= 2 * np.pi * 440 / 16000 * 1.01 + 1e-4


def test_smooths():
    assert run(1, 1) < 0.5 * run(1, 0)
