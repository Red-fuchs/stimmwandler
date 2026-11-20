import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def resample(x, a, b):
    if a == b:
        return x
    g = gcd(a, b)
    return resample_poly(x, b // g, a // g).astype(np.float32)


def read_mono(path, sr):
    x, r = sf.read(path, dtype="float32", always_2d=True)
    return resample(x.mean(1), r, sr)


def write(path, x, sr):
    sf.write(path, x, sr, subtype="PCM_16")


class Xfade:
    """Linear crossfade between consecutive blocks; adds `ov` samples latency."""

    def __init__(self, ov):
        self.ov, self.tail = ov, None

    def push(self, s):
        ov = min(self.ov, len(s)); k = len(s) - ov
        if self.tail is not None and len(self.tail) == ov:
            w = (np.arange(ov, dtype=np.float32) + 1) / (ov + 1)
            head = self.tail * (1 - w) + s[:ov] * w
        else:
            head = s[:ov]
        self.tail = s[k:].copy()
        return np.concatenate([head, s[ov:k]]).astype(np.float32)
