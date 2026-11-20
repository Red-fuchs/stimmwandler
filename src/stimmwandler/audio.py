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
