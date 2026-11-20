import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("STIMMWANDLER_HOME", Path.home() / ".stimmwandler"))
HF = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
HUB_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]


def paths(a):
    return Path(a.hubert or HOME / "hubert_base"), Path(a.rmvpe or HOME / "rmvpe.pt")


def dl(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    print("downloading", url, file=sys.stderr)
    urllib.request.urlretrieve(url, dst)


def cmd_download(a):
    for f in HUB_FILES:
        dl(HF + "hubert_base/" + f, HOME / "hubert_base" / f)
    dl(HF + "rmvpe.pt", HOME / "rmvpe.pt")
    print("models in", HOME)


def cmd_convert(a):
    from . import audio
    from .core import SR, Converter
    hub, rm = paths(a)
    c = Converter(a.model, hub, rm, a.device)
    x = audio.read_mono(a.input, SR)
    t = time.time()
    y = c.convert(x, a.pitch, a.index, a.index_rate, a.protect, a.rms_mix, a.sid)
    audio.write(a.output, y, c.sr)
    print(f"{len(x)/SR:.1f}s audio -> {a.output} ({c.sr} Hz) in {time.time()-t:.1f}s on {c.dev}")


def cmd_stream(a):
    import queue
    import numpy as np
    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return
    from . import audio
    from .core import SR, Converter
    hub, rm = paths(a)
    c = Converter(a.model, hub, rm, a.device)
    ctx = int(a.context * SR)
    dev = (a.input_device, a.output_device)
    sr = int(sd.query_devices(a.output_device, "output")["default_samplerate"])
    iq, oq = queue.Queue(), queue.Queue()

    def duplex(inp, out, frames, t, st):
        iq.put(inp[:, 0].copy())
        try:
            o = oq.get_nowait()
        except queue.Empty:
            o = np.zeros(0, np.float32)
        out[:] = 0
        n = min(frames, len(o))
        out[:n, :] = o[:n, None]

    print("streaming; Ctrl+C to stop", file=sys.stderr)
    hist = np.zeros(ctx, np.float32)
    with sd.Stream(samplerate=sr, blocksize=int(a.block * sr), device=dev, channels=(1, 2), dtype="float32", callback=duplex):
        while True:
            x = audio.resample(iq.get(), sr, SR)
            seg = np.concatenate([hist, x])
            hist = seg[-ctx:]
            y = c.convert(seg, a.pitch, a.index, a.index_rate, a.protect, 1.0, a.sid)
            oq.put(audio.resample(y[-int(len(x) * c.sr / SR):], c.sr, sr))


def cmd_export(a):
    from .export import export_all
    hub, rm = paths(a)
    export_all(a.model, hub, rm, Path(a.out), a.index, a.deterministic)


def cmd_prep_index(a):
    import numpy as np
    from .core import load_index
    np.save(a.output, load_index(a.index))
    print("saved", a.output)


def main():
    p = argparse.ArgumentParser(prog="stimmwandler")
    s = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("-m", "--model", required=True, help="RVC .pth")
        q.add_argument("--index", help=".index or .npy")
        q.add_argument("--index-rate", type=float, default=0.5)
        q.add_argument("--pitch", type=int, default=0, help="semitones")
        q.add_argument("--protect", type=float, default=0.33)
        q.add_argument("--sid", type=int, default=0)
        q.add_argument("--device", default="auto")
        q.add_argument("--hubert"), q.add_argument("--rmvpe")
    s.add_parser("download", help="fetch hubert_base + rmvpe from the official RVC HF repo").set_defaults(f=cmd_download)
    q = s.add_parser("convert"); common(q)
    q.add_argument("input"), q.add_argument("-o", "--output", required=True)
    q.add_argument("--rms-mix", type=float, default=0.25)
    q.set_defaults(f=cmd_convert)
    q = s.add_parser("stream"); common(q)
    q.add_argument("--input-device"), q.add_argument("--output-device")
    q.add_argument("--block", type=float, default=0.4), q.add_argument("--context", type=float, default=1.0)
    q.add_argument("--list-devices", action="store_true")
    q.set_defaults(f=cmd_stream)
    q = s.add_parser("export-onnx", help="export model + hubert + rmvpe to ONNX for the Rust app"); common(q)
    q.add_argument("--out", required=True)
    q.add_argument("--deterministic", action="store_true", help="zero noise (parity tests only)")
    q.set_defaults(f=cmd_export)
    q = s.add_parser("prep-index"); q.add_argument("index"), q.add_argument("output")
    q.set_defaults(f=cmd_prep_index)
    a = p.parse_args()
    a.f(a)
