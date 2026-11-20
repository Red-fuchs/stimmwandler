# Stimmwandler handoff

## Done (3 local commits, nothing pushed)
- Python pkg `src/stimmwandler`: vendored RVC model code (MIT, `rvc/`; attention rewritten with gather/one-hot for dynamic ONNX length, verified equal to upstream), `core.py` (Converter, RMVPE, kNN index blend), `cli.py` (download/convert/stream/export-onnx/prep-index), `export.py`.
- Parity (tests/parity.py): hubert/rmvpe/synth ONNX vs torch rel err <1.5e-4 at several lengths. tests/e2e_compare.py: Python vs Rust CLI corr=1.0000 (with and without faiss index; random-weight dummy model, noise zeroed).
- Rust `rust/`: ort rc.15 (Cargo says rc.12 caret, resolves rc.15, needs Environment), cpal 0.18, eframe 0.37. bins: stimmwandler-rs (convert/stream), stimmwandler-gui. Builds on macOS.
- CI `.github/workflows/release.yml` (untriggered), README, LICENSE, THIRD_PARTY_NOTICES.

## Not done / next
1. kNN-VC (bshall/knn-vc, MIT, clone to $TMPDIR/upstream/knn-vc already present): Python port (WavLM-Large layer 6 + prematched HiFi-GAN, weights from the repo's GitHub releases), `stimmwandler knn` CLI, optionally ONNX.
2. Realtime stream quality untested with real audio devices (no GUI/mic run); no crossfade between blocks yet (clicks likely at block edges).
3. Windows/Linux builds untested; ort download-binaries lib copying in CI is a guess (dll/so must sit next to exe; verify).
4. No real voice model tested (only random weights); nono (no-f0) models only in Python, not ONNX.
5. Rust highpass is forward-backward SOS without edge padding (differs marginally from scipy filtfilt). Rust lacks fcpe/pm f0.
6. Test data: base models cached in $TMPDIR/models (may be cleared); dummy in $TMPDIR/t.
