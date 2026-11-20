# Stimmwandler handoff

## Done (3 local commits, nothing pushed)
- Python pkg `src/stimmwandler`: vendored RVC model code (MIT, `rvc/`; attention rewritten with gather/one-hot for dynamic ONNX length, verified equal to upstream), `core.py` (Converter, RMVPE, kNN index blend), `cli.py` (download/convert/stream/export-onnx/prep-index), `export.py`.
- Parity (tests/parity.py): hubert/rmvpe/synth ONNX vs torch rel err <1.5e-4 at several lengths. tests/e2e_compare.py: Python vs Rust CLI corr=1.0000 (with and without faiss index; random-weight dummy model, noise zeroed).
- Rust `rust/`: ort rc.15 (Cargo says rc.12 caret, resolves rc.15, needs Environment), cpal 0.18, eframe 0.37. bins: stimmwandler-rs (convert/stream), stimmwandler-gui. Builds on macOS.
- CI `.github/workflows/release.yml` (untriggered), README, LICENSE, THIRD_PARTY_NOTICES.

## Update 2026-11-20
- Done: ort pinned =2.0.0-rc.15; Xfade (50 ms) in Rust dsp.rs + Python audio.py with tests; kNN-VC Python engine (src/stimmwandler/knn, `stimmwandler knn`), tested with macOS `say` audio; README/NOTICES updated. kNN-VC ONNX/Rust not done (documented Python-only).
## Not done / next
1. Real-device streaming, Windows/Linux builds + CI lib copying untested.
2. No real voice model tested; nono models Python only; Rust lacks fcpe/pm; highpass differs marginally.
3. kNN-VC: optional ONNX export, streaming variant.
