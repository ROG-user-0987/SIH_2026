# Model Files Directory

Place trained `model` files here and set `MODEL_DIR` env var (or leave unset to use this folder).

## Expected files (real mode)

| File | Detector | Notes |
|------|----------|-------|
| `ssl_aasist.onnx` | SSL-AASIST | XLS-R encoder → AASIST graph-attention. Input: log-mel (128 x ~200), output: spoof prob |
| `aasist.onnx` | AASIST (raw) | Conv + graph-attention path. Input: 80-band log-mel |
| `prosody_xgb.json` | Prosody | XGBoost classifier trained on the ~31-dim feature vector (see `backend/detectors/prosody.py`) |
| `fusion_lr.pkl` | Fusion | sklearn LogisticRegression (or joblib MLP) trained on (ssl, aasist, prosody) sub-scores |

## How to generate

1. Export AASIST / SSL-AASIST PyTorch weights to ONNX with `torch.onnx.export`.
2. Train prosody XGBoost:
   `python -m backend.detectors.prosody --train` (add a CLI trainer later, or use your own script on test-audio features).
3. Train fusion LR on your labelled validation set from the three detector sub-scores.

In mock mode (`MOCK_MODE=true`), detectors run with heuristic/random fallback so the full pipeline is demoable without these files.