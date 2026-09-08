"""One-shot verify: score real + fake clips through the GPU inference server.
Usage:  python verify_clips.py
Expects remote server up on 127.0.0.1:8001. Expects REAL<0.2, FAKE>0.8.
"""
import warnings
warnings.filterwarnings("ignore")

import httpx
import numpy as np
import scipy.signal as sg
import soundfile as sf

URL = "http://127.0.0.1:8001/analyze"
CLIPS = [
    (r"C:\Users\Rohith\Downloads\SIH\test-audio\real_human.wav", "REAL"),
    (r"C:\Users\Rohith\Downloads\SIH\test-audio\fake_neerja.wav", "FAKE"),
]


def score(path: str):
    w, sr = sf.read(path)
    if sr != 16000:
        w = sg.resample_poly(w, 16000, sr)
    w = w.astype(np.float32)
    z = (w - w.mean()) / (w.std() + 1e-8)
    r = httpx.post(URL, params={"fmt": "float32", "sr": 16000}, content=z.tobytes(), timeout=30)
    r.raise_for_status()
    d = r.json()
    return d["spoof_prob"], d.get("latency_ms"), d.get("model")


ok = True
for path, label in CLIPS:
    try:
        p, lat, model = score(path)
        good = (p < 0.2) if label == "REAL" else (p > 0.8)
        ok = ok and good
        print(f"[{'PASS' if good else 'FAIL'}] {label:5s} spoof_prob={p:.4f} ({lat}ms) {model}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {label} error: {e}")

print("\nALL OK" if ok else "\nSOME FAILED - check server/model")