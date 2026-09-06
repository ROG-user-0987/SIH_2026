"""Smoke test for the remote inference server (backend/remote_server/server.py).

Assumes the server is already running (e.g. MODEL_NAME=aasist_local on :8001).
"""
import io
import sys

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio
import urllib.request
import json

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
N = 64600


def post_analyze(wav: np.ndarray) -> dict:
    req = urllib.request.Request(
        f"{BASE}/analyze?fmt=float32&sr=16000",
        data=wav[:N].astype(np.float32).tobytes() if len(wav) >= N else np.pad(
            wav, (0, N - len(wav))).astype(np.float32).tobytes(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def get_real(i):
    t = pd.read_parquet(r"C:\Users\Rohith\Downloads\SIH\real_speech.parquet")
    row = t.iloc[i]["audio"]
    data = row.get("bytes", b"") if isinstance(row, dict) else row
    a, sr = sf.read(io.BytesIO(data), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    return np.ascontiguousarray(a, dtype=np.float32)


def get_ai(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != 16000:
        a = torchaudio.transforms.Resample(sr, 16000)(torch.from_numpy(a).unsqueeze(0)).squeeze(0).numpy()
    return np.ascontiguousarray(a, dtype=np.float32)


with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
    health = json.loads(r.read().decode())
print("health:", health)
assert health.get("status") == "ok", f"server unhealthy: {health}"

for name, w in [
    ("real0", get_real(0)),
    ("ai_edge_tts", get_ai(r"C:\Users\Rohith\AppData\Local\Temp\opencode\ai_0.mp3")),
]:
    out = post_analyze(w)
    print(f"{name:12s} probs={out['probs']} spoof={out['spoof_prob']} lat={out['latency_ms']}ms device={out['device']}")

print("REMOTE SERVER SMOKE TEST PASSED")