"""Enroll a known speaker from any audio file into the running backend.
Usage:  python enroll_speaker.py <speaker_id> <audio_file>
POSTs audio to http://localhost:8000/enroll (backend must be running).
"""
import base64
import sys

import httpx
import soundfile as sf

URL = "http://localhost:8000/enroll"


def main(speaker_id: str, path: str) -> None:
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        import scipy.signal as sg
        w = sg.resample_poly(w, 16000, sr)
    peak = max(float(w.max()), abs(float(w.min())), 1e-8)
    pcm = (w / peak * 32767).astype("int16").tobytes()
    body = {
        "speaker_id": speaker_id,
        "audio_b64": base64.b64encode(pcm).decode(),
        "sample_rate": 16000,
    }
    r = httpx.post(URL, json=body, timeout=120)
    r.raise_for_status()
    print(f"Enrolled '{speaker_id}' ->", r.json()["enrolled"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])