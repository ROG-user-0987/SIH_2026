"""Stream any audio file through the full WS pipeline (like the frontend does).
Usage:  python stream_file_test.py <path-to-wav>
Expects: backend on ws://localhost:8000/ws/analyze
"""
import asyncio
import json
import sys

import numpy as np
import soundfile as sf
import websockets

URI = "ws://localhost:8000/ws/analyze"
WINDOW_SAMPLES = 16000 * 4
BYTES_PER_FRAME = (16000 // 4) * 2  # 250ms = 8000 bytes


def to_pcm16(path, sr=16000):
    a, a_sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if a_sr != sr:
        import scipy.signal as sg
        a = sg.resample_poly(a, sr, a_sr)
    if len(a) < WINDOW_SAMPLES:
        a = np.pad(a, (0, WINDOW_SAMPLES - len(a)))
    a = a / (np.max(np.abs(a)) + 1e-8)
    return (a * 32767).astype(np.int16)


async def main(path: str, label: str) -> None:
    pcm = to_pcm16(path)
    async with websockets.connect(URI, open_timeout=10) as ws:
        await ws.send(json.dumps({
            "type": "session_init",
            "session_id": f"file-test-{label}",
            "sample_rate": 16000,
        }))
        ack = json.loads(await ws.recv())
        assert ack["type"] == "session_ack", f"Expected ack, got {ack}"

        sent = 0
        while sent < len(pcm):
            await ws.send(pcm[sent:sent + BYTES_PER_FRAME].tobytes())
            sent += BYTES_PER_FRAME
            await asyncio.sleep(0.01)

        result = json.loads(await asyncio.wait_for(ws.recv(), timeout=90))
        print(f"[{label}] verdict: {result['verdict']} | risk: {result['risk_score']} | signals:",
              [(s["name"], round(s["score"], 3)) for s in result["signals"]])
        return result


if __name__ == "__main__":
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "clip"
    asyncio.run(main(path, label))