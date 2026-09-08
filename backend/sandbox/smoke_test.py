import asyncio
import io
import json

import numpy as np
import pandas as pd
import soundfile as sf
import websockets

URI = "ws://localhost:8000/ws/analyze"
WINDOW_SAMPLES = 16000 * 4  # 4-second window at 16kHz


def real_speech_pcm(n=WINDOW_SAMPLES):
    t = pd.read_parquet(r"C:\Users\Rohith\Downloads\SIH\real_speech.parquet")
    row = t.iloc[0]["audio"]
    data = row.get("bytes", b"") if isinstance(row, dict) else row
    a, sr = sf.read(io.BytesIO(data), dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    return (a[:n] * 32767).astype(np.int16)


async def main() -> None:
    async with websockets.connect(URI, open_timeout=10) as ws:
        await ws.send(json.dumps({
            "type": "session_init",
            "session_id": "smoke-test-01",
            "sample_rate": 16000,
        }))
        ack = json.loads(await ws.recv())
        assert ack["type"] == "session_ack", f"Expected ack, got {ack}"
        print("session_ack OK:", ack["status"])

        pcm = real_speech_pcm()
        bytes_per_frame = (16000 // 4) * 2  # 250ms = 4000 samples = 8000 bytes
        sent = 0
        while sent < len(pcm):
            chunk = pcm[sent:sent + bytes_per_frame].tobytes()
            await ws.send(chunk)
            sent += bytes_per_frame
            await asyncio.sleep(0.01)

        result = await asyncio.wait_for(ws.recv(), timeout=60)
        data = json.loads(result)
        assert data["type"] == "result", f"Expected result, got {data}"
        print("result verdict:", data["verdict"], "| risk:", data["risk_score"])
        print("signals:", [(s["name"], round(s["score"], 4)) for s in data["signals"]])
        print("optional:", data["optional_signals"])
        print("explanation:", data["explanation"])
        assert len(data["signals"]) == 4
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())