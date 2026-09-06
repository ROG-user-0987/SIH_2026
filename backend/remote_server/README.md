# Remote Inference Mode

The heavyweight deepfake model (Spectra-AASIST3, XLS-R 300M front-end: ~3.3 GB in
weights) **lives on a GPU machine you rent/free-tier**, not on your laptop. Your
laptop only runs the thin FastAPI backend + Next.js frontend and streams audio to
the remote inference service over HTTPS.

```
Browser mic/file audio
   -> local backend  (/ws/analyze, WebSocket, 4 s windows)
        -> remote GPU server  (POST /analyze, raw PCM float32)
        <- logits / spoof probability
   -> fusion + explanation (local) -> JSON result back to browser
```

## 1. Deploy the remote server (choose ONE)

### A) Hugging Face Spaces (easiest, HTTPS + public URL built-in)
1. Create a new Space (Docker SDK, "Free" generous for SpaceX… GPU spaces are
   available on paid tiers; a CPU space also works, just slower).
2. Upload these into the Space repo:
   - `server.py`
   - `spectra_aasist3_model.py`
   - `requirements-remote.txt` (rename to `requirements.txt`)
   - `Dockerfile`
3. Set env var `MODEL_NAME=spectra`. The Space auto-builds and downloads the model
   on first start. Your Space's public URL is `https://<user>-<space>.hf.space`.
4. Point the local app at it (step 2).

### B) RunPod / Lambda / any NVIDIA GPU pod
1. Launch a pod with a `pytorch/pytorch` runtime (T4/A10/P100 is plenty — 6 GB+ VRAM).
2. Copy this folder to the pod: `backend/remote_server/`.
3. `pip install -r requirements-remote.txt`
4. `MODEL_NAME=spectra uvicorn server:app --host 0.0.0.0 --port 8001`
5. Expose port 8001 -> RunPod gives you a public `https://...runpod.net` URL.

### C) Google Colab (free T4)
1. Upload this project (or just `backend/remote_server/`).
2. In a cell:
   ```
   %pip install -q -r backend/remote_server/requirements-remote.txt
   %run backend/remote_server/colab_run.py
   ```
3. The printed `https://...trycloudflare.com` URL is your endpoint.
   (Colab replaces the GPU periodically; restart and rerun as needed. Best for
   demo time, not long-running service.)

> First start downloads ~3.3 GB of weights **on the remote machine only** — once.
> Your laptop downloads nothing.

## 2. Point the local backend at it

In `backend/.env`:

```
MOCK_MODE=false
REMOTE_INFERENCE_URL=https://<your-remote-host>/   # <- from step 1
```

The WebSocket app already forwards every 4-second window to that URL and returns
the fused result to the browser. If the remote URL is unreachable, the backend logs
an error and returns a low-confidence fallback (instead of crashing).

## 3. Local-only test mode (no downloads)

You can also run the server with `MODEL_NAME=aasist_local`, which serves the tiny
clovaai AASIST checkpoint already present in `backend/models/aasist/` on this repo:

```
MODEL_NAME=aasist_local PORT=8001 python server.py
```

Same API; useful to validate the pipeline before renting a GPU.

## API

- `GET  /health` -> `{status, model, device, torch_cuda}`
- `POST /analyze` — body: **raw PCM mono float32 @ 16 kHz** (4 s window = 256 000
  bytes). Query params: `fmt=float32|int16`.
  Returns:
  ```json
  {
    "model": "spectra-aasist3",
    "device": "cuda:0",
    "latency_ms": 412.0,
    "nb_samp": 64600,
    "logits": [-1.05, 2.31],
    "probs": [0.03, 0.97],
    "spoof_prob": 0.97
  }
  ```

> `spoof_prob` is the probability of the **spoofed** class per the served model's
> convention (`spectra`: column 1; `aasist_local`: column 0).