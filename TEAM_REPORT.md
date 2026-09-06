# SIH26104 — Voice Cloning Detection System: Team Report

**Date:** 2026-09-06
**Version:** 0.3 (real-mode, remote-inference architecture)
**Problem track:** SIH26104 — detect real-time voice cloning / deepfake speech and flag it as FAKE with honest confidence, while keeping genuine speech as REAL.

---

## 1. What We Built

A working end-to-end web MVP: a **live-mic voice-cloning detector** that streams audio from a browser, runs a deepfake classifier, and shows a live **risk score, verdict (FAKE/REAL), confidence, and an explainable breakdown** of *why*.

Key decisions locked in so far:

- **Frontend:** Next.js 14 + Tailwind (deployable to Vercel), streams 16 kHz PCM over WebSocket.
- **Backend:** FastAPI, consumes binary PCM frames, buffers 4-second analysis windows, runs VAD → detector(s) → fusion → explainability, and pushes a JSON `result` per window.
- **Inference strategy (final):** the heavy deepfake model lives on a **remote GPU** (Colab/RunPod/HF Space), NOT on team laptops. Laptops only run the thin backend + frontend. Remote model is swap-able without touching the UI contract.
- **Honesty/Explainability:** every result carries `rationale_tags`, per-signal scores, dominant signal, and model availability flags. The UI shows confidence, not a bare verdict.

---

## 2. Verified Results (empirically measured on this machine, 2026-09-06)

| Input | AASIST 2019-LA (clovaai, our local test model) | Consequence |
|---|---|---|
| Real LibriSpeech speech ("real0") | spoof prob **0.0000** → REAL | deterministic, correct |
| Real speech row 1 | spoof 0.0000 → REAL | correct |
| Edge-TTS neural voice (`ai_0.mp3`) | spoof **0.0008** → REAL | **limitation (documented)** |
| White noise | spoof 0.9902 → FAKE | correct-ish (not speech) |
| 220 Hz sine | spoof 1.0000 → FAKE | correct-ish |

Full-stack smoke test (WebSocket, real speech, remote-mode) — **PASSED**:

```
verdict: REAL | risk: 52
signals: [(ssl_aasist, 0.0), (aasist, 0.0), (prosody, 0.4505)]
optional: {remote_model: aasist-2019la, remote_latency_ms: 467.5, remote_loaded: True}
```

> **Critical, must be told to judges:** the *classic* AASIST (trained on ASVspoof2019-LA) does **NOT generalize to 2025-era neural TTS** (Edge-TTS/HiFi-GAN-style). It calls modern AI voices "REAL". This is a known, well-documented problem (Generalization gap; EER ≈ 43 % on InTheWild). **Any modern-AI voice that goes "REAL" is the 2019 checkpoint's failure, not a bug in the pipeline.** The fix is a modern checkpoint (Spectra-AASIST3 / SSL-AASIST / AASIST3 trained on ASVspoof5+MLAAD) — served remotely, once, on the GPU box (see §7).

Also measured while chasing options (so nobody re-wastes bandwidth):
- **MTUCI/AASIST3 released weights are degenerate** — output near-constant "bonafide" for silence/noise/speech. Do not use. (Upstream itself labels it deprecated.)
- The lab's replacement, **Spectra-AASIST3** (`lab260/Spectra-AASIST3`), is the one to serve remotely: self-contained, PyTorchModelHubMixin, XLS-R-300M front-end, KAN graph attention, **~3.3 GB weights** (300 M SSL model). Unranked but is the maintainers' recommended update.

---

## 3. Architecture (current)

```
Browser (Next.js)
  │  getUserMedia / WebSocket: int16 PCM ~250 ms frames
  ▼
Local FastAPI backend  (ws://<host>:8000/ws/analyze)
  │  session_init → session_ack handshake
  │  buffer → 4 s window (64000 samples @ 16 kHz)
  │  VAD (energy-based, zero-download; optional Silero)
  │  ▼
  │  detector paths (env-selected):
  │   ├─ REMOTE: POST /analyze (raw float32 PCM) → remote GPU server
  │   │          returns {logits, probs, spoof_prob, latency_ms, model}
  │   ├─ LOCAL : ONNX AASIST (needs local checkpoint, layout in backend/models)
  │   └─ MOCK  : deterministic-ish placeholder (offline demo only)
  │  ▼
  │  Prosody heuristics (fast fallback — see §6)
  │  │  Fusion (weighted logit → 0–100 risk, verdict, confidence)
  │  │  Explainability (rationale tags, dominant signal)
  │  ▼ result JSON
  ▼
Browser UI: RiskGauge + ResultPanel + ExplanationPanel + ConnectionStatus
```

**Signal semantics (UI-facing):** every `score` is **probability of SPOOF (0–1)**. Verdict thresholds: `risk_score >= 70 → FAKE`, else `REAL` (this maps Fusion output; window too quiet → `INSUFFICIENT_SPEECH`).

---

## 4. File Map

```
SIH/
├─ backend/                          FastAPI + Python
│  ├─ main.py                        app, env config, /health, /ws/analyze
│  ├─ .env.example                   config template (NEW remote vars)
│  ├─ vad/silero_vad.py              VAD (energy default; Silero via torch.hub if USE_SILERO_VAD=true)
│  ├─ detectors/
│  │  ├─ aasist.py                   local AASIST (ONNX path or mock fallback)
│  │  ├─ ssl_aasist.py               local SSL-AASIST (same pattern)
│  │  ├─ prosody.py                  prosody feature extractor + heuristic (librosa gated)
│  │  ├─ remote.py                   ★ RemoteDetector — calls remote /analyze over HTTPS
│  │  ├─ aasist/AASIST.py            vendored clovaai AASIST model class (self-contained)
│  │  └─ __init__.py
│  ├─ models/aasist/{AASIST.pth, aasist.onnx, meta.yaml, trt_aasist.py}  2019-LA weights
│  ├─ fusion/model.py                weighted fusion (ssl=.45 aasist=.35 prosody=.20)
│  ├─ explainability/                rationale tags + dominant-signal normalizer
│  ├─ websocket_handler/handler.py   session/streaming/windowing logic
│  ├─ sandbox/                       mock.py, smoke_test.py, remote_server_test.py
│  └─ remote_server/                 ★ deployable GPU inference service (goes to the GPU box)
│     ├─ server.py                   FastAPI: GET /health, POST /analyze
│     ├─ spectra_aasist3_model.py    vendored Spectra-AASIST3 model class
│     ├─ spectra_aasist3_model.py.d  (n/a)
│     ├─ requirements-remote.txt
│     ├─ Dockerfile
│     ├─ colab_run.py                boots server + cloudflared tunnel, prints public URL
│     └─ README.md                   full remote-deployment guide
├─ frontend/                         Next.js 14.2.35 + Tailwind
│  ├─ src/app/page.tsx               mic capture, Start/Stop, live results
│  ├─ src/lib/websocket-client.ts    WS client (session_init, sendAudio, result handling)
│  └─ src/components/                RiskGauge, ResultPanel, ExplanationPanel, ConnectionStatus
├─ start-backend.cmd / start-frontend.cmd   verified launchers
├─ real_speech.parquet               real-speech regression corpus (test-only)
└─ .env.example                      backend + frontend env template
```

---

## 5. WebSocket / API Contract (verified)

**Connect:** `ws://<host>:8000/ws/analyze`
1. Client → `{"type":"session_init","session_id":"...","sample_rate":16000}` → server `session_ack`.
2. Client streams **binary chunks: int16 PCM mono, little-endian, 16 kHz** (~250 ms each).
3. Server emits a `result` per 4 s window:

```json
{
  "type": "result",
  "window_ts_start": "...", "window_ts_end": "...",
  "risk_score": 52,
  "verdict": "REAL",
  "confidence": 0.51,
  "signals": [
    {"name":"ssl_aasist","category":"spectral","score":0.0,"weight":0.36,"loaded":true},
    {"name":"aasist","category":"spectral","score":0.0,"weight":0.28,"loaded":true},
    {"name":"prosody","category":"prosody","score":0.4505,"weight":0.16,"top_feature":"..."}
  ],
  "optional_signals": {"remote_model":"aasist-2019la","remote_latency_ms":467.5,"remote_loaded":true},
  "explanation": {"dominant_signal":"aasist","dominant_category":"spectral","rationale_tags":["natural_spectral_pattern","low_confidence_signal"],"explainability_version":"explain-v0.1"},
  "model_versions": {"fusion":"fusion-v0.3","ssl_aasist":"ssl-aasist-v0.1","aasist":"aasist-v0.1","prosody":"prosody-v0.1"}
}
```

**Remote inference HTTP API** (`backend/remote_server/server.py`):
- `GET  /health` → `{status, model, device, torch_cuda}`
- `POST /analyze?fmt=float32|int16&sr=16000` — body = raw PCM → `{model, model_display, device, latency_ms, nb_samp, len_in, logits, probs, spoof_prob}`
- Class convention per model: `spectra` → `probs[1]` is spoof; `aasist_local` (clovaai) → `probs[0]` is spoof.

---

## 6. Key Engineering Decisions & Findings

1. **Mock mode was the user-visible "random" bug.** Original default `MOCK_MODE=true` returned inverted/random labels. Real inference (even the imperfect AASIST) is deterministic and correct on real speech. Default is now `MOCK_MODE=false`.
2. **Preprocessing convention (must match the model):** AASIST takes raw waveform, **mono 16 kHz float32**, **window = exactly 64600 samples** (`_pad_trim` in server). Clovaai convention: `softmax(logits)[0] = spoof probability`. Empirically nailed down with probes before wiring.
3. **The 2019 generalization gap is the #1 demo risk.** See §2. Mitigation: serve a modern checkpoint remotely; keep honest UX (confidence + rationale), never fake a verdict.
4. **Slow local libraries kill real-time:** librosa `pyin` on 4 s windows took seconds on CPU. Gated behind `USE_LIBROSA_PROSODY` (default off → fast autocorrelation fallback). VAD defaults to energy-based so nothing downloads at runtime (`USE_SILERO_VAD=true` opts back in).
5. **Zero team-laptop downloads:** all heavy artifacts (weights, SSL encoder) download **once, on the GPU host**. Local machine needs only pip deps (already installed) + npm packages.
6. **Robustness:** on remote unreachable, backend returns a neutral low-confidence fallback (`remote_loaded:false`) instead of crashing — the UI can show "model unavailable" honestly.
7. **Fusion weights** `[ssl .45, aasist .35, prosody .20]` — in remote mode the strong model feeds both the `ssl_aasist` and `aasist` slots (documented), so it dominates; prosody stays as a secondary heuristic. Retrain/retune when a labelled validation set exists.

---

## 7. Deployment (How to run it)

### A) Remote GPU service (the only place with the big model)
Pick one (details in `backend/remote_server/README.md`):
- **Hugging Face Space (Docker)** — easiest public HTTPS URL. Upload `server.py`, `spectra_aasist3_model.py`, `requirements-remote.txt`(→`requirements.txt`), `Dockerfile`. Set `MODEL_NAME=spectra`. First boot downloads ~3.3 GB on the Space.
- **RunPod / Lambda** — any NVIDIA GPU pod (T4/A10+, ≥6 GB VRAM): `pip install -r requirements-remote.txt` then `MODEL_NAME=spectra uvicorn server:app --port 8001`, expose port.
- **Google Colab (free T4)** — `%pip install -r requirements-remote.txt` then `%run colab_run.py` → prints a `trycloudflare.com` public URL. Session-restart caveat: re-run when Colab recycles.

### B) Local backend + frontend
```
# backend (project root)
set MOCK_MODE=false
set REMOTE_INFERENCE_URL=https://<your-remote-url>
backend\.venv\Scripts\python -m uvicorn backend.main:app --port 8000

# frontend
cd frontend && npm install && npm run dev
# open http://localhost:3000
```
Vercel: set `NEXT_PUBLIC_WS_URL=wss://<tunnel-or-backend-host>/ws/analyze`.

### C) Local-only test mode (dev, no GPU)
`MODEL_NAME=aasist_local PORT=8001 python backend/remote_server/server.py` serves the small 2019 checkpoint we already ship. Points the local backend at `http://127.0.0.1:8001`.

---

## 8. Configuration Reference

| Env var | Default | Purpose |
|---|---|---|
| `MOCK_MODE` | `false` | `true` = offline placeholder signals only |
| `REMOTE_INFERENCE_URL` | — | remote GPU service base URL (recommended mode) |
| `MODEL_DIR` | — | local model dir if using local ONNX checkpoints |
| `USE_SILERO_VAD` | `false` | `true` = download/use Silero VAD at runtime |
| `USE_LIBROSA_PROSODY` | `false` | `true` = slower but richer prosody features |
| `ALLOWED_ORIGINS` | `*` | CORS |
| `PORT` / `HOST` | 8000 / 0.0.0.0 | backend bind |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws/analyze` | frontend WS endpoint |
| (remote server) `MODEL_NAME` | `spectra` | `spectra` \| `aasist_local` |
| (remote server) `PORT` | 8001 | remote service port |

---

## 9. Tests (all green, ready to re-run)

```
backend\.venv\Scripts\python backend\sandbox\remote_server_test.py http://127.0.0.1:8001   # remote API
backend\.venv\Scripts\python backend\sandbox\smoke_test.py                                   # full WS loop
```
`smoke_test.py` streams real LibriSpeech PCM over the WebSocket and asserts a `result` with 3 signals + explanation. Mock-mode plumbing was proven earlier; real-mode remote plumbing proven today.

---

## 10. Known Limitations & Risks (be upfront in the demo)

1. **Modern TTS may be labelled REAL** by the 2019 AASIST (see §2). Demo with a modern checkpoint served remotely, or acknowledge the limitation and show explainability.
2. **Remote inference latency**: ~0.4–0.6 s per 4 s window measured (local test); add WAN round-trip (~0.1–0.5 s). Returns one result per ~4.5 s of audio — acceptable for the demo.
3. **CPU-only torch locally**: heavy models cannot run in real time on laptops (measured: CPU torch build only). That's why they're remote.
4. **File upload is now implemented** (frontend decodes any audio file in-browser via `AudioContext.decodeAudioData`, resamples to 16 kHz mono, and streams the same binary path the mic uses; progress shown per chunk).
5. **Fusion/prosody are heuristic, not trained**: fine for demo; retrain once labelled data exists.
6. **Privacy flip**: audio leaves the device to the remote GPU. Fine for a competition demo; note it if the brief demands on-prem processing.

## 11. Next Steps (suggested order)

1. **Deploy the remote SDK on a GPU host** (only remaining item depending on external infra): set `REMOTE_INFERENCE_URL`; verify Edge-TTS clip → FAKE with Spectra-AASIST3 *before the demo*.
2. Batch on the test corpus to calibrate fusion weights + verdict thresholds.
3. Add a small "model status / limitations" banner in the UI that reflects `remote_loaded` and `model`.
4. Containerize the local backend (`Dockerfile` + `docker-compose` with trusted tunnel) for a stable prod path to Vercel.

> Completed since v0.3 report: file-upload analysis (mic + file share the same streaming path), full-stack remote-mode smoke test, remote API smoke test.