# Voice Cloning Detection System — Web MVP

**SIH26104** | PRD v0.3 | September 2026

## Architecture

```
Browser (Next.js on Vercel)
  └─ WebSocket (PCM audio frames, ~250ms)
      └─ Cloudflare Tunnel
          └─ FastAPI (GPU machine)
              ├─ Silero VAD → 2-4s windows
              ├─ SSL-AASIST (XLS-R encoder → graph-attention)
              ├─ AASIST (raw conv + graph-attention)
              ├─ Prosody/XGBoost (~30-dim feature vector)
              ├─ Fusion (logistic regression, calibrated 0-100)
              └─ Explainability (rationale tags, signal breakdown)
```

## Quick Start

### Backend (GPU machine)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Mock mode (no models needed) — run from the project root
set MOCK_MODE=true
..\backend\.venv\Scripts\python -m uvicorn backend.main:app --port 8000

# Real mode (requires model files in backend/models/)
set MOCK_MODE=false
set MODEL_DIR=models
..\backend\.venv\Scripts\python -m uvicorn backend.main:app --port 8000
```

### Frontend

```bash
cd frontend
npm install
set NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/analyze
npm run dev
```

Open http://localhost:3000

## Deployment

### Vercel (Hobby, free)
1. Push frontend to GitHub
2. Import in Vercel → automatic Next.js detection
3. Set env var `NEXT_PUBLIC_WS_URL=wss://<your-tunnel-url>/ws/analyze`
4. Deploy

### Cloudflare Tunnel
```bash
cloudflared tunnel --url localhost:8000
```
For persistent URL, use a named tunnel with a custom domain.

## Model Files

Download to `backend/models/` (or set `MODEL_DIR`):

| Model | Format | Source |
|-------|--------|--------|
| AASIST | ONNX | [AASIST repo](https://github.com/clovaai/AASIST) |
| SSL-AASIST | ONNX | SSL-AASIST checkpoints |
| Prosody XGBoost | JSON | Train from prosody features |
| Fusion LR | PKL | Train on labelled validation set |

In mock mode, all detectors return plausible synthetic/real scores.

## Project Structure

```
SIH/
├── frontend/          Next.js 14 + Tailwind
│   └── src/
│       ├── app/       Pages + layout
│       ├── components/  RiskGauge, ExplanationPanel, etc.
│       └── lib/       WebSocket client
├── backend/           FastAPI + Python
│   ├── vad/           Silero VAD
│   ├── detectors/     AASIST, SSL-AASIST, Prosody
│   ├── fusion/        Logistic regression fusion
│   ├── explainability/ Tag vocabulary, rationale generation
│   ├── websocket_handler/ Session management
│   └── sandbox/       Mock mode
└── test-audio/        Test samples with provenance
```

## WebSocket Contract

**Connect:** `ws://<host>:8000/ws/analyze`

1. Client sends `session_init` JSON with `session_id` and `sample_rate`
2. Server responds `session_ack`
3. Client streams binary frames (16-bit PCM mono, ~250ms each)
4. Server returns `result` JSON per processed 4s window:

```json
{
  "type": "result",
  "risk_score": 78,
  "verdict": "FAKE",
  "confidence": 0.89,
  "signals": [...],
  "explanation": {
    "dominant_signal": "ssl_aasist",
    "dominant_category": "spectral",
    "rationale_tags": ["unnatural_spectral_pattern"],
    "explainability_version": "explain-v0.1"
  }
}
```

## License

SIH26104 — Smart India Hackathon 2026
"# SIH_2026" 
"# AI-Clone-Detector" 
