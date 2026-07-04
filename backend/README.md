---
title: LeafLens API
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# LeafLens API

FastAPI backend for **LeafLens** — an AI crop-disease diagnosis and grounded
treatment-advice service for Bangladeshi farmers.

It serves two AI layers:

1. **Vision** — a fine-tuned MobileNetV2 that classifies a leaf photo into a
   crop/disease class with a confidence score.
2. **Advisory** — grounded treatment facts from a curated knowledge base, translated
   to Bangla + summarised by a Groq LLM (the LLM never invents treatments).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + whether the model is loaded |
| GET | `/labels` | Class list the model can predict |
| POST | `/predict` | Image -> diagnosis only |
| POST | `/advise` | Disease -> grounded bilingual advice |
| POST | `/diagnose` | Image -> diagnosis + advice + saved to history |
| GET | `/history` | Recent scans |

Interactive docs at `/docs`.

## Environment variables

- `GROQ_API_KEY` — Groq key (set as a Space secret)
- `GROQ_MODEL` — default `llama-3.3-70b-versatile`
- `FRONTEND_ORIGIN` — allowed CORS origin(s), comma-separated
- `FASAL_CONF_THRESHOLD` — min confidence before naming a disease (default 0.55)

The frontend lives in a separate repo and is deployed on Vercel.
