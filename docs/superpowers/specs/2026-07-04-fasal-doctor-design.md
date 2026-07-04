# LeafLens (ফসল ডাক্তার) — Design Spec

**Date:** 2026-07-04
**Author:** Md. Shadman Sakib Rahman
**Type:** AI Capstone Project (Frontend + Backend + AI/ML)

## 1. Problem statement

Agriculture employs roughly 40% of Bangladesh's workforce, and crop disease is one of
the biggest causes of yield and income loss. Most smallholder farmers have no timely
access to an agronomist, so diseases get misdiagnosed and pesticides get over- or
mis-applied — which wastes money, harms soil and health, and still loses the crop.

**LeafLens** is a mobile-first web app that turns a phone camera into a pocket
agronomist. A farmer photographs a diseased leaf and instantly gets (1) a diagnosis with
a confidence score and (2) a grounded, plain-Bangla treatment and prevention plan.

## 2. Goals and non-goals

**Goals**
- Diagnose common diseases across major Bangladeshi crops from a single leaf photo.
- Give actionable, *grounded* advice (organic option, chemical option, prevention) that
  the model cannot fabricate.
- Ship a real, decoupled, production-shaped system: React frontend + FastAPI backend +
  database, both deployed and reachable from one public URL.
- Report real accuracy metrics on a held-out test set.

**Non-goals (explicitly cut for scope)**
- User accounts / authentication.
- Offline/PWA mode.
- Languages beyond Bangla + English.
- Pest/insect identification (disease only).

## 3. Architecture (decoupled)

```
React / Next.js (Vercel)            FastAPI backend (HF Spaces, Docker)
  camera + upload      --HTTPS-->     POST /predict   image -> {crop, disease, conf}
  result card                        POST /advise    disease -> grounded advice
  scan history         <--JSON---     GET  /history   past scans
  BN / EN toggle                     GET  /health     liveness
                                     SQLite: scans table
```

- **Frontend:** Next.js (React) on Vercel. The single URL a grader opens. Mobile-first,
  camera/upload input, result card, scan-history screen, Bangla/English toggle, and a
  friendly "backend waking up" state for free-tier cold starts.
- **Backend:** FastAPI in a Docker container on Hugging Face Spaces (free tier has enough
  RAM to serve a PyTorch model reliably). Serves both AI layers and the database.
- **Database:** SQLite table of scans (id, crop, disease, confidence, advice, created_at).

## 4. AI design (two models, working together)

### Layer 1 — Vision (diagnosis)
- Transfer learning on **MobileNetV2 / EfficientNet-B0** (ImageNet-pretrained backbone,
  fine-tuned classifier head).
- Trained on a curated multi-crop dataset (Bangladeshi staples: potato, tomato, maize,
  pepper, and rice if a clean public source is available) — disease classes + healthy.
- Output: predicted class + softmax **confidence**.
- **Confidence gate:** below a threshold, the app declines to guess and tells the user to
  retake the photo or consult an expert. This is the anti-hallucination / safety behavior
  for the vision side.
- Reported metrics: overall accuracy, per-class precision/recall/F1, confusion matrix.

### Layer 2 — Advisory (RAG + LLM)
- A curated **treatment knowledge base** (JSON): one entry per disease with symptoms,
  organic treatment, chemical treatment, prevention, and a "when to call an expert" note.
- On diagnosis, the backend **retrieves** the KB entry for that disease and passes those
  facts to a **Groq LLM** (llama-3.3-70b) which rewrites them into friendly Bangla +
  English. The LLM is instructed to use ONLY the retrieved facts — it cannot invent
  treatments. This is the anti-hallucination guarantee.
- **Fallback:** if Groq is unreachable, the backend returns the raw KB entry directly, so
  the feature degrades gracefully instead of failing.

## 5. Data flow

1. Farmer selects/takes a leaf photo in the React app.
2. `POST /predict` (multipart image) -> backend runs vision model -> `{crop, disease, confidence}`.
3. `POST /advise` (`{disease}`) -> backend retrieves KB entry -> Groq composes grounded
   BN/EN advice -> returns structured advice.
4. Result card renders; the scan is written to SQLite.
5. `GET /history` lists recent scans on the history screen.

## 6. Error handling & robustness (Backend criterion)

- Validate upload MIME type and size; reject non-images with clear errors.
- Low-confidence path returns an "uncertain" result rather than a wrong label.
- Groq failure -> KB fallback (never a hard failure).
- CORS restricted to the frontend origin.
- `/health` endpoint for liveness and cold-start detection.
- Request logging and timeouts.

## 7. Deployment

- Frontend -> Vercel (public URL, calls the backend).
- Backend + model -> Hugging Face Spaces (Docker).
- **Single-deploy fallback:** if wiring two hosts runs long, the FastAPI app can serve the
  built React static bundle itself on one Space, guaranteeing "accessible via URL."

## 8. Repository & deliverables

- One monorepo `fasal-doctor` under `ShadmanSakibRahman`: `frontend/`, `backend/`, `ml/`,
  `docs/`. Clean incremental commits.
- `README.md` — plain student voice, setup instructions, screenshots, methodology.
- `docs/proposal.md` — 1–2 page project proposal.
- `docs/final_report.md` — 3–5 page final report with metrics, design decisions,
  AI workflow, honest limitations, conclusions.
- `.env` gitignored; post-push secret scan must return 0.

## 9. Rubric mapping (out of 100)

| Criterion | Marks | How this design earns it |
|---|---|---|
| Problem Definition | 10 | Compelling, quantified BD agriculture problem; clear scope/goals above. |
| Frontend | 10 | Polished mobile-first Next.js UI, camera input, BN/EN, usable result cards. |
| Backend | 15 | Real decoupled FastAPI, REST endpoints, SQLite, validation + error handling. |
| AI Integration | 25 | Two integrated models (vision + RAG/LLM), reported metrics, confidence gate. |
| Deployment | 10 | Live on Vercel + HF Spaces, verified end-to-end, single URL. |
| Git/GitHub | 10 | Organized monorepo, incremental commits, clear README + docs. |
| Documentation & Report | 15 | Proposal + 3–5pg report with results and insights. |
| Innovation & Impact | 5 | Bangla-first, grounded anti-hallucination advisor for a real BD problem. |

## 10. Known risks & mitigations

- **No GPU** -> train a compact model on CPU on a subsampled dataset; keep epochs low.
- **Dataset availability** -> source from Hugging Face Hub (no Kaggle key); include rice
  only if a clean public source exists, otherwise document it as future work honestly.
- **Free-tier cold start** -> "waking up" UI state; `/health` ping on load.
- **Two-deploy fragility** -> single-deploy fallback kept ready.
