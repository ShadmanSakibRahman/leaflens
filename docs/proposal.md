# Project Proposal — LeafLens

**Author:** Md. Shadman Sakib Rahman
**Project type:** AI Capstone (Frontend + Backend + AI/ML)
**Date:** July 2026

## Problem statement

Agriculture supports close to 40% of the workforce in Bangladesh, and most of that is
smallholder farming. One of the biggest, most avoidable causes of crop loss is disease
that gets spotted too late or treated wrongly. A farmer who sees strange spots on a leaf
usually has no quick way to know what it is. The nearest agriculture officer may be a bus
ride away, and by the time advice arrives the disease has spread. The common fallback —
asking a shopkeeper and spraying whatever pesticide is on hand — wastes money, harms the
soil, and often does not even target the real problem.

So the core problem is a **diagnosis-and-advice gap**: farmers can see symptoms but cannot
reliably name the disease or find the right, safe treatment in time and in a language they
read comfortably.

## Proposed solution

**LeafLens** is a mobile-first web app that turns a phone camera into a pocket crop
doctor. The farmer photographs an affected leaf and, within a couple of seconds, gets:

1. a **diagnosis** — which crop and which disease, with a confidence score, and
2. a **treatment plan** in plain Bangla (and English) — organic/cultural steps, chemical
   options, prevention, and when to consult a real expert.

Two design choices make it trustworthy rather than a gimmick:

- If the model is not confident, it **refuses to guess** and tells the farmer to retake the
  photo or consult an officer. A wrong confident answer is worse than an honest "not sure".
- The treatment facts come from a **curated knowledge base**, not from an LLM's imagination.
  The language model is only used to translate those verified facts into friendly Bangla.
  This keeps the advice grounded and safe.

## AI approach

The system uses **two AI components that work together**:

- **Vision model (diagnosis):** transfer learning on **MobileNetV2** (pretrained on
  ImageNet, with a fine-tuned classifier head). It is trained on real leaf-photo datasets
  covering major Bangladeshi crops — rice, potato, tomato, maize and pepper — across roughly
  twenty disease/healthy classes. On a held-out test set it reaches about **95.9%**
  accuracy. Output is a class label plus a softmax confidence used by the safety gate.

- **Advisory model (grounded generation):** a retrieval step pulls the treatment sheet for
  the diagnosed disease from the knowledge base, and a **Groq-hosted LLM
  (llama-3.3-70b)** rewrites those exact facts into simple Bangla with a short summary. The
  LLM is instructed to use only the provided facts, so it cannot fabricate treatments. If
  the LLM service is down, the app falls back to the verified English facts.

## Technology stack

- **Frontend:** Next.js (React), mobile-first, camera/upload, Bangla⇄English toggle, scan
  history. The live demo is deployed on **GitHub Pages** as an on-device app (the model is
  exported to ONNX and runs in the browser via onnxruntime-web), which keeps it reachable
  from Bangladesh.
- **Backend:** **FastAPI** (Python), containerised with Docker. REST endpoints for
  prediction, advice, diagnosis and history; input validation and graceful error handling.
  Runs locally and deploys to any Docker host.
- **Database:** SQLite for scan history (browser localStorage in the on-device build).
- **AI/ML:** PyTorch + torchvision (MobileNetV2), scikit-learn for evaluation, Groq API for
  the advisory language layer.
- **Tooling:** Git/GitHub, Hugging Face `datasets` for data, Colab for GPU training.

## Expected impact

A farmer with a cheap smartphone gets an instant, grounded second opinion in their own
language, reducing wrong pesticide use and catching disease earlier. The same architecture
extends naturally to more crops and diseases, and could later run as an offline app or plug
into a government agriculture hotline.
