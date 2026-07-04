# Final Report — LeafLens

**An AI crop-disease diagnosis and advice system for Bangladeshi farmers**

**Author:** Md. Shadman Sakib Rahman
**Course:** AI Capstone Project
**Date:** July 2026

---

## 1. Introduction and the problem I chose

I grew up around people for whom farming is not a hobby but the whole household income, so
I wanted my capstone to solve something that actually matters to them rather than another
generic demo. In Bangladesh agriculture supports close to 40% of the workforce, and a huge
share of avoidable loss comes from crop disease that is caught late or treated with the
wrong chemical.

The specific gap I set out to close is this: a farmer can *see* that a leaf looks wrong, but
has no fast, trustworthy way to know *what* it is or *what to do* about it, in a language
they read comfortably. The usual path — ask a shop, buy some pesticide, spray and hope — is
expensive and often misses the real disease.

**LeafLens** is my answer: photograph a leaf, get a diagnosis and a grounded treatment
plan in Bangla in a couple of seconds.

## 2. What I built (system overview)

The system is deliberately built as three separable parts so each one can be understood and
tested on its own:

- **Frontend** — a mobile-first Next.js (React) web app. The farmer takes or uploads a
  photo, sees the diagnosis and advice, can flip the whole interface between Bangla and
  English, and can scroll a short history of past scans.
- **Backend** — a FastAPI service that runs the AI and stores history. It exposes a small
  REST API (`/predict`, `/advise`, `/diagnose`, `/history`, `/health`) and a SQLite
  database.
- **AI** — two models working together: a vision model that names the disease, and a
  retrieval-plus-LLM layer that turns the diagnosis into safe, grounded advice.

The frontend is deployed on Vercel and the backend on Hugging Face Spaces; the farmer only
ever sees one URL.

```
React app (Vercel)  --HTTPS-->  FastAPI (Hugging Face Spaces)
  camera + advice               vision model + knowledge base + Groq
                                SQLite history
```

## 3. The AI workflow in detail

### 3.1 Vision model — naming the disease

I used **transfer learning** on **MobileNetV2**, which is pretrained on ImageNet. I froze
the convolutional backbone and trained only a new classifier head sized to my classes. I
picked MobileNetV2 on purpose: it is small and fast enough to run on a free CPU server,
which matters because the whole point is cheap, accessible deployment.

**Data.** I combined two public, no-login datasets from the Hugging Face Hub:
`BrandonFors/Plant-Diseases-PlantVillage-Dataset` (for potato, tomato, maize and pepper) and
`sharmin3/Rice-Leaf-Disease` (for rice). I mapped both into one clean set of canonical
`Crop___Disease` classes so the model labels, the treatment knowledge base and the UI all
line up. I capped images per class to keep the classes balanced and training quick, and
split the data into train/validation/test in a stratified way so every class appears in
every split.

**Training and results.** The model was trained for a modest number of epochs with Adam and
cross-entropy loss, keeping the best validation checkpoint. On the held-out **test set** it
reached:

- **Test accuracy: {{TEST_ACC}}**
- **Macro F1: {{MACRO_F1}}**
- Number of classes: **{{NUM_CLASSES}}** across 5 crops

Per-class precision/recall/F1 and a confusion matrix are saved with the model
(`backend/model/metrics.json`, `confusion_matrix.png`). {{METRIC_COMMENT}}

**Safety gate.** The model returns a softmax confidence. Below a threshold (0.55) the app
does **not** name a disease — it tells the farmer to retake a clearer photo or consult an
officer. I added this because in this domain a confident wrong answer (spray the wrong
chemical) is more harmful than an honest "I'm not sure".

### 3.2 Advisory layer — turning a label into safe advice

This is the part I most wanted to get right, because it is where language models usually go
wrong: they happily invent plausible-sounding but fake treatments. My rule was that the app
must never fabricate agricultural advice.

So the treatment facts live in a **curated knowledge base** (`knowledge_base.json`): one
entry per disease with symptoms, organic/cultural control, chemical control, prevention, and
a "when to call an expert" note. When the vision model returns a disease, the backend
**retrieves** that entry. The **Groq LLM (llama-3.3-70b)** is then given *only* those facts
and asked to translate them into simple Bangla and write a two-sentence summary. It is
explicitly instructed not to add anything not in the facts.

If Groq is unreachable, the backend simply returns the verified English facts. The feature
degrades; it never fabricates.

## 4. Design decisions and trade-offs

- **Decoupled frontend/backend instead of one Streamlit app.** More work (two deploys,
  CORS), but it gives a real API and database, which is both better engineering and a more
  honest "backend". It also means the model server and the UI can scale or change
  independently.
- **MobileNetV2, frozen backbone.** I traded a little accuracy for a model that trains in
  minutes on a CPU and serves fast on free hosting. For this use case, deployability beats a
  fractional accuracy gain.
- **Knowledge base + LLM instead of a pure LLM.** Slightly less "magical", but grounded and
  safe. In agriculture, wrong advice has a real cost, so I chose trust over fluency.
- **Confidence gate.** I accept that the app will sometimes say "not sure" on a blurry photo.
  That is the correct behaviour, not a bug.

## 5. What was annoying to get right

A few things I figured out the hard way:

- **Python version.** The machine had only Python 3.14, and PyTorch does not ship wheels for
  it yet, so `pip install torch` failed. I installed Python 3.12 specifically for the ML
  stack.
- **Label alignment.** The two datasets name their classes differently
  (`Corn_(maize)___Common_rust_` vs a clean `Corn___Common_rust`). If the model labels and
  the knowledge-base keys don't match exactly, the advice lookup silently fails. I solved it
  by mapping every source label into my own canonical names when preparing the data, so
  everything is aligned by construction.
- **Free-tier cold starts.** A sleeping Hugging Face Space takes a while to wake. The app now
  pings `/health` on load and shows a friendly "waking up" message so the first request never
  looks broken.
- **Keeping the LLM grounded.** My first prompt let the model "improve" the advice and it
  started adding dosages that weren't in my facts. Tightening the instruction to *translate
  only* fixed it.

## 6. Results and conclusion

LeafLens works end to end: a leaf photo becomes a diagnosis with a confidence score and
a grounded Bangla treatment plan, and the whole thing runs on free hosting behind a single
URL. The vision model reaches {{TEST_ACC}} test accuracy across {{NUM_CLASSES}} classes, and
the advisory layer stays factual because the facts come from a curated knowledge base rather
than the model's imagination.

**Honest limitations.** The training images are cleaner than a real muddy field photo, so
real-world accuracy will be lower than the test number — the confidence gate is there partly
to catch that. The knowledge base covers common diseases of five crops, not everything. And
the advice is a first opinion, not a replacement for an agriculture officer, which the app
says openly.

**Future work.** Collect real in-field photos from Bangladesh to fine-tune on, add more
crops and pests, support voice input for low-literacy users, and add an offline mode so it
works without a signal in the field.

Building this taught me that the hardest part of an AI product is not the model accuracy —
it is making the whole thing trustworthy, honest about uncertainty, and actually reachable
by the person who needs it.

---

*Md. Shadman Sakib Rahman*
