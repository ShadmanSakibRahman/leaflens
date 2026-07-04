# Project Proposal: LeafLens

**Author:** Md. Shadman Sakib Rahman
**Project type:** AI Capstone (Frontend + Backend + AI/ML)
**Date:** July 2026

## Problem statement

Agriculture is close to 40% of the workforce in Bangladesh, and most of it is small farmers. One of the biggest and most avoidable causes of crop loss is disease that gets noticed too late or treated the wrong way. A farmer who sees strange spots on a leaf usually has no fast way to know what it actually is. The nearest agriculture officer might be a bus ride away, and by the time real advice comes the disease has already spread. The usual backup plan is to ask a shopkeeper and spray whatever pesticide is around, which costs money, is bad for the soil, and often does not even hit the real problem.

So the main problem I am trying to solve is this gap between seeing symptoms and getting a real answer. Farmers can see that something is wrong, but they cannot easily name the disease or find the right and safe treatment in time, in a language they read comfortably.

## Proposed solution

LeafLens is a mobile first web app that turns a phone camera into a crop doctor. The farmer takes a photo of an affected leaf and within a couple of seconds gets:

1. a diagnosis, meaning which crop and which disease, along with a confidence score, and
2. a treatment plan in plain Bangla (and English), with organic steps, chemical options, prevention, and a note about when to go see a real expert.

Two decisions make it something a farmer can actually trust:

- If the model is not confident, it refuses to guess and instead tells the farmer to retake the photo or ask an officer. A confident wrong answer is worse than an honest "not sure".
- The treatment facts come from a fixed knowledge file that I wrote, not from the AI making things up. The language model is only used to translate those checked facts into simple Bangla. That way the advice stays factual and safe.

## AI approach

The system uses two AI parts that work together.

The first is the vision model that does the diagnosis. It is transfer learning on MobileNetV2 (already trained on ImageNet, with a new classifier head that I trained). It is trained on real leaf photo datasets covering the main Bangladeshi crops, which are rice, potato, tomato, maize and pepper, across about twenty disease and healthy classes. On the held out test set it reaches about 95.9% accuracy. It outputs a class label plus a confidence value that the "not sure" check uses.

The second is the advice part. When a disease is found, the app looks up the treatment sheet for that disease in the knowledge file, and a Groq LLM (llama-3.3-70b) rewrites those exact facts into simple Bangla with a short summary. The LLM is told to only use the facts it is given, so it cannot invent a treatment. If the LLM is not available, the app just shows the checked English facts instead.

## Technology stack

- Frontend: Next.js (React), mobile first, with camera or upload, a Bangla and English toggle, and scan history. The live version runs on GitHub Pages as an on device app (the model is exported to ONNX and runs in the browser with onnxruntime-web), which keeps it reachable from Bangladesh.
- Backend: FastAPI (Python) with a Dockerfile. It has REST endpoints for prediction, advice, diagnosis and history, with input checking and error handling. It runs locally and can be deployed to any Docker host.
- Database: SQLite for scan history (the on device build uses browser localStorage).
- AI/ML: PyTorch and torchvision for the MobileNetV2 model, scikit-learn for the evaluation, and the Groq API for the Bangla translation.
- Tooling: Git and GitHub, Hugging Face `datasets` for the data, and Google Colab for GPU training.

## Expected impact

A farmer with a cheap smartphone gets a fast second opinion in their own language, which should cut down on wrong pesticide use and catch disease earlier. The same setup can be extended to more crops and diseases later, and could eventually run fully offline or be connected to a government agriculture helpline.
