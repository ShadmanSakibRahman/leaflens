# Final Report: LeafLens

**An AI crop disease diagnosis and advice system for Bangladeshi farmers**

**Author:** Md. Shadman Sakib Rahman
**Course:** AI Capstone Project
**Date:** July 2026

## 1. Introduction and the problem I chose

I grew up around people for whom farming is not a hobby but the whole household income, so I wanted my capstone to be something that actually helps them and not just another generic demo. In Bangladesh agriculture is close to 40% of the workforce, and a large share of avoidable loss comes from crop disease that is caught late or treated with the wrong chemical.

The specific gap I wanted to close is this. A farmer can see that a leaf looks wrong, but has no fast and trustworthy way to know what it is or what to do about it, in a language they read comfortably. The usual path is to ask a shop, buy some pesticide, spray it and hope, which is expensive and often misses the real disease.

LeafLens is my answer to that. You photograph a leaf and get a diagnosis and a treatment plan in Bangla in a couple of seconds.

## 2. What I built (system overview)

I built the system as three separate parts so each one can be understood and tested on its own.

The frontend is a mobile first Next.js (React) web app. The farmer takes or uploads a photo, sees the diagnosis and advice, can switch the whole interface between Bangla and English, and can scroll through a short history of past scans.

The backend is a FastAPI service that runs the AI and stores history. It has a small REST API (`/predict`, `/advise`, `/diagnose`, `/history`, `/health`) and a SQLite database.

The AI is two models working together: a vision model that names the disease, and a retrieval plus LLM layer that turns that diagnosis into safe advice.

The live demo is deployed as an in browser app on GitHub Pages (section 5 explains why), and the full FastAPI backend also lives in the repo and runs the same models on the server side. In the live version the React app loads the model as ONNX and runs it in the browser. In the server version the React app talks to FastAPI over HTTP, and FastAPI runs the model, the knowledge base, Groq and SQLite.

## 3. The AI workflow in detail

### 3.1 Vision model: naming the disease

I used transfer learning on MobileNetV2, which is already trained on ImageNet. I froze the convolutional part and trained only a new classifier head sized to my classes. I picked MobileNetV2 on purpose, because it is small and fast enough to run on a free CPU, and the whole point of the project is cheap and easy to reach.

For the data I combined two public datasets from the Hugging Face Hub that need no login: `BrandonFors/Plant-Diseases-PlantVillage-Dataset` for potato, tomato, maize and pepper, and `sharmin3/Rice-Leaf-Disease` for rice. I mapped both of them into one clean set of `Crop___Disease` class names so that the model labels, the treatment knowledge base and the UI all match. I capped the number of images per class to keep things balanced and training fast, and split the data into train, validation and test in a stratified way so every class shows up in every split.

The model was trained for a modest number of epochs with Adam and cross entropy loss, keeping the best validation checkpoint. On the held out test set it reached:

- Test accuracy: 95.9%
- Macro F1: 0.96 (weighted F1 is also 0.96)
- 20 classes across 5 crops

The per class precision, recall and F1, plus a confusion matrix, are saved with the model in `backend/model/metrics.json` and `confusion_matrix.png`. The easiest classes were rice tungro, rice healthy and potato healthy, which all got an F1 of 1.00. The hardest were the tomato blights: tomato early blight (F1 0.83), late blight (0.86) and septoria leaf spot (0.89). That makes sense, because those three genuinely look similar even to a person, and the confusion matrix shows they get mixed up with each other rather than with unrelated crops.

The model also returns a softmax confidence. Below a threshold of 0.55 the app does not name a disease at all, and instead tells the farmer to retake a clearer photo or ask an officer. I added this because in this kind of app a confident wrong answer, meaning spraying the wrong chemical, is worse than an honest "I am not sure".

### 3.2 Advisory layer: turning a label into safe advice

This is the part I most wanted to get right, because this is where language models usually go wrong. They will happily invent treatments that sound correct but are made up. My rule was that the app must never make up farming advice.

So the treatment facts live in a knowledge file that I wrote myself (`knowledge_base.json`). Each disease has one entry with symptoms, organic control, chemical control, prevention, and a note about when to call an expert. When the vision model returns a disease, the backend looks up that entry. Then the Groq LLM (llama-3.3-70b) is given only those facts and asked to translate them into simple Bangla and write a two sentence summary. It is clearly told not to add anything that is not already in the facts.

If Groq cannot be reached, the backend just returns the checked English facts. The feature gets simpler, but it never makes anything up.

## 4. Design decisions and trade-offs

I kept the frontend and backend separate instead of putting everything in one Streamlit app. It was more work (two things to run, plus CORS), but it gives a real API and a real database, which is both better engineering and a more honest backend. It also means the model server and the UI can be changed on their own.

I used a frozen MobileNetV2 backbone. I gave up a little accuracy in exchange for a model that trains in minutes on a CPU and runs fast on free hosting. For this project, being easy to deploy was worth more than a small accuracy gain.

I used a knowledge file plus an LLM instead of a pure LLM. It is a bit less impressive to look at, but it is safe. In agriculture a wrong answer has a real cost, so I chose trust over sounding clever.

I kept the confidence gate even though it means the app sometimes says "not sure" on a blurry photo. That is the correct behaviour, not a bug.

## 5. What was annoying to get right

A few things I only figured out the hard way.

The Python version. The machine only had Python 3.14, and PyTorch does not have a build for it yet, so `pip install torch` just failed. I installed Python 3.12 for the ML side to fix it.

Label alignment. The two datasets name their classes differently (one had `Corn_(maize)___Common_rust_`, and I wanted a clean `Corn___Common_rust`). If the model labels and the knowledge file keys do not match exactly, the advice lookup quietly returns nothing. I fixed it by renaming every source label into my own naming when I prepare the data, so everything lines up.

The host was blocked in Bangladesh. I first deployed the backend to Hugging Face Spaces and it ran fine, but then I realised `*.hf.space` is blocked on my internet here (every Space, even famous ones, gave me the same error, while the Hugging Face API still worked). Since my grader is probably on a Bangladeshi network too, a blocked demo would be useless. So I changed the plan. I exported the model to ONNX, pre translated all the advice into Bangla at build time, and rebuilt the app so it runs completely in the browser on GitHub Pages, which does load here. In the end the limitation made the product better, because now it needs no server and costs nothing per request. The full FastAPI backend is still in the repo.

Keeping the LLM honest. My first prompt let the model "improve" the advice and it started adding pesticide dosages that were not in my facts. I had to make the instruction stricter (basically translate only, add nothing) before it stopped.

## 6. Results and conclusion

LeafLens works from end to end. A leaf photo becomes a diagnosis with a confidence score and a Bangla treatment plan, and the live version runs on free hosting on one link. The vision model gets 95.9% test accuracy across 20 classes, and the advice stays factual because the facts come from a knowledge file I wrote rather than from the model's imagination.

Honest limitations. The training images are cleaner than a real muddy field photo, so real accuracy in a field will be lower than the test number. That is partly why the "not sure" behaviour is there. The knowledge file only covers common diseases of five crops, not everything. And the advice is a first opinion, not a replacement for a real agriculture officer, which the app also says.

Future work. Collect real in field photos from Bangladesh to fine tune on, add more crops and pests, add voice input for people who do not read much, and add an offline mode so it works without a signal in the field.

Building this taught me that the hardest part of an AI product is not really the model accuracy. It is making the whole thing trustworthy, honest about what it does not know, and actually reachable by the person who needs it.

Md. Shadman Sakib Rahman
