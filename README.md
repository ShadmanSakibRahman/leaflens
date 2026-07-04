# LeafLens

Take a photo of a sick crop leaf and the app makes a quick guess at what disease it is, then tells you what to do about it, in Bangla.

This is my AI capstone project. The basic idea is simple: a farmer points their phone at a bad looking leaf, the app figures out the crop and the disease, and then it gives some treatment advice (organic options, chemical options, and how to stop it next time) in Bangla or English.

## Live demo

https://shadmansakibrahman.github.io/leaflens/

Just open the link, upload or take a photo of a leaf, and it tells you what it thinks. The AI model actually runs inside your browser, so the photo never leaves your phone.

One thing about the hosting. I first put the backend on Hugging Face Spaces and it worked, but then I found out that `*.hf.space` is blocked on my internet here in Bangladesh (even the big popular Spaces would not load for me), which means my teacher probably could not open it either. So I exported the model to ONNX and made the whole thing run in the browser on GitHub Pages instead, and that loads fine here. The full FastAPI backend is still in the repo under `backend/` if you want to run the server version.

## Screenshots

| Upload (Bangla) | Result in Bangla | Result in English | History |
|---|---|---|---|
| ![upload](docs/screenshots/01-upload.png) | ![result Bangla](docs/screenshots/02-result-bn.png) | ![result English](docs/screenshots/03-result-en.png) | ![history](docs/screenshots/04-history.png) |

## The problem I picked

Around 40% of people in Bangladesh work in agriculture, and a lot of avoidable crop loss happens because a disease gets noticed too late or gets treated with the wrong pesticide. Most farmers cannot just call an agriculture officer whenever they want, so they end up guessing, and guessing wrong wastes money and is bad for the soil. I wanted to make something that gives a fast second opinion in the language they actually read.

## How it works

There are two AI parts.

The first part does the diagnosis. It is a MobileNetV2 network that was already trained on ImageNet, and I fine tuned it on real photos of diseased leaves. It handles 5 crops (rice, potato, tomato, maize, pepper) across 20 classes, and it gets about 95.9% accuracy on the held out test set. It also gives a confidence number, and if it is not confident it says so instead of guessing. I felt that mattered, because a confident wrong answer (spray the wrong thing) is worse than an honest "not sure".

The second part gives the advice. I did not want the AI making up treatments, so all the treatment text sits in a JSON file that I wrote myself (symptoms, organic fix, chemical fix, prevention, and when to call an expert). When a disease is found, the app grabs that entry and only uses the Groq LLM (llama-3.3-70b) to translate it into simple Bangla. The model is told to use just the facts it is given, so it cannot invent a fake pesticide. If the LLM is down, it falls back to showing the English facts.

## Tech stack

- Frontend: Next.js (React). The live version runs on GitHub Pages with the model running in the browser using onnxruntime-web.
- Backend: FastAPI (Python) with a Dockerfile, and SQLite for saving past scans.
- ML: PyTorch and torchvision for the MobileNetV2 model, scikit-learn for the metrics, Groq for the Bangla translation.
- Data: two public datasets from the Hugging Face Hub (a PlantVillage one for potato, tomato, maize and pepper, and a rice leaf one for rice).

## Project structure

```
leaflens/
├── frontend/          Next.js app (the UI, and the in-browser model)
├── backend/           FastAPI service
│   ├── app/           main.py, model.py, advisor.py, db.py, knowledge_base.json
│   ├── model/         weights.pt, labels.json, metrics.json, confusion_matrix.png
│   └── Dockerfile
├── ml/                data prep, training, and the ONNX export
│   ├── prepare_data.py
│   ├── train.py
│   └── train_colab.ipynb
└── docs/              proposal and final report
```

## Running it yourself

### Backend

```bash
cd backend
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Then the API is at http://localhost:8000 and the docs are at /docs. You need a trained model in `backend/model/` (`weights.pt` and `labels.json`), which is already committed. To retrain, see below.

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

Then the app is at http://localhost:3000.

### Retrain the model

```bash
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r ml/requirements.txt
venv\Scripts\python.exe ml/prepare_data.py --cap 500
venv\Scripts\python.exe ml/train.py --epochs 10
```

Or open `ml/train_colab.ipynb` in Google Colab for a free GPU.

## Stuff that went wrong (and how I fixed it)

- PyTorch would not install at first. The PC only had Python 3.14 and torch does not have a build for it yet, so `pip install torch` just failed. I installed Python 3.12 for the ML side and that sorted it out.
- The advice kept coming back empty for a while. It turned out the two datasets name their classes differently (one had `Corn_(maize)___Common_rust_`, and I wanted a clean `Corn___Common_rust`), so my lookups were missing. I fixed it by renaming everything to my own naming when I prepare the data, so the model, the advice file, and the screen all match.
- The LLM kept trying to be helpful and added pesticide amounts that I never gave it. I had to make the prompt stricter (basically "just translate, do not add anything") before it stopped.
- The hf.space blocking problem I mentioned up top. That is what pushed me to make it run in the browser, which honestly ended up being better since now it does not need a server at all.

## Limitations

The training images are cleaner than a real muddy field photo, so real accuracy out in a field will be lower than the test number. That is partly why I added the "not sure" behaviour. The advice only covers common diseases of five crops, not everything. And it is meant to be a first opinion, not a replacement for a real agriculture officer, which the app also says.

## Author

Md. Shadman Sakib Rahman
