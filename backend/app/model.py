"""Vision layer: load the fine-tuned MobileNetV2 and classify a leaf image.

The model file (`weights.pt`) and the class list (`labels.json`) are produced by
`ml/train.py`. This module only does inference, so the backend has no training
dependencies. Labels follow the PlantVillage convention `Crop___Condition`
(e.g. `Tomato___Late_blight`, `Potato___healthy`), which we split into a friendly
crop name and disease name.
"""

import os
import json
import io

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.pt")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# Below this softmax confidence we refuse to name a disease and ask the user to
# retake the photo / consult an expert. This is the vision-side safety gate.
CONFIDENCE_THRESHOLD = float(os.environ.get("FASAL_CONF_THRESHOLD", "0.55"))

IMG_SIZE = 224
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)

_model = None
_labels = None


def _prettify(raw_label):
    """`Tomato___Late_blight` -> ('Tomato', 'Late blight', is_healthy=False)."""
    if "___" in raw_label:
        crop, condition = raw_label.split("___", 1)
    else:
        crop, condition = raw_label, raw_label
    crop = crop.replace("_", " ").strip().title()
    is_healthy = condition.strip().lower() == "healthy"
    disease = condition.replace("_", " ").strip()
    disease = "Healthy" if is_healthy else disease[:1].upper() + disease[1:]
    return crop, disease, is_healthy


def load_model():
    """Lazily load weights + labels once, then reuse."""
    global _model, _labels
    if _model is not None:
        return _model, _labels

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        _labels = json.load(f)

    net = models.mobilenet_v2(weights=None)
    net.classifier[1] = torch.nn.Linear(net.last_channel, len(_labels))
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    _model = net
    return _model, _labels


def predict(image_bytes):
    """Run inference on raw image bytes. Returns a structured result dict."""
    model, labels = load_model()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Could not read the uploaded file as an image.") from exc

    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)

    raw_label = labels[int(idx)]
    crop, disease, is_healthy = _prettify(raw_label)
    confidence = float(conf)
    uncertain = confidence < CONFIDENCE_THRESHOLD

    # top-3 for transparency in the UI / report
    top_conf, top_idx = torch.topk(probs, k=min(3, len(labels)))
    top3 = []
    for c, i in zip(top_conf.tolist(), top_idx.tolist()):
        tc, td, _ = _prettify(labels[i])
        top3.append({"label": labels[i], "crop": tc, "disease": td, "confidence": round(c, 4)})

    return {
        "raw_label": raw_label,
        "crop": crop,
        "disease": disease,
        "is_healthy": is_healthy,
        "confidence": round(confidence, 4),
        "uncertain": uncertain,
        "top3": top3,
    }
