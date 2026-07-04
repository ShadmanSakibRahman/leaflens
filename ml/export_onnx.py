"""Export the trained MobileNetV2 to ONNX so it can run in the browser with
onnxruntime-web, and verify the ONNX outputs match PyTorch on a real image.

Outputs:
  frontend/public/model/leaflens.onnx
  frontend/public/model/labels.json  (copied)
"""

import json
import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "backend", "model")
OUT_DIR = os.path.join(HERE, "..", "frontend", "public", "model")
os.makedirs(OUT_DIR, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

labels = json.load(open(os.path.join(MODEL_DIR, "labels.json"), encoding="utf-8"))

net = models.mobilenet_v2(weights=None)
net.classifier[1] = nn.Linear(net.last_channel, len(labels))
net.load_state_dict(torch.load(os.path.join(MODEL_DIR, "weights.pt"), map_location="cpu", weights_only=True))
net.eval()

dummy = torch.randn(1, 3, 224, 224)
onnx_path = os.path.join(OUT_DIR, "leaflens.onnx")
torch.onnx.export(
    net, dummy, onnx_path,
    input_names=["input"], output_names=["logits"],
    opset_version=13, dynamic_axes=None,
)
print("Exported", onnx_path, "size", os.path.getsize(onnx_path) // 1024, "KB")

# copy labels for the frontend
json.dump(labels, open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---- verify torch vs onnx on real images ----
tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

mismatches = 0
tested = 0
for cls in ["Rice___Blast", "Tomato___Late_blight", "Potato___Early_blight", "Corn___Common_rust", "Pepper___Bacterial_spot"]:
    imgs = sorted(glob.glob(os.path.join(HERE, "data", cls, "*.jpg")))[:2]
    for p in imgs:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            t_out = net(x).numpy()
        o_out = sess.run(None, {"input": x.numpy()})[0]
        t_arg = int(t_out.argmax()); o_arg = int(o_out.argmax())
        maxdiff = float(np.abs(t_out - o_out).max())
        tested += 1
        ok = (t_arg == o_arg) and maxdiff < 1e-3
        if not ok:
            mismatches += 1
        print(f"{cls:30s} torch={labels[t_arg]:28s} onnx={labels[o_arg]:28s} maxdiff={maxdiff:.2e} {'OK' if ok else 'MISMATCH'}")

print(f"\n{tested} images tested, {mismatches} mismatches")
