"""Train the LeafLens leaf-disease classifier.

Transfer learning on MobileNetV2 (ImageNet-pretrained). Because we freeze the
backbone and only train the classifier head, we extract the backbone's pooled
features ONCE (a single pass over the images) and then train a linear head on those
cached features for many epochs in seconds. This is a standard "linear probe" and is
far faster on CPU than recomputing the frozen features every epoch.

Reads a standard ImageFolder at ml/data/, splits train/val/test in a stratified way,
trains, evaluates, and writes weights + labels + metrics + a confusion matrix into
backend/model/ so the API can serve them directly.

Usage:
    python ml/train.py --epochs 40 --per-class-cap 500
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data")
DEFAULT_OUT = os.path.join(HERE, "..", "backend", "model")

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def eval_transform():
    # Same preprocessing the backend uses at inference, so features match.
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def capped_indices(targets, cap, seed):
    if cap is None or cap <= 0:
        return list(range(len(targets)))
    rng = random.Random(seed)
    by_class = {}
    for i, t in enumerate(targets):
        by_class.setdefault(int(t), []).append(i)
    keep = []
    for t, idxs in by_class.items():
        rng.shuffle(idxs)
        keep.extend(idxs[:cap])
    rng.shuffle(keep)
    return keep


def split_indices(indices, targets, seed, val_frac=0.15, test_frac=0.15):
    rng = random.Random(seed)
    by_class = {}
    for i in indices:
        by_class.setdefault(int(targets[i]), []).append(i)
    train, val, test = [], [], []
    for t, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test.extend(idxs[:n_test])
        val.extend(idxs[n_test : n_test + n_val])
        train.extend(idxs[n_test + n_val :])
    rng.shuffle(train)
    return train, val, test


def extract_features(backbone, dataset, indices, device, batch_size=64):
    """One forward pass through the frozen backbone -> pooled 1280-d features."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    feats, labels = [], []
    backbone.eval()
    done = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = backbone.features(x)
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
            feats.append(f.cpu())
            labels.append(y)
            done += x.size(0)
            print(f"    extracted {done}/{len(indices)}", end="\r")
    print()
    return torch.cat(feats), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--per-class-cap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = datasets.ImageFolder(args.data_dir, transform=eval_transform())
    classes = ds.classes
    targets = ds.targets
    print(f"Found {len(classes)} classes, {len(targets)} images total.")

    kept = capped_indices(targets, args.per_class_cap, args.seed)
    tr_idx, va_idx, te_idx = split_indices(kept, targets, args.seed)
    print(f"Split -> train {len(tr_idx)}, val {len(va_idx)}, test {len(te_idx)}")

    # Build the network once; use its frozen backbone for feature extraction and
    # later attach the trained head for saving.
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    net = net.to(device)

    print("Extracting features (one pass through the frozen backbone)...")
    print("  train:")
    Xtr, ytr = extract_features(net, ds, tr_idx, device)
    print("  val:")
    Xva, yva = extract_features(net, ds, va_idx, device)
    print("  test:")
    Xte, yte = extract_features(net, ds, te_idx, device)

    # Train a linear head on cached features (fast).
    head = nn.Linear(net.last_channel, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)

    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xva, yva = Xva.to(device), yva.to(device)

    best_val = 0.0
    best_state = None
    n = Xtr.size(0)
    bs = 256
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            optimizer.zero_grad()
            loss = criterion(head(Xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_acc = (head(Xva).argmax(1) == yva).float().mean().item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{args.epochs}  val_acc={val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)

    # Test evaluation
    head.eval()
    with torch.no_grad():
        y_pred = head(Xte.to(device)).argmax(1).cpu().numpy().tolist()
    y_true = yte.numpy().tolist()
    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    print(f"\nBest val accuracy: {best_val:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")

    # Attach trained head to the full network and save the full state_dict, so the
    # backend can load mobilenet_v2 + this head exactly as it does at inference.
    net.classifier[1] = nn.Linear(net.last_channel, len(classes))
    net.classifier[1].load_state_dict(head.state_dict())
    net.eval()
    torch.save(net.state_dict(), os.path.join(args.out_dir, "weights.pt"))
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    metrics = {
        "test_accuracy": round(float(test_acc), 4),
        "best_val_accuracy": round(float(best_val), 4),
        "num_classes": len(classes),
        "num_images_total": len(targets),
        "train_size": len(tr_idx),
        "val_size": len(va_idx),
        "test_size": len(te_idx),
        "epochs": args.epochs,
        "backbone": "mobilenet_v2 (ImageNet, frozen) + linear head (linear probe)",
        "per_class": {
            c: {
                "precision": round(report[c]["precision"], 4),
                "recall": round(report[c]["recall"], 4),
                "f1": round(report[c]["f1-score"], 4),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.55), max(6, len(classes) * 0.55)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test acc {test_acc:.2%})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=120)
    print(f"Saved weights, labels, metrics, confusion matrix to {args.out_dir}")


if __name__ == "__main__":
    main()
