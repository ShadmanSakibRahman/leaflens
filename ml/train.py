"""Train the Fasal Doctor leaf-disease classifier.

Transfer learning on MobileNetV2 (ImageNet-pretrained). Reads a standard
ImageFolder at ml/data/ (one subfolder per class), splits train/val/test, trains
the classifier head, evaluates, and writes weights + labels + metrics into
backend/model/ so the API can serve them directly.

CPU-friendly by design: the backbone is frozen (only the head trains) and images
per class are capped, so this finishes in minutes without a GPU.

Usage:
    python ml/train.py --epochs 8 --per-class-cap 400
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
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


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tf, eval_tf


def capped_indices(targets, cap, seed):
    """Return a subset of indices with at most `cap` samples per class."""
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
    """Stratified split so every class appears in every split."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--per-class-cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_tf, eval_tf = build_transforms()
    # Two views of the same folder so train gets augmentation, val/test don't.
    full_train = datasets.ImageFolder(args.data_dir, transform=train_tf)
    full_eval = datasets.ImageFolder(args.data_dir, transform=eval_tf)
    classes = full_train.classes
    targets = full_train.targets
    print(f"Found {len(classes)} classes, {len(targets)} images total.")

    kept = capped_indices(targets, args.per_class_cap, args.seed)
    tr_idx, va_idx, te_idx = split_indices(kept, targets, args.seed)
    print(f"Split -> train {len(tr_idx)}, val {len(va_idx)}, test {len(te_idx)}")

    train_ds = Subset(full_train, tr_idx)
    val_ds = Subset(full_eval, va_idx)
    test_ds = Subset(full_eval, te_idx)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in net.features.parameters():
        p.requires_grad = False  # freeze backbone -> fast on CPU
    net.classifier[1] = nn.Linear(net.last_channel, len(classes))
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr
    )

    best_val = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        net.train()
        running = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(1, len(tr_idx))

        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                pred = net(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(1, total)
        print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)

    # Final test evaluation
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_ld:
            x = x.to(device)
            preds = net(x).argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    test_acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save artifacts the backend needs
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
        "backbone": "mobilenet_v2 (ImageNet, frozen features)",
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

    # Confusion matrix figure for the report/README
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.6), max(5, len(classes) * 0.6)))
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
