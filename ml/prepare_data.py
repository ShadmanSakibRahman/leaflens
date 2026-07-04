"""Download the source datasets from the Hugging Face Hub (no auth needed) and
materialise them into a clean ImageFolder at ml/data/ using OUR canonical class
names, so the model labels, the knowledge base keys, and the UI all line up.

Sources (both public, non-gated, pure parquet):
  - BrandonFors/Plant-Diseases-PlantVillage-Dataset  (potato, tomato, corn, pepper)
  - sharmin3/Rice-Leaf-Disease                       (rice)

We keep only the Bangladesh-relevant classes we have treatment sheets for, cap the
number of images per class (CPU-friendly), and save as JPEG.

Usage:
    python ml/prepare_data.py --cap 500
"""

import argparse
import os

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

PLANTVILLAGE_REPO = "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
RICE_REPO = "sharmin3/Rice-Leaf-Disease"


def map_plantvillage(name):
    """Map a source PlantVillage class name to our canonical label, or None to skip."""
    s = name.lower()
    if "potato" in s:
        if "early" in s:
            return "Potato___Early_blight"
        if "late" in s:
            return "Potato___Late_blight"
        if "healthy" in s:
            return "Potato___healthy"
    if "tomato" in s:
        if "bacterial" in s:
            return "Tomato___Bacterial_spot"
        if "early" in s:
            return "Tomato___Early_blight"
        if "late" in s:
            return "Tomato___Late_blight"
        if "mold" in s or "mould" in s:
            return "Tomato___Leaf_Mold"
        if "septoria" in s:
            return "Tomato___Septoria_leaf_spot"
        if "curl" in s:
            return "Tomato___Yellow_Leaf_Curl_Virus"
        if "healthy" in s:
            return "Tomato___healthy"
        return None  # skip spider mites, target spot, mosaic
    if "corn" in s or "maize" in s:
        if "rust" in s:
            return "Corn___Common_rust"
        if "northern" in s:
            return "Corn___Northern_Leaf_Blight"
        if "healthy" in s:
            return "Corn___healthy"
        return None  # skip cercospora / gray leaf spot
    if "pepper" in s:
        if "bacterial" in s:
            return "Pepper___Bacterial_spot"
        if "healthy" in s:
            return "Pepper___healthy"
    return None


def map_rice(name):
    s = name.lower().replace(" ", "")
    if "bacterial" in s:
        return "Rice___Bacterial_leaf_blight"
    if "blast" in s:
        return "Rice___Blast"
    if "brown" in s:
        return "Rice___Brown_spot"
    if "tungro" in s:
        return "Rice___Tungro"
    if "healthy" in s:
        return "Rice___healthy"
    return None


def harvest(repo, config, mapper, cap, counts):
    print(f"\nLoading {repo} ...")
    ds = load_dataset(repo, config) if config else load_dataset(repo)
    split = "train" if "train" in ds else list(ds.keys())[0]
    data = ds[split]
    names = data.features["label"].names
    print(f"  split '{split}': {len(data)} images, {len(names)} source classes")

    # Which source indices map to a canonical target we want.
    idx_to_canon = {}
    for i, nm in enumerate(names):
        canon = mapper(nm)
        if canon:
            idx_to_canon[i] = canon
    wanted = set(idx_to_canon)
    if not wanted:
        print("  WARNING: no matching classes found!")
        return

    # Filter by label only (no image decode) then shuffle for a fair sample.
    subset = data.filter(lambda lab: lab in wanted, input_columns=["label"])
    subset = subset.shuffle(seed=42)
    print(f"  {len(subset)} images match the classes we keep")

    saved = 0
    for row in subset:
        canon = idx_to_canon[row["label"]]
        if counts.get(canon, 0) >= cap:
            continue
        out_dir = os.path.join(DATA_DIR, canon)
        os.makedirs(out_dir, exist_ok=True)
        n = counts.get(canon, 0)
        try:
            img = row["image"].convert("RGB")
            img.save(os.path.join(out_dir, f"{n:04d}.jpg"), "JPEG", quality=90)
        except Exception as exc:  # noqa: BLE001
            print(f"    skip one image ({exc})")
            continue
        counts[canon] = n + 1
        saved += 1
        # Stop early once every wanted class is full.
        if all(counts.get(c, 0) >= cap for c in set(idx_to_canon.values())):
            break
    print(f"  saved {saved} images from {repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=500, help="max images per class")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    counts = {}
    harvest(PLANTVILLAGE_REPO, None, map_plantvillage, args.cap, counts)
    harvest(RICE_REPO, None, map_rice, args.cap, counts)

    print("\n=== Final per-class counts ===")
    for c in sorted(counts):
        print(f"  {c:40s} {counts[c]}")
    print(f"Total classes: {len(counts)}  Total images: {sum(counts.values())}")


if __name__ == "__main__":
    main()
