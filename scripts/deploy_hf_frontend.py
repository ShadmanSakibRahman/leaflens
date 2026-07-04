"""Deploy the built LeafLens frontend (static export in frontend/out/) to a
Hugging Face *static* Space.

Build the frontend first with the backend URL baked in:
    cd frontend
    NEXT_PUBLIC_API_BASE=https://<user>-leaflens-api.hf.space npm run build

Env vars:
  HF_TOKEN             Hugging Face write token (required)
  FRONTEND_SPACE_NAME  Space name (default: leaflens)

Usage:
  HF_TOKEN=hf_xxx python scripts/deploy_hf_frontend.py
"""

import os
import sys

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "frontend", "out")

README = """---
title: LeafLens
emoji: 🌿
colorFrom: green
colorTo: green
sdk: static
pinned: false
license: mit
---

# LeafLens

AI crop-disease diagnosis with grounded Bangla treatment advice for Bangladeshi farmers.
This static Space serves the Next.js frontend; it talks to the LeafLens API Space.
"""


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is required")
    if not os.path.exists(os.path.join(OUT, "index.html")):
        sys.exit("frontend/out/index.html not found — build the frontend first.")

    space_name = os.environ.get("FRONTEND_SPACE_NAME", "leaflens")

    # Static Spaces need a README.md with frontmatter at the repo root.
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{space_name}"
    print(f"Deploying frontend to static Space: {repo_id}")

    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        folder_path=OUT,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy LeafLens frontend",
    )

    url = f"https://{user}-{space_name}.hf.space".replace("_", "-")
    print("\nDone.")
    print(f"Frontend URL: {url}")


if __name__ == "__main__":
    main()
