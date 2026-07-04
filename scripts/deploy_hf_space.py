"""Deploy the LeafLens backend to a Hugging Face Docker Space.

Creates (or updates) the Space, uploads the backend/ folder, and sets the runtime
secrets. Run after the model is trained (weights.pt + labels.json exist).

Env vars:
  HF_TOKEN         Hugging Face write token (required)
  GROQ_API_KEY     Groq key to set as a Space secret (required)
  FRONTEND_ORIGIN  Allowed CORS origin(s) for the Space (optional; can be set later)
  SPACE_NAME       Space name (default: leaflens-api)

Usage:
  HF_TOKEN=hf_xxx GROQ_API_KEY=gsk_xxx python scripts/deploy_hf_space.py
"""

import os
import sys

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..", "backend")


def main():
    token = os.environ.get("HF_TOKEN")
    groq = os.environ.get("GROQ_API_KEY")
    frontend_origin = os.environ.get("FRONTEND_ORIGIN", "")
    space_name = os.environ.get("SPACE_NAME", "leaflens-api")

    if not token:
        sys.exit("HF_TOKEN is required")
    if not groq:
        sys.exit("GROQ_API_KEY is required")

    # sanity: model must be trained
    if not os.path.exists(os.path.join(BACKEND, "model", "weights.pt")):
        sys.exit("backend/model/weights.pt not found — train the model first.")

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{space_name}"
    print(f"Deploying to Space: {repo_id}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=BACKEND,
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            ".env",
            ".env.*",
            "*.db",
            "__pycache__/*",
            "**/__pycache__/*",
            ".dockerignore",
        ],
        commit_message="Deploy LeafLens backend",
    )

    api.add_space_secret(repo_id, "GROQ_API_KEY", groq)
    api.add_space_secret(repo_id, "GROQ_MODEL", os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    if frontend_origin:
        api.add_space_secret(repo_id, "FRONTEND_ORIGIN", frontend_origin)

    url = f"https://huggingface.co/spaces/{repo_id}"
    api_url = f"https://{user}-{space_name}.hf.space".replace("_", "-")
    print("\nDone.")
    print(f"Space page: {url}")
    print(f"API base:   {api_url}")
    print("The Space will build the Docker image now (takes a few minutes).")


if __name__ == "__main__":
    main()
