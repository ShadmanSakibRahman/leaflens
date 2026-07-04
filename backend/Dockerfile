# LeafLens backend — container for Hugging Face Spaces (SDK: docker).
# HF Spaces serves on port 7860 and mounts a read-only filesystem except /tmp,
# so the SQLite DB is written under /tmp.

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASAL_DB_PATH=/tmp/scans.db

# Install CPU-only torch first (small, no CUDA), then the rest from PyPI.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
