"""Fasal Doctor API.

Endpoints
  GET  /health    liveness + whether the model is loaded (frontend uses this to
                  detect free-tier cold starts and show a "waking up" state)
  GET  /labels    the class list the model can predict (handy for docs/demo)
  POST /predict   image -> diagnosis only (vision layer)
  POST /advise    {raw_label,...} -> grounded bilingual advice (advisory layer)
  POST /diagnose  image -> diagnosis + advice + saved to history (what the UI calls)
  GET  /history   recent scans
"""

import os
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import model as vision
from . import advisor
from . import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fasal")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

app = FastAPI(
    title="LeafLens API",
    description="AI crop-disease diagnosis and grounded treatment advice for Bangladeshi farmers.",
    version="1.0.0",
)

# CORS: allow the deployed frontend + local dev. FRONTEND_ORIGIN can be a comma list.
_origins_env = os.environ.get("FRONTEND_ORIGIN", "")
_allowed = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    db.init_db()
    # Warm the model at boot so the first real request isn't slow. If weights are
    # missing (e.g. before training) we log and let /health report not-ready.
    try:
        vision.load_model()
        log.info("Model loaded and ready.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Model not loaded at startup: %s", exc)


async def _read_valid_image(file: UploadFile) -> bytes:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file (JPEG or PNG).")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 8 MB).")
    return data


@app.get("/")
def root():
    return {"service": "Fasal Doctor API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    ready = True
    try:
        vision.load_model()
    except Exception:  # noqa: BLE001
        ready = False
    return {"status": "ok", "model_ready": ready}


@app.get("/labels")
def labels():
    try:
        _, lbls = vision.load_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail="Model not available yet.") from exc
    return {"count": len(lbls), "labels": lbls}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await _read_valid_image(file)
    try:
        result = vision.predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("predict failed")
        raise HTTPException(status_code=503, detail="Model not available yet.") from exc
    return result


class AdviseIn(BaseModel):
    raw_label: str
    crop: str = "Unknown"
    disease: str = "Unknown"
    is_healthy: bool = False
    uncertain: bool = False


@app.post("/advise")
def advise(body: AdviseIn):
    return advisor.advise(
        body.raw_label, body.crop, body.disease, body.is_healthy, body.uncertain
    )


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    """Primary endpoint: vision + advice + persistence in one call."""
    data = await _read_valid_image(file)
    try:
        pred = vision.predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("diagnose/predict failed")
        raise HTTPException(status_code=503, detail="Model not available yet.") from exc

    advice = advisor.advise(
        pred["raw_label"],
        pred["crop"],
        pred["disease"],
        is_healthy=pred["is_healthy"],
        uncertain=pred["uncertain"],
    )

    try:
        scan_id = db.save_scan(
            crop=pred["crop"],
            disease=pred["disease"] if not pred["uncertain"] else "Uncertain",
            is_healthy=pred["is_healthy"],
            confidence=pred["confidence"],
            uncertain=pred["uncertain"],
            advice=advice,
        )
    except Exception:  # noqa: BLE001 - never fail the request just because history write failed
        log.exception("save_scan failed")
        scan_id = None

    return {"id": scan_id, "prediction": pred, "advice": advice}


@app.get("/history")
def history(limit: int = 20):
    limit = max(1, min(limit, 100))
    return {"scans": db.recent_scans(limit)}
