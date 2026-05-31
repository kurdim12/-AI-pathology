"""
Naseej REST API — lab-workflow integration.

A thin HTTP layer over the triage pipeline so a LIS / lab system can POST slide
images and get back structured triage decisions (no UI required). This is the
integration path the roadmap calls for.

Endpoints:
  * ``GET  /health``          — liveness + whether a trained model is loaded.
  * ``POST /predict``         — one image (multipart ``file``) -> triage JSON.
  * ``POST /triage``          — many images (multipart ``files``) -> sorted
                                worklist (most urgent first).

Run::

    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn app.api:app --host 0.0.0.0 --port 8000
    # interactive docs at http://localhost:8000/docs

``fastapi`` is an optional dependency; importing this module without it raises a
clear message. The core demo (``app/app.py``) does not need it.
"""

from __future__ import annotations

import io

try:
    from fastapi import FastAPI, File, UploadFile
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The REST API needs FastAPI: pip install fastapi \"uvicorn[standard]\" "
        "python-multipart. The Gradio demo (app/app.py) does not require it."
    ) from exc

from PIL import Image

from src.inference import analyze, load_model, load_thresholds
from src.triage import worklist_from_paths  # noqa: F401  (kept for parity / future use)

app = FastAPI(
    title="Naseej · AI Pathology Triage",
    description="Decision support, not diagnosis. Re-orders the queue; the "
                "pathologist makes the final call.",
    version="1.0.0",
)

# Load once at import so every request reuses the same weights.
MODEL, BACKBONE, TRAINED = load_model()


def _read_image(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _result_to_dict(result, filename: str | None = None) -> dict:
    payload = {
        "label": result.label,
        "prob_malignant": round(result.prob_malignant, 6),
        "confidence": round(result.confidence, 6),
        "priority": result.priority,
        "trained": result.trained,
    }
    if filename is not None:
        payload["filename"] = filename
    return payload


@app.get("/health")
def health() -> dict:
    """Liveness + model status. ``trained=False`` means predictions are placeholders."""
    urgent, review = load_thresholds()
    return {
        "status": "ok",
        "model_loaded": True,
        "trained": TRAINED,
        "backbone": BACKBONE,
        "thresholds": {"urgent": urgent, "review": review},
        "disclaimer": "Decision support, not diagnosis.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Triage a single slide image."""
    image = _read_image(await file.read())
    result = analyze(image, model=MODEL, backbone=BACKBONE, trained=TRAINED,
                     with_heatmap=False)
    return _result_to_dict(result, filename=file.filename)


@app.post("/triage")
async def triage(files: list[UploadFile] = File(...)) -> dict:
    """Triage a batch and return a worklist sorted most-urgent-first."""
    rows = []
    for f in files:
        image = _read_image(await f.read())
        result = analyze(image, model=MODEL, backbone=BACKBONE, trained=TRAINED,
                         with_heatmap=False)
        rows.append(_result_to_dict(result, filename=f.filename))

    rank = {"URGENT": 0, "REVIEW": 1, "ROUTINE": 2}
    rows.sort(key=lambda r: (rank[r["priority"]], -r["prob_malignant"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    counts = {p: sum(1 for r in rows if r["priority"] == p) for p in rank}
    return {"count": len(rows), "counts": counts, "worklist": rows,
            "trained": TRAINED}


def main() -> None:  # pragma: no cover - convenience launcher
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
