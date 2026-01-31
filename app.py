import joblib
import json
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

MODEL_PATH = "adhd_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    # Surface a helpful message in logs so the Space build shows it
    print(f"Failed to load model from {MODEL_PATH}: {e}")
    raise


def _parse_input(inp):
    # Accept string (CSV or JSON), list (single row or batch)
    if isinstance(inp, str):
        # try JSON first
        try:
            parsed = json.loads(inp)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            arr = np.array(parsed, dtype=float)
            if arr.ndim == 1:
                return arr.reshape(1, -1)
            return arr
        # treat as CSV
        parts = [p.strip() for p in inp.split(",") if p.strip() != ""]
        return np.array([float(p) for p in parts], dtype=float).reshape(1, -1)

    if isinstance(inp, list):
        arr = np.array(inp, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    raise ValueError("Unsupported input format. Send a CSV string or a JSON list of numbers.")


def predict(data):
    """Predict wrapper for Gradio/Spaces.

    Accepts a single argument which is either:
    - a string like "0.1,0.2,0.3"
    - a JSON list of numbers [0.1, 0.2, 0.3]

    Returns a JSON-able dict with `prediction` and optional `probabilities`.
    """
    try:
        X = _parse_input(data)
    except Exception as e:
        return {"error": f"Input parsing error: {e}"}

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        return {
            "error": (
                f"Model expects {expected} features, but received {X.shape[1]}. "
                f"Please send a CSV string or JSON list with {expected} values per row."
            )
        }

    try:
        pred = model.predict(X).tolist()
    except Exception as e:
        return {"error": f"Model prediction error: {e}"}

    out = {"prediction": pred}
    if hasattr(model, "predict_proba"):
        try:
            out["probabilities"] = model.predict_proba(X).tolist()
        except Exception:
            pass
    return out


# Replace Gradio with a small FastAPI application for reliable programmatic endpoints.
app = FastAPI(title="ADHD Noise Classifier API")

# Allow CORS from any origin for easy testing; tighten in production if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_input_from_payload(payload):
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list) and len(payload["data"]) > 0:
            return payload["data"][0]
        if "inputs" in payload and isinstance(payload["inputs"], list) and len(payload["inputs"]) > 0:
            return payload["inputs"][0]
        return payload
    return payload


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON payload"}, status_code=400)

    data = _extract_input_from_payload(payload)
    try:
        out = predict(data)
    except Exception as e:
        return JSONResponse({"error": f"Prediction error: {e}"}, status_code=500)
    return JSONResponse(out)


@app.get("/")
async def root():
    return {"info": "POST JSON to /predict with {\"data\": [[...features...]]} (expects 23 features)."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
