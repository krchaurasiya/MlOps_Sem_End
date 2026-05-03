from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import os
import logging
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Model loading  (must be defined before app)
# ─────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/rank_model.pkl")
_model_data = None
_startup_time = datetime.now(timezone.utc)


def load_model():
    global _model_data
    try:
        _model_data = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.warning("⚠️  Model file not found at %s – predictions will fail.", MODEL_PATH)
        _model_data = None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Application lifespan: runs load_model on startup."""
    load_model()
    yield
    # shutdown logic here if needed


# ─────────────────────────────────────────────
# App setup  (lifespan is now defined above)
# ─────────────────────────────────────────────
app = FastAPI(
    title="🌴 Goa AI Recommender",
    description="ML-powered popularity ranking for Goa tourist spots.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """Liveness / readiness probe used by Docker & CI."""
    return {
        "status": "ok",
        "model_loaded": _model_data is not None,
        "uptime_seconds": (datetime.now(timezone.utc) - _startup_time).total_seconds(),
        "version": "1.0.0",
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Redirect browsers to the Swagger docs."""
    return HTMLResponse(
        "<html><body><h2>🌴 Goa AI Recommender API</h2>"
        "<p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p></body></html>"
    )


@app.post("/predict", tags=["ml"])
def predict(input_data: dict):
    """
    Predict popularity score for a Goa tourist spot.

    **Body**
    ```json
    {
      "rating": 4.5,
      "review_count": 120,
      "category": "beach",
      "text": "sunset beach party vibes"
    }
    ```
    """
    if _model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded – check server logs.")

    required_fields = {"rating", "review_count", "category", "text"}
    missing = required_fields - set(input_data.keys())
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {missing}")

    try:
        model = _model_data["model"]
        tfidf = _model_data["tfidf"]
        le    = _model_data["label_encoder"]

        category_encoded = le.transform([input_data["category"]])[0]
        text_features    = tfidf.transform([input_data["text"]]).toarray()

        final_input = np.hstack([
            [[input_data["rating"], input_data["review_count"], category_encoded]],
            text_features,
        ])

        prediction = model.predict(final_input)
        score      = round(float(prediction[0]), 4)

        logger.info("Prediction %.4f for category=%s", score, input_data["category"])

        return {
            "prediction": score,
            "category": input_data["category"],
            "rating": input_data["rating"],
            "review_count": input_data["review_count"],
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error.")


@app.get("/info", tags=["ops"])
def info():
    """Return API metadata."""
    return {
        "service": "Goa AI Recommender",
        "version": "1.0.0",
        "model_path": MODEL_PATH,
        "model_loaded": _model_data is not None,
    }