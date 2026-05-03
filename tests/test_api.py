"""
Unit tests for the Goa AI Recommender FastAPI application.
Run with: pytest tests/ -v --cov=app
"""
import numpy as np
import pytest
import pytest_asyncio


# ─────────────────────────────────────────────
# Minimal model stub so tests run without .pkl
# ─────────────────────────────────────────────
class _FakeModel:
    def predict(self, X):
        return np.array([4.2])


class _FakeTfidf:
    def transform(self, texts):
        class _R:
            def toarray(self):
                return np.zeros((1, 10))
        return _R()


class _FakeLabelEncoder:
    def transform(self, labels):
        return np.array([0])


FAKE_MODEL_DATA = {
    "model": _FakeModel(),
    "tfidf": _FakeTfidf(),
    "label_encoder": _FakeLabelEncoder(),
}

# ─────────────────────────────────────────────
# Patch joblib BEFORE importing app
# ─────────────────────────────────────────────
import joblib                                  # noqa: E402
joblib.load = lambda _path: FAKE_MODEL_DATA    # noqa: E731

import app.main as main_module                 # noqa: E402
main_module._model_data = FAKE_MODEL_DATA

from httpx import AsyncClient, ASGITransport   # noqa: E402
from app.main import app                       # noqa: E402


# ─────────────────────────────────────────────
# Async client fixture
# ─────────────────────────────────────────────
@pytest_asyncio.fixture
async def ac():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


# ─────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health_returns_200(ac):
    r = await ac.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_health_body_structure(ac):
    data = (await ac.get("/health")).json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "uptime_seconds" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health_model_loaded(ac):
    data = (await ac.get("/health")).json()
    assert data["model_loaded"] is True


# ─────────────────────────────────────────────
# Predict endpoint
# ─────────────────────────────────────────────
VALID_PAYLOAD = {
    "rating": 4.5,
    "review_count": 120,
    "category": "beach",
    "text": "beautiful sunset at the beach",
}


@pytest.mark.asyncio
async def test_predict_returns_200(ac):
    r = await ac.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_predict_has_prediction(ac):
    data = (await ac.post("/predict", json=VALID_PAYLOAD)).json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


@pytest.mark.asyncio
async def test_predict_echoes_category(ac):
    data = (await ac.post("/predict", json=VALID_PAYLOAD)).json()
    assert data["category"] == "beach"


@pytest.mark.asyncio
async def test_predict_echoes_rating(ac):
    data = (await ac.post("/predict", json=VALID_PAYLOAD)).json()
    assert data["rating"] == 4.5


@pytest.mark.asyncio
@pytest.mark.parametrize("cat", ["food", "nightlife", "temple", "nature"])
async def test_predict_categories(ac, cat):
    payload = {**VALID_PAYLOAD, "category": cat}
    r = await ac.post("/predict", json=payload)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_predict_missing_field(ac):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "text"}
    r = await ac.post("/predict", json=bad)
    assert r.status_code in (400, 422)


@pytest.mark.asyncio
async def test_predict_empty_payload(ac):
    r = await ac.post("/predict", json={})
    assert r.status_code in (400, 422)


# ─────────────────────────────────────────────
# Info endpoint
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_info_returns_200(ac):
    r = await ac.get("/info")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_info_structure(ac):
    data = (await ac.get("/info")).json()
    assert "service" in data
    assert "version" in data
    assert "model_loaded" in data


# ─────────────────────────────────────────────
# Root endpoint
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_root_returns_html(ac):
    r = await ac.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
