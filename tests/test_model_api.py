import io

import pytest

from tests.conftest import FakePredictor, image_bytes


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "s3cret")
    from model_api.app import create_app
    return create_app(predictor=FakePredictor()).test_client()


def test_health_needs_no_key(api):
    r = api.get("/health")
    assert r.status_code == 200 and r.json["auth_required"] is True


def test_predict_requires_key(api):
    r = api.post("/predict", data=image_bytes())
    assert r.status_code == 401


def test_predict_multipart(api):
    r = api.post("/predict", headers={"X-API-Key": "s3cret"},
                 data={"file": (io.BytesIO(image_bytes("JPEG")), "a.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 200
    assert set(r.json) >= {"label", "confidence", "p_ai", "threshold", "model_version"}


def test_predict_raw_body_and_bad_image(api):
    ok = api.post("/predict", headers={"X-API-Key": "s3cret"}, data=image_bytes())
    assert ok.status_code == 200
    bad = api.post("/predict", headers={"X-API-Key": "s3cret"}, data=b"garbage")
    assert bad.status_code == 422
