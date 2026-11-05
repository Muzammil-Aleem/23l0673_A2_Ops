import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visionguard.inference import Prediction  # noqa: E402


class FakePredictor:
    """Stands in for the Keras model so tests run without TensorFlow."""
    model_name, model_version, threshold, img_size = "fake", "0", 0.5, 224

    def predict_bytes(self, raw, filename=None):
        from visionguard.preprocess import preprocess_bytes
        x = preprocess_bytes(raw)  # exercise the real preprocessing path
        p = float(x.mean() / 255.0)
        label = "AI-generated" if p >= self.threshold else "Real"
        return Prediction(label, p if p >= 0.5 else 1 - p, p, self.threshold, self.model_name,
                          self.model_version, 1.0)


def image_bytes(fmt="PNG", size=(64, 48), color=(200, 30, 30), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, color if mode == "RGB" else None).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def app(tmp_path):
    from webapp import create_app
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 't.db'}",
                      "LOAD_MODEL": False, "MAX_CONTENT_LENGTH": 200_000},
                     predictor=FakePredictor())
    yield app


@pytest.fixture
def client(app):
    return app.test_client()
