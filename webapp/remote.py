"""Task 8 — client for the cloud-hosted model API. Same interface as KerasPredictor,
so the Flask views do not care whether the model runs in-process or in the cloud."""
import time

import requests

from visionguard.inference import Prediction


class RemotePredictorError(RuntimeError):
    pass


class RemotePredictor:
    def __init__(self, url: str, api_key: str | None, timeout: float = 30):
        if not url:
            raise ValueError("MODEL_API_URL must be set when INFERENCE_MODE=remote")
        self.url = url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.timeout = timeout
        self.session = requests.Session()  # keep-alive: reuse the TLS connection between calls
        self.model_name, self.model_version, self.threshold = "remote", "?", None

    def health(self) -> dict:
        r = self.session.get(self.url.rsplit("/predict", 1)[0] + "/health", timeout=self.timeout)
        r.raise_for_status()
        info = r.json()
        self.model_name = info.get("model_name", self.model_name)
        self.model_version = str(info.get("model_version", self.model_version))
        self.threshold = info.get("threshold")
        return info

    def predict_bytes(self, raw: bytes, filename: str = "upload.jpg") -> Prediction:
        t0 = time.perf_counter()
        try:
            r = self.session.post(self.url, headers=self.headers, timeout=self.timeout,
                                  files={"file": (filename, raw, "application/octet-stream")})
        except requests.RequestException as exc:
            raise RemotePredictorError(f"model API unreachable: {exc}") from exc
        if r.status_code != 200:
            raise RemotePredictorError(f"model API returned {r.status_code}: {r.text[:200]}")
        d = r.json()
        return Prediction(label=d["label"], confidence=d["confidence"], p_ai=d["p_ai"],
                          threshold=d["threshold"], model_name=d["model_name"],
                          model_version=str(d["model_version"]),
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))
