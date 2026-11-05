import io

from tests.conftest import image_bytes


def upload(client, data: bytes, name: str):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_pages_render(client):
    for url in ("/", "/history"):
        assert client.get(url).status_code == 200


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok" and r.json["database"] == "ok" and r.json["debug"] is False


def test_valid_upload_is_logged(client):
    r = upload(client, image_bytes("PNG"), "../../etc/My Photo.png")
    assert r.status_code == 200
    assert b"Confidence" in r.data and b"Caveat" in r.data
    rows = client.get("/api/history").json
    assert len(rows) == 1
    assert rows[0]["filename"] == "etc_My_Photo.png"  # sanitised, no path traversal
    assert rows[0]["label"] in ("AI-generated", "Real")
    assert rows[0]["model_name"] == "fake"
    assert b"etc_My_Photo.png" in client.get("/history").data


def test_wrong_file_type_rejected(client):
    r = upload(client, b"MZ\x90\x00 fake exe", "virus.exe")
    assert r.status_code == 415
    assert b"not accepted" in r.data


def test_too_large_rejected(client):
    r = upload(client, b"\xff" * 300_000, "big.jpg")
    assert r.status_code == 413
    assert b"limit" in r.data


def test_undecodable_image_rejected(client):
    r = upload(client, b"this is text pretending to be a jpeg", "fake.jpg")
    assert r.status_code == 422
    assert b"could not be decoded" in r.data


def test_rejections_are_not_logged(client):
    upload(client, b"nope", "fake.jpg")
    assert client.get("/api/history").json == []


def test_missing_file(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_404_friendly(client):
    r = client.get("/result/999")
    assert r.status_code == 404
    assert b"Try another image" in r.data
