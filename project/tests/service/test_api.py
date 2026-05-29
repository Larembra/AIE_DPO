from service import api
from fastapi.testclient import TestClient


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_ready_flags(client: TestClient):
    api.state.ready_classifier = False
    api.state.ready_spans = False
    api.state.ready_detox = False
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is False
    api.state.ready_classifier = True
    api.state.ready_spans = True
    api.state.ready_detox = True
    r2 = client.get("/ready")
    assert r2.json()["ready"] is True


def test_reload_monkeypatch(client: TestClient, monkeypatch):
    monkeypatch.setattr(api, "load_weights", lambda: None)
    r = client.post("/reload")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_detox_endpoint_smoke(client: TestClient, monkeypatch):
    monkeypatch.setattr(api, "predict_toxicity", lambda *a, **k: {"label": "normal", "probs": [0.1, 0.9]})
    monkeypatch.setattr(api, "predict_toxic_tokens", lambda *a, **k: ["bad"])
    class DummyDetox:
        def generate(self, texts):
            return ["detoxified " + t for t in texts]

    api.state.detox_model = DummyDetox()
    api.state.ready_classifier = True
    api.state.ready_spans = True
    api.state.ready_detox = True

    r = client.post("/detox", json={"text": "some bad text"})
    assert r.status_code == 200
    data = r.json()
    assert "toxicity_type" in data
    assert isinstance(data.get("toxic_words"), list)
    assert isinstance(data.get("detox_text"), str)


