from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.inference import (
    load_multilabel_model,
    load_spans_model,
    load_detox_transformer,
    predict_toxicity,
    predict_toxic_tokens,
)


class HealthResponse(BaseModel):
    status: str


class ModelInfo(BaseModel):
    name: str
    version: str
    ready: bool | None = None


class ReadyResponse(BaseModel):
    ready: bool
    models: list[ModelInfo]


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class DetoxRequest(BaseModel):
    text: str


class DetoxResponse(BaseModel):
    toxicity_type: str
    toxic_words: list[str]
    detox_text: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _config_dir(name: str) -> Path:
    return _project_root() / "configs" / name


def _artifacts_dir(name: str) -> Path:
    return _project_root() / "artifacts" / name


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    import json

    with path.open(encoding="utf-8") as f:
        return json.load(f)


class ModelState:
    def __init__(self):
        self.classifier = None
        self.classifier_vocab = None
        self.classifier_labels = None
        self.classifier_max_len = None
        self.spans_model = None
        self.spans_vocab = None
        self.spans_max_len = None
        self.detox_model = None
        self.ready_classifier = False
        self.ready_spans = False
        self.ready_detox = False


state = ModelState()
app = FastAPI(title="AI Detox Service", version="1.0")


def load_weights() -> None:
    device = _device()

    try:
        cfg = _config_dir("multilabel_model")
        art = _artifacts_dir("multilabel_model")
        model, word2idx, labels = load_multilabel_model(art, cfg, device)
        inf = _read_json(cfg / "inference_params.json")
        state.classifier = model
        state.classifier_vocab = word2idx
        state.classifier_labels = labels
        state.classifier_max_len = int(inf.get("max_len", 100))
        state.ready_classifier = True
    except Exception:
        state.ready_classifier = False

    try:
        cfg = _config_dir("spans_model")
        art = _artifacts_dir("spans_model")
        model, word2idx, max_len = load_spans_model(art, cfg, device)
        state.spans_model = model
        state.spans_vocab = word2idx
        state.spans_max_len = max_len
        state.ready_spans = True
    except Exception:
        state.ready_spans = False

    try:
        art = _artifacts_dir("detox_model")
        state.detox_model = load_detox_transformer(art, device)
        state.ready_detox = True
    except Exception:
        state.ready_detox = False


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    models = [
        ModelInfo(name="toxicity_classifier", version="1.0", ready=state.ready_classifier),
        ModelInfo(name="toxic_token_detector", version="1.2", ready=state.ready_spans),
        ModelInfo(name="detox_generator", version="2.0", ready=state.ready_detox),
    ]
    return ReadyResponse(ready=all(m.ready for m in models), models=models)


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(
        models=[
            ModelInfo(name="toxicity_classifier", version="1.0"),
            ModelInfo(name="toxic_token_detector", version="1.2"),
        ]
    )


@app.post("/load_weights")
@app.post("/reload")
def load_weights_endpoint() -> dict:
    load_weights()
    return {"status": "ok"}


@app.post("/detox", response_model=DetoxResponse)
def detox(req: DetoxRequest) -> DetoxResponse:
    if not (state.ready_classifier and state.ready_spans and state.ready_detox):
        raise HTTPException(status_code=503, detail="models not ready")

    device = _device()
    clf = predict_toxicity(
        req.text,
        state.classifier,
        state.classifier_vocab,
        state.classifier_labels,
        state.classifier_max_len,
        device,
    )
    toxic_words = predict_toxic_tokens(
        req.text,
        state.spans_model,
        state.spans_vocab,
        state.spans_max_len,
        device,
    )
    detox_text = state.detox_model.generate([req.text])[0]

    return DetoxResponse(
        toxicity_type=clf["label"],
        toxic_words=toxic_words,
        detox_text=detox_text,
    )


@app.on_event("startup")
def _startup() -> None:
    load_weights()

