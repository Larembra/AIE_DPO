from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from features.text import tokenize
from models.architectures import GRUTextClassifier, GRUTagger, TransformerDetox
from models.utils import texts_to_sequences, tokens_to_padded_indices


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_multilabel_model(
    artifacts_dir: Path,
    configs_dir: Path,
    device: torch.device,
) -> tuple[GRUTextClassifier, dict[str, int], list[str]]:
    hyper = _read_json(configs_dir / "hyperparams.json")
    inf = _read_json(configs_dir / "inference_params.json")
    vocab_path = artifacts_dir / "word2idx.json"
    labels_path = artifacts_dir / "labels.json"

    with vocab_path.open(encoding="utf-8") as f:
        word2idx = json.load(f)

    if labels_path.exists():
        with labels_path.open(encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = ["normal", "insult", "threat", "obscenity"]

    model = GRUTextClassifier(
        vocab_size=len(word2idx),
        embed_dim=int(hyper.get("embed_dim", 100)),
        hidden_size=int(hyper.get("hidden_size", 64)),
        num_layers=int(hyper.get("num_layers", 2)),
        dropout=float(hyper.get("dropout", 0.3)),
        num_classes=len(labels),
    ).to(device)
    state = torch.load(artifacts_dir / "best_multilabel_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, word2idx, labels


def predict_toxicity(
    text: str,
    model: GRUTextClassifier,
    word2idx: dict[str, int],
    labels: list[str],
    max_len: int,
    device: torch.device,
) -> dict:
    seq = texts_to_sequences([text], word2idx, max_len)
    with torch.no_grad():
        logits = model(torch.LongTensor(seq).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return {"label": labels[idx], "probs": probs.tolist()}


def load_spans_model(
    artifacts_dir: Path,
    configs_dir: Path,
    device: torch.device,
) -> tuple[GRUTagger, dict[str, int], int]:
    hyper = _read_json(configs_dir / "hyperparams.json")
    inf = _read_json(configs_dir / "inference_params.json")
    max_len = int(inf.get("max_len", 100))

    with (artifacts_dir / "word2idx.json").open(encoding="utf-8") as f:
        word2idx = json.load(f)

    model = GRUTagger(
        vocab_size=len(word2idx),
        embed_dim=int(hyper.get("embed_dim", 100)),
        hidden_size=int(hyper.get("hidden_size", 128)),
        num_layers=int(hyper.get("num_layers", 2)),
        dropout=float(hyper.get("dropout", 0.3)),
    ).to(device)
    state = torch.load(artifacts_dir / "best_spans_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, word2idx, max_len


def predict_toxic_tokens(
    text: str,
    model: GRUTagger,
    word2idx: dict[str, int],
    max_len: int,
    device: torch.device,
    threshold: float = 0.5,
) -> list[str]:
    tokens = tokenize(text)
    seqs, masks = tokens_to_padded_indices([tokens], word2idx, max_len)
    seq = torch.LongTensor(seqs).to(device)
    mask = torch.FloatTensor(masks).to(device)
    with torch.no_grad():
        logits = model(seq)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    toxic = []
    for token, prob, m in zip(tokens, probs[: len(tokens)], mask.cpu().numpy()[0][: len(tokens)]):
        if m > 0 and prob >= threshold:
            toxic.append(token)
    return toxic


def load_detox_transformer(
    artifacts_dir: Path,
    device: torch.device,
) -> TransformerDetox:
    model_dir = artifacts_dir / "transformer"
    if model_dir.exists():
        model = TransformerDetox(model_name=str(model_dir))
    else:
        model = TransformerDetox()
    return model.to(device)

