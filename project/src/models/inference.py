from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from features.text import tokenize
from models.architectures import GRUTextClassifier, GRUTagger, TransformerDetox
from models.utils import texts_to_sequences, tokens_to_padded_indices

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    if not path.exists():
        logger.debug("[read_json] file not found: %s", path)
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("[read_json] loaded %s (keys: %s)", path, list(data.keys()))
        return data
    except Exception as e:
        logger.exception("[read_json] error reading %s: %s", path, e)
        return {}


def load_multilabel_model(
    artifacts_dir: Path,
    configs_dir: Path,
    device: torch.device,
) -> tuple[GRUTextClassifier, dict[str, int], list[str]]:
    logger.info("[load_multilabel_model] artifacts_dir=%s", artifacts_dir)
    logger.info("[load_multilabel_model] configs_dir=%s", configs_dir)
    hyper = _read_json(configs_dir / "hyperparams.json")
    inf = _read_json(configs_dir / "inference_params.json")
    vocab_path = artifacts_dir / "word2idx.json"

    try:
        logger.debug("[load_multilabel_model] checking %s exists=%s", vocab_path, vocab_path.exists())
        if artifacts_dir.exists():
            logger.debug("[load_multilabel_model] artifacts files: %s", [p.name for p in artifacts_dir.iterdir()])
    except Exception:
        logger.exception("[load_multilabel_model] error listing artifacts_dir: %s", artifacts_dir)

    try:
        with vocab_path.open(encoding="utf-8") as f:
            word2idx = json.load(f)
        logger.info("[load_multilabel_model] loaded vocab size: %d", len(word2idx))
    except Exception:
        logger.exception("[load_multilabel_model] failed to load vocab at %s", vocab_path)
        word2idx = {}

    labels = ["normal", "insult", "threat", "obscenity"]

    model = GRUTextClassifier(
        vocab_size=int(hyper.get("vocab_size", 2)),
        embed_dim=int(hyper.get("embed_dim", 100)),
        hidden_size=int(hyper.get("hidden_size", 64)),
        num_layers=int(hyper.get("num_layers", 2)),
        dropout=float(hyper.get("dropout", 0.3)),
        num_classes=len(labels),
    ).to(device)

    state_path = artifacts_dir / "best_multilabel_model.pt"
    logger.debug("[load_multilabel_model] looking for model state at %s exists=%s", state_path, state_path.exists())
    if state_path.exists():
        try:
            state = torch.load(state_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            logger.info("[load_multilabel_model] model loaded and set to eval from %s", state_path)
        except Exception:
            logger.exception("[load_multilabel_model] error loading state_dict from %s", state_path)
    else:
        logger.warning("[load_multilabel_model] model file not found: %s", state_path)
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
    logger.info("[load_spans_model] artifacts_dir=%s", artifacts_dir)
    logger.info("[load_spans_model] configs_dir=%s", configs_dir)
    hyper = _read_json(configs_dir / "hyperparams.json")
    inf = _read_json(configs_dir / "inference_params.json")
    max_len = int(inf.get("max_len", 100))

    vocab_path = artifacts_dir / "word2idx.json"
    logger.debug("[load_spans_model] checking vocab at %s exists=%s", vocab_path, vocab_path.exists())
    try:
        if artifacts_dir.exists():
            logger.debug("[load_spans_model] artifacts files: %s", [p.name for p in artifacts_dir.iterdir()])
    except Exception:
        logger.exception("[load_spans_model] error listing artifacts_dir: %s", artifacts_dir)

    if not vocab_path.exists():
        logger.error("[load_spans_model] vocab file not found: %s", vocab_path)
        raise FileNotFoundError(f"vocab file not found: {vocab_path}")

    try:
        with vocab_path.open(encoding="utf-8") as f:
            word2idx = json.load(f)
        logger.info("[load_spans_model] loaded vocab size: %d", len(word2idx))
    except Exception:
        logger.exception("[load_spans_model] failed to load vocab at %s", vocab_path)
        raise

    model = GRUTagger(
        vocab_size=len(word2idx),
        embed_dim=int(hyper.get("embed_dim", 100)),
        hidden_size=int(hyper.get("hidden_size", 128)),
        num_layers=int(hyper.get("num_layers", 2)),
        dropout=float(hyper.get("dropout", 0.3)),
    ).to(device)

    state_path = artifacts_dir / "best_spans_model.pt"
    logger.debug("[load_spans_model] looking for model state at %s exists=%s", state_path, state_path.exists())
    if not state_path.exists():
        logger.error("[load_spans_model] state dict not found: %s", state_path)
        raise FileNotFoundError(f"state dict not found: {state_path}")

    try:
        state = torch.load(state_path, map_location=device)
        try:
            model.load_state_dict(state)
            model.eval()
            logger.info("[load_spans_model] model loaded and set to eval from %s", state_path)
        except Exception as e_load:
            logger.warning("[load_spans_model] direct load failed: %s. Attempting key-mapping...", e_load)
            # try to remap common rnn/gru name differences
            sd_keys = list(state.keys())
            model_keys = list(model.state_dict().keys())
            mapped = None
            if any(k.startswith("rnn.") for k in sd_keys) and any(k.startswith("gru.") for k in model_keys):
                mapped = {k.replace("rnn.", "gru."): v for k, v in state.items()}
                logger.info("[load_spans_model] remapping keys rnn.->gru.")
            elif any(k.startswith("gru.") for k in sd_keys) and any(k.startswith("rnn.") for k in model_keys):
                mapped = {k.replace("gru.", "rnn."): v for k, v in state.items()}
                logger.info("[load_spans_model] remapping keys gru.->rnn.")

            if mapped is not None:
                try:
                    model.load_state_dict(mapped)
                    model.eval()
                    logger.info("[load_spans_model] model loaded after key-mapping from %s", state_path)
                except Exception:
                    logger.exception("[load_spans_model] mapped load also failed")
                    raise
            else:
                logger.exception("[load_spans_model] no viable key mapping found")
                raise
    except Exception:
        logger.exception("[load_spans_model] final error loading state_dict from %s", state_path)
        raise

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
    model_dir = artifacts_dir / "best_detox_model.pt"
    logger.info("[load_detox_transformer] artifacts_dir=%s model_dir=%s exists=%s", artifacts_dir, model_dir, model_dir.exists())
    if model_dir.exists():
        logger.info("[load_detox_transformer] loading transformer state from %s", model_dir)
        try:
            model = TransformerDetox()
            state = torch.load(model_dir, map_location=device)
            model.model.load_state_dict(state)
            logger.info("[load_detox_transformer] loaded state_dict into transformer model from %s", model_dir)
        except Exception:
            logger.exception("[load_detox_transformer] failed to load state_dict from %s", model_dir)
            logger.info("[load_detox_transformer] defaulting to hub model")
            model = TransformerDetox()
    else:
        # try to load folder with pretrained saved model
        model_folder = artifacts_dir / "transformer"
        if model_folder.exists():
            try:
                model = TransformerDetox(model_name=str(model_folder))
                logger.info("[load_detox_transformer] loaded transformer from folder %s", model_folder)
            except Exception:
                logger.exception("[load_detox_transformer] failed to load transformer from folder %s", model_folder)
                model = TransformerDetox()
        else:
            logger.info("[load_detox_transformer] defaulting to hub model %s", getattr(TransformerDetox, 'model_name', 'cointegrated/rut5-small'))
            model = TransformerDetox()
    model = model.to(device)
    logger.info("[load_detox_transformer] transformer ready")
    return model
