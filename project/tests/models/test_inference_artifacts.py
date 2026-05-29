import pytest
from pathlib import Path
from models import inference
import torch


def test_load_multilabel_model_fallback(tmp_path: Path, cpu_device: torch.device):
    art = tmp_path / "artifacts"
    cfg = tmp_path / "configs"
    art.mkdir()
    cfg.mkdir()
    model, word2idx, labels = inference.load_multilabel_model(art, cfg, cpu_device)
    import torch.nn as nn
    assert isinstance(model, nn.Module)
    assert isinstance(word2idx, dict)
    assert isinstance(labels, list)


def test_load_spans_model_missing_vocab(tmp_path: Path, cpu_device: torch.device):
    art = tmp_path / "artifacts2"
    cfg = tmp_path / "configs2"
    art.mkdir()
    cfg.mkdir()
    with pytest.raises(FileNotFoundError):
        inference.load_spans_model(art, cfg, cpu_device)

