from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from models.architectures import GRUTextClassifier, GRUTagger
from models.utils import (
    build_word2idx_from_texts,
    build_word2idx_from_tokens,
    texts_to_sequences,
    tokens_to_padded_indices,
    labels_to_padded,
    parse_list_column,
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def train_multilabel_gru(
    data_dir: Path,
    artifacts_dir: Path,
    configs_dir: Path,
    epochs: int,
    device: torch.device,
) -> Path:
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    label_cols = ["normal", "insult", "threat", "obscenity"]

    max_len = int(_read_json(configs_dir / "inference_params.json").get("max_len", 100))
    hyper = _read_json(configs_dir / "hyperparams.json")

    word2idx = build_word2idx_from_texts(train_df["text"].astype(str).tolist())
    X_train = texts_to_sequences(train_df["text"].astype(str).tolist(), word2idx, max_len)
    X_val = texts_to_sequences(val_df["text"].astype(str).tolist(), word2idx, max_len)

    y_train = train_df[label_cols].values.astype(np.float32)
    y_val = val_df[label_cols].values.astype(np.float32)

    train_ds = TensorDataset(torch.LongTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.LongTensor(X_val), torch.FloatTensor(y_val))

    model = GRUTextClassifier(
        vocab_size=len(word2idx),
        embed_dim=int(hyper.get("embed_dim", 100)),
        hidden_size=int(hyper.get("hidden_size", 64)),
        num_layers=int(hyper.get("num_layers", 2)),
        dropout=float(hyper.get("dropout", 0.3)),
        num_classes=len(label_cols),
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += loss.item()
        val_loss = val_loss / max(1, len(val_loader))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "best_multilabel_model.pt"
    torch.save(model.state_dict(), model_path)

    vocab_path = artifacts_dir / "word2idx.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    labels_path = artifacts_dir / "labels.json"
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(label_cols, f, ensure_ascii=False, indent=2)

    return model_path


def train_spans_gru(
    data_dir: Path,
    artifacts_dir: Path,
    configs_dir: Path,
    epochs: int,
    device: torch.device,
) -> Path:
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")

    train_tokens = [parse_list_column(x) for x in train_df["tokens"]]
    val_tokens = [parse_list_column(x) for x in val_df["tokens"]]
    train_labels = [parse_list_column(x) for x in train_df["labels"]]
    val_labels = [parse_list_column(x) for x in val_df["labels"]]

    max_len = int(_read_json(configs_dir / "inference_params.json").get("max_len", 100))
    hyper = _read_json(configs_dir / "hyperparams.json")

    word2idx = build_word2idx_from_tokens(train_tokens)

    X_train, M_train = tokens_to_padded_indices(train_tokens, word2idx, max_len)
    X_val, M_val = tokens_to_padded_indices(val_tokens, word2idx, max_len)
    y_train = labels_to_padded(train_labels, max_len)
    y_val = labels_to_padded(val_labels, max_len)

    train_ds = TensorDataset(torch.LongTensor(X_train), torch.FloatTensor(y_train), torch.FloatTensor(M_train))
    val_ds = TensorDataset(torch.LongTensor(X_val), torch.FloatTensor(y_val), torch.FloatTensor(M_val))

    model = GRUTagger(
        vocab_size=len(word2idx),
        embed_dim=int(hyper.get("embed_dim", 100)),
        hidden_size=int(hyper.get("hidden_size", 128)),
        num_layers=int(hyper.get("num_layers", 2)),
        dropout=float(hyper.get("dropout", 0.3)),
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss = (loss * mb).sum() / (mb.sum() + 1e-9)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss = (loss * mb).sum() / (mb.sum() + 1e-9)
                val_loss += loss.item()
        val_loss = val_loss / max(1, len(val_loader))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "best_spans_model.pt"
    torch.save(model.state_dict(), model_path)

    vocab_path = artifacts_dir / "word2idx.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    return model_path


def train_detox_transformer(
    data_dir: Path,
    artifacts_dir: Path,
    configs_dir: Path,
    epochs: int,
    device: torch.device,
) -> Path:
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")

    train_df = train_df.dropna()
    val_df = val_df.dropna()
    train_df = train_df[train_df["ru_toxic_comment"].str.len() > 0]
    train_df = train_df[train_df["ru_neutral_comment"].str.len() > 0]
    val_df = val_df[val_df["ru_toxic_comment"].str.len() > 0]
    val_df = val_df[val_df["ru_neutral_comment"].str.len() > 0]

    inf_params = _read_json(configs_dir / "inference_params.json")
    max_len = int(inf_params.get("max_len", 128))

    model_name = "cointegrated/rut5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def encode_batch(df: pd.DataFrame):
        inputs = tokenizer(
            ["детоксифицируй текст: " + str(x) for x in df["ru_toxic_comment"].tolist()],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        labels = tokenizer(
            df["ru_neutral_comment"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return {"inputs": inputs, "labels": labels}

    train_data = encode_batch(train_df)
    val_data = encode_batch(val_df)

    class DetoxSeq2SeqDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __len__(self):
            return self.labels.size(0)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.inputs.items()}
            item["labels"] = self.labels[idx]
            return item

    train_ds = DetoxSeq2SeqDataset(train_data["inputs"], train_data["labels"])
    val_ds = DetoxSeq2SeqDataset(val_data["inputs"], val_data["labels"])

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    best_state = None
    best_val = float("inf")

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = out.loss
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss
                if torch.isnan(loss):
                    continue
                total += loss.item()
                n += 1
        val_loss = total / max(1, n)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifacts_dir / "transformer"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    return model_dir

