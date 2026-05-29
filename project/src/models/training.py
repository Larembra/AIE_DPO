from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.architectures import GRUTextClassifier, GRUTagger
from models.utils import (
    build_word2idx_from_texts,
    build_word2idx_from_tokens,
    texts_to_sequences,
    tokens_to_padded_indices,
    labels_to_padded,
    parse_list_column,
)

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def train_multilabel_gru(
    data_dir: Path,
    artifacts_dir: Path,
    configs_dir: Path,
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Path:
    set_seed(seed)
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

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            steps += 1

        train_loss = train_loss / max(1, steps)

        model.eval()
        val_loss = 0.0
        vsteps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += loss.item()
                vsteps += 1
        val_loss = val_loss / max(1, vsteps)

        logger.info("[multilabel] epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

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

    # evaluate on test (if available) otherwise on val
    test_df = None
    test_path = data_dir / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
    else:
        test_df = val_df

    # prepare test loader
    X_test = texts_to_sequences(test_df["text"].astype(str).tolist(), word2idx, max_len)
    y_test = test_df[label_cols].values.astype(np.float32)
    test_ds = TensorDataset(torch.LongTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # compute metrics
    y_true = []
    y_score = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(yb.numpy().tolist())

    try:
        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)
        y_pred = (y_score_arr > 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred)),
            "precision": float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr, average="macro"))
        except Exception:
            metrics["roc_auc"] = None
    except Exception as e:
        logger.exception("Error while computing multilabel metrics: %s", e)
        metrics = {}

    # save metrics
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("[multilabel] test_metrics=%s", metrics)

    return model_path


def train_spans_gru(
    data_dir: Path,
    artifacts_dir: Path,
    configs_dir: Path,
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Path:
    set_seed(seed)
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

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0
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
            train_loss += loss.item()
            steps += 1

        train_loss = train_loss / max(1, steps)

        model.eval()
        val_loss = 0.0
        vsteps = 0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss = (loss * mb).sum() / (mb.sum() + 1e-9)
                val_loss += loss.item()
                vsteps += 1
        val_loss = val_loss / max(1, vsteps)

        logger.info("[spans] epoch=%d train_loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

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

    # evaluate on test (if available) otherwise on val
    test_df = None
    test_path = data_dir / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
    else:
        test_df = val_df

    test_tokens = [parse_list_column(x) for x in test_df["tokens"]]
    test_labels = [parse_list_column(x) for x in test_df["labels"]]
    X_test, M_test = tokens_to_padded_indices(test_tokens, word2idx, max_len)
    y_test = labels_to_padded(test_labels, max_len)

    y_true_all = []
    y_score_all = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_test), 32):
            xb = torch.LongTensor(X_test[i : i + 32]).to(device)
            mb = torch.FloatTensor(M_test[i : i + 32]).to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            mb_np = mb.cpu().numpy()
            for j in range(probs.shape[0]):
                mask = mb_np[j]
                probs_row = probs[j]
                labs_row = y_test[j + i]
                for p, t, m in zip(probs_row[: len(labs_row)], labs_row[: len(labs_row)], mask[: len(labs_row)]):
                    if m > 0:
                        y_score_all.append(float(p))
                        y_true_all.append(int(t))

    try:
        y_pred = [1 if s >= 0.5 else 0 for s in y_score_all]
        metrics = {
            "accuracy": float(accuracy_score(y_true_all, y_pred)),
            "precision": float(precision_score(y_true_all, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_all, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true_all, y_pred, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_all, y_score_all))
        except Exception:
            metrics["roc_auc"] = None
    except Exception as e:
        logger.exception("Error while computing spans metrics: %s", e)
        metrics = {}

    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("[spans] test_metrics=%s", metrics)

    return model_path


def train_detox_transformer(
    data_dir: Path,
    artifacts_dir: Path,
    configs_dir: Path,
    epochs: int,
    device: torch.device,
    seed: int = 42,
) -> Path:
    set_seed(seed)
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
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collator, generator=gen)
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

        logger.info("[detox-transformer] val_loss=%.6f", val_loss)

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

    # evaluate on test (if available) otherwise on val: generate outputs and compute BLEU/ROUGE/BERTScore
    test_df = None
    test_path = data_dir / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
    else:
        test_df = val_df

    try:
        import evaluate
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")
        gen_texts = []
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = model.to(device)
        batchsize = 8
        for i in range(0, len(test_df), batchsize):
            batch = test_df["ru_toxic_comment"].astype(str).tolist()[i : i + batchsize]
            preds = []
            try:
                preds = model.generate(
                    **tokenizer(["детоксифицируй текст: " + t for t in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device),
                    max_new_tokens=64,
                    num_beams=4,
                )
            except Exception:
                # fallback simple decode via pipeline
                continue
            decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
            gen_texts.extend([d.strip() for d in decoded])

        refs = test_df["ru_neutral_comment"].astype(str).tolist()[: len(gen_texts)]
        results = {}
        if gen_texts:
            try:
                bleu_score = bleu.compute(predictions=gen_texts, references=[[r] for r in refs])["bleu"]
            except Exception:
                bleu_score = 0.0
            rouge_scores = rouge.compute(predictions=gen_texts, references=refs)
            bert_scores = bertscore.compute(predictions=gen_texts, references=refs, lang="ru")
            results = {
                "bleu": float(bleu_score),
                "rouge1": rouge_scores.get("rouge1"),
                "rouge2": rouge_scores.get("rouge2"),
                "rougeL": rouge_scores.get("rougeL"),
                "bertscore_f1": float(np.mean(bert_scores["f1"])) if bert_scores and "f1" in bert_scores else None,
            }
        else:
            results = {}
    except Exception as e:
        logger.exception("Could not compute transformer metrics: %s", e)
        results = {}

    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("[detox-transformer] test_metrics=%s", results)

    return model_dir

