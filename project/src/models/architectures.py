from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class GRUTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        last_hidden = out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.head(out)


class GRUTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # use attribute name 'rnn' to be compatible with notebooks that saved state_dict with 'rnn.*' keys
        self.rnn = nn.GRU(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        out = self.dropout(out)
        logits = self.head(out).squeeze(-1)
        return logits


class TransformerDetox:
    def __init__(self, model_name: str = "cointegrated/rut5-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def to(self, device: torch.device) -> "TransformerDetox":
        self.model = self.model.to(device)
        return self

    def generate(
        self,
        texts: list[str],
        max_length: int = 128,
        max_new_tokens: int = 64,
        num_beams: int = 4,
    ) -> list[str]:
        inputs = self.tokenizer(
            ["детоксифицируй текст: " + str(x) for x in texts],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [x.strip() for x in decoded]
