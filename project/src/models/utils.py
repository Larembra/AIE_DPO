from __future__ import annotations

import ast
from collections import Counter

import numpy as np

from features.text import tokenize


def build_word2idx_from_texts(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in counter.items():
        if count >= min_freq:
            word2idx[word] = idx
            idx += 1
    return word2idx


def build_word2idx_from_tokens(token_lists: list[list[str]], min_freq: int = 1) -> dict[str, int]:
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in counter.items():
        if count >= min_freq:
            word2idx[word] = idx
            idx += 1
    return word2idx


def texts_to_sequences(texts: list[str], word2idx: dict[str, int], max_len: int) -> np.ndarray:
    sequences = []
    for text in texts:
        seq = [word2idx.get(word, 1) for word in tokenize(text)][:max_len]
        seq += [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences, dtype=np.int64)


def tokens_to_padded_indices(tokens_list: list[list[str]], word2idx: dict[str, int], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    seqs = []
    masks = []
    for tokens in tokens_list:
        idxs = [word2idx.get(t, 1) for t in tokens][:max_len]
        mask = [1] * len(idxs)
        if len(idxs) < max_len:
            pad = [0] * (max_len - len(idxs))
            idxs = idxs + pad
            mask = mask + [0] * (max_len - len(mask))
        seqs.append(idxs)
        masks.append(mask)
    return np.array(seqs, dtype=np.int64), np.array(masks, dtype=np.float32)


def labels_to_padded(labels_list: list[list[int]], max_len: int) -> np.ndarray:
    arr = []
    for labels in labels_list:
        labels = labels[:max_len]
        if len(labels) < max_len:
            labels = labels + [0] * (max_len - len(labels))
        arr.append(labels)
    return np.array(arr, dtype=np.float32)


def parse_list_column(value) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []

