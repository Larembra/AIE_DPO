import numpy as np
from models.utils import (
    build_word2idx_from_texts,
    build_word2idx_from_tokens,
    texts_to_sequences,
    tokens_to_padded_indices,
    labels_to_padded,
    parse_list_column,
)


def test_build_word2idx_and_sequences():
    texts = ["hello world", "hello"]
    w2i = build_word2idx_from_texts(texts)
    assert w2i["<PAD>"] == 0
    assert w2i["<UNK>"] == 1
    assert "hello" in w2i
    seqs = texts_to_sequences(["hello world"], w2i, max_len=5)
    assert seqs.shape == (1, 5)


def test_tokens_to_padded_and_labels():
    tokens = [["a", "b", "c"], ["d"]]
    w2i = build_word2idx_from_tokens(tokens)
    seqs, masks = tokens_to_padded_indices(tokens, w2i, max_len=4)
    assert seqs.shape == (2, 4)
    assert masks.shape == (2, 4)
    lbls = labels_to_padded([[1, 0, 1], [0]], max_len=4)
    assert lbls.shape == (2, 4)


def test_parse_list_column():
    assert parse_list_column([1, 2]) == [1, 2]
    assert parse_list_column(None) == []
    assert parse_list_column("[1,2]") == [1, 2]
    assert parse_list_column("notalist") == []

