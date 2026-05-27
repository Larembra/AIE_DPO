from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from features.text import tokenize, comma_words_to_token_lists

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except Exception:
    MultilabelStratifiedShuffleSplit = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def load_detox_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = {"train": "train.tsv", "validation": "dev.tsv"}
    df_train = pd.read_csv("hf://datasets/s-nlp/ru_paradetox/" + splits["train"], sep="\t")
    df_val = pd.read_csv("hf://datasets/s-nlp/ru_paradetox/" + splits["validation"], sep="\t")
    return df_train, df_val


def clean_detox(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    sample_frac: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = df_train.drop_duplicates().reset_index(drop=True)
    df_val = df_val.drop_duplicates().reset_index(drop=True)

    mask_error_train = df_train.astype(str).apply(lambda col: col.str.contains(r"#ERROR!", na=False)).any(axis=1)
    mask_error_val = df_val.astype(str).apply(lambda col: col.str.contains(r"#ERROR!", na=False)).any(axis=1)

    df_train = df_train.loc[~mask_error_train].reset_index(drop=True)
    df_val = df_val.loc[~mask_error_val].reset_index(drop=True)

    if 0 < sample_frac < 1:
        df_train = df_train.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        df_val = df_val.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    df_train, df_test = train_test_split(df_train, test_size=test_size, random_state=random_state)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_val, df_test


def save_detox_splits(out_dir: Path, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)


def load_multilabel_raw(path: Path) -> pd.DataFrame:
    data_list = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            labels = line.split()[0]
            text = line[len(labels) + 1 :].strip()
            labels = labels.split(",")
            mask = [
                1 if "__label__NORMAL" in labels else 0,
                1 if "__label__INSULT" in labels else 0,
                1 if "__label__THREAT" in labels else 0,
                1 if "__label__OBSCENITY" in labels else 0,
            ]
            data_list.append((text, *mask))
    return pd.DataFrame(data_list, columns=["text", "normal", "insult", "threat", "obscenity"])


def clean_multilabel(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["normal", "insult", "threat", "obscenity"]
    df = df.copy()
    df["labels_tuple"] = df[label_cols].apply(tuple, axis=1)
    labels_nunique = df.groupby("text")["labels_tuple"].nunique()
    conflict_texts = labels_nunique[labels_nunique > 1].index.tolist()
    conflict_set = set(conflict_texts)

    mask_nonconflict = ~df["text"].isin(conflict_set)
    df_nonconflict = df[mask_nonconflict].copy()

    exact_dup_mask = df_nonconflict.duplicated(subset=["text"] + label_cols, keep="first")
    df_clean = df_nonconflict.drop_duplicates(subset=["text"] + label_cols, keep="first").reset_index(drop=True)

    if "labels_tuple" in df_clean.columns:
        df_clean = df_clean.drop(columns=["labels_tuple"])

    return df_clean


def sample_multilabel(
    df: pd.DataFrame,
    sample_frac: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    if not (0 < sample_frac < 1):
        return df.reset_index(drop=True)

    label_cols = ["normal", "insult", "threat", "obscenity"]
    X = df["text"].astype(str)
    y = df[label_cols]
    sample_size = max(1, int(len(df) * sample_frac))

    if MultilabelStratifiedShuffleSplit is not None:
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - (sample_size / len(df)), random_state=random_state
        )
        for sample_idx, _ in msss.split(X, y):
            sample_idx = X.index[sample_idx]
            return df.loc[sample_idx].reset_index(drop=True)

    strat = y.sum(axis=1)
    sample_indices = train_test_split(
        df.index,
        train_size=sample_size / len(df),
        random_state=random_state,
        stratify=strat,
    )[0]
    return df.loc[sample_indices].reset_index(drop=True)


def split_multilabel(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_cols = ["normal", "insult", "threat", "obscenity"]
    X = df["text"].astype(str)
    y = df[label_cols]

    try:
        if MultilabelStratifiedShuffleSplit is not None:
            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
            for train_idx, temp_idx in msss.split(X, y):
                X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
                y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]
            msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
            for val_idx_rel, test_idx_rel in msss2.split(X_temp, y_temp):
                val_idx = X_temp.index[val_idx_rel]
                test_idx = X_temp.index[test_idx_rel]
                X_val, X_test = X.loc[val_idx], X.loc[test_idx]
                y_val, y_test = y.loc[val_idx], y.loc[test_idx]
        else:
            strat = y.sum(axis=1)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=random_state, stratify=strat
            )
            strat_temp = y_temp.sum(axis=1)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=strat_temp
            )
    except Exception:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    return train_df, val_df, test_df


def save_multilabel_splits(
    out_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


def load_spans_raw(split: str = "ru") -> pd.DataFrame:
    if load_dataset is None:
        raise RuntimeError("datasets is not available")
    ds = load_dataset("textdetox/multilingual_toxic_spans", split=split)
    return pd.DataFrame(ds)


def build_span_labels(tokens: list[str], toxic_words: str) -> list[int]:
    labels = [0] * len(tokens)
    toxic_list = comma_words_to_token_lists(toxic_words)
    for toxic in toxic_list:
        n = len(toxic)
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == toxic:
                for j in range(i, i + n):
                    labels[j] = 1
    return labels


def prepare_spans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tokens"] = df["Sentence"].apply(tokenize)
    df["labels"] = df.apply(lambda row: build_span_labels(row["tokens"], row["Negative Connotations"]), axis=1)
    return df


def split_spans(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["has_toxic"] = df["labels"].apply(lambda lbl: int(any(lbl)))
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["has_toxic"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_spans_splits(out_dir: Path, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

