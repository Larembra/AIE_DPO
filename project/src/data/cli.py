from __future__ import annotations

from pathlib import Path

import logging


import pandas as pd
import typer

from data.datasets import (
    load_detox_raw,
    clean_detox,
    save_detox_splits,
    load_multilabel_raw,
    clean_multilabel,
    sample_multilabel,
    split_multilabel,
    save_multilabel_splits,
    load_spans_raw,
    prepare_spans,
    split_spans,
    save_spans_splits,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Data loading and cleaning commands (detox, multilabel, spans)")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@app.command("load-detox", help="Load raw detox dataset from HF and save to folder (train_raw.csv, val_raw.csv)")
def load_detox(out_dir: str = typer.Option("data/detox_raw")) -> None:
    root = _project_root()
    logger.info("Loading detox raw dataset")
    df_train, df_val = load_detox_raw()
    out_path = root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_path / "train_raw.csv", index=False)
    df_val.to_csv(out_path / "val_raw.csv", index=False)
    logger.info("Saved detox raw to %s (train=%d, val=%d)", out_path, len(df_train), len(df_val))
    typer.echo(str(out_path))


@app.command("clean-detox", help="Clean detox raw files and save splits (train/val/test). Logs number of removed rows if any.")
def clean_detox_cmd(
    in_dir: str = typer.Option("data/detox_raw"),
    out_dir: str = typer.Option("data/detox"),
    sample_frac: float = typer.Option(0.2),
    test_size: float = typer.Option(0.1),
) -> None:
    root = _project_root()
    in_path = root / in_dir
    raw_train_path = in_path / "train_raw.csv"
    raw_val_path = in_path / "val_raw.csv"
    df_train_raw = pd.read_csv(raw_train_path)
    df_val_raw = pd.read_csv(raw_val_path)
    logger.info("Cleaning detox datasets: raw train=%d rows, raw val=%d rows", len(df_train_raw), len(df_val_raw))
    train_df, val_df, test_df, stats = clean_detox(
        df_train=df_train_raw,
        df_val=df_val_raw,
        sample_frac=sample_frac,
        test_size=test_size,
    )
    logger.info(
        "Detox cleaning: initial train=%d val=%d; duplicates removed train=%d val=%d; errors removed train=%d val=%d; removed train=%d val=%d due to sample_frac; final splits train=%d val=%d test=%d; total_removed=%d",
        stats.get("initial_train", 0),
        stats.get("initial_val", 0),
        stats.get("duplicates_removed_train", 0),
        stats.get("duplicates_removed_val", 0),
        stats.get("errors_removed_train", 0),
        stats.get("errors_removed_val", 0),
        stats.get("sample_removed_train", 0),
        stats.get("sample_removed_val", 0),
        stats.get("final_train", 0),
        stats.get("final_val", 0),
        stats.get("final_test", 0),
        stats.get("total_removed", 0),
    )
    save_detox_splits(root / out_dir, train_df, val_df, test_df)
    logger.info("Saved detox splits to %s (train=%d val=%d test=%d)", root / out_dir, len(train_df), len(val_df), len(test_df))
    typer.echo(str(root / out_dir))


@app.command("load-multilabel", help="Load multilabel raw dataset (dataset.txt) and save to folder multilabel_raw")
def load_multilabel(
    dataset_path: str = typer.Option("data/dataset.txt"),
    out_dir: str = typer.Option("data/multilabel_raw"),
) -> None:
    root = _project_root()
    logger.info("Loading multilabel raw from %s", dataset_path)
    df = load_multilabel_raw(root / dataset_path)
    out_path = root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "dataset_raw.csv"
    df.to_csv(out_file, index=False)
    logger.info("Saved multilabel raw to %s (rows=%d)", out_file, len(df))
    typer.echo(str(out_path))


@app.command("clean-multilabel", help="Clean multilabel raw and save splits; logs duplicates/conflicts removed")
def clean_multilabel_cmd(
    in_path: str = typer.Option("data/multilabel_raw/dataset_raw.csv"),
    out_dir: str = typer.Option("data/multilabel"),
    sample_frac: float = typer.Option(0.1),
) -> None:
    root = _project_root()
    in_file = root / in_path
    df_raw = pd.read_csv(in_file)
    initial_rows = len(df_raw)

    label_cols = ["normal", "insult", "threat", "obscenity"]
    df_tmp = df_raw.copy()
    df_tmp["labels_tuple"] = df_tmp[label_cols].apply(tuple, axis=1)
    labels_nunique = df_tmp.groupby("text")["labels_tuple"].nunique()
    conflict_texts = labels_nunique[labels_nunique > 1].index.tolist()
    num_conflicting = int(df_tmp[df_tmp["text"].isin(conflict_texts)].shape[0])

    df_nonconflict = df_tmp[~df_tmp["text"].isin(conflict_texts)].copy()
    exact_duplicates_removed = int(df_nonconflict.duplicated(subset=["text"] + label_cols, keep="first").sum())

    df_clean = clean_multilabel(df_raw)
    removed_total = int(initial_rows - len(df_clean))

    other_anomalies_removed = int(max(0, removed_total - (exact_duplicates_removed + num_conflicting)))

    logger.info(
        "Multilabel cleaning: initial=%d, after=%d, removed_total=%d, exact_duplicates_removed=%d, conflicting_rows_removed=%d, other_anomalies_removed=%d",
        initial_rows,
        len(df_clean),
        removed_total,
        exact_duplicates_removed,
        num_conflicting,
        other_anomalies_removed,
    )

    sampled = sample_multilabel(df_clean, sample_frac=sample_frac)
    sample_removed = int(len(df_clean) - len(sampled))
    train_df, val_df, test_df = split_multilabel(sampled)
    save_multilabel_splits(root / out_dir, train_df, val_df, test_df)
    logger.info(
        "Saved multilabel splits to %s (train=%d val=%d test=%d) sample_removed=%d",
        root / out_dir,
        len(train_df),
        len(val_df),
        len(test_df),
        sample_removed,
    )
    typer.echo(str(root / out_dir))


@app.command("load-spans", help="Load spans dataset and save raw into folder spans_raw")
def load_spans(out_dir: str = typer.Option("data/spans_raw")) -> None:
    root = _project_root()
    logger.info("Loading spans dataset from HF")
    df = load_spans_raw()
    out_path = root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "full_raw.csv"
    df.to_csv(out_file, index=False)
    logger.info("Saved spans raw to %s (rows=%d)", out_file, len(df))
    typer.echo(str(out_path))


@app.command("clean-spans", help="Prepare spans (tokens/labels) and save train/val splits; logs basic stats")
def clean_spans_cmd(
    in_path: str = typer.Option("data/spans_raw/full_raw.csv"),
    out_dir: str = typer.Option("data/spans"),
) -> None:
    root = _project_root()
    in_file = root / in_path
    df_raw = pd.read_csv(in_file)
    logger.info("Preparing spans: input rows=%d", len(df_raw))
    df = prepare_spans(df_raw)
    try:
        num_with_toxic = int(df['labels'].apply(lambda x: any(eval(x) if isinstance(x, str) else x)).sum()) if 'labels' in df.columns else 0
    except Exception:
        # labels may already be lists
        num_with_toxic = int(df['labels'].apply(lambda x: int(any(x))).sum()) if 'labels' in df.columns else 0
    train_df, val_df = split_spans(df)
    save_spans_splits(root / out_dir, train_df, val_df)
    logger.info("Saved spans splits to %s (train=%d val=%d) toxic_examples=%d", root / out_dir, len(train_df), len(val_df), num_with_toxic)
    typer.echo(str(root / out_dir))


if __name__ == "__main__":
    app()
