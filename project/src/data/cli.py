from __future__ import annotations

from pathlib import Path

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

app = typer.Typer()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@app.command("load-detox")
def load_detox(out_dir: str = typer.Option("data/detox_raw")) -> None:
    root = _project_root()
    df_train, df_val = load_detox_raw()
    out_path = root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_path / "train_raw.csv", index=False)
    df_val.to_csv(out_path / "val_raw.csv", index=False)
    typer.echo(str(out_path))


@app.command("clean-detox")
def clean_detox_cmd(
    in_dir: str = typer.Option("data/detox_raw"),
    out_dir: str = typer.Option("data/detox"),
    sample_frac: float = typer.Option(0.2),
    test_size: float = typer.Option(0.1),
) -> None:
    root = _project_root()
    in_path = root / in_dir
    df_train = (in_path / "train_raw.csv")
    df_val = (in_path / "val_raw.csv")
    train_df, val_df, test_df = clean_detox(
        df_train=pd.read_csv(df_train),
        df_val=pd.read_csv(df_val),
        sample_frac=sample_frac,
        test_size=test_size,
    )
    save_detox_splits(root / out_dir, train_df, val_df, test_df)
    typer.echo(str(root / out_dir))


@app.command("load-multilabel")
def load_multilabel(
    dataset_path: str = typer.Option("data/dataset.txt"),
    out_path: str = typer.Option("data/multilabel_raw.csv"),
) -> None:
    root = _project_root()
    df = load_multilabel_raw(root / dataset_path)
    out_file = root / out_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    typer.echo(str(out_file))


@app.command("clean-multilabel")
def clean_multilabel_cmd(
    in_path: str = typer.Option("data/multilabel_raw.csv"),
    out_dir: str = typer.Option("data/multilabel"),
    sample_frac: float = typer.Option(0.1),
) -> None:
    root = _project_root()
    df = pd.read_csv(root / in_path)
    df = clean_multilabel(df)
    df = sample_multilabel(df, sample_frac=sample_frac)
    train_df, val_df, test_df = split_multilabel(df)
    save_multilabel_splits(root / out_dir, train_df, val_df, test_df)
    typer.echo(str(root / out_dir))


@app.command("load-spans")
def load_spans(out_path: str = typer.Option("data/spans_raw.csv")) -> None:
    root = _project_root()
    df = load_spans_raw()
    out_file = root / out_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    typer.echo(str(out_file))


@app.command("clean-spans")
def clean_spans_cmd(
    in_path: str = typer.Option("data/spans_raw.csv"),
    out_dir: str = typer.Option("data/spans"),
) -> None:
    root = _project_root()
    df = pd.read_csv(root / in_path)
    df = prepare_spans(df)
    train_df, val_df = split_spans(df)
    save_spans_splits(root / out_dir, train_df, val_df)
    typer.echo(str(root / out_dir))


if __name__ == "__main__":
    app()
