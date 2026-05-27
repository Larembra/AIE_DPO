from __future__ import annotations

from pathlib import Path

import torch
import typer

from models.training import train_multilabel_gru, train_spans_gru, train_detox_transformer

app = typer.Typer()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.command("train-multilabel")
def train_multilabel(
    data_dir: str = typer.Option("data/multilabel"),
    artifacts_dir: str = typer.Option("artifacts/multilabel_model"),
    configs_dir: str = typer.Option("configs/multilabel_model"),
    epochs: int = typer.Option(10),
) -> None:
    root = _project_root()
    model_path = train_multilabel_gru(
        data_dir=root / data_dir,
        artifacts_dir=root / artifacts_dir,
        configs_dir=root / configs_dir,
        epochs=epochs,
        device=_device(),
    )
    typer.echo(str(model_path))


@app.command("train-spans")
def train_spans(
    data_dir: str = typer.Option("data/spans"),
    artifacts_dir: str = typer.Option("artifacts/spans_model"),
    configs_dir: str = typer.Option("configs/spans_model"),
    epochs: int = typer.Option(10),
) -> None:
    root = _project_root()
    model_path = train_spans_gru(
        data_dir=root / data_dir,
        artifacts_dir=root / artifacts_dir,
        configs_dir=root / configs_dir,
        epochs=epochs,
        device=_device(),
    )
    typer.echo(str(model_path))


@app.command("train-detox")
def train_detox(
    data_dir: str = typer.Option("data/detox"),
    artifacts_dir: str = typer.Option("artifacts/detox_model"),
    configs_dir: str = typer.Option("configs/detox_model"),
    epochs: int = typer.Option(10),
) -> None:
    root = _project_root()
    model_path = train_detox_transformer(
        data_dir=root / data_dir,
        artifacts_dir=root / artifacts_dir,
        configs_dir=root / configs_dir,
        epochs=epochs,
        device=_device(),
    )
    typer.echo(str(model_path))


if __name__ == "__main__":
    app()

