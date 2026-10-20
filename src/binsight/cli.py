"""`binsight` command line."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from .config import Config, load_config

app = typer.Typer(add_completion=False, help="BinSight: smart waste-sorting bin demo.")

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config.yaml.")
ModelOpt = typer.Option(None, "--model", "-m", help="Model id, or 'fallback' for CLIP.")
ThresholdOpt = typer.Option(None, "--threshold", "-t", help="UNSURE confidence threshold.")


def _load(config: Path | None, model: str | None) -> tuple[Config, str]:
    cfg = load_config(config)
    model_id = cfg.fallback_model if model == "fallback" else (model or cfg.model)
    return cfg, model_id


def _classifier(cfg: Config, model_id: str):
    from .model import Classifier  # heavy import, kept off the --help path

    typer.echo(f"Loading {model_id} ...")
    start = time.perf_counter()
    clf = Classifier(cfg, model_id)
    typer.echo(f"Ready on {clf.device} in {time.perf_counter() - start:.1f}s")
    return clf


@app.command()
def live(
    model: str | None = ModelOpt,
    config: Path | None = ConfigOpt,
    threshold: float | None = ThresholdOpt,
    camera: int | None = typer.Option(None, "--camera", help="OpenCV camera index."),
) -> None:
    """Open the webcam and sort whatever is placed on the tray."""
    from .live import run_live

    cfg, model_id = _load(config, model)
    if camera is not None:
        cfg.camera = camera
    clf = _classifier(cfg, model_id)
    try:
        run_live(cfg, clf, threshold)
    except RuntimeError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(1) from err
    except KeyboardInterrupt:
        pass


@app.command()
def classify(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, help="Image file."),
    model: str | None = ModelOpt,
    config: Path | None = ConfigOpt,
    threshold: float | None = ThresholdOpt,
) -> None:
    """Print the class scores and routing decision for one image file."""
    from PIL import Image

    cfg, model_id = _load(config, model)
    clf = _classifier(cfg, model_id)
    result = clf.classify(Image.open(image), threshold)

    typer.echo(f"\n{image.name}")
    for name, p in result.ranked():
        typer.echo(f"  {name:<8} {p:6.3f}  {'#' * round(p * 40)}")
    why = f"  ({result.reason})" if result.reason else ""
    typer.echo(f"-> {result.compartment}{why}   [{result.latency_ms:.0f} ms]")
