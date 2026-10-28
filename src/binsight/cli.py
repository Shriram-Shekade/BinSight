"""`binsight` command line."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from .config import Config, load_config

app = typer.Typer(add_completion=False, help="BinSight: smart waste-sorting bin demo.")

ConfigOpt = typer.Option(None, "--config", "-c", help="Path to config.yaml.")
ModelOpt = typer.Option(None, "--model", "-m", help="Backbone id, or 'fallback' for CLIP.")
ThresholdOpt = typer.Option(None, "--threshold", "-t", help="UNSURE confidence threshold.")
ZeroShotOpt = typer.Option(False, "--zero-shot", help="Ignore the trained head (v1 behaviour).")


def _load(config: Path | None, model: str | None) -> tuple[Config, str]:
    cfg = load_config(config)
    model_id = cfg.fallback_model if model == "fallback" else (model or cfg.model)
    return cfg, model_id


def _classifier(cfg: Config, model_id: str, zero_shot: bool):
    from .model import Classifier  # heavy import, kept off the --help path

    typer.echo(f"Loading {model_id} ...")
    start = time.perf_counter()
    clf = Classifier(cfg, model_id, use_head=not zero_shot)
    if clf.note:
        typer.echo(f"note: {clf.note}")
    typer.echo(
        f"Ready on {clf.backbone.device} ({clf.mode}, threshold {clf.threshold:.2f}) "
        f"in {time.perf_counter() - start:.1f}s"
    )
    return clf


@app.command()
def prepare(
    config: Path | None = ConfigOpt,
    model: str | None = ModelOpt,
) -> None:
    """Download the public datasets, embed them, de-duplicate and split. Cached; re-runnable."""
    from .data import prepare as run
    from .model import Backbone

    cfg, model_id = _load(config, model)
    stats = run(Backbone(model_id, cfg.device, cfg.dtype), cfg)
    typer.echo(
        f"{stats['images_mapped']} usable images -> {stats['images_kept']} after removing "
        f"{stats['duplicate_links']} duplicates and {stats['conflicting_images']} images with "
        "conflicting labels"
    )
    for name, counts in stats["per_split_class"].items():
        typer.echo(f"  {name:<5} " + "  ".join(f"{c} {n}" for c, n in counts.items()))


@app.command()
def train(config: Path | None = ConfigOpt) -> None:
    """Train the classifier head on the prepared embeddings and save it to config `head`."""
    from .train import train as run

    run(load_config(config), log=typer.echo)


@app.command()
def evaluate(
    config: Path | None = ConfigOpt,
    out: Path = typer.Option(Path("reports"), "--out", help="Report folder."),
) -> None:
    """Score zero-shot and trained models on the held-out test split; write the report."""
    from .evaluate import evaluate as run

    path = run(load_config(config), out, log=typer.echo)
    typer.echo(f"wrote {path}")


@app.command()
def live(
    model: str | None = ModelOpt,
    config: Path | None = ConfigOpt,
    threshold: float | None = ThresholdOpt,
    camera: int | None = typer.Option(None, "--camera", help="OpenCV camera index."),
    zero_shot: bool = ZeroShotOpt,
) -> None:
    """Open the webcam and sort whatever is placed on the tray."""
    from .live import run_live

    cfg, model_id = _load(config, model)
    if camera is not None:
        cfg.camera = camera
    clf = _classifier(cfg, model_id, zero_shot)
    try:
        run_live(cfg, clf, threshold)
    except RuntimeError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(1) from err
    except KeyboardInterrupt:
        pass


@app.command()
def classify(
    images: list[Path] = typer.Argument(..., exists=True, dir_okay=False, help="Image files."),
    model: str | None = ModelOpt,
    config: Path | None = ConfigOpt,
    threshold: float | None = ThresholdOpt,
    zero_shot: bool = ZeroShotOpt,
) -> None:
    """Print the class scores and routing decision for image files."""
    from PIL import Image

    cfg, model_id = _load(config, model)
    clf = _classifier(cfg, model_id, zero_shot)
    for image in images:
        result = clf.classify(Image.open(image), threshold)
        typer.echo(f"\n{image.name}")
        for name, p in result.ranked():
            typer.echo(f"  {name:<8} {p:6.3f}  {'#' * round(p * 40)}")
        why = f"  ({result.reason})" if result.reason else ""
        typer.echo(f"-> {result.compartment}{why}   [{result.latency_ms:.0f} ms]")
