"""Load config.yaml into typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNSURE = "UNSURE"


@dataclass
class TriggerConfig:
    diff_level: int
    motion_level: int
    occupied_fraction: float
    clear_fraction: float
    still_fraction: float
    still_frames: int
    clear_frames: int
    warmup_frames: int
    background_frames: int
    adapt_rate: float
    rearm_still_frames: int
    rearm_skip_frames: int


@dataclass
class DataConfig:
    dir: Path
    seed: int
    val_fraction: float
    test_fraction: float
    dedupe_cosine: float
    augment_views: int
    batch_size: int
    workers: int
    sources: dict[str, dict[str, str | None]]


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    label_smoothing: float
    dropout: float
    hidden: list[int]
    lr: list[float]
    weight_decay: list[float]
    max_wrong_sort: float


@dataclass
class Config:
    model: str
    fallback_model: str
    device: str
    dtype: str
    head: Path
    prompt_template: str
    threshold: float | str
    threshold_step: float
    reject_class: str
    guard: float | None
    prompts: dict[str, list[str]]
    data: DataConfig
    train: TrainConfig
    camera: int
    roi_fraction: float
    display_width: int
    trigger: TriggerConfig
    colours: dict[str, str]

    @property
    def classes(self) -> list[str]:
        return list(self.prompts)

    @property
    def compartments(self) -> list[str]:
        sortable = [c.upper() for c in self.prompts if c != self.reject_class]
        return [*sortable, UNSURE]


def load_config(path: Path | None = None) -> Config:
    """Read `path`, else ./config.yaml, else the project's config.yaml.

    Relative paths inside it (head, data dir) resolve against the config file's folder.
    """
    if path is None:
        path = Path("config.yaml")
        if not path.exists():
            path = PROJECT_ROOT / "config.yaml"
    base = path.resolve().parent
    raw = yaml.safe_load(path.read_text())
    data = raw.pop("data")
    data["dir"] = base / data["dir"]
    return Config(
        trigger=TriggerConfig(**raw.pop("trigger")),
        data=DataConfig(**data),
        train=TrainConfig(**raw.pop("train")),
        head=base / raw.pop("head"),
        **raw,
    )


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> OpenCV (b, g, r)."""
    v = value.lstrip("#")
    return int(v[4:6], 16), int(v[2:4], 16), int(v[0:2], 16)
