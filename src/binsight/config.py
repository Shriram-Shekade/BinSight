"""Load config.yaml into typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
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
    turn_frames: int


@dataclass
class TrayCheckConfig:
    min_brightness: float
    max_edges: float
    max_unevenness: float


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
class DoubleCheck:
    always: list[str]
    suspect: dict[str, float]
    below: float | None


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
    unsure_classes: list[str]
    guard: float | None
    double_check: DoubleCheck | None
    min_confidence_per_class: dict[str, float]
    prompts: dict[str, list[str]]
    data: DataConfig
    train: TrainConfig
    camera: int
    roi_fraction: float
    display_width: int
    trigger: TriggerConfig
    vote_frames: int
    retry_seconds: float
    tray_check: TrayCheckConfig
    colours: dict[str, str]

    @property
    def classes(self) -> list[str]:
        return list(self.prompts)

    @property
    def compartments(self) -> list[str]:
        sortable = [c.upper() for c in self.prompts if c not in self.unsure_classes]
        return [*sortable, UNSURE]


def load_config(path: Path | None = None) -> Config:
    """Read `path`, else ./config.yaml, else the project's, else the copy bundled in the wheel.

    Relative paths inside it (head, data dir) resolve against the config file's folder.
    """
    if path is None:
        found = [Path("config.yaml"), PROJECT_ROOT / "config.yaml", PACKAGE_DIR / "config.yaml"]
        path = next((p for p in found if p.exists()), found[1])
    base = path.resolve().parent
    raw = yaml.safe_load(path.read_text())
    data = raw.pop("data")
    data["dir"] = base / data["dir"]
    check = raw.pop("double_check")
    return Config(
        double_check=DoubleCheck(**check) if check else None,
        trigger=TriggerConfig(**raw.pop("trigger")),
        tray_check=TrayCheckConfig(**raw.pop("tray_check")),
        data=DataConfig(**data),
        train=TrainConfig(**raw.pop("train")),
        head=base / raw.pop("head"),
        **raw,
    )


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> OpenCV (b, g, r)."""
    v = value.lstrip("#")
    return int(v[4:6], 16), int(v[2:4], 16), int(v[0:2], 16)
