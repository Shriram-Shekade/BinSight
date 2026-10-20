"""Load config.yaml into typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
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


@dataclass
class Config:
    model: str
    fallback_model: str
    device: str
    prompt_template: str
    threshold: float
    threshold_step: float
    reject_class: str
    prompts: dict[str, list[str]]
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
    """Read `path`, else ./config.yaml, else the config.yaml next to the package."""
    if path is None:
        path = Path("config.yaml")
        if not path.exists():
            path = PROJECT_CONFIG
    raw = yaml.safe_load(path.read_text())
    trigger = TriggerConfig(**raw.pop("trigger"))
    return Config(trigger=trigger, **raw)


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> OpenCV (b, g, r)."""
    v = value.lstrip("#")
    return int(v[4:6], 16), int(v[2:4], 16), int(v[0:2], 16)
