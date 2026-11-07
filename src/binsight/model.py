"""Frozen SigLIP2 / CLIP backbone, trained classifier head, zero-shot scoring and routing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoProcessor

from .config import UNSURE, Config

SIGLIP_TEXT_LENGTH = 64
ZERO_SHOT_THRESHOLD = 0.6  # used for threshold: auto when there is no trained head


def pick_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pick_dtype(preference: str, device: torch.device) -> torch.dtype:
    if preference != "auto":
        return getattr(torch, preference)
    # fp16 is ~3x faster on Apple GPUs and matches fp32 embeddings to cosine 0.99999.
    return torch.float16 if device.type == "mps" else torch.float32


def _features(out) -> torch.Tensor:
    """get_*_features returns a tensor on older transformers, a model output on newer."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class Backbone:
    """Pretrained image/text encoder. Its weights are never changed."""

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto"):
        self.model_id = model_id
        self.device = pick_device(device)
        self.dtype = pick_dtype(dtype, self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device, self.dtype).eval()
        self.is_siglip = self.model.config.model_type.startswith("siglip")
        self.scale = self.model.logit_scale.exp().item()

    def pixel_values(self, images: list[Image.Image]) -> torch.Tensor:
        rgb = [im.convert("RGB") for im in images]
        return self.processor.image_processor(images=rgb, return_tensors="pt")["pixel_values"]

    @torch.inference_mode()
    def embed_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Unit-length image embeddings, float32 on CPU."""
        out = self.model.get_image_features(pixel_values=pixels.to(self.device, self.dtype))
        return F.normalize(_features(out).float(), dim=-1).cpu()

    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        return self.embed_pixels(self.pixel_values(images))

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        tokenizer = self.processor.tokenizer
        if self.is_siglip:
            # SigLIP was trained on lowercase text padded to 64 tokens, without a mask;
            # other padding gives much worse scores.
            tokens = tokenizer(
                [t.lower() for t in texts],
                padding="max_length",
                max_length=SIGLIP_TEXT_LENGTH,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {"input_ids": tokens["input_ids"]}
        else:
            inputs = dict(tokenizer(texts, padding=True, truncation=True, return_tensors="pt"))
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return F.normalize(_features(self.model.get_text_features(**inputs)).float(), dim=-1).cpu()

    def class_text_embeds(self, prompts: dict[str, list[str]], template: str) -> torch.Tensor:
        """One unit vector per class: mean of its prompt embeddings, re-normalised."""
        embeds = []
        for items in prompts.values():
            vectors = self.embed_texts([template.format(p) for p in items])
            embeds.append(F.normalize(vectors.mean(dim=0), dim=-1))
        return torch.stack(embeds)


class Head(nn.Module):
    """The trained part: linear or one-hidden-layer classifier on frozen embeddings."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
        # Softmax temperature fitted on validation data so confidences mean what they say.
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.inference_mode()
    def probs(self, x: torch.Tensor) -> torch.Tensor:
        return (self(x) / self.temperature).softmax(dim=-1)


def save_head(path: Path, head: Head, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_head(path: Path) -> tuple[Head, dict]:
    meta = json.loads(path.with_suffix(".json").read_text())
    head = Head(meta["dim"], len(meta["classes"]), meta["hidden"])
    head.load_state_dict(load_file(path))
    return head.eval(), meta


@dataclass(frozen=True)
class Policy:
    """Everything that decides between a compartment and UNSURE."""

    classes: tuple[str, ...]
    unsure: tuple[str, ...]  # classes with no compartment
    threshold: float
    # Zero-shot probability of the `unsure` classes at or above which the item goes to UNSURE.
    guard: float | None = None
    # Per class: confidence bar stricter than `threshold`, for compartments where one wrong
    # item does the most harm.
    min_confidence: dict[str, float] = field(default_factory=dict)
    # Second opinion: zero-shot must agree when the head's top class is in `check_always`,
    # gives a `check_suspect` class at least that probability, or is under `check_below` sure.
    check_always: tuple[str, ...] = ()
    check_suspect: dict[str, float] = field(default_factory=dict)
    check_below: float | None = None

    def with_threshold(self, threshold: float) -> Policy:
        return replace(self, threshold=threshold)

    @classmethod
    def from_meta(cls, meta: dict, unsure: list[str]) -> Policy:
        check = meta.get("double_check") or {}
        return cls(
            tuple(meta["classes"]),
            tuple(unsure),
            meta["threshold"],
            meta.get("guard"),
            meta.get("min_confidence") or {},
            tuple(check.get("always", ())),
            check.get("suspect") or {},
            check.get("below"),
        )

    def needs_second_opinion(self, probs: np.ndarray, top: str, confidence: float) -> bool:
        if top in self.check_always:
            return True
        if any(probs[self.classes.index(c)] >= p for c, p in self.check_suspect.items()):
            return True
        return self.check_below is not None and confidence < self.check_below


def route(probs: np.ndarray, zero_shot: np.ndarray | None, policy: Policy) -> tuple[str, str]:
    """Pick the compartment and, for UNSURE, the reason. Anything doubtful goes to UNSURE.

    The zero-shot checks are skipped when `zero_shot` is None (zero-shot is the decider).
    """
    classes = policy.classes
    i = int(np.argmax(probs))
    top, confidence = classes[i], float(probs[i])
    if top in policy.unsure:
        return UNSURE, f"top class is '{top}'"
    if confidence < policy.threshold:
        return UNSURE, f"confidence {confidence:.2f} < {policy.threshold:.2f}"
    bar = policy.min_confidence.get(top)
    if bar is not None and confidence < bar:
        return UNSURE, f"confidence {confidence:.2f} < {bar:.2f} needed for {top}"
    if zero_shot is None:
        return top.upper(), ""
    if policy.guard is not None:
        out = float(sum(zero_shot[classes.index(c)] for c in policy.unsure))
        if out >= policy.guard:
            return UNSURE, f"zero-shot guard: {'/'.join(policy.unsure)} {out:.2f}"
    if policy.needs_second_opinion(probs, top, confidence):
        second = classes[int(np.argmax(zero_shot))]
        if second != top:
            return UNSURE, f"zero-shot disagrees: '{second}'"
    return top.upper(), ""


@dataclass
class Result:
    scores: dict[str, float]  # class -> probability, sums to 1
    top: str
    confidence: float
    compartment: str
    reason: str  # why it went to UNSURE; empty otherwise
    latency_ms: float
    views: int = 1  # camera frames averaged into this decision

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)


class Classifier:
    """Backbone + trained head when one matches, otherwise zero-shot prompts."""

    def __init__(self, cfg: Config, model_id: str | None = None, use_head: bool = True):
        self.cfg = cfg
        self.classes = cfg.classes
        self.backbone = Backbone(model_id or cfg.model, cfg.device, cfg.dtype)
        self.model_id = self.backbone.model_id
        self.text_embeds = self.backbone.class_text_embeds(cfg.prompts, cfg.prompt_template)
        self.head, self.meta, self.note = None, {}, ""
        if use_head and cfg.head.exists():
            head, meta = load_head(cfg.head)
            if meta["backbone"] != self.model_id:
                self.note = f"head was trained on {meta['backbone']}; using zero-shot"
            elif meta["classes"] != self.classes:
                self.note = "head classes differ from config prompts; using zero-shot"
            else:
                self.head, self.meta = head, meta
        elif use_head:
            self.note = f"no trained head at {cfg.head}; using zero-shot"
        self.mode = "trained" if self.head else "zero-shot"
        if self.head:
            self.policy = Policy.from_meta(self.meta, cfg.unsure_classes)
        else:
            self.policy = Policy(
                tuple(self.classes), tuple(cfg.unsure_classes), ZERO_SHOT_THRESHOLD
            )
        if cfg.threshold != "auto":
            self.policy = self.policy.with_threshold(float(cfg.threshold))
        self.threshold = self.policy.threshold
        self._warmup()

    def zero_shot_probs(self, embeds: torch.Tensor) -> torch.Tensor:
        return (self.backbone.scale * embeds @ self.text_embeds.T).softmax(dim=-1)

    def probs(self, embeds: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(probabilities used for the decision, zero-shot probabilities) per embedding."""
        zero_shot = self.zero_shot_probs(embeds)
        main = self.head.probs(embeds) if self.head else zero_shot
        return main.numpy(), zero_shot.numpy()

    def _warmup(self) -> None:
        # The GPU compiles each batch size once; do it now for single images and the live vote.
        blank = Image.new("RGB", (256, 256))
        for n in sorted({1, max(self.cfg.vote_frames, 1)}):
            self.backbone.embed_images([blank] * n)

    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        """Unit-length embeddings, one per image (frames of the same item are averaged later)."""
        return self.backbone.embed_images(images)

    def classify(self, image: Image.Image, threshold: float | None = None) -> Result:
        start = time.perf_counter()
        embeds = self.embed([image])
        return self.decide(embeds, threshold, (time.perf_counter() - start) * 1000)

    def decide(
        self, embeds: torch.Tensor, threshold: float | None = None, latency_ms: float = 0.0
    ) -> Result:
        """Route one item from one or more embeddings of it (e.g. several camera frames)."""
        policy = self.policy if threshold is None else self.policy.with_threshold(threshold)
        mean = F.normalize(embeds.mean(dim=0, keepdim=True), dim=-1)
        probs, zero_shot = self.probs(mean)
        # In zero-shot mode the zero-shot scores are the decision, not a second opinion.
        second = zero_shot[0] if self.head else None
        compartment, reason = route(probs[0], second, policy)
        scores = dict(zip(self.classes, probs[0].tolist(), strict=True))
        top = max(scores, key=scores.__getitem__)
        return Result(scores, top, scores[top], compartment, reason, latency_ms, len(embeds))
