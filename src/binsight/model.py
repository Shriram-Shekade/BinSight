"""Zero-shot classifier: SigLIP2 or CLIP with cached class embeddings."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .config import UNSURE, Config

SIGLIP_TEXT_LENGTH = 64


@dataclass
class Result:
    scores: dict[str, float]  # class -> softmax probability, sums to 1
    top: str
    confidence: float
    compartment: str
    reason: str  # why it went to UNSURE; empty otherwise
    latency_ms: float

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)


def pick_device(preference: str) -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _features(out) -> torch.Tensor:
    """get_*_features returns a tensor on older transformers, a model output on newer."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def route(top: str, confidence: float, threshold: float, reject_class: str) -> tuple[str, str]:
    """Pick the compartment. Anything doubtful goes to UNSURE."""
    if top == reject_class:
        return UNSURE, f"top class is '{top}'"
    if confidence < threshold:
        return UNSURE, f"confidence {confidence:.2f} < {threshold:.2f}"
    return top.upper(), ""


class Classifier:
    def __init__(self, cfg: Config, model_id: str | None = None):
        self.cfg = cfg
        self.model_id = model_id or cfg.model
        self.device = pick_device(cfg.device)
        self.classes = cfg.classes
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self.is_siglip = self.model.config.model_type.startswith("siglip")
        self.scale = self.model.logit_scale.exp().item()
        self.class_embeds = self._encode_classes()
        self._warmup()

    @torch.inference_mode()
    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
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
        return F.normalize(_features(self.model.get_text_features(**inputs)).float(), dim=-1)

    def _encode_classes(self) -> torch.Tensor:
        """One unit vector per class: mean of its prompt embeddings, re-normalised."""
        template = self.cfg.prompt_template
        embeds = []
        for name in self.classes:
            prompts = self.cfg.prompts[name]
            vectors = self._encode_texts([template.format(p) for p in prompts])
            embeds.append(F.normalize(vectors.mean(dim=0), dim=-1))
        return torch.stack(embeds)

    @torch.inference_mode()
    def _score(self, image: Image.Image) -> torch.Tensor:
        pixels = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
        feats = _features(self.model.get_image_features(pixel_values=pixels.to(self.device)))
        feats = F.normalize(feats.float(), dim=-1)
        return (self.scale * feats @ self.class_embeds.T).softmax(dim=-1)[0].cpu()

    def _warmup(self) -> None:
        self._score(Image.new("RGB", (256, 256)))

    def classify(self, image: Image.Image, threshold: float | None = None) -> Result:
        threshold = self.cfg.threshold if threshold is None else threshold
        start = time.perf_counter()
        probs = self._score(image.convert("RGB"))
        latency = (time.perf_counter() - start) * 1000
        scores = dict(zip(self.classes, probs.tolist(), strict=True))
        top = max(scores, key=scores.__getitem__)
        compartment, reason = route(top, scores[top], threshold, self.cfg.reject_class)
        return Result(scores, top, scores[top], compartment, reason, latency)
