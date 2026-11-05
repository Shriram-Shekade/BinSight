"""Leave-one-dataset-out test: train without a source, test on all of that source.

The main test split shares photographers, backgrounds and cameras with training. Holding a
whole dataset out asks the harder question: how well does the recipe work on photos from
somewhere it has never seen? That is closer to a new webcam on a new campus.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import SPLITS, load_split, load_train_views
from .metrics import decisions, operating_point, pick_threshold, sweep, true_compartments
from .model import ZERO_SHOT_THRESHOLD, Backbone, load_head
from .train import fit, fit_temperature


def crosscheck(cfg: Config, out: Path, log=print) -> dict:
    split = load_split(cfg)
    _, meta = load_head(cfg.head)  # reuse the chosen hyperparameters; no re-tuning per fold
    x_aug, y_aug, src_aug = load_train_views(cfg, split)
    backbone = Backbone(meta["backbone"], cfg.device, cfg.dtype)
    text = backbone.class_text_embeds(cfg.prompts, cfg.prompt_template)
    best = max(meta["grid"], key=lambda r: r["val_macro_f1"])
    classes, rc, n = cfg.classes, cfg.reject_class, len(cfg.classes)
    as_t = lambda a: torch.from_numpy(np.asarray(a, np.float32))  # noqa: E731

    def zero_shot(x):
        return (backbone.scale * x @ text.T).softmax(-1).numpy()

    results = {}
    for k, source in enumerate(cfg.data.sources):
        held = split["src"] == k
        if not held.any():
            continue
        train = ~held & (split["split"] == SPLITS.index("train"))
        val = ~held & (split["split"] == SPLITS.index("val"))
        x = as_t(np.concatenate([split["emb"][train], x_aug[src_aug != k]]))
        y = torch.from_numpy(np.concatenate([split["y"][train], y_aug[src_aug != k]])).long()
        xv, yv = as_t(split["emb"][val]), torch.from_numpy(split["y"][val]).long()
        head, _, _ = fit(x, y, xv, yv, n, best["hidden"], best["lr"], best["weight_decay"],
                         cfg, cfg.data.seed)  # fmt: skip
        with torch.no_grad():
            head.temperature.fill_(fit_temperature(head(xv), yv))
        v_probs, v_zs = head.probs(xv).numpy(), zero_shot(xv)
        threshold = pick_threshold(
            sweep(v_probs, v_zs, yv.numpy(), classes, rc, cfg.guard, agree=meta["agree"]),
            meta["max_wrong_sort"],
        )

        xt, yt = as_t(split["emb"][held]), split["y"][held]
        t_probs, t_zs = head.probs(xt).numpy(), zero_shot(xt)
        truth = true_compartments(yt, classes, rc)
        trained = operating_point(
            decisions(t_probs, t_zs, classes, threshold, rc, cfg.guard, meta["agree"]), truth
        )
        zs = operating_point(decisions(t_zs, None, classes, ZERO_SHOT_THRESHOLD, rc, None), truth)
        trained_classes = set(np.unique(y.numpy()).tolist())
        unseen = [c for i, c in enumerate(classes) if (yt == i).any() and i not in trained_classes]
        results[source] = {
            "test_images": int(held.sum()),
            "classes_missing_from_training": unseen,
            "classes_present": {c: int((yt == i).sum()) for i, c in enumerate(classes)},
            "threshold": threshold,
            "trained": {"accuracy": float((t_probs.argmax(1) == yt).mean()), **trained},
            "zero_shot": {"accuracy": float((t_zs.argmax(1) == yt).mean()), **zs},
        }
        r = results[source]
        log(
            f"  held out {source}: {r['test_images']} images | trained acc "
            f"{r['trained']['accuracy']:.1%}, wrong-sort {r['trained']['wrong_sort_rate']:.2%} "
            f"| v1 zero-shot acc {r['zero_shot']['accuracy']:.1%}, wrong-sort "
            f"{r['zero_shot']['wrong_sort_rate']:.2%}"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "crosscheck.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
