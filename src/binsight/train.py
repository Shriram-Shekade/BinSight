"""Train the classifier head on frozen embeddings, calibrate it, and pick the threshold."""

from __future__ import annotations

import copy
import itertools
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import SPLITS, load_split, load_train_views
from .metrics import macro_f1, pick_threshold, sweep
from .model import Backbone, Head, save_head


def _arrays(cfg: Config):
    split = load_split(cfg)
    parts = {name: split["split"] == i for i, name in enumerate(SPLITS)}
    x_aug, y_aug = load_train_views(cfg, split)
    x_train = np.concatenate([split["emb"][parts["train"]], x_aug])
    y_train = np.concatenate([split["y"][parts["train"]], y_aug])
    as_t = lambda a: torch.from_numpy(np.asarray(a, np.float32))  # noqa: E731
    return {
        "train": (as_t(x_train), torch.from_numpy(y_train).long()),
        "val": (
            as_t(split["emb"][parts["val"]]),
            torch.from_numpy(split["y"][parts["val"]]).long(),
        ),
        "backbone": str(split["backbone"]),
    }


def fit(x, y, xv, yv, n_classes, hidden, lr, wd, cfg: Config, seed: int):
    """One training run; returns the head at its best validation macro-F1 epoch."""
    torch.manual_seed(seed)
    tc = cfg.train
    head = Head(x.shape[1], n_classes, hidden, tc.dropout)
    counts = torch.bincount(y, minlength=n_classes).float()
    weight = counts.sum() / (n_classes * counts.clamp(min=1))  # classes count equally
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    steps = tc.epochs * math.ceil(len(y) / tc.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)
    best = (-1.0, 0, None)
    for epoch in range(tc.epochs):
        head.train()
        for idx in torch.randperm(len(y), generator=gen).split(tc.batch_size):
            loss = F.cross_entropy(
                head(x[idx]), y[idx], weight=weight, label_smoothing=tc.label_smoothing
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            score = macro_f1(yv.numpy(), head(xv).argmax(1).numpy(), n_classes)
        if score > best[0]:
            best = (score, epoch + 1, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[2])
    return head.eval(), best[0], best[1]


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Temperature minimising validation NLL (grid search: robust, no tuning)."""
    grid = torch.logspace(-1.5, 1.5, 301)
    nll = torch.stack([F.cross_entropy(logits / t, y) for t in grid])
    return float(grid[nll.argmin()])


def train(cfg: Config, log=print) -> dict:
    arrays = _arrays(cfg)
    x, y = arrays["train"]
    xv, yv = arrays["val"]
    n = len(cfg.classes)
    log(f"train {len(y)} embeddings (incl. augmented views), val {len(yv)}")

    runs = []
    tc = cfg.train
    for hidden, lr, wd in itertools.product(tc.hidden, tc.lr, tc.weight_decay):
        head, score, epoch = fit(x, y, xv, yv, n, hidden, lr, wd, cfg, cfg.data.seed)
        runs.append({"hidden": hidden, "lr": lr, "weight_decay": wd, "val_macro_f1": score,
                     "best_epoch": epoch, "head": head})  # fmt: skip
        log(f"  hidden={hidden:<4} lr={lr:<7} wd={wd:<7} val macro-F1 {score:.4f} @ epoch {epoch}")
    best = max(runs, key=lambda r: r["val_macro_f1"])
    head = best["head"]

    with torch.no_grad():
        logits = head(xv)
    before = F.cross_entropy(logits, yv).item()
    t = fit_temperature(logits, yv)
    head.temperature.fill_(t)
    after = F.cross_entropy(logits / t, yv).item()
    log(f"temperature {t:.3f}: val NLL {before:.4f} -> {after:.4f}")

    # Threshold for the deployed system: head + zero-shot guard, routed exactly as live.
    backbone = Backbone(arrays["backbone"], cfg.device, cfg.dtype)
    text = backbone.class_text_embeds(cfg.prompts, cfg.prompt_template)
    zero_shot = (backbone.scale * xv @ text.T).softmax(-1).numpy()
    probs = head.probs(xv).numpy()
    points = sweep(probs, zero_shot, yv.numpy(), cfg.classes, cfg.reject_class, cfg.guard)
    threshold = pick_threshold(points, tc.max_wrong_sort)
    chosen = next(p for p in points if p["threshold"] == threshold)
    log(
        f"threshold {threshold:.2f}: val wrong-sort {chosen['wrong_sort_rate']:.2%}, "
        f"coverage {chosen['coverage']:.1%} (target wrong-sort <= {tc.max_wrong_sort:.1%})"
    )

    meta = {
        "backbone": arrays["backbone"],
        "classes": cfg.classes,
        "dim": int(x.shape[1]),
        "hidden": best["hidden"],
        "temperature": t,
        "threshold": threshold,
        "guard": cfg.guard,
        "max_wrong_sort": tc.max_wrong_sort,
        "val_operating_point": chosen,
        "val_macro_f1": best["val_macro_f1"],
        "grid": [{k: v for k, v in r.items() if k != "head"} for r in runs],
        "train_embeddings": int(len(y)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_head(cfg.head, head, meta)
    log(f"saved {cfg.head}")
    return meta
