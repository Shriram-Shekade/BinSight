"""Classification and bin-routing metrics shared by training and evaluation."""

from __future__ import annotations

import numpy as np

from .config import UNSURE
from .model import route


def confusion(y: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    """Rows are true classes, columns predicted."""
    return np.bincount(y * n + pred, minlength=n * n).reshape(n, n)


def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "support": cm.sum(axis=1)}


def macro_f1(y: np.ndarray, pred: np.ndarray, n: int) -> float:
    return float(per_class(confusion(y, pred, n))["f1"].mean())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-class confidence."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def decisions(
    probs: np.ndarray,
    zero_shot: np.ndarray | None,
    classes: list[str],
    threshold: float,
    reject_class: str,
    guard: float | None,
) -> np.ndarray:
    """Compartment chosen for every row, through the same `route` the live demo uses."""
    zs = zero_shot if zero_shot is not None else [None] * len(probs)
    return np.array(
        [
            route(p, z, classes, threshold, reject_class, guard)[0]
            for p, z in zip(probs, zs, strict=True)
        ]
    )


def true_compartments(y: np.ndarray, classes: list[str], reject_class: str) -> np.ndarray:
    names = np.array([UNSURE if c == reject_class else c.upper() for c in classes])
    return names[y]


def operating_point(chosen: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """What a bin would do. Out-of-scope items auto-sorted anywhere count as wrong sorts."""
    sorted_ = chosen != UNSURE
    wrong = sorted_ & (chosen != truth)
    sortable = truth != UNSURE
    return {
        "coverage": float(sorted_.mean()),  # share of all items auto-sorted
        "wrong_sort_rate": float(wrong.mean()),  # share of all items put in a wrong compartment
        "unsure_rate": float((~sorted_).mean()),
        "sortable_recall": float((sorted_ & ~wrong)[sortable].mean()) if sortable.any() else 0.0,
        "precision_of_sorts": float(1 - wrong.sum() / max(sorted_.sum(), 1)),
    }


def sweep(probs, zero_shot, y, classes, reject_class, guard, grid=None) -> list[dict]:
    """Operating point at each threshold."""
    grid = np.round(np.arange(0.30, 0.995, 0.01), 2) if grid is None else grid
    truth = true_compartments(y, classes, reject_class)
    out = []
    for t in grid:
        chosen = decisions(probs, zero_shot, classes, float(t), reject_class, guard)
        out.append({"threshold": float(t), **operating_point(chosen, truth)})
    return out


def pick_threshold(points: list[dict], max_wrong_sort: float) -> float:
    """Lowest threshold (most auto-sorting) whose wrong-sort rate meets the target."""
    ok = [p for p in points if p["wrong_sort_rate"] <= max_wrong_sort]
    if ok:
        return min(ok, key=lambda p: p["threshold"])["threshold"]
    return min(points, key=lambda p: (p["wrong_sort_rate"], -p["threshold"]))["threshold"]


def bootstrap_ci(values: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean of per-item 0/1 values."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
