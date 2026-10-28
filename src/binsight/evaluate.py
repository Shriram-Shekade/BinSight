"""Held-out evaluation: zero-shot (v1) vs trained head, written up as reports/REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import UNSURE, Config
from .data import SPLITS, load_split
from .metrics import (
    bootstrap_ci,
    confusion,
    decisions,
    ece,
    operating_point,
    per_class,
    pick_threshold,
    sweep,
    true_compartments,
)
from .model import ZERO_SHOT_THRESHOLD, Backbone, load_head

# Reference data-viz palette (light surface).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _system(name, probs, zero_shot, y, classes, cfg, threshold, guard, logits_raw=None):
    truth = true_compartments(y, classes, cfg.reject_class)
    chosen = decisions(probs, zero_shot, classes, threshold, cfg.reject_class, guard)
    pred = probs.argmax(1)
    cm = confusion(y, pred, len(classes))
    wrong = ((chosen != UNSURE) & (chosen != truth)).astype(float)
    correct = (pred == y).astype(float)
    contamination = {}
    for comp in sorted(set(truth) - {UNSURE}):
        into = chosen == comp
        contamination[comp] = float((truth[into] != comp).mean()) if into.any() else 0.0
    out = {
        "name": name,
        "threshold": threshold,
        "guard": guard,
        "accuracy": float(correct.mean()),
        "accuracy_ci": bootstrap_ci(correct),
        "macro_f1": float(per_class(cm)["f1"].mean()),
        "ece": ece(probs, y),
        **operating_point(chosen, truth),
        "wrong_sort_ci": bootstrap_ci(wrong),
        "contamination": contamination,
        "confusion": cm.tolist(),
        "per_class": {k: v.tolist() for k, v in per_class(cm).items()},
    }
    if logits_raw is not None:
        out["ece_uncalibrated"] = ece(torch.from_numpy(logits_raw).softmax(-1).numpy(), y)
    return out


def _plot_confusions(systems: list[dict], classes: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(1, len(systems), figsize=(5.2 * len(systems), 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(np.atleast_1d(axes), systems, strict=True):
        cm = np.array(s["confusion"], float)
        rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = rates[i, j]
                ax.text(j, i, f"{v:.0%}" if v >= 0.005 else "·", ha="center", va="center",
                        fontsize=9, color="white" if v > 0.55 else INK)  # fmt: skip
        ax.set_xticks(range(len(classes)), classes, color=INK_2, fontsize=9)
        ax.set_yticks(range(len(classes)), classes, color=INK_2, fontsize=9)
        ax.set_xlabel("predicted", color=INK_2)
        ax.set_ylabel("true", color=INK_2)
        ax.set_title(f"{s['name']}\naccuracy {s['accuracy']:.1%}", color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _plot_tradeoff(
    curves: list[tuple[str, list[dict], dict]], reference: dict, target: float, path: Path
) -> None:
    """Sortable items sorted correctly vs wrong sorts, one curve per system over thresholds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for colour, (name, points, chosen) in zip(SERIES, curves, strict=False):
        xs = [p["wrong_sort_rate"] * 100 for p in points]
        ys = [p["sortable_recall"] * 100 for p in points]
        # Curves sit close together near the target, so the operating point goes in the
        # legend rather than as on-plot labels that would collide.
        label = (
            f"{name}: {chosen['sortable_recall']:.0%} sorted right, "
            f"{chosen['wrong_sort_rate']:.1%} wrong (threshold {chosen['threshold']:.2f})"
        )
        ax.plot(xs, ys, color=colour, lw=2, label=label, solid_capstyle="round")
        point = ([chosen["wrong_sort_rate"] * 100], [chosen["sortable_recall"] * 100])
        ax.scatter(*point, s=64, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3,
                   clip_on=False)  # fmt: skip
    # Where v1 actually ran: on the zero-shot curve, hollow ring in the zero-shot colour.
    ax.scatter(
        [reference["wrong_sort_rate"] * 100],
        [reference["sortable_recall"] * 100],
        s=72, facecolor=SURFACE, edgecolor=SERIES[0], linewidth=2, zorder=4,
        label=(
            f"v1 as shipped: {reference['sortable_recall']:.0%} sorted right, "
            f"{reference['wrong_sort_rate']:.1%} wrong (threshold {reference['threshold']:.2f})"
        ),
    )  # fmt: skip
    ax.axvline(target * 100, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.text(target * 100, 3, f" target {target:.0%}", color=INK_2, fontsize=8)
    ax.set_xlabel("wrong sorts (% of all items)", color=INK_2)
    ax.set_ylabel("sorted into the right compartment\n(% of items that belong in one)", color=INK_2)
    ax.set_title("Test set: sorting more vs sorting wrong", color=INK, fontsize=11, loc="left")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(10.0, target * 1000))  # the region a bin can live with
    ax.grid(color=GRID, lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK_2, length=0)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left",
              bbox_to_anchor=(0, -0.14))  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _ci(ci) -> str:
    return f"{ci[0] * 100:.1f}–{ci[1] * 100:.1f}%"


def _report(cfg: Config, stats: dict, meta: dict, systems: list[dict], per_source: dict,
            n_test: int) -> str:  # fmt: skip
    zs_v1, zs_tuned, head, deployed = systems
    classes = cfg.classes
    head_kind = "linear" if meta["hidden"] == 0 else f"MLP, {meta['hidden']} hidden"
    other = deployed["per_class"]["support"][classes.index(cfg.reject_class)]
    sortable_share = 1 - other / n_test
    lines = [
        "# BinSight accuracy report",
        "",
        f"Backbone `{meta['backbone']}` (frozen) + trained head ({head_kind}), "
        f"trained {meta['trained_at']}. Test set: {n_test} held-out images never used for "
        "training, model selection, calibration or threshold choice.",
        "",
        "## Summary",
        "",
        f"- **Deployed system** (trained head + zero-shot guard, threshold "
        f"{deployed['threshold']:.2f}): puts **{_pct(deployed['wrong_sort_rate'])}** of items in "
        f"the wrong compartment (95% CI {_ci(deployed['wrong_sort_ci'])}) and sorts "
        f"**{_pct(deployed['sortable_recall'])}** of the items that belong in a compartment into "
        "the right one; everything else goes to UNSURE.",
        f"- **v1 zero-shot** at its 0.60 threshold: {_pct(zs_v1['wrong_sort_rate'])} wrong sorts "
        f"(CI {_ci(zs_v1['wrong_sort_ci'])}), {_pct(zs_v1['sortable_recall'])} of sortable items "
        "sorted right.",
        f"- 5-class accuracy: trained **{_pct(head['accuracy'])}** "
        f"(CI {_ci(head['accuracy_ci'])}) vs zero-shot {_pct(zs_v1['accuracy'])} "
        f"(CI {_ci(zs_v1['accuracy_ci'])}).",
        "",
        "Wrong sorts are what matter for a bin: one wrong item contaminates a compartment, while "
        "an UNSURE only costs a manual sort. The threshold was chosen on the validation split as "
        f"the lowest one keeping wrong sorts at or below {_pct(meta['max_wrong_sort'])}.",
        "",
        "## Results on the test set",
        "",
        "| System | Threshold | Accuracy | Macro-F1 | Wrong sorts | Sortable sorted right | "
        "Auto-sorted | UNSURE | ECE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in systems:
        lines.append(
            f"| {s['name']} | {s['threshold']:.2f} | {_pct(s['accuracy'])} | "
            f"{s['macro_f1']:.3f} | {_pct(s['wrong_sort_rate'])} | {_pct(s['sortable_recall'])} | "
            f"{_pct(s['coverage'])} | "
            f"{_pct(s['unsure_rate'])} | {s['ece']:.3f} |"
        )
    lines += [
        "",
        "Accuracy and macro-F1 use the top class over all five classes (no UNSURE). Wrong sorts, "
        "auto-sorted and UNSURE are shares of all test items after routing; an out-of-scope item "
        "('other') auto-sorted into any compartment counts as a wrong sort. 'Sortable sorted "
        "right' is the share of plastic/paper/metal/organic items that reached the right "
        f"compartment; auto-sorted can be at most {_pct(sortable_share)} "
        "here because the rest of the test set is 'other'. ECE is expected "
        "calibration error of the confidence (lower is better).",
        "",
        "![Coverage vs wrong sorts](tradeoff.png)",
        "",
        "## Per class (deployed system)",
        "",
        "| Class | Precision | Recall | F1 | Test images |",
        "| --- | --- | --- | --- | --- |",
    ]
    pc = deployed["per_class"]
    for i, c in enumerate(classes):
        lines.append(
            f"| {c} | {pc['precision'][i]:.3f} | {pc['recall'][i]:.3f} | {pc['f1'][i]:.3f} | "
            f"{int(pc['support'][i])} |"
        )
    lines += [
        "",
        "Compartment contamination (share of auto-sorted items that do not belong there):",
        "",
        "| Compartment | v1 zero-shot | Deployed |",
        "| --- | --- | --- |",
    ]
    for comp, v in deployed["contamination"].items():
        lines.append(f"| {comp} | {_pct(zs_v1['contamination'][comp])} | {_pct(v)} |")
    lines += [
        "",
        "![Confusion matrices](confusion.png)",
        "",
        "## Per source dataset (5-class accuracy)",
        "",
        "| Source | Test images | v1 zero-shot | Trained |",
        "| --- | --- | --- | --- |",
    ]
    for source, v in per_source.items():
        lines.append(f"| `{source}` | {v['n']} | {_pct(v['zero_shot'])} | {_pct(v['trained'])} |")
    lines += [
        "",
        "## Calibration",
        "",
        f"Softmax temperature {meta['temperature']:.2f}, fitted on validation. Test ECE "
        f"{head.get('ece_uncalibrated', float('nan')):.3f} before -> {head['ece']:.3f} after. "
        "Calibrated confidences make the threshold mean roughly what it says.",
        "",
        "## Data",
        "",
        "| Source | License | Images mapped | Kept after de-dup |",
        "| --- | --- | --- | --- |",
    ]
    licenses = {
        "omasteam/waste-garbage-management-dataset": "MIT",
        "garythung/trashnet": "MIT",
        "viola77data/recycling-dataset": "Apache-2.0",
    }
    for source in cfg.data.sources:
        lines.append(
            f"| `{source}` | {licenses.get(source, '?')} | {stats['per_source_mapped'][source]} | "
            f"{stats['per_source_kept'][source]} |"
        )
    lines += [
        "",
        "- Labels were mapped to the five classes in `config.yaml`; mixed labels (generic "
        "'trash', takeaway cups, disposable plates, paper towels, polystyrene) were dropped "
        "rather than guessed.",
        f"- Near-duplicates (cosine >= {cfg.data.dedupe_cosine} between embeddings, or identical "
        f"image hash) were merged before splitting, so no image leaks between train and test: "
        f"{stats['duplicate_links']} duplicate links in {stats['duplicate_groups']} groups; "
        f"{stats['conflicting_groups']} groups whose copies had different labels "
        f"({stats['conflicting_images']} images) were dropped.",
        f"- Stratified {1 - cfg.data.val_fraction - cfg.data.test_fraction:.0%} / "
        f"{cfg.data.val_fraction:.0%} / {cfg.data.test_fraction:.0%} train / val / test split "
        f"(seed {cfg.data.seed}). Training also used {cfg.data.augment_views} augmented views "
        "per image (crop, flip, rotation, colour, blur).",
        "",
        "| Split | " + " | ".join(classes) + " |",
        "| --- | " + " | ".join("---" for _ in classes) + " |",
    ]
    for split, counts in stats["per_split_class"].items():
        lines.append(f"| {split} | " + " | ".join(str(counts[c]) for c in classes) + " |")
    lines += [
        "",
        "## Method",
        "",
        "1. Every image is embedded once by the frozen SigLIP2 vision encoder (fp16 on MPS).",
        "2. A small head (grid over linear / MLP, learning rate, weight decay; best validation "
        "macro-F1 kept) is trained on those embeddings with class-balanced cross-entropy.",
        "3. A softmax temperature is fitted on validation.",
        "4. The zero-shot prompts from v1 still run on the same embedding as a guard: if they say "
        f"'other' with probability >= {cfg.guard}, the item goes to UNSURE. This covers things "
        "the datasets have no pictures of (hands, faces, an empty tray, phones).",
        "5. The UNSURE threshold is the lowest one meeting the wrong-sort target on validation.",
        "",
        "Validation grid:",
        "",
        "| Hidden | LR | Weight decay | Val macro-F1 | Best epoch |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in meta["grid"]:
        lines.append(
            f"| {r['hidden']} | {r['lr']} | {r['weight_decay']} | {r['val_macro_f1']:.4f} | "
            f"{r['best_epoch']} |"
        )
    lines += [
        "",
        "Weight decay made no measurable difference at these settings; the hidden layer and "
        "learning rate did.",
        "",
        "## Limitations",
        "",
        "- The test images come from the same web datasets as the training images: mostly "
        "single items, well lit, plain backgrounds. A webcam over a real tray will look "
        "different, so expect lower accuracy live; the threshold can be raised with `+`.",
        "- De-duplication removes copies, not different photos of the same object; such "
        "near-twins can still sit on both sides of the split and flatter these numbers.",
        "- 'other' in the data is glass, batteries, clothes and shoes only. Hands, faces, phones "
        "and an empty tray rely on the zero-shot guard, which this test set does not measure.",
        "- Class mapping is a policy choice (e.g. cardboard is paper, soft plastics are plastic, "
        "glass is UNSURE because the bin has no glass compartment).",
        "",
    ]
    return "\n".join(lines)


def evaluate(cfg: Config, out: Path, log=print) -> Path:
    split = load_split(cfg)
    if not cfg.head.exists():
        raise FileNotFoundError(f"{cfg.head} missing: run `binsight train` first")
    head, meta = load_head(cfg.head)
    stats = json.loads((cfg.data.dir / "stats.json").read_text())
    classes = cfg.classes

    backbone = Backbone(meta["backbone"], cfg.device, cfg.dtype)
    text = backbone.class_text_embeds(cfg.prompts, cfg.prompt_template)

    def scores(part: str):
        m = split["split"] == SPLITS.index(part)
        x = torch.from_numpy(split["emb"][m].astype(np.float32))
        with torch.no_grad():
            raw = head(x).numpy()
        zero_shot = (backbone.scale * x @ text.T).softmax(-1).numpy()
        return m, head.probs(x).numpy(), zero_shot, raw, split["y"][m]

    _, v_probs, v_zs, _, v_y = scores("val")
    t_mask, t_probs, t_zs, t_raw, t_y = scores("test")
    log(f"evaluating on {len(t_y)} test images")

    rc = cfg.reject_class
    zs_threshold = pick_threshold(sweep(v_zs, None, v_y, classes, rc, None), meta["max_wrong_sort"])
    head_threshold = pick_threshold(
        sweep(v_probs, None, v_y, classes, rc, None), meta["max_wrong_sort"]
    )
    systems = [
        _system("v1 zero-shot", t_zs, None, t_y, classes, cfg, ZERO_SHOT_THRESHOLD, None),
        _system("zero-shot, tuned threshold", t_zs, None, t_y, classes, cfg, zs_threshold, None),
        _system("trained head", t_probs, None, t_y, classes, cfg, head_threshold, None, t_raw),
        _system(
            "trained + guard (deployed)",
            t_probs,
            t_zs,
            t_y,
            classes,
            cfg,
            meta["threshold"],
            meta["guard"],
        ),  # fmt: skip
    ]
    for s in systems:
        log(
            f"  {s['name']:<28} acc {s['accuracy']:.3f}  wrong-sort {s['wrong_sort_rate']:.3%}  "
            f"auto-sorted {s['coverage']:.1%}  (threshold {s['threshold']:.2f})"
        )

    per_source = {}
    src = split["src"][t_mask]
    for k, source in enumerate(cfg.data.sources):
        m = src == k
        if m.any():
            per_source[source] = {
                "n": int(m.sum()),
                "zero_shot": float((t_zs[m].argmax(1) == t_y[m]).mean()),
                "trained": float((t_probs[m].argmax(1) == t_y[m]).mean()),
            }

    out.mkdir(parents=True, exist_ok=True)
    _plot_confusions([systems[0], systems[2]], classes, out / "confusion.png")
    curves = [
        ("zero-shot", sweep(t_zs, None, t_y, classes, rc, None), systems[1]),
        ("trained", sweep(t_probs, None, t_y, classes, rc, None), systems[2]),
        ("trained + guard", sweep(t_probs, t_zs, t_y, classes, rc, meta["guard"]), systems[3]),
    ]
    _plot_tradeoff(curves, systems[0], meta["max_wrong_sort"], out / "tradeoff.png")
    (out / "metrics.json").write_text(
        json.dumps({"systems": systems, "per_source": per_source, "data": stats}, indent=2) + "\n"
    )
    path = out / "REPORT.md"
    path.write_text(_report(cfg, stats, meta, systems, per_source, len(t_y)))
    return path
