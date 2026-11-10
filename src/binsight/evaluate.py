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
from .model import ZERO_SHOT_THRESHOLD, Backbone, Policy, load_head

# Model release shown on the report and graphs (git tag of the committed head).
RELEASE = "v3"

# Reference data-viz palette (light surface).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _system(name, probs, zero_shot, y, policy: Policy, logits_raw=None):
    classes = list(policy.classes)
    truth = true_compartments(y, classes, policy.unsure)
    chosen = decisions(probs, zero_shot, policy)
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
        "threshold": policy.threshold,
        "guard": policy.guard,
        "check_below": policy.check_below,
        "min_confidence": policy.min_confidence,
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
            f"zero-shot at 0.60 (v1 setting): {reference['sortable_recall']:.0%} sorted right, "
            f"{reference['wrong_sort_rate']:.1%} wrong (threshold {reference['threshold']:.2f})"
        ),
    )  # fmt: skip
    ax.axvline(target * 100, color=INK_2, lw=1, ls=(0, (3, 3)))
    ax.text(target * 100, 3, f" target {target:.0%}", color=INK_2, fontsize=8)
    ax.set_xlabel("wrong sorts (% of all items)", color=INK_2)
    ax.set_ylabel("sorted into the right compartment\n(% of items that belong in one)", color=INK_2)
    ax.set_title(
        f"BinSight {RELEASE} on the test set: sorting more vs sorting wrong",
        color=INK,
        fontsize=11,
        loc="left",
    )
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
            n_test: int, cross: dict | None) -> str:  # fmt: skip
    zs_v1, zs_tuned, head, guarded, deployed = systems
    classes = cfg.classes
    head_kind = "linear" if meta["hidden"] == 0 else f"MLP, {meta['hidden']} hidden"
    other = sum(deployed["per_class"]["support"][classes.index(c)] for c in cfg.unsure_classes)
    sortable_share = 1 - other / n_test
    lines = [
        f"# BinSight {RELEASE} accuracy report",
        "",
        f"Backbone `{meta['backbone']}` (frozen) + trained head ({head_kind}), "
        f"trained {meta['trained_at']}. Test set: {n_test} held-out images never used for "
        "training, model selection, calibration or threshold choice.",
        "",
        "## Summary",
        "",
        f"- **{RELEASE} deployed system** (trained head + zero-shot guard + second opinion, "
        f"threshold {deployed['threshold']:.2f}): puts "
        f"**{_pct(deployed['wrong_sort_rate'])}** of items in "
        f"the wrong compartment (95% CI {_ci(deployed['wrong_sort_ci'])}) and sorts "
        f"**{_pct(deployed['sortable_recall'])}** of the items that belong in a compartment into "
        "the right one; everything else goes to UNSURE.",
        "- **Zero-shot only** (the v1 approach, with the current prompts) at 0.60: "
        f"{_pct(zs_v1['wrong_sort_rate'])} wrong sorts "
        f"(CI {_ci(zs_v1['wrong_sort_ci'])}), {_pct(zs_v1['sortable_recall'])} of sortable items "
        "sorted right.",
        f"- 6-class accuracy: trained **{_pct(head['accuracy'])}** "
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
        f"Accuracy and macro-F1 use the top class over all {len(classes)} classes (no UNSURE). "
        "Wrong sorts, auto-sorted and UNSURE are shares of all test items after routing; an item "
        f"with no compartment ({' or '.join(cfg.unsure_classes)}) auto-sorted into any "
        "compartment counts as a wrong sort. 'Sortable sorted "
        "right' is the share of plastic/paper/metal/organic items that reached the right "
        f"compartment; auto-sorted can be at most {_pct(sortable_share)} "
        f"here because the rest of the test set is {' or '.join(cfg.unsure_classes)}. "
        "ECE is expected calibration error of the confidence (lower is better).",
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
        "| Compartment | Zero-shot only | Deployed |",
        "| --- | --- | --- |",
    ]
    for comp, v in deployed["contamination"].items():
        lines.append(f"| {comp} | {_pct(zs_v1['contamination'][comp])} | {_pct(v)} |")
    lines += [
        "",
        "![Confusion matrices](confusion.png)",
        "",
        "## Per source dataset (6-class accuracy)",
        "",
        "| Source | Test images | Zero-shot only | Trained |",
        "| --- | --- | --- | --- |",
    ]
    for source, v in per_source.items():
        lines.append(f"| `{source}` | {v['n']} | {_pct(v['zero_shot'])} | {_pct(v['trained'])} |")
    if cross:
        lines += [
            "",
            "## Photos from an unseen source",
            "",
            "The test set above shares photographers, cameras and backgrounds with training. "
            "This harder test trains without one whole dataset and tests on all of it, which is "
            "closer to a new webcam on a new campus. Same recipe and settings as deployed.",
            "",
            "| Held-out dataset | Images | Trained accuracy | Trained wrong sorts | "
            "zero-shot accuracy | zero-shot wrong sorts |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for source, r in cross.items():
            missing = r.get("classes_missing_from_training", [])
            if missing:
                lines.append(
                    f"| `{source}` | {r['test_images']} | not comparable: training had no "
                    f"{', '.join(missing)} | | {_pct(r['zero_shot']['accuracy'])} | "
                    f"{_pct(r['zero_shot']['wrong_sort_rate'])} |"
                )
                continue
            lines.append(
                f"| `{source}` | {r['test_images']} | {_pct(r['trained']['accuracy'])} | "
                f"{_pct(r['trained']['wrong_sort_rate'])} | {_pct(r['zero_shot']['accuracy'])} | "
                f"{_pct(r['zero_shot']['wrong_sort_rate'])} |"
            )
        lines += [
            "",
            "What this found (history, measured with three datasets before RealWaste was added): "
            "plastic in the two larger datasets is mostly bottles, so a head trained without the "
            "recycling-dataset mistook soft plastics (bags, wrappers) and hard plastic containers "
            "for paper or 'other', confidently: 12.0% wrong sorts on it. Zero-shot SigLIP2 knows "
            "what a wrapper is. Requiring the trained head and zero-shot to agree cut that to "
            "2.2% but sent much of the paper to UNSURE. The deployed rule asks for the second "
            "opinion only when the item might be glass or organic, or the head is under "
            f"{((meta.get('double_check') or {}).get('below') or 0):.0%} sure. Checking only "
            "possible glass/organic would sort a little more but quadrupled wrong sorts on unseen "
            "datasets, so the confidence condition stays (comparison in `scripts/rule_ab.py`, "
            "chosen on validation and these unseen-dataset folds). On the main test set: "
            f"sortable items sorted right {_pct(guarded['sortable_recall'])} without a second "
            f"opinion, {_pct(deployed['sortable_recall'])} with it.",
            "",
            "## Organic safety",
            "",
            "Glass or plastic in food waste can hurt animals that eat it, so ORGANIC is stricter "
            "than the other compartments: zero-shot must always agree, and the trained head needs "
            f"{(meta.get('min_confidence') or {}).get('organic', 0):.2f} confidence. When the bar "
            "was set, 3 non-organic test items had reached ORGANIC with only the usual rules: "
            "green glass ornaments shaped like beans (both models were fooled), a burlap sack "
            "covered in grass, and a plastic item; validation had none, so it could not be used "
            "to choose the bar. The 0.98 bar was therefore set as a safety policy after seeing "
            "those test items. On the current test set, reshuffled when a fifth dataset was "
            "added, it again stops 3 items (one glass, one plastic, one other). Some of the "
            "photos it was chosen on may still be in the test set, so the result for ORGANIC "
            "leaks is not an independent measurement. Cost: about 2% of organic items go to "
            "UNSURE.",
        ]
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
        "Tatyr2210/waste-classification-vit": "MIT upload of RealWaste (CC BY 4.0)",
        "TrashBusters/combined": "MIT (merges four Kaggle datasets, incl. real-world photos)",
    }
    for source in cfg.data.sources:
        lines.append(
            f"| `{source}` | {licenses.get(source, '?')} | {stats['per_source_mapped'][source]} | "
            f"{stats['per_source_kept'][source]} |"
        )
    lines += [
        "",
        "- Labels were mapped to the six classes in `config.yaml`; mixed labels (generic "
        "'trash', takeaway and paper cups, disposable plates, paper towels, polystyrene, "
        "Tetra Pak, medicine blister strips, tea bags, aerosol cans) were dropped rather than "
        "guessed.",
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
        f"{' or '.join(cfg.unsure_classes)} with probability >= {cfg.guard}, the item goes to "
        "UNSURE. This covers things "
        "the datasets have no pictures of (hands, faces, an empty tray, phones).",
        "5. Second opinion: when the item might be glass or organic, or the head is under "
        f"{((meta.get('double_check') or {}).get('below') or 0):.0%} sure, it is auto-sorted only "
        "if the zero-shot prompts pick the same class; organic also needs higher head "
        "confidence (see the unseen-source test and organic safety).",
        "6. The UNSURE threshold is the lowest one meeting the wrong-sort target on validation.",
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
        "- 'other' in the data is batteries, clothes, shoes, textiles, ceramics, gloves, masks, "
        "bandages, nappies, construction scrap and dead animals. Hands, faces, "
        "phones and an empty tray rely on the zero-shot guard, which this test set does not "
        "measure.",
        "- Class mapping is a policy choice (e.g. cardboard is paper, soft plastics are plastic, "
        "glass is UNSURE because the bin has no glass compartment).",
        "- Some photos labelled glass in the datasets look like clear PET plastic (screw-thread "
        "bottles, a disposable champagne flute). Part of the measured 'glass in PLASTIC' wrong "
        "sorts is likely label noise, and clear glass and clear plastic can look identical in a "
        "photo: a real bin would add a cheap physical check (weight, or the sound of a tap).",
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

    rc = tuple(cfg.unsure_classes)
    target = meta["max_wrong_sort"]
    deployed = Policy.from_meta(meta, rc)
    plain = Policy(tuple(classes), rc, ZERO_SHOT_THRESHOLD)
    guarded = Policy(tuple(classes), rc, 0.0, meta["guard"])

    def tuned(policy, probs, zero_shot):
        return policy.with_threshold(pick_threshold(sweep(probs, zero_shot, v_y, policy), target))

    zs_tuned = tuned(plain, v_zs, None)
    head_tuned = tuned(plain, v_probs, None)
    guard_tuned = tuned(guarded, v_probs, v_zs)
    systems = [
        _system("zero-shot only", t_zs, None, t_y, plain),
        _system("zero-shot, tuned threshold", t_zs, None, t_y, zs_tuned),
        _system("trained head", t_probs, None, t_y, head_tuned, t_raw),
        _system("trained + guard", t_probs, t_zs, t_y, guard_tuned),
        _system("trained + guard + second opinion (deployed)", t_probs, t_zs, t_y, deployed),
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
    trained = {**systems[2], "name": f"{RELEASE} trained head"}
    _plot_confusions([systems[0], trained], classes, out / "confusion.png")
    curves = [
        ("zero-shot", sweep(t_zs, None, t_y, plain), systems[1]),
        ("trained", sweep(t_probs, None, t_y, plain), systems[2]),
        ("trained + guard", sweep(t_probs, t_zs, t_y, guarded), systems[3]),
        (f"{RELEASE} deployed", sweep(t_probs, t_zs, t_y, deployed), systems[4]),
    ]
    _plot_tradeoff(curves, systems[0], meta["max_wrong_sort"], out / "tradeoff.png")
    (out / "metrics.json").write_text(
        json.dumps({"systems": systems, "per_source": per_source, "data": stats}, indent=2) + "\n"
    )
    path = out / "REPORT.md"
    cross_path = out / "crosscheck.json"
    cross = json.loads(cross_path.read_text()) if cross_path.exists() else None
    path.write_text(_report(cfg, stats, meta, systems, per_source, len(t_y), cross))
    return path
