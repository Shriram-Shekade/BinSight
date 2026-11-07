# Experiment (not part of the package): run with `uv run python scripts/rule_ab.py`.
"""Compare second-opinion rules on main and unseen-dataset folds."""

import numpy as np
import torch

from binsight.config import load_config
from binsight.data import SPLITS, load_split, load_train_views
from binsight.metrics import operating_point, true_compartments
from binsight.model import Backbone
from binsight.train import fit, fit_temperature

cfg = load_config()
split = load_split(cfg)
xa, ya, sa = load_train_views(cfg, split)
bb = Backbone(cfg.model, cfg.device, cfg.dtype)
text = bb.class_text_embeds(cfg.prompts, cfg.prompt_template)
C = cfg.classes
U = tuple(cfg.unsure_classes)
G, OTH, ORG = C.index("glass"), C.index("other"), C.index("organic")


def T(a):
    return torch.from_numpy(np.asarray(a, np.float32))


def zs(x):
    return (bb.scale * x @ text.T).softmax(-1).numpy()


NAMES = np.array(["UNSURE" if c in U else c.upper() for c in C])


def decide(hp, zp, th, rule, s=0.05, guard=0.5):
    top, conf, ztop = hp.argmax(1), hp.max(1), zp.argmax(1)
    ok = ~np.isin(top, [G, OTH]) & (conf >= th) & ~((top == ORG) & (conf < 0.98))
    if guard is not None:
        ok &= (zp[:, G] + zp[:, OTH]) < guard
    if rule.startswith("hybrid"):
        need = (top == ORG) | (hp[:, G] >= s) | (conf < float(rule[6:]))
    else:
        need = {
            "old": np.where(top == ORG, True, conf < 0.99),
            "user": (top == ORG) | (hp[:, G] >= s),
            "none": np.zeros(len(top), bool),
        }[rule]
    ok &= ~need | (ztop == top)
    return np.where(ok, NAMES[top], "UNSURE")


def pick(hp, zp, y, **kw):
    truth, best = true_compartments(y, C, U), 0.99
    for th in np.round(np.arange(0.99, 0.295, -0.01), 2):
        if operating_point(decide(hp, zp, th, **kw), truth)["wrong_sort_rate"] <= 0.01:
            best = th
        else:
            break
    return best


tr, va, te = (split["split"] == SPLITS.index(p) for p in SPLITS)
folds = {}


def head_for(train_mask, aug_mask, val_mask):
    x = T(np.concatenate([split["emb"][train_mask], xa[aug_mask]]))
    y = torch.from_numpy(np.concatenate([split["y"][train_mask], ya[aug_mask]])).long()
    xv, yv = T(split["emb"][val_mask]), torch.from_numpy(split["y"][val_mask]).long()
    h, _, _ = fit(x, y, xv, yv, len(C), 512, 0.001, 0.0001, cfg, 0)
    with torch.no_grad():
        h.temperature.fill_(fit_temperature(h(xv), yv))
    return h


folds["main val"] = (head_for(tr, np.ones(len(ya), bool), va), va, va)
folds["main test"] = (folds["main val"][0], va, te)
for k, s in enumerate(cfg.data.sources):
    held = split["src"] == k
    folds["unseen " + s.split("/")[0][:8]] = (
        head_for(tr & ~held, sa != k, va & ~held),
        va & ~held,
        held,
    )
cache = {
    n: (
        h.probs(T(split["emb"][vm])).numpy(),
        zs(T(split["emb"][vm])),
        split["y"][vm],
        h.probs(T(split["emb"][tm])).numpy(),
        zs(T(split["emb"][tm])),
        split["y"][tm],
    )
    for n, (h, vm, tm) in folds.items()
}
variants = [
    ("old: agree unless 99% sure", dict(rule="old")),
    ("user: glass>=0.02 or organic", dict(rule="user", s=0.02)),
]
variants += [
    (f"hybrid: glass/organic or <{c}", dict(rule=f"hybrid{c}", s=0.02))
    for c in (0.9, 0.95, 0.97, 0.99)
]
print(f"{'variant':<30}" + "".join(f"{n:>22}" for n in folds))
for name, kw in variants:
    row = []
    for _n, (hv, zv, yv, ht, zt, yt) in cache.items():
        th = pick(hv, zv, yv, **kw)
        op = operating_point(decide(ht, zt, th, **kw), true_compartments(yt, C, U))
        row.append(f"{op['sortable_recall']:6.1%} / {op['wrong_sort_rate']:5.2%}")
    print(f"{name:<30}" + "".join(f"{r:>22}" for r in row))
print("cells: sortable items sorted right / wrong sorts (share of all items)")
