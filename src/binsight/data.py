"""Training data: public datasets -> frozen-backbone embeddings -> de-duplicated splits.

Stages, each cached under data.dir:
  1. <source>.npz      centre-view embedding + image hash for every usable row of a source
  2. split.npz         all sources merged, near-duplicates collapsed, stratified train/val/test
  3. <source>_aug.npz  extra augmented-view embeddings for that source's training rows
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm import tqdm

from .config import Config
from .model import Backbone

SPLITS = ("train", "val", "test")
PARQUET_REVISION = "refs/convert/parquet"
DECODE_SIZE = 512  # JPEGs are decoded at reduced size; the backbone sees 256 px anyway

# Approximates the gap between web photos and a webcam over a tray: framing, mirror,
# tilt, exposure and white balance, and softness.
AUGMENT = T.Compose(
    [
        T.RandomResizedCrop(320, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20, fill=128),
        T.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.3, hue=0.03),
    ]
)


def slug(source: str) -> str:
    return source.replace("/", "__")


def dhash(image: Image.Image) -> np.ndarray:
    """64-bit difference hash as 8 bytes: equal for resized / recompressed copies."""
    g = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    return np.packbits(g[:, 1:] > g[:, :-1])


def load_source(source: str):
    """All of a dataset's splits as one, images left undecoded (decoding happens in workers).

    We make our own split, so a dataset's train/validation/test files are simply concatenated.

    Reads the Hub's auto-converted Parquet copy: a few large files instead of one request
    per image, which is hours faster for image-folder datasets.
    """
    import datasets
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(source, repo_type="dataset", revision=PARQUET_REVISION))
    files = sorted(str(p) for p in root.glob("*/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"{source}: no Parquet files at {PARQUET_REVISION}")
    ds = datasets.load_dataset("parquet", data_files=files, split="train")
    return ds.cast_column("image", datasets.Image(decode=False))


class _Rows(Dataset):
    """Decodes dataset rows into backbone pixel tensors; runs inside DataLoader workers."""

    def __init__(self, ds, rows: np.ndarray, image_processor, augment: bool):
        self.ds, self.rows, self.image_processor, self.augment = ds, rows, image_processor, augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        item = self.ds[int(self.rows[i])]["image"]
        image = Image.open(io.BytesIO(item["bytes"]) if item["bytes"] else item["path"])
        image.draft("RGB", (DECODE_SIZE, DECODE_SIZE))
        image = image.convert("RGB")
        image.thumbnail((DECODE_SIZE, DECODE_SIZE))
        code = dhash(image)
        if self.augment:
            image = AUGMENT(image)
            if torch.rand(()) < 0.3:
                image = image.filter(
                    ImageFilter.GaussianBlur(float(torch.empty(()).uniform_(0.5, 1.5)))
                )
        pixels = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixels, torch.from_numpy(code)


def embed_rows(backbone: Backbone, ds, rows, cfg: Config, augment: bool, desc: str):
    """(float16 embeddings, uint8 hashes) for the given rows, in order."""
    loader = DataLoader(
        _Rows(ds, rows, backbone.processor.image_processor, augment),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.workers,
        generator=torch.Generator().manual_seed(cfg.data.seed),
    )
    embeds, codes = [], []
    for pixels, code in tqdm(loader, desc=desc, unit="batch"):
        embeds.append(backbone.embed_pixels(pixels).half().numpy())
        codes.append(code.numpy())
    return np.concatenate(embeds), np.concatenate(codes)


def _usable_rows(ds, source: str, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Rows whose label maps to one of our classes, and that class index.

    A source maps its `label` column, or names another column: {column: ..., labels: {...}}.
    """
    spec = cfg.data.sources[source]
    column, mapping = (spec["column"], spec["labels"]) if "labels" in spec else ("label", spec)
    feature, values = ds.features[column], ds[column]
    if hasattr(feature, "names"):  # class-label column: values are indices into names
        values = [feature.names[v] for v in values]
    unknown = set(values) - set(mapping)
    if unknown:
        raise ValueError(f"{source}: no mapping in config.yaml for labels {sorted(unknown)}")
    to_class = {n: cfg.classes.index(c) if c else -1 for n, c in mapping.items()}
    y = np.array([to_class[v] for v in values])
    rows = np.flatnonzero(y >= 0)
    return rows, y[rows]


def _cached(path: Path, backbone_id: str) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    cache = dict(np.load(path))
    return cache if str(cache["backbone"]) == backbone_id else None


def _merge(cache, rows, arrays: dict[str, np.ndarray], row_axis: dict[str, int]):
    """Add newly embedded rows to a cache, kept sorted by row."""
    if cache is None:
        merged = {"rows": rows, **arrays}
    else:
        merged = {"rows": np.concatenate([cache["rows"], rows])}
        for k, a in arrays.items():
            merged[k] = np.concatenate([cache[k], a], axis=row_axis.get(k, 0))
    order = np.argsort(merged["rows"])
    return {k: np.take(v, order, axis=row_axis.get(k, 0)) for k, v in merged.items()}


def embed_sources(backbone: Backbone, cfg: Config) -> None:
    """Stage 1: embeds only rows not cached yet, so new datasets or mappings are cheap."""
    cfg.data.dir.mkdir(parents=True, exist_ok=True)
    for source in cfg.data.sources:
        path = cfg.data.dir / f"{slug(source)}.npz"
        cache = _cached(path, backbone.model_id)
        ds = load_source(source)
        rows, _ = _usable_rows(ds, source, cfg)
        missing = rows if cache is None else np.setdiff1d(rows, cache["rows"])
        if not len(missing):
            continue
        emb, codes = embed_rows(backbone, ds, missing, cfg, augment=False, desc=source)
        merged = _merge(cache, missing, {"emb": emb, "hash": codes}, {})
        np.savez(path, **merged, backbone=backbone.model_id)


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def build_split(cfg: Config, backbone_id: str) -> dict:
    """Stage 2: merge sources, collapse near-duplicates, stratified split. Returns stats."""
    parts = []
    for source in cfg.data.sources:
        # Labels come from the current mapping, not the cache, so mapping edits take effect.
        rows, labels = _usable_rows(load_source(source), source, cfg)
        cache = np.load(cfg.data.dir / f"{slug(source)}.npz")
        keep = np.isin(cache["rows"], rows)
        lookup = dict(zip(rows.tolist(), labels.tolist(), strict=True))
        parts.append({
            "rows": cache["rows"][keep],
            "y": np.array([lookup[r] for r in cache["rows"][keep].tolist()], dtype=int),
            "emb": cache["emb"][keep],
            "hash": cache["hash"][keep],
        })  # fmt: skip
    src = np.concatenate([np.full(len(p["rows"]), k) for k, p in enumerate(parts)])
    rows = np.concatenate([p["rows"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    emb = np.concatenate([p["emb"] for p in parts])
    codes = np.concatenate([p["hash"] for p in parts])
    n = len(y)

    # Near-duplicate pairs: very high cosine, or identical hash with high cosine.
    x = torch.from_numpy(emb).float()
    code_key = codes.view(np.uint64).ravel()
    parent = np.arange(n)
    pairs = 0
    for start in range(0, n, 2048):
        sims = x[start : start + 2048] @ x.T
        block = torch.arange(start, min(start + 2048, n))
        sims[torch.arange(len(block)), block] = -1  # self
        same_hash = torch.from_numpy(code_key[start : start + 2048, None] == code_key[None, :])
        hit = (sims >= cfg.data.dedupe_cosine) | (same_hash & (sims >= 0.9))
        for i, j in torch.nonzero(hit).tolist():
            a, b = _find(parent, start + i), _find(parent, j)
            if a != b:
                parent[max(a, b)] = min(a, b)
                pairs += 1
    root = np.array([_find(parent, i) for i in range(n)])

    # One image per group; groups whose copies disagree on the label are dropped.
    keep = np.zeros(n, bool)
    conflicts = conflict_images = 0
    sizes = np.bincount(root, minlength=n)
    order = np.argsort(root, kind="stable")
    for members in np.split(order, np.flatnonzero(np.diff(root[order])) + 1):
        if len(set(y[members].tolist())) > 1:
            conflicts += 1
            conflict_images += len(members)
        else:
            keep[members[0]] = True

    # Stratified split per class.
    rng = np.random.default_rng(cfg.data.seed)
    split = np.full(n, -1)
    for c in range(len(cfg.classes)):
        idx = rng.permutation(np.flatnonzero(keep & (y == c)))
        n_test = round(len(idx) * cfg.data.test_fraction)
        n_val = round(len(idx) * cfg.data.val_fraction)
        split[idx[:n_test]] = SPLITS.index("test")
        split[idx[n_test : n_test + n_val]] = SPLITS.index("val")
        split[idx[n_test + n_val :]] = SPLITS.index("train")

    sel = split >= 0
    np.savez(
        cfg.data.dir / "split.npz",
        src=src[sel], row=rows[sel], y=y[sel], split=split[sel], emb=emb[sel],
        backbone=backbone_id,
    )  # fmt: skip

    sources = list(cfg.data.sources)
    stats = {
        "backbone": backbone_id,
        "images_mapped": int(n),
        "duplicate_links": int(pairs),
        "duplicate_groups": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()),
        "conflicting_groups": int(conflicts),
        "conflicting_images": int(conflict_images),
        "images_kept": int(sel.sum()),
        "per_source_mapped": {s: int((src == k).sum()) for k, s in enumerate(sources)},
        "per_source_kept": {s: int((src[sel] == k).sum()) for k, s in enumerate(sources)},
        "per_split_class": {
            name: {c: int(((split == si) & (y == ci)).sum()) for ci, c in enumerate(cfg.classes)}
            for si, name in enumerate(SPLITS)
        },
    }
    (cfg.data.dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def embed_augmented(backbone: Backbone, cfg: Config) -> None:
    """Stage 3: augmented views of every training image; only uncached rows are embedded."""
    split = np.load(cfg.data.dir / "split.npz")
    train = split["split"] == SPLITS.index("train")
    for k, source in enumerate(cfg.data.sources):
        rows = np.sort(split["row"][train & (split["src"] == k)])
        path = cfg.data.dir / f"{slug(source)}_aug.npz"
        cache = _cached(path, backbone.model_id)
        if cache is not None and len(cache["emb"]) != cfg.data.augment_views:
            cache = None
        missing = rows if cache is None else np.setdiff1d(rows, cache["rows"])
        if not len(missing):
            continue
        ds = load_source(source)
        views = [
            embed_rows(backbone, ds, missing, cfg, augment=True, desc=f"{source} aug {v + 1}")[0]
            for v in range(cfg.data.augment_views)
        ]
        merged = _merge(cache, missing, {"emb": np.stack(views)}, {"emb": 1})
        np.savez(path, **merged, backbone=backbone.model_id)


def prepare(backbone: Backbone, cfg: Config) -> dict:
    embed_sources(backbone, cfg)
    stats = build_split(cfg, backbone.model_id)
    if cfg.data.augment_views:
        embed_augmented(backbone, cfg)
    return stats


def load_split(cfg: Config) -> dict[str, np.ndarray]:
    path = cfg.data.dir / "split.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `binsight prepare` first")
    return dict(np.load(path))


def load_train_views(
    cfg: Config, split: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augmented training embeddings, their labels and source index, matched to split rows."""
    embs, labels, srcs = [], [], []
    train = split["split"] == SPLITS.index("train")
    for k, source in enumerate(cfg.data.sources):
        path: Path = cfg.data.dir / f"{slug(source)}_aug.npz"
        if not path.exists():
            continue
        aug = np.load(path)
        mine = train & (split["src"] == k)
        pos = {r: i for i, r in enumerate(aug["rows"].tolist())}
        idx = np.array([pos[r] for r in split["row"][mine].tolist()], dtype=int)
        for view in aug["emb"]:
            embs.append(view[idx])
            labels.append(split["y"][mine])
            srcs.append(np.full(len(idx), k))
    if not embs:
        empty = np.zeros(0, int)
        return np.zeros((0, split["emb"].shape[1]), np.float16), empty, empty
    return np.concatenate(embs), np.concatenate(labels), np.concatenate(srcs)
