# BinSight

Demo of a campus smart bin. An item goes on a tray, a camera classifies it with a pretrained
zero-shot model (no dataset, no training), and the tray drops it into PLASTIC, PAPER, METAL or
ORGANIC. Anything uncertain or out of scope goes to UNSURE for manual sorting: a wrong auto-sort
is worse than an UNSURE.

Default model is `google/siglip2-so400m-patch16-256`; `openai/clip-vit-large-patch14` is the
fallback. Runs on Apple-silicon MPS with CPU fallback.

## Setup

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The first run downloads the model (about 4.5 GB for SigLIP2) into the Hugging Face cache.

### Camera access

macOS only lets an app use the camera if you allow it. Allow the terminal app you run
`binsight` from (Terminal, iTerm, ...) in
**System Settings → Privacy & Security → Camera**, then restart that app. Without it
`binsight live` stops with "Cannot open camera". If a phone nearby takes over camera 0
(Continuity Camera), use `--camera 1`.

## Commands

```bash
uv run binsight live                 # webcam demo
uv run binsight classify photo.jpg   # scores for one image file
```

Both accept `--model <hf id>` (or `--model fallback` for CLIP), `--config <path>` and
`--threshold <0-1>`. `live` also takes `--camera <index>`.

### `binsight live`

Hold an item inside the white square (the tray). With the tray empty at start, the app grabs a
background; when the square differs from it and the item stays still for about 10 frames, it
classifies once, shows the result, slides a block to the chosen compartment, and re-arms when you
take the item away. The display is mirrored; the classified crop is not, so label text stays
readable. Nothing is written to disk. Each decision is also printed in the terminal.

| Key | Action |
| --- | --- |
| `SPACE` | classify the tray now |
| `B` | re-capture the empty-tray background (do this if lighting changes) |
| `+` / `-` | raise / lower the UNSURE threshold by 0.05 |
| `Q` / `Esc` | quit |

Tips: keep the tray empty for the first second after start, and keep your hand out of the square
after placing the item (a hand in view is scored as "other" and goes to UNSURE).

## Configuration

Everything is in [`config.yaml`](config.yaml): model ids, prompts per class, prompt template,
threshold, ROI size, trigger sensitivity and compartment colours.

How scoring works: every class has several text prompts; their embeddings are averaged into one
vector per class (encoded once at startup). For an item, only the image is encoded; scores are the
scaled cosine similarity to each class vector, softmaxed over the five classes so they sum to 1.
The item goes to UNSURE if the top class is `other` or its confidence is below the threshold
(default 0.6).

## Development

```bash
uv run ruff format . && uv run ruff check .
```
