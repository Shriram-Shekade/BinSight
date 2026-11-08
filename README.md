# BinSight v2

A campus smart bin. An item goes on a tray, a camera classifies it, and the tray drops it into
PLASTIC, PAPER, METAL or ORGANIC. Anything uncertain or out of scope goes to UNSURE for manual
sorting, because one wrong item contaminates a whole compartment: a wrong auto-sort is much worse
than an UNSURE.

v1 asked a pretrained model (SigLIP2) to match photos against text prompts. v2 trains its own
classifier on ~28k public waste photos on top of that model, calibrates it, and picks the UNSURE
threshold from data to keep wrong sorts under 1%. See **[reports/REPORT.md](reports/REPORT.md)**
for the measured accuracy against v1, and **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)** for a
plain-language tour of how the code works.

## Results

On 5,984 held-out test photos (never used for training, tuning or picking the threshold):

| | Zero-shot only (v1 approach) | v2 deployed |
| --- | --- | --- |
| 6-class accuracy | 87.8% | **98.6%** |
| Items put in the wrong compartment | 2.8% | **0.5%** |
| Sortable items sorted into the right compartment | 79.8% | **91.4%** |
| Wrong items in the ORGANIC bin | 9.4% | **0%** |
| Wrong items in the PLASTIC bin | 7.2% | **1.0%** |
| Wrong items in the METAL bin | 5.8% | **2.2%** |
| Wrong sorts on real rubbish it never trained on (RealWaste held out) | 5.7% | **1.8%** |
| Wrong sorts on soft plastics it never trained on (recycling-dataset held out) | 2.3% | **1.3%** |
| Wrong sorts on household photos it never trained on (TrashBusters held out) | 3.2% | **4.2%** |
| Time per decision (M5, fp16) | ~34 ms (1 frame) | ~135 ms (5-frame vote) |

The zero-shot column uses v2's prompts, which give glass its own class; v1 itself (glass lumped
into "other") made 5.7% wrong sorts on the previous version of this test set. The test set includes real rubbish
photographed at a recycling facility (RealWaste), which is harder than studio photos.

The TrashBusters row is the one place zero-shot does better: about two thirds of those wrong
sorts are items such as porcelain, gloves and masks that no other dataset shows, so a model
trained without TrashBusters has never seen them. The deployed model has.

Adding TrashBusters as a fifth, harder dataset changed the test set. The previous model scored
99.1% with 0.43% wrong sorts on a smaller one without those photos, so those numbers can't be
compared directly. Held-out datasets can: at the same sorting rate, training with TrashBusters made
fewer wrong sorts on three of the four original datasets and about the same on RealWaste.

Glass has its own class (it still goes to UNSURE: there is no glass compartment), so the model
learns the glass-vs-clear-plastic boundary instead of lumping glass in with clothes and shoes.
A second opinion from the zero-shot prompts is required when an item might be glass or organic,
or the model is under 95% sure. ORGANIC is the strictest bin: the second opinion always applies
and the model needs 98% confidence, because glass or plastic in food waste can hurt animals.

![Sorting more vs sorting wrong](reports/tradeoff.png)

These are web photos, so expect lower numbers from a real webcam; the full write-up and its
limitations are in [reports/REPORT.md](reports/REPORT.md).

Why the guard matters: on a photo of an open hand, an early trained head alone said *organic,
0.73*, because the training data has no hands. The zero-shot guard scored it as out of scope
(0.99) and it went to UNSURE instead.

## How it works

```
webcam crop ──► SigLIP2 vision encoder (frozen, open source) ──► 1152-d embedding
                                                                   │
                         ┌─────────────────────────────────────────┤
                         ▼                                         ▼
              trained head (this repo)                 zero-shot prompts (v1)
              plastic/paper/metal/organic/glass/other  "is it a hand, a face, ...?"
                         │                                         │
                         └──────────► route ◄──────────────────────┘
               UNSURE if the top class has no bin (glass, other), confidence is
               too low, the zero-shot guard says glass/other, or a needed second
               opinion disagrees; otherwise that compartment
```

- **Backbone**: `google/siglip2-so400m-patch16-256`, frozen. Only the head's weights are
  trained. With about 28k training images, a frozen strong encoder generalises better than fine-tuning
  400M parameters, and it keeps the zero-shot prompts working as a guard for things the datasets
  have no pictures of (hands, faces, an empty tray).
- **Second opinion**: zero-shot must pick the same class when the item might be glass or
  organic, or the head is under 95% sure (`double_check` in `config.yaml`). Checking only
  glass/organic sorted a little more but quadrupled wrong sorts on unseen datasets
  (`scripts/rule_ab.py`).
- **Head**: linear or one-hidden-layer MLP, picked by a small grid on validation macro-F1;
  class-balanced loss; label smoothing; temperature-calibrated softmax.
- **Data**: five openly licensed Hugging Face datasets (MIT / Apache-2.0 / CC BY 4.0), labels mapped to the
  six classes in `config.yaml`, near-duplicates merged before splitting so nothing leaks from
  train to test, 70/15/15 stratified split, two augmented views per training image.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and about 10 GB of disk for the model and datasets.

```bash
uv sync
```

The trained head is committed in `models/`, so `live` and `classify` work straight away. To
rebuild everything from the public data:

```bash
uv run binsight prepare    # download datasets, embed, de-duplicate, split (~35 min on an M5)
uv run binsight train      # train the head, calibrate, pick the threshold (~ a few minutes)
uv run binsight crosscheck # harder test: leave each dataset out in turn
uv run binsight evaluate   # held-out test results -> reports/REPORT.md
```

`prepare` caches its work in `data/` and can be re-run.

### Camera access

macOS only lets an app use the camera if you allow it. Allow the terminal app you run
`binsight` from in **System Settings → Privacy & Security → Camera**, then restart that app.
If a phone nearby takes over camera 0 (Continuity Camera), use `--camera 1`.

## Commands

```bash
uv run binsight live                  # webcam demo
uv run binsight classify a.jpg b.jpg  # scores and decision for image files
```

Options for both: `--threshold <0-1>` (default: the value `train` chose), `--zero-shot` (ignore
the trained head, i.e. v1 behaviour), `--model <hf id>` (or `--model fallback` for CLIP; a head
trained on a different backbone is ignored), `--config <path>`. `live` also takes
`--camera <index>`.

### `binsight live`

Hold an item inside the white square (the tray). With the tray empty at start, the app grabs a
background and checks the tray (tips appear top right if it is too dark, too busy or unevenly
lit). When the square differs from the background and the item stays still for about 10 frames,
it averages the last 5 frames into one decision, shows the result, slides a block to the chosen
compartment, and re-arms when you take the item away.

If it is unsure because the model is not confident or the two models disagree, it asks you to
**turn the item and hold it still** (6 seconds) and decides again from both looks. Items that
clearly have no bin (glass, clothes, batteries) go straight to UNSURE.

The display is mirrored; the classified crop is not, so label text stays readable. Frames are
kept in memory only; nothing is written to disk. Each decision is also printed in the terminal.

**Tray setup matters more than any setting**: the model learned from plain, evenly lit photos.
Use a plain light sheet or board, light it evenly without hard shadows, and take your hand out
of the square once the item is down.

| Key | Action |
| --- | --- |
| `SPACE` | classify the tray now |
| `B` | re-capture the empty-tray background (do this if lighting changes) |
| `+` / `-` | raise / lower the UNSURE threshold by 0.05 |
| `Q` / `Esc` | quit |

The hint under the tray shows how much of it differs from the background; near 0% means empty.

## Browser demo

[`space/`](space) is a small web app (Gradio) for trying BinSight from any browser or phone:
take or upload a photo of one item and it shows the compartment and, for UNSURE, why. It uses
the same classifier and rules as `binsight classify` (one photo, so no frame vote or second look).

```bash
uv run --group demo python space/app.py   # then open http://localhost:7860
```

To put it online for free as a [Hugging Face Space](https://huggingface.co/spaces):

1. Sign in at huggingface.co, click **New Space**, pick SDK **Gradio** and hardware
   **CPU basic (free)**.
2. In the Space's **Files** tab, upload everything inside `space/`: `app.py`,
   `requirements.txt`, `README.md` and the `examples` folder.
3. The first build takes about 10 minutes (it downloads the 4 GB SigLIP2 model). After that the
   Space link works for anyone, at roughly a second per photo.

The Space installs this repository at tag `v3`, which includes the trained head. To deploy a
newer version, move the tag or change `space/requirements.txt`. Free Spaces sleep after two days
without visits; the next visit wakes them in a few minutes, so open the link before showing it.

## Configuration

Everything is in [`config.yaml`](config.yaml): backbone, head path, prompts, threshold, guard,
second-opinion rule, per-compartment confidence bars, dataset label mapping, split and training
settings, ROI size, trigger sensitivity, frame vote and second look, tray-check limits and
colours.

## Development

```bash
uv run ruff format . && uv run ruff check .
```

## License and credits

BinSight's code and trained head (`models/`) are released under the [MIT License](LICENSE).

It builds on others' work, each under its own license:

| What | Used for | License |
| --- | --- | --- |
| [google/siglip2-so400m-patch16-256](https://huggingface.co/google/siglip2-so400m-patch16-256) | frozen image/text encoder, downloaded at first run (not redistributed here) | Apache-2.0 |
| [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | optional fallback encoder | MIT |
| [omasteam/waste-garbage-management-dataset](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset) | training and test images | MIT |
| [garythung/trashnet](https://huggingface.co/datasets/garythung/trashnet) | training and test images | MIT |
| [viola77data/recycling-dataset](https://huggingface.co/datasets/viola77data/recycling-dataset) | training and test images | Apache-2.0 |
| [Tatyr2210/waste-classification-vit](https://huggingface.co/datasets/Tatyr2210/waste-classification-vit), which appears to be [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) (Single, Iranmanesh & Raad, 2023) | training and test images (real rubbish at a recycling facility) | MIT (upload); RealWaste is CC BY 4.0 |
| [TrashBusters/combined](https://huggingface.co/datasets/TrashBusters/combined) | training and test images (includes real-world household photos) | MIT |

The datasets are downloaded by `binsight prepare` and are not included in this repository.
