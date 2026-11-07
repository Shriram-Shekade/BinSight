"""BinSight browser demo: one photo in, a compartment (or UNSURE) out.

Runs the same classifier and routing rules as `binsight classify`, on one photo at a time (the
webcam app's frame vote and second look need a live camera).
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import gradio as gr
from PIL import Image

from binsight.config import UNSURE, load_config
from binsight.model import Classifier, Result

HERE = Path(__file__).parent
REPO = "https://github.com/Shriram-Shekade/BinSight"

cfg = load_config()
clf = Classifier(cfg)


def pct(p: float) -> str:
    return "over 99%" if p >= 0.995 else f"{p:.0%}"


def explain(result: Result) -> str:
    """The routing reason in plain words."""
    reason = result.reason
    if not reason:
        return f"The model is {pct(result.confidence)} sure and the safety checks pass."
    if reason == "top class is 'glass'":
        return "It looks like glass. There is no glass compartment, so a person sorts it."
    if reason == "top class is 'other'":
        return (
            "It doesn't look like plastic, paper, metal, food or glass "
            "(e.g. a battery, clothes, a hand or an empty tray)."
        )
    if reason.startswith("zero-shot guard"):
        return "The second model thinks it could be glass or something with no bin."
    if reason.startswith("zero-shot disagrees"):
        other = reason.split("'")[1]
        return (
            f"The model says {result.top}, but the second model says {other}. "
            "When they disagree, it plays safe."
        )
    if "needed for organic" in reason:
        return (
            f"ORGANIC needs 98% confidence (this was {result.confidence:.0%}), because glass "
            "or plastic in food waste can hurt animals."
        )
    if reason.startswith("confidence"):
        return f"The model is only {result.confidence:.0%} sure, which is not enough."
    return reason


def card(result: Result) -> str:
    colour = cfg.colours[result.compartment]
    text = "#1f2937" if result.compartment in ("PAPER", "METAL") else "#ffffff"
    title = "UNSURE: a person sorts it" if result.compartment == UNSURE else result.compartment
    return (
        f'<div style="background:{colour};color:{text};border-radius:12px;padding:20px 24px">'
        f'<div style="font-size:0.85rem;opacity:0.85">Goes to</div>'
        f'<div style="font-size:2rem;font-weight:700;line-height:1.2">{escape(title)}</div>'
        f'<div style="margin-top:8px">{escape(explain(result))}</div></div>'
    )


def sort(image: Image.Image | None):
    if image is None:
        return "", None, ""
    result = clf.classify(image.convert("RGB"))
    detail = f"Decided in {result.latency_ms:.0f} ms on {clf.backbone.device.type.upper()}."
    if result.reason:
        detail += f" Rule: `{result.reason}`."
    return card(result), result.scores, detail


INTRO = f"""
# BinSight
A campus smart bin that sorts rubbish into **PLASTIC**, **PAPER**, **METAL** or **ORGANIC**.
Anything it is not sure about goes to **UNSURE** for a person, because one wrong item spoils a
whole compartment. Take a photo of one item on a plain background, or try an example.
[Code and accuracy report]({REPO})
"""

HOW = f"""
- A frozen open-source image model,
  [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-256), turns the photo into
  1,152 numbers.
- A small classifier trained on about 26,000 public waste photos reads those numbers and picks
  plastic, paper, metal, organic, glass or other.
- A second opinion (the image model matching the photo against text descriptions) has to agree
  when the item might be glass or food, or the classifier is under 95% sure.
- Glass never goes in a bin: there is no glass compartment. ORGANIC needs 98% confidence.

On 3,945 test photos it never saw in training, 0.43% of items went to a wrong compartment and
nothing wrong reached ORGANIC. Real photos are harder than dataset photos, so expect more
UNSURE here. Details are in the [report]({REPO}/blob/main/reports/REPORT.md).

Photos are not stored by BinSight; Gradio's temporary upload files are deleted within about
20 minutes. Example photos: [TrashNet](https://github.com/garythung/trashnet) (MIT).
"""

EXAMPLES = [str(HERE / "examples" / f"{n}.jpg") for n in ("plastic", "paper", "metal", "glass")]

with gr.Blocks(title="BinSight", delete_cache=(600, 600)) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            photo = gr.Image(
                type="pil",
                sources=["webcam", "upload", "clipboard"],
                webcam_options=gr.WebcamOptions(mirror=False),
                label="Item",
                height=360,
            )
            button = gr.Button("Sort it", variant="primary")
            gr.Examples(EXAMPLES, inputs=photo)
        with gr.Column():
            decision = gr.HTML()
            scores = gr.Label(label="What the classifier sees", num_top_classes=6)
            detail = gr.Markdown()
    with gr.Accordion("How it decides", open=False):
        gr.Markdown(HOW)
    outputs = [decision, scores, detail]
    button.click(sort, photo, outputs)
    photo.change(sort, photo, outputs)

if __name__ == "__main__":
    demo.launch()
