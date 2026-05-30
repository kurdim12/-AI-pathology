"""
Naseej — Gradio demo (the booth interface).

Two tabs:
  * **Single slide** — upload one image (scanner output *or* a phone photo of a
    microscope field) and see the label, confidence, a colour-coded triage
    badge, and a Grad-CAM heatmap.
  * **Triage queue** — upload many slides at once and get back a prioritised
    worklist (URGENT first): the actual product, re-ordering the queue so the
    dangerous cases reach a pathologist first.

Run::

    python -m app.app
"""

from __future__ import annotations

import gradio as gr
from PIL import Image

import config
from src.inference import (
    PRIORITY_REVIEW,
    PRIORITY_ROUTINE,
    PRIORITY_URGENT,
    analyze,
    load_model,
    load_thresholds,
)
from src.triage import worklist_from_paths

# Load the model once at startup (falls back to a pretrained backbone if no
# checkpoint exists yet, so the booth always runs).
MODEL, BACKBONE, TRAINED = load_model()

# Bundled sample slides for the booth: prefer a checked-in assets/samples/ dir;
# otherwise fall back to a few images from the local dataset if present.
import glob as _glob
import os as _os


def _sample_images(max_n: int = 6):
    roots = [
        _os.path.join(config.ROOT, "assets", "samples"),
        config.TRAIN_DIR,
    ]
    found = []
    for root in roots:
        if _os.path.isdir(root):
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                found += _glob.glob(_os.path.join(root, "**", ext), recursive=True)
        if found:
            break
    return sorted(found)[:max_n]

PRIORITY_STYLE = {
    PRIORITY_URGENT: ("#b00020", "🔴", "Likely malignant — move to the front of the queue."),
    PRIORITY_REVIEW: ("#c77700", "🟠", "Uncertain — a pathologist should review soon."),
    PRIORITY_ROUTINE: ("#1b7a3d", "🟢", "Likely benign — routine queue."),
}

_DISCLAIMER = (
    "Decision support, not diagnosis. Naseej re-orders the queue and highlights "
    "regions of interest — the pathologist always makes the final call."
)


def _badge_html(priority: str, prob_malignant: float) -> str:
    color, dot, blurb = PRIORITY_STYLE[priority]
    prob_str = "—" if prob_malignant != prob_malignant else f"{prob_malignant:.1%}"  # NaN-safe
    return (
        f"<div style='border-left:8px solid {color};padding:12px 16px;"
        f"background:rgba(0,0,0,0.03);border-radius:6px'>"
        f"<div style='font-size:22px;font-weight:700;color:{color}'>{dot} {priority}</div>"
        f"<div style='margin-top:4px;color:#333'>P(malignant) = "
        f"<b>{prob_str}</b></div>"
        f"<div style='margin-top:4px;color:#555;font-size:14px'>{blurb}</div>"
        f"</div>"
    )


def _untrained_note() -> str:
    return (
        "<div style='margin-top:10px;color:#b00020;font-size:13px'>"
        "⚠ No trained checkpoint found — running on an un-fine-tuned model. "
        "These predictions are <b>not</b> meaningful. Run "
        "<code>python -m src.train</code> to train, then reload.</div>"
    )


def run_single(image, check_quality=True):
    if image is None:
        return "Please upload a slide image.", None, None

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    result = analyze(image, model=MODEL, backbone=BACKBONE, trained=TRAINED,
                     check_quality=check_quality)

    # Quality-rejected capture: explain and ask for a re-capture, no overlay.
    if not result.quality_ok:
        summary = (
            "<div style='border-left:8px solid #c77700;padding:12px 16px;"
            "background:rgba(0,0,0,0.03);border-radius:6px'>"
            "<div style='font-size:20px;font-weight:700;color:#c77700'>⚠ Image not usable</div>"
            f"<div style='margin-top:6px;color:#555'>{result.quality_reason}</div>"
            "<div style='margin-top:6px;color:#555;font-size:14px'>Re-capture the "
            "field (more tissue in frame, refocus) and try again.</div></div>"
            f"<div style='margin-top:10px;color:#888;font-size:12px'>{_DISCLAIMER}</div>"
        )
        return summary, None, None

    summary = _badge_html(result.priority, result.prob_malignant)
    summary += (
        f"<div style='margin-top:10px;color:#333'>Prediction: <b>{result.label}</b> "
        f"(confidence {result.confidence:.1%})</div>"
    )
    if result.uncertain:
        summary += (
            "<div style='margin-top:8px;color:#c77700;font-size:13px'>"
            "⚖ Low-confidence call (probability near 50%) — flagged for "
            "mandatory human review.</div>"
        )
    if not result.trained:
        summary += _untrained_note()
    summary += f"<div style='margin-top:10px;color:#888;font-size:12px'>{_DISCLAIMER}</div>"

    return summary, result.overlay, result.heatmap


def run_queue(files):
    if not files:
        return "Upload one or more slide images to build a worklist.", []

    paths = [getattr(f, "name", f) for f in files]
    items = worklist_from_paths(paths, model=MODEL, backbone=BACKBONE, trained=TRAINED)

    counts = {PRIORITY_URGENT: 0, PRIORITY_REVIEW: 0, PRIORITY_ROUTINE: 0}
    for it in items:
        counts[it.priority] += 1

    summary = (
        f"<div style='font-size:16px'><b>{len(items)} slides triaged</b> &nbsp; "
        f"<span style='color:#b00020'>🔴 {counts[PRIORITY_URGENT]} urgent</span> &nbsp; "
        f"<span style='color:#c77700'>🟠 {counts[PRIORITY_REVIEW]} review</span> &nbsp; "
        f"<span style='color:#1b7a3d'>🟢 {counts[PRIORITY_ROUTINE]} routine</span></div>"
    )
    if not TRAINED:
        summary += _untrained_note()

    rows = [
        [it.rank, it.priority, f"{it.prob_malignant:.1%}", it.label, it.filename]
        for it in items
    ]
    return summary, rows


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="نسيج · Naseej — AI Pathology Triage") as demo:
        gr.Markdown(
            "# نسيج · Naseej — AI Pathology Triage\n"
            "AI pathology triage for the labs that can't afford the scanner. "
            "Upload biopsy slide images — clean scans **or** phone photos through "
            "a microscope — to get a malignancy estimate, a triage priority, and "
            "a Grad-CAM heatmap."
        )
        if not TRAINED:
            gr.Markdown(
                "> ⚠ **Demo mode:** no trained checkpoint found, so the model is "
                "running on ImageNet weights with an untrained head. Predictions "
                "are placeholders until you run `python -m src.train`."
            )

        with gr.Tab("Single slide"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(type="pil", label="Slide image (scanner or phone photo)")
                    quality_chk = gr.Checkbox(
                        value=True,
                        label="Reject unusable captures (blank / out-of-focus) before analysing",
                    )
                    btn = gr.Button("Analyse slide", variant="primary")
                    _samples = _sample_images()
                    if _samples:
                        gr.Examples(examples=_samples, inputs=inp,
                                    label="Sample slides (click to load)")
                with gr.Column(scale=1):
                    out_summary = gr.HTML(label="Triage")
                    with gr.Row():
                        out_overlay = gr.Image(label="Grad-CAM overlay")
                        out_heatmap = gr.Image(label="Heatmap")
            btn.click(run_single, inputs=[inp, quality_chk],
                      outputs=[out_summary, out_overlay, out_heatmap])

        with gr.Tab("Triage queue"):
            gr.Markdown(
                "Upload a batch of slides and Naseej returns a **prioritised "
                "worklist** — the most likely-malignant case is rank 1."
            )
            q_files = gr.File(
                file_count="multiple", type="filepath",
                label="Slide images (upload many)",
            )
            q_btn = gr.Button("Build worklist", variant="primary")
            q_summary = gr.HTML()
            q_table = gr.Dataframe(
                headers=["#", "priority", "P(malignant)", "label", "file"],
                datatype=["number", "str", "str", "str", "str"],
                label="Worklist (urgent first)",
                wrap=True,
            )
            q_btn.click(run_queue, inputs=q_files, outputs=[q_summary, q_table])

        # Show the thresholds actually used for the badges (calibrated if a
        # thresholds.json exists, else the config defaults) so the footer never
        # disagrees with the decisions.
        urgent_t, review_t = load_thresholds()
        gr.Markdown(
            f"*{_DISCLAIMER}* &nbsp;|&nbsp; backbone: `{BACKBONE}` &nbsp;|&nbsp; "
            f"thresholds — urgent ≥ {urgent_t:.2f}, review ≥ {review_t:.2f}"
        )
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
