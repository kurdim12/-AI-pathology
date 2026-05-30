"""
Naseej — Gradio demo (the booth interface).

A judge uploads a slide image (scanner output *or* a phone photo of one) and
watches the system flag it live: predicted label, confidence, a colour-coded
triage badge, and a Grad-CAM heatmap showing the tissue that drove the call.

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
)

# Load the model once at startup (falls back to a pretrained backbone if no
# checkpoint exists yet, so the booth always runs).
MODEL, BACKBONE, TRAINED = load_model()

PRIORITY_STYLE = {
    PRIORITY_URGENT: ("#b00020", "🔴", "Likely malignant — move to the front of the queue."),
    PRIORITY_REVIEW: ("#c77700", "🟠", "Uncertain — a pathologist should review soon."),
    PRIORITY_ROUTINE: ("#1b7a3d", "🟢", "Likely benign — routine queue."),
}


def _badge_html(priority: str, prob_malignant: float) -> str:
    color, dot, blurb = PRIORITY_STYLE[priority]
    return (
        f"<div style='border-left:8px solid {color};padding:12px 16px;"
        f"background:rgba(0,0,0,0.03);border-radius:6px'>"
        f"<div style='font-size:22px;font-weight:700;color:{color}'>"
        f"{dot} {priority}</div>"
        f"<div style='margin-top:4px;color:#333'>P(malignant) = "
        f"<b>{prob_malignant:.1%}</b></div>"
        f"<div style='margin-top:4px;color:#555;font-size:14px'>{blurb}</div>"
        f"</div>"
    )


_DISCLAIMER = (
    "Decision support, not diagnosis. Naseej re-orders the queue and highlights "
    "regions of interest — the pathologist always makes the final call."
)


def run(image):
    if image is None:
        return "Please upload a slide image.", None, None

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    result = analyze(image, model=MODEL, backbone=BACKBONE, trained=TRAINED)

    summary = _badge_html(result.priority, result.prob_malignant)
    summary += (
        f"<div style='margin-top:10px;color:#333'>"
        f"Prediction: <b>{result.label}</b> "
        f"(confidence {result.confidence:.1%})</div>"
    )
    if not result.trained:
        summary += (
            "<div style='margin-top:10px;color:#b00020;font-size:13px'>"
            "⚠ No trained checkpoint found — running on an un-fine-tuned model. "
            "These predictions are <b>not</b> meaningful. Run "
            "<code>python -m src.train</code> to train, then reload.</div>"
        )
    summary += (
        f"<div style='margin-top:10px;color:#888;font-size:12px'>{_DISCLAIMER}</div>"
    )

    return summary, result.overlay, result.heatmap


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="نسيج · Naseej — AI Pathology Triage") as demo:
        gr.Markdown(
            "# نسيج · Naseej — AI Pathology Triage\n"
            "AI pathology triage for the labs that can't afford the scanner. "
            "Upload a biopsy slide image — a clean scan **or** a phone photo "
            "through a microscope — to get a malignancy estimate, a triage "
            "priority, and a Grad-CAM heatmap."
        )
        if not TRAINED:
            gr.Markdown(
                "> ⚠ **Demo mode:** no trained checkpoint found, so the model is "
                "running on ImageNet weights with an untrained head. Predictions "
                "are placeholders until you run `python -m src.train`."
            )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Slide image (scanner or phone photo)")
                btn = gr.Button("Analyse slide", variant="primary")
            with gr.Column(scale=1):
                out_summary = gr.HTML(label="Triage")
                with gr.Row():
                    out_overlay = gr.Image(label="Grad-CAM overlay")
                    out_heatmap = gr.Image(label="Heatmap")

        btn.click(run, inputs=inp, outputs=[out_summary, out_overlay, out_heatmap])
        inp.upload(run, inputs=inp, outputs=[out_summary, out_overlay, out_heatmap])

        gr.Markdown(
            f"*{_DISCLAIMER}* &nbsp;|&nbsp; backbone: `{BACKBONE}` &nbsp;|&nbsp; "
            f"thresholds — urgent ≥ {config.URGENT_THRESHOLD:.2f}, "
            f"review ≥ {config.REVIEW_THRESHOLD:.2f}"
        )
    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
