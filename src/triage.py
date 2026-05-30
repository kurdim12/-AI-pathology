"""
Naseej triage queue — turn a pile of slides into a prioritised worklist.

The product is not a single prediction; it is *re-ordering the queue* so the
likely-malignant cases reach a pathologist first. This module runs the model
over a folder of slide images and returns a worklist sorted by P(malignant),
most urgent first, so the bench works the dangerous cases before the routine
ones.

Command line::

    python -m src.triage path/to/image.png          # single slide
    python -m src.triage path/to/folder             # whole queue -> worklist
    python -m src.triage path/to/folder --save-overlays --csv outputs/worklist.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass
from glob import glob
from typing import List, Optional

from PIL import Image

import config
from src.inference import TriageResult, analyze, load_model

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Sort order for ties / grouping: most urgent priority first.
_PRIORITY_RANK = {"URGENT": 0, "REVIEW": 1, "ROUTINE": 2}


@dataclass
class WorklistItem:
    """One slide's place in the triage queue."""

    rank: int
    filename: str
    path: str
    label: str
    prob_malignant: float
    confidence: float
    priority: str


def list_images(folder: str) -> List[str]:
    """Return sorted image paths under ``folder`` (non-recursive then recursive)."""
    paths: List[str] = []
    for ext in IMAGE_EXTS:
        paths.extend(glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob(os.path.join(folder, f"*{ext.upper()}")))
        paths.extend(glob(os.path.join(folder, "**", f"*{ext}"), recursive=True))
        paths.extend(glob(os.path.join(folder, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(paths))


def build_worklist(
    folder: str,
    model=None,
    backbone: str = config.BACKBONE,
    trained: bool = True,
    save_overlays_to: Optional[str] = None,
) -> List[WorklistItem]:
    """Run the model over every image in ``folder`` and return a sorted worklist.

    Sorted by triage priority then by descending P(malignant), so the single
    most dangerous slide is rank 1.
    """
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"No images found under {folder!r} ({IMAGE_EXTS}).")
    return worklist_from_paths(
        paths, model=model, backbone=backbone, trained=trained,
        save_overlays_to=save_overlays_to,
    )


def worklist_from_paths(
    paths: List[str],
    model=None,
    backbone: str = config.BACKBONE,
    trained: bool = True,
    save_overlays_to: Optional[str] = None,
) -> List[WorklistItem]:
    """Score an explicit list of image paths and return a sorted worklist.

    Shared by the CLI (folder scan) and the demo (uploaded files).
    """
    if model is None:
        model, backbone, trained = load_model()

    if save_overlays_to:
        os.makedirs(save_overlays_to, exist_ok=True)

    scored: List[WorklistItem] = []
    for path in paths:
        try:
            image = Image.open(path)
        except Exception as exc:  # skip unreadable files, keep the queue moving
            print(f"[naseej] skipping {path}: {exc}")
            continue

        result: TriageResult = analyze(
            image,
            model=model,
            backbone=backbone,
            trained=trained,
            with_heatmap=bool(save_overlays_to),
        )

        if save_overlays_to and result.overlay is not None:
            stem = os.path.splitext(os.path.basename(path))[0]
            result.overlay.save(os.path.join(save_overlays_to, f"{stem}_gradcam.png"))

        scored.append(
            WorklistItem(
                rank=0,  # filled after sorting
                filename=os.path.basename(path),
                path=path,
                label=result.label,
                prob_malignant=result.prob_malignant,
                confidence=result.confidence,
                priority=result.priority,
            )
        )

    scored.sort(key=lambda it: (_PRIORITY_RANK[it.priority], -it.prob_malignant))
    for i, item in enumerate(scored, start=1):
        item.rank = i
    return scored


def write_worklist_csv(items: List[WorklistItem], csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(items[0]).keys()) if items else
                                ["rank", "filename", "path", "label", "prob_malignant", "confidence", "priority"])
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))


def print_worklist(items: List[WorklistItem]) -> None:
    counts = {p: 0 for p in _PRIORITY_RANK}
    for it in items:
        counts[it.priority] += 1

    print("\n===================== Naseej · triage worklist =====================")
    print(f"  {len(items)} slides   |   "
          f"🔴 URGENT {counts['URGENT']}   "
          f"🟠 REVIEW {counts['REVIEW']}   "
          f"🟢 ROUTINE {counts['ROUTINE']}")
    print("  --------------------------------------------------------------")
    print(f"  {'#':>3}  {'priority':<8}  {'P(malig)':>8}  {'label':<10}  file")
    for it in items:
        dot = {"URGENT": "🔴", "REVIEW": "🟠", "ROUTINE": "🟢"}[it.priority]
        print(f"  {it.rank:>3}  {dot}{it.priority:<7}  {it.prob_malignant:>7.1%}  "
              f"{it.label:<10}  {it.filename}")
    print("====================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Naseej triage: prioritise a queue of slides.")
    parser.add_argument("path", help="a single image file or a folder of slides")
    parser.add_argument("--csv", default=None, help="write the worklist to this CSV path")
    parser.add_argument("--save-overlays", action="store_true",
                        help="save a Grad-CAM overlay per slide")
    parser.add_argument("--overlay-dir", default=os.path.join(config.OUTPUT_DIR, "overlays"),
                        help="where to write overlays (with --save-overlays)")
    parser.add_argument("--report", choices=["en", "ar", "bilingual"], default=None,
                        help="for a single image, print a localised triage report")
    args = parser.parse_args()

    model, backbone, trained = load_model()
    if not trained:
        print("[naseej] warning: no trained checkpoint found — predictions are "
              "placeholders. Run `python -m src.train` first.")

    if os.path.isfile(args.path):
        result = analyze(Image.open(args.path), model=model, backbone=backbone, trained=trained)
        if args.report:
            from src.report import render_bilingual_text, render_text
            print("\n" + (render_bilingual_text(result) if args.report == "bilingual"
                          else render_text(result, args.report)) + "\n")
        else:
            dot = {"URGENT": "🔴", "REVIEW": "🟠", "ROUTINE": "🟢"}[result.priority]
            print(f"\n{dot} {result.priority}   {result.label}   "
                  f"P(malignant)={result.prob_malignant:.1%}   "
                  f"(confidence {result.confidence:.1%})\n")
        return

    if os.path.isdir(args.path):
        overlay_dir = args.overlay_dir if args.save_overlays else None
        items = build_worklist(args.path, model=model, backbone=backbone,
                               trained=trained, save_overlays_to=overlay_dir)
        print_worklist(items)
        csv_path = args.csv or os.path.join(config.OUTPUT_DIR, "worklist.csv")
        write_worklist_csv(items, csv_path)
        print(f"[naseej] worklist written -> {csv_path}")
        if overlay_dir:
            print(f"[naseej] overlays written -> {overlay_dir}")
        return

    raise FileNotFoundError(f"Not a file or folder: {args.path!r}")


if __name__ == "__main__":
    main()
