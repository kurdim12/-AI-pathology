"""
One-command booth demo bootstrap.

Gets a judge from a fresh clone to a running, *meaningful* demo with a single
command — no dataset, no manual steps:

  1. generate a realistic synthetic dataset (if none present),
  2. train a quick model + fit temperature + calibrate thresholds (if no
     checkpoint present),
  3. bundle sample slides for the UI,
  4. launch the Gradio app.

Anything already present is reused, so re-running is fast. As always, the
synthetic data makes the numbers believable-but-meaningless — real data gives
real results.

Run::

    python -m scripts.demo
    python -m scripts.demo --epochs 8     # train a little longer
    python -m scripts.demo --no-launch    # set everything up but don't launch
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import config


def _run(label: str, args: list) -> None:
    print(f"\n[demo] {label}: {' '.join(args)}")
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command Naseej booth demo.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--no-launch", action="store_true",
                        help="prepare data/model/samples but don't start the app")
    parser.add_argument("--force", action="store_true",
                        help="regenerate data and retrain even if present")
    args = parser.parse_args()

    have_data = os.path.isdir(config.TRAIN_DIR) and os.listdir(config.TRAIN_DIR)
    if args.force or not have_data:
        _run("generate realistic synthetic data",
             ["-m", "scripts.make_demo_data", "--difficulty", "realistic",
              "--per-class", str(args.per_class)])
    else:
        print(f"[demo] reusing existing data at {config.TRAIN_DIR}")

    if args.force or not os.path.exists(config.BEST_MODEL_PATH):
        _run("train (recall-prioritised, + temperature scaling)",
             ["-m", "src.train", "--no-pretrained", "--backbone", "resnet18",
              "--epochs", str(args.epochs), "--batch-size", "64"])
        _run("calibrate triage thresholds (95% sensitivity floor)",
             ["-m", "src.calibrate", "--target-sensitivity", "0.95"])
    else:
        print(f"[demo] reusing existing checkpoint at {config.BEST_MODEL_PATH}")

    _run("bundle sample slides for the booth", ["-m", "scripts.make_samples"])

    print("\n[demo] setup complete.")
    if args.no_launch:
        print("[demo] --no-launch set; skipping app launch. Start it with: "
              "python -m app.app")
        return
    print("[demo] launching the Gradio app...\n")
    _run("launch app", ["-m", "app.app"])


if __name__ == "__main__":
    main()
