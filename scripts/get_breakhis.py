"""
Download a real BreaKHis (400X) subset into Naseej's expected layout.

The official BreaKHis archive (web.inf.ufpr.br / Kaggle / Mendeley) is gated or
sits behind hosts that aren't always reachable. This script pulls a **real**
BreaKHis 400X subset that is mirrored, unpacked as PNGs, in a public GitHub
repository (``PerceptiLabs/breakhis-400x``) — so it downloads over plain HTTPS
with no account or API key — and arranges it as::

    data/train/Benign/*.png
    data/train/Malignant/*.png

These are genuine histopathology images (authentic BreaKHis ``SOB_*`` filenames),
~1,700 images at 400X magnification, with the real ~1:2 benign:malignant
imbalance. For the full multi-magnification dataset, request access from the
official source (see ``scripts/get_data.py``).

Run::

    python -m scripts.get_breakhis
    python -m scripts.get_breakhis --keep-zip   # keep the downloaded archive
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
import zipfile

import config

# Real BreaKHis 400X subset, mirrored as PNGs in a public GitHub repo.
ZIP_URL = "https://codeload.github.com/PerceptiLabs/breakhis-400x/zip/refs/heads/main"
# Path components inside the archive that indicate class membership.
CLASS_MAP = {"benign": "Benign", "malignant": "Malignant"}


def _download(url: str, dest: str) -> None:
    print(f"[naseej] downloading real BreaKHis 400X subset...\n  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out)
    print(f"[naseej] downloaded {os.path.getsize(dest) / 1e6:.0f} MB -> {dest}")


def _extract(zip_path: str) -> dict:
    counts = {v: 0 for v in CLASS_MAP.values()}
    for cls in CLASS_MAP.values():
        os.makedirs(os.path.join(config.TRAIN_DIR, cls), exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".png"):
                continue
            parts = {p.lower() for p in name.split("/")}
            cls = next((CLASS_MAP[k] for k in CLASS_MAP if k in parts), None)
            if cls is None:
                continue
            target = os.path.join(config.TRAIN_DIR, cls, os.path.basename(name))
            with z.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            counts[cls] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a real BreaKHis 400X subset.")
    parser.add_argument("--keep-zip", action="store_true", help="keep the downloaded archive")
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    zip_path = os.path.join(config.DATA_DIR, "_breakhis400x.zip")

    if not os.path.exists(zip_path):
        _download(ZIP_URL, zip_path)
    else:
        print(f"[naseej] reusing existing archive {zip_path}")

    counts = _extract(zip_path)
    total = sum(counts.values())
    if total == 0:
        raise SystemExit("[naseej] no images extracted — the mirror layout may have changed.")

    if not args.keep_zip:
        os.remove(zip_path)

    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"[naseej] extracted real BreaKHis 400X: {summary} "
          f"(total {total}) -> {config.TRAIN_DIR}")
    print("[naseej] verify with `python -m scripts.get_data`, then train with "
          "`python -m src.train --no-pretrained --backbone resnet18 --epochs 15`.")


if __name__ == "__main__":
    main()
