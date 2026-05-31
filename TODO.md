# Naseej — Project TODO / Roadmap

Status legend: ✅ done · 🔶 in progress · ⬜ not started · 🔴 blocked

## PHASE A — REAL DATA (top priority)
- 🔶 A1. Acquire a real histopathology dataset (BreaKHis / PCam / PathMNIST / other)
- ⬜ A2. Train on real data; fill the README Results table with real numbers
- ⬜ A3. Calibrate thresholds + run robustness benchmark on real data
- ⬜ A4. Record real metrics in MODEL_CARD.md

## PHASE B — TEST THE UNTESTED CORE
- ⬜ B1. Unit tests for src/train.py (validate(), temperature fit, two-phase, early stop)
- ⬜ B2. Unit tests for src/evaluate.py (compute_metrics, triage_breakdown, json export)
- ⬜ B3. Test the foundation-model loader (src/model.build_foundation_model, timm)

## PHASE C — PACKAGING & HYGIENE
- ⬜ C1. pyproject.toml (installable package + metadata)
- ⬜ C2. Pin dependency versions (upper bounds) in requirements + pyproject
- ⬜ C3. Console entry-points (naseej-train, naseej-triage, naseej-eval, ...)
- ⬜ C4. CONTRIBUTING.md, CHANGELOG.md
- ⬜ C5. Linting/format config (ruff) + pre-commit, wire into CI

## PHASE D — REAL-DATA READINESS (turnkey for the user's machine)
- ⬜ D1. Colab/Jupyter notebook: clone → data → train → eval → demo
- ⬜ D2. results-export script that auto-fills the README Results table
- ⬜ D3. Checkpoint resume / GPU-aware config polish

(Owner notes: the actual large-scale real training run needs the user's
hardware/internet; everything here makes that one run turnkey and trustworthy.)
