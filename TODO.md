# Naseej — Project TODO / Roadmap

Status legend: ✅ done · 🔶 in progress · ⬜ not started · 🔴 blocked

## PHASE A — REAL DATA (top priority) ✅ DONE
- ✅ A1. Acquired REAL BreaKHis 400X (1,693 images) via scripts/get_breakhis.py
        (public GitHub mirror — no account, fully turnkey)
- ✅ A2. Trained on real data; README Results table filled with real numbers
        (AUC 0.959, sensitivity 0.973 on 338 held-out images)
- ✅ A3. Calibrated thresholds (sens 0.95 / spec 0.87) + robustness benchmark on real data
- ✅ A4. Recorded real metrics in MODEL_CARD.md

## PHASE B — TEST THE UNTESTED CORE ✅ DONE
- ✅ B1. tests/test_train.py (validate binary+multiclass, temperature roundtrip, two-phase)
- ✅ B2. tests/test_evaluate.py (compute_metrics, triage_breakdown, collect_predictions)
- ✅ B3. tests/test_foundation.py (loader contract + real build when timm present)

## PHASE C — PACKAGING & HYGIENE ✅ DONE
- ✅ C1. pyproject.toml (installable package + metadata + classifiers)
- ✅ C2. Pinned dependency bounds in pyproject (core + optional extras)
- ✅ C3. Console entry-points (naseej-train/eval/calibrate/triage/robustness/demo)
- ✅ C4. CONTRIBUTING.md (CHANGELOG: optional, deferred)
- ✅ C5. ruff config + .pre-commit-config.yaml (CI wiring: see D)

## PHASE D — REAL-DATA READINESS (turnkey)
- ✅ D-core. scripts/get_breakhis.py makes the real run one command
- ⬜ D1. Colab/Jupyter notebook (nice-to-have; CLI path already turnkey)
- ⬜ D2. results-export script that auto-fills the README table
- ⬜ D3. Add ruff lint + a real-data smoke job to CI; CHANGELOG.md

Remaining (lower priority): D1/D2/D3 polish. Core project is now validated on
real data, tested, and installable.
