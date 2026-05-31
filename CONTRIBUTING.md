# Contributing to Naseej

Thanks for your interest. Naseej is a research prototype for AI pathology
triage — **decision support, not a diagnostic device**. Please keep that framing
in any contribution (UI text, docs, model behaviour).

## Development setup

```bash
git clone https://github.com/kurdim12/naseej.git
cd naseej
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo,api]"        # editable install + dev tools
pre-commit install                       # optional: run lint/format on commit
```

## Run the checks

```bash
ruff check .            # lint
ruff format .           # format
pytest                  # full test suite (CPU, no dataset, no downloads)
```

The test suite runs entirely on CPU with synthetic fixtures — no dataset and no
network. Please keep new tests that way so CI stays fast and hermetic.

## Guidelines

- **Correctness first.** This is a medical-adjacent tool; a silent wrong triage
  decision is the worst failure. Add a regression test with any bug fix.
- **Keep binary the default.** Multi-class subtype grading is opt-in; don't
  break the binary benign/malignant path.
- **Route P(malignant) through `inference.malignant_probability`** — never index
  a hard-coded malignant column. This is what keeps binary and multi-class
  consistent.
- **Calibrated thresholds, not magic numbers.** New decision logic should read
  thresholds via `inference.load_thresholds`.
- **Match the surrounding style** — type hints, docstrings explaining *why*, and
  the existing naming. Run `ruff` before pushing.

## Reporting issues

Include: what you ran, what you expected, what happened, and your environment
(OS, Python, torch version). For suspected wrong triage behaviour, attach a
minimal reproducing input if possible (never real patient data).

## Ethics

No patient-identifiable data in the repo or issues. Any use of real patient
samples requires institutional review and informed consent. Naseej is **not** a
medical device.
