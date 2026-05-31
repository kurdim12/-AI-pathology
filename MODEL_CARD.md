# Model Card — نسيج · Naseej

A living model card for the Naseej pathology-triage model. Fill the metric
placeholders after training on your data; the framing below is fixed by design.

## Model details
- **Task:** histopathology image classification with a triage layer
  (`URGENT` / `REVIEW` / `ROUTINE`) on top of P(malignant). Binary
  (benign vs malignant) by default; optional multi-class **subtype grading**
  that rolls up to the same benign/malignant triage decision.
- **Architecture:** transfer learning from an ImageNet-pretrained CNN
  (ResNet-50 default; ResNet-18 / EfficientNet-B0 selectable) with a new 2-class
  head. Optional upgrade path to an open pathology foundation model (CTransPath /
  Phikon / UNI) via `timm` (`src/model.py:build_foundation_model`).
- **Training:** recall-prioritised, class-weighted cross-entropy
  (`config.CLASS_WEIGHTS`), two-phase fine-tuning (head warm-up → full
  fine-tune), best checkpoint selected by validation AUC.
- **Explainability:** from-scratch Grad-CAM (`src/gradcam.py`).
- **Version:** prototype / research. **Not a medical device.**

## Intended use
- **Intended:** decision support that re-orders a pathologist's queue so
  likely-malignant slides are reviewed first, especially in labs without a
  whole-slide scanner (input may be a phone photo through a microscope).
- **Users:** pathologists and lab staff, under professional oversight.
- **Out of scope:** autonomous diagnosis; final clinical decisions; any use
  without a qualified pathologist signing off; non-histopathology images.

## Training data
- **Anchor:** BreaKHis (~7,900 breast histopathology images, benign vs
  malignant). Automated alternative: PatchCamelyon via
  `python -m scripts.download_pcam`. Also supported: LC25000.
- No patient-identifiable data is stored in this repository.

## Phone-capture robustness
Training augmentations simulate phone-through-microscope conditions — variable
lighting, defocus/motion blur, rotation, colour/white-balance shift, and
perspective skew (`src/data.py`). This is the key difference from scanner-only
tools and the main reason the model is expected to degrade more gracefully on
messy phone images.

## Metrics — real data (BreaKHis 400X, held-out split, n=338)
ResNet-18 trained from scratch (ImageNet weights unavailable in our
environment); single seeded split; lead with sensitivity.

| Metric | Value |
|---|---|
| Sensitivity (recall, malignant) | **0.973** |
| Specificity | 0.730 |
| AUC | 0.959 |
| Accuracy | 0.891 |
| Malignant recall @ calibrated REVIEW | 0.973 (sens 0.95 / spec 0.87 at the cut-off) |

Phone-capture robustness (simulated degradation): AUC degrades gracefully
0.959 → 0.795 from clean to worst severity while calibrated REVIEW recall stays
≥ 0.94. Produced by `python -m src.evaluate`, `src.calibrate`, `src.robustness`
(figures in `assets/results/`).

**Scope of these numbers:** the 400X subset only, one split, trained from
scratch on CPU — a credible proof the pipeline learns real histopathology, not a
multi-magnification / cross-validated / externally-validated clinical result.

## Limitations & risks
- **Domain shift:** trained on public (largely Western) data; performance on
  Jordanian/MENA staining and local phone cameras is **unvalidated**.
- **Phone images are harder** than scanner whole-slide images; augmentation
  mitigates but does not eliminate the gap.
- **Thresholds** must be **re-calibrated on local data** before any real use.
  `python -m src.calibrate` does this — it sets the `REVIEW` cut-off to the
  highest threshold that still guarantees a target malignant recall (default
  95%) and writes `outputs/thresholds.json`, which inference/triage/API/demo
  pick up automatically. The `config.py` defaults are only a starting point.
- **Probability calibration:** training fits a temperature scalar on validation
  (`src/temperature.py`) so confidences are meaningful and triage thresholds are
  less brittle. For deployment to phone-capture conditions, prefer
  robustness-aware thresholds (`src.calibrate --robust <severity>`), which hold
  the sensitivity floor under simulated field degradation.
- A false "ROUTINE" on a malignant slide is the most harmful error — hence the
  recall-prioritised loss and an explicit `REVIEW` band, but residual risk
  remains.

## Ethics
Any future use of real patient samples requires institutional review and
informed consent. Naseej is a research prototype and is **not** a medical device.
