# Model Card — نسيج · Naseej

A living model card for the Naseej pathology-triage model. Fill the metric
placeholders after training on your data; the framing below is fixed by design.

## Model details
- **Task:** binary histopathology image classification (benign vs malignant)
  with a triage layer (`URGENT` / `REVIEW` / `ROUTINE`) on top of P(malignant).
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

## Metrics (fill after training on a held-out split — lead with sensitivity)
| Metric | Value |
|---|---|
| Sensitivity (recall, malignant) | _TODO_ |
| Specificity | _TODO_ |
| AUC | _TODO_ |
| Accuracy | _TODO_ |

Produced by `python -m src.evaluate` (also writes a confusion matrix to
`outputs/`).

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
- A false "ROUTINE" on a malignant slide is the most harmful error — hence the
  recall-prioritised loss and an explicit `REVIEW` band, but residual risk
  remains.

## Ethics
Any future use of real patient samples requires institutional review and
informed consent. Naseej is a research prototype and is **not** a medical device.
