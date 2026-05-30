# نسيج · Naseej — AI Pathology Triage

**AI pathology triage for the labs that can't afford the scanner.**

Naseej is a software-only AI system that analyses a biopsy slide image — including
a photo taken through an ordinary microscope with a phone — and flags suspicious
tissue, assigning each case a **triage priority** so the likely-malignant slides
reach a pathologist first. It is built for the reality of Jordanian and MENA
labs, where the expensive whole-slide scanners that commercial pathology-AI
requires simply do not exist.

> **Decision support, not diagnosis.** Naseej re-orders the queue and highlights
> regions of interest. The pathologist always makes the final call.

---

## 1. The problem

Cancer is diagnosed late in Jordan, and late diagnosis is what kills.

- Cancer is a leading cause of death in Jordan (~15% of all deaths), and incidence
  rose over 60% in 13 years.
- About **two-thirds of breast cancer patients in Jordan present at an advanced
  stage**, after the disease has already spread.
- Early, localised breast cancer has a **5-year survival above 99%** — yet Jordan's
  overall breast-cancer survival has historically sat near **59%**. The gap is
  largely *how late it is found*.
- Every diagnosis runs through a biopsy and a pathologist — but **only ~10% of
  Jordanian pathologists use digital pathology** in routine work, and **funding is
  the number-one barrier** to adopting the AI tools that could speed it up.

The bottleneck is the pathologist: too few of them, and a manual, first-in-first-out
workflow that buries the urgent malignant cases behind a pile of benign ones
(70–80% of biopsies turn out benign).

## 2. Why existing solutions don't reach here

State-of-the-art pathology AI (Paige, PathAI, Ibex) is excellent — and locked
behind a **whole-slide scanner that costs USD 50,000–300,000+**, plus IT, storage,
and training. That is exactly why most Jordanian labs are not digitised. The
budget players still sell a *scanner*. **Naseej eliminates the scanner.**

| | Commercial pathology AI | **Naseej** |
|---|---|---|
| Hardware required | $50k–300k whole-slide scanner | A phone + the microscope the lab already owns |
| Input | Pristine whole-slide image | Phone photo through the eyepiece |
| Market | Western, scanner-equipped labs | Labs the giants will never sell to |
| Our claim | More accurate | More *reachable* — good-enough triage that actually runs here |

We are honest about this: we are not trying to out-accuracy Paige. We make
computational pathology *possible* where it is currently impossible.

## 3. What Naseej does

1. Takes a biopsy slide image (scanner output **or** a phone photo).
2. Runs a fine-tuned CNN to estimate the probability of malignancy.
3. Produces a **Grad-CAM heatmap** showing the tissue regions that drove the call
   (explainability the pathologist can audit).
4. Assigns a **triage priority**: `URGENT` → `REVIEW` → `ROUTINE`, re-ordering the
   queue so dangerous cases surface first.

## 4. AI / technical approach

- **Paradigm:** supervised binary image classification (benign vs malignant) on
  histopathology, with a triage layer on top of the probability.
- **Model:** transfer learning from an ImageNet-pretrained CNN (ResNet-50 by
  default; ResNet-18 / EfficientNet-B0 selectable), with a new 2-class head,
  fine-tuned on histopathology.
- **Recall-prioritised training:** the loss weights the malignant class higher
  (a missed cancer is far costlier than a false alarm), pushing the model toward
  high sensitivity.
- **Phone-capture robustness *(our contribution)*:** training augmentations
  simulate real phone-through-microscope conditions — variable lighting, blur,
  rotation, colour shift — so the model holds up on messy phone images rather than
  only clean scanner output. This is the key difference from scanner-only tools.
- **Explainability:** a from-scratch Grad-CAM implementation (`src/gradcam.py`).
- **Upgrade path:** swap the backbone for an open pathology foundation model
  (CTransPath, Phikon, UNI) for state-of-the-art features with little extra data —
  see the stub in `src/model.py`.

**This is not an API wrapper.** The proprietary work is the fine-tuning, the
phone-robustness pipeline, and the triage logic — all in this repo.

## 5. System architecture

```
 Slide image (scanner OR phone photo)
        │
        ▼
 Preprocess + normalise  (src/data.py)
        │
        ▼
 Fine-tuned CNN backbone  (src/model.py)
        │
        ├──► P(malignant) ──► Triage priority  (src/inference.py)
        │
        └──► Grad-CAM heatmap  (src/gradcam.py)
        │
        ▼
 Result + heatmap + priority  →  Web demo (app/app.py)
```

## 6. Project structure

```
naseej/
├── README.md            ← this file
├── requirements.txt
├── config.py            ← all tunable settings
├── src/
│   ├── model.py         ← backbone + classification head (+ foundation-model stub)
│   ├── data.py          ← data loading + phone-capture augmentations
│   ├── gradcam.py       ← Grad-CAM (explainability)
│   ├── inference.py     ← predict + triage priority + heatmap overlay
│   ├── train.py         ← training (recall-prioritised)
│   └── evaluate.py      ← sensitivity / specificity / AUC / confusion matrix
├── app/
│   └── app.py           ← Gradio demo (the booth interface)
└── scripts/
    └── get_data.py      ← dataset layout helper
```

## 7. Setup

```bash
git clone https://github.com/kurdim12/naseej.git
cd naseej
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

## 8. Get the data

Anchor dataset: **BreaKHis** (~7,900 breast histopathology images, benign vs
malignant). Request access and download from the link in `scripts/get_data.py`,
then arrange as:

```
data/train/Benign/*.png
data/train/Malignant/*.png
```

Verify the layout:

```bash
python -m scripts.get_data
```

(Alternatives: PatchCamelyon, LC25000 — also public.)

## 9. Train

```bash
python -m src.train
```

Saves the best checkpoint (by validation AUC) to `checkpoints/best_model.pt`.

## 10. Evaluate

```bash
python -m src.evaluate
```

Reports **sensitivity (recall)** — the headline metric for a triage tool —
plus specificity, AUC, accuracy, and the confusion matrix.

## 11. Run the demo

```bash
python -m app.app
```

Open the printed local URL, upload a slide image, and see the label, confidence,
triage badge, and Grad-CAM heatmap. This is the booth demo: a judge can upload an
image (or a phone photo of one) and watch the system flag it live.

## 12. Results

> Fill in after training on your machine. Report on a held-out split:
> sensitivity, specificity, AUC, accuracy. Lead with sensitivity.

| Metric | Value |
|---|---|
| Sensitivity (recall) | _TODO_ |
| Specificity | _TODO_ |
| AUC | _TODO_ |
| Accuracy | _TODO_ |

## 13. Limitations & honest framing

- **Decision support, not a diagnostic device.** The pathologist signs off.
- **Not yet validated on Jordanian samples.** Trained on public (largely Western)
  data; domain shift to local staining and phone cameras is a real risk. Jordanian
  field validation is the explicit next step (see roadmap).
- **Phone images are harder** than scanner whole-slide images; the augmentation
  pipeline mitigates but does not eliminate this.
- Thresholds in `config.py` should be **re-calibrated on local data** before any
  real use.

## 14. Roadmap

1. Field validation on a small set of real Jordanian biopsy images (with a partner lab).
2. Swap to an open pathology foundation model (CTransPath / Phikon / UNI).
3. Multi-class grading beyond binary benign/malignant.
4. Arabic-language reporting and a lab-workflow integration.

## 15. Ethics

No patient-identifiable data is used in this repository. Any future use of real
patient samples requires institutional review and informed consent. Naseej is a
research prototype and is **not** a medical device.

## 16. Acknowledgements

Built on PyTorch and torchvision. Public datasets: BreaKHis, PatchCamelyon,
LC25000. Open pathology foundation models referenced: CTransPath, Phikon, UNI.

## License

MIT — see [LICENSE](LICENSE).
