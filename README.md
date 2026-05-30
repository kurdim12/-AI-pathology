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
  (CTransPath, Phikon, UNI) for state-of-the-art features with little extra data
  — `build_foundation_model` in `src/model.py` is a real, optional `timm`-based
  loader (set `config.FOUNDATION_MODEL`); Grad-CAM finds the target layer
  automatically.

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
├── MODEL_CARD.md        ← intended use, data, metrics, limitations, ethics
├── Dockerfile           ← containerised demo
├── requirements.txt
├── Makefile             ← one-command workflows (make train / eval / test / ...)
├── config.py            ← all tunable settings
├── src/
│   ├── model.py         ← backbone + classification head (+ foundation-model loader)
│   ├── data.py          ← data loading + phone-capture augmentations
│   ├── gradcam.py       ← Grad-CAM (explainability)
│   ├── inference.py     ← predict + triage priority + heatmap (+ TTA, calibrated thresholds)
│   ├── triage.py        ← batch triage queue → prioritised worklist (CLI)
│   ├── train.py         ← training (recall-prioritised, two-phase, early stop)
│   ├── evaluate.py      ← sensitivity / specificity / AUC / confusion matrix + triage bands
│   ├── calibrate.py     ← data-driven thresholds (sensitivity floor; robust mode)
│   ├── temperature.py   ← temperature scaling (probability calibration)
│   ├── phone_sim.py     ← deterministic phone-capture degradation (for evaluation)
│   ├── robustness.py    ← benchmark: triage vs phone-capture degradation
│   └── report.py        ← bilingual (Arabic / English) triage reports
├── app/
│   ├── app.py           ← Gradio demo (single slide + triage-queue tabs)
│   └── api.py           ← REST API (FastAPI) for lab-workflow integration
├── scripts/
│   ├── get_data.py      ← dataset layout helper
│   ├── download_pcam.py ← automated public-dataset download (PatchCamelyon)
│   └── make_demo_data.py ← synthetic fixture so the pipeline runs with no downloads
├── tests/               ← CPU tests (no dataset / no downloads)
└── .github/workflows/   ← CI (tests + synthetic end-to-end on every push)
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

**Automated option (no access form):** PatchCamelyon downloads directly via
torchvision and is exported straight into the layout above —

```bash
python -m scripts.download_pcam --per-class 1500   # quick start
python -m scripts.download_pcam --full             # everything (large)
```

(Alternatives: BreaKHis as above, LC25000 — also public.)

### Try the whole pipeline with zero downloads

No dataset yet? Generate a tiny **synthetic** fixture (not real histopathology —
a sanity signal only) and run the full chain in seconds, on CPU:

```bash
python -m scripts.make_demo_data            # writes data/train/{Benign,Malignant}
python -m src.train --no-pretrained --epochs 2
python -m src.evaluate
python -m src.calibrate
python -m src.triage data/train --csv outputs/worklist.csv
```

## 9. Train

```bash
python -m src.train                          # ResNet-50, ImageNet weights
python -m src.train --backbone resnet18 --epochs 20 --lr 1e-4
python -m src.train --no-pretrained          # random init (offline)
```

Two-phase fine-tuning (head warm-up → full network), early stopping on
validation AUC, mixed precision on CUDA. Saves the best checkpoint (by AUC) to
`checkpoints/best_model.pt`, plus `outputs/history.json` and `train_summary.json`.

## 10. Evaluate & calibrate

```bash
python -m src.evaluate                        # metrics + triage-band breakdown
python -m src.calibrate                        # data-driven thresholds
python -m src.calibrate --target-sensitivity 0.98
```

`evaluate` reports **sensitivity (recall)** — the headline metric for a triage
tool — plus specificity, AUC, accuracy, the confusion matrix, and how the
triage bands bucket the validation set (it writes `outputs/metrics.json`).

`calibrate` is the operating-point step a real deployment needs. It picks the
`REVIEW` cut-off as the **highest threshold that still guarantees a target
malignant recall** (default 95%), and `URGENT` at the most-separating point
(Youden's J). Results are written to `outputs/thresholds.json` and picked up
automatically by inference, the triage CLI, the API, and the demo — so the
thresholds in `config.py` stop being guesses and start being measured.

### Measure phone-capture robustness

The project's central claim — *it works on phone photos, not just clean scans* —
is something to **measure, not assert**. This benchmark re-scores the validation
set under increasing, controlled phone-capture degradation (lighting, blur, JPEG,
rotation, colour cast) and reports how the triage-critical metrics — above all
the malignant recall still captured at the `REVIEW` threshold — hold up:

```bash
python -m src.robustness --severities 0,0.25,0.5,0.75,1.0
```

Writes `outputs/robustness.json` and a degradation curve to `outputs/robustness.png`.
A sharp drop in recall@REVIEW as severity rises is a sign the calibrated
threshold is too brittle for field conditions — exactly the kind of finding this
tool is meant to surface before deployment.

**Two fixes for that brittleness, both built in:**

- **Temperature scaling** (`src/temperature.py`): training automatically fits a
  scalar `T` on the validation set (Guo et al. 2017) and stores it in the
  checkpoint. It corrects over/under-confidence so probabilities mean what they
  say — without changing predictions (accuracy/AUC are untouched). Inference,
  evaluation, calibration, and triage all apply it transparently.
- **Robustness-aware calibration**: calibrate the thresholds *under* simulated
  phone degradation so the sensitivity floor holds in the field, not just on
  clean scans:

  ```bash
  python -m src.calibrate --target-sensitivity 0.95 --robust 0.5
  ```

## 11. Run the demo

```bash
python -m app.app
```

Open the printed local URL. Two tabs:
- **Single slide** — upload one image (or a phone photo of one) and see the
  label, confidence, triage badge, and Grad-CAM heatmap, live.
- **Triage queue** — upload many slides and get a prioritised worklist back,
  most-urgent first.

### Triage a whole queue (CLI)

The product is re-ordering the queue. Point Naseej at a folder of slides and get
a sorted worklist (URGENT → ROUTINE), written to CSV:

```bash
python -m src.triage path/to/folder --csv outputs/worklist.csv --save-overlays
python -m src.triage path/to/one_slide.png          # single slide
python -m src.triage path/to/one_slide.png --report bilingual   # Arabic + English
```

### Bilingual reports (Arabic / English)

For MENA-lab workflows, a single slide's triage result can be rendered as a
clean report in Arabic, English, or both — every report repeats, in both
languages, that this is decision support and the pathologist decides. Use
`--report {en,ar,bilingual}` on the CLI above, or `src.report.render_html` to
embed a report in a UI.

### REST API (lab-workflow integration)

A LIS or lab system can POST images and get structured triage JSON back — no UI:

```bash
pip install fastapi "uvicorn[standard]" python-multipart
uvicorn app.api:app --port 8000           # interactive docs at /docs
```

- `GET /health` — liveness + whether a trained model is loaded.
- `POST /predict` — one image (`file`) → `{label, prob_malignant, priority, …}`.
- `POST /triage` — many images (`files`) → worklist sorted most-urgent-first.

### Run with Docker

```bash
docker build -t naseej .
docker run -p 7860:7860 \
  -v "$PWD/checkpoints:/app/checkpoints" \   # mount a trained checkpoint
  naseej
```

### One-command workflows (Makefile)

```bash
make demo-data        # synthetic fixture
make train            # train (override: make train ARGS="--backbone resnet18")
make eval calibrate   # evaluate then calibrate thresholds
make test             # run the CPU test suite
make app              # launch the Gradio demo
make api              # launch the REST API
```

## 12. Results

> **Real-data results are still TODO** — fill the "real data" column after
> training on BreaKHis / PCam on a machine with a GPU and dataset access. Lead
> with sensitivity.

| Metric | Synthetic sanity run ⚠️ | Real data (BreaKHis/PCam) |
|---|---|---|
| Sensitivity (recall) | 1.000 | _TODO_ |
| Specificity | 1.000 | _TODO_ |
| AUC | 1.000 | _TODO_ |
| Accuracy | 1.000 | _TODO_ |

⚠️ **The synthetic column is a pipeline sanity check, not a pathology result.**
It comes from training ResNet-18 (random init, 6 epochs, early-stopped) on the
1,600-image fixture from `scripts/make_demo_data.py`, evaluated on a 320-image
held-out split. The signal there (benign≈bluish, malignant≈reddish) is trivially
separable, so perfect scores are *expected* and say nothing about real tissue.
They only confirm the train → evaluate → calibrate → triage machinery works
end-to-end. Reproduce with:

```bash
python -m scripts.make_demo_data --per-class 800
python -m src.train --no-pretrained --backbone resnet18 --epochs 10 --batch-size 64
python -m src.evaluate && python -m src.calibrate
```

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, training data, metrics,
limitations, and ethics in one place.

### Tests

CPU smoke tests run without any dataset or downloads (and in CI on every push):

```bash
python -m pytest tests/ -q
```

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
