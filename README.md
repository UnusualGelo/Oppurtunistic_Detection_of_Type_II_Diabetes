# Opportunistic Detection of Type II Diabetes

Opportunistic screening for Type 2 diabetes (T2D) from routine frontal chest radiographs, with sociodemographic indicators to be added next.

Research use only. This is not a diagnostic tool.

---

## Main model: [`main_model/`](main_model/)

The model the team builds on is a **DenseNet-121 pretrained on chest X-rays** (TorchXRayVision `densenet121-res224-all`), fine-tuned to predict T2D from a single frontal chest X-ray.

| Evaluation (image only, 948 cleaned frontal CXRs) | ROC-AUC (95% CI) | AP | Sens / Spec @ 0.488 | F1 |
|---|---|---|---|---|
| 5-fold patient-grouped CV × 3 seeds (n = 758) | 0.624 (0.579–0.664) | 0.445 | 0.687 / 0.542 | 0.527 |
| Held-out test, scored once (n = 190, 55 T2D) | 0.703 (0.616–0.781) | 0.496 | 0.727 / 0.519 | 0.500 |

[`main_model/README.md`](main_model/README.md) covers:
- how to run it, locally or in Colab
- the fixed split in `main_model/splits/splits.csv`, which every new experiment must use
- how to add patient metadata (age, gender, ethnicity, SDI) without leaking test data

## Repository structure

```
├── main_model/                 DenseNet-121 T2D model (active development)
├── data_prep/
│   └── frontal_cxr_cleaner/    frontal vs lateral classifier and QC; produces the cleaned image set
└── archive/                    earlier team experiments, kept for reference (not maintained)
```

[`archive/README.md`](archive/README.md) lists each archived approach, who built it, and its reported results.
To see the repository exactly as it was before this reorganization, check out the tag `pre-reorg-2026-11`.

## Setup

```bash
pip install -r requirements.txt
```

## Data privacy and HIPAA compliance

To comply with HIPAA Safe Harbor and healthcare privacy standards:
- No raw medical radiographs (`.png`, `.jpg`, `.dcm`), binary embeddings (`.npy`) or patient-level identifiable health information (PHI) are tracked in this repository.
- All exclusion rules are maintained in `.gitignore`.
- Never commit images, clinical CSVs, model checkpoints or `results/` folders.
