# Archive

These are earlier experiments, kept for reference and credit.

- **Not maintained:** the code is kept as it was, and some scripts use relative paths that no longer resolve after the move into `archive/`. To run anything in its original layout, check out the tag `pre-reorg-2026-11`.
- **Results aren't directly comparable:** each project used its own data split, image set and evaluation protocol. The numbers below are copied from each folder's own README or results files, as reported by their authors, and have not been re-verified. To compare an archived approach with the main model fairly, re-run it on `main_model/splits/splits.csv`.

| Folder | Author(s) | Approach | Reported result | Where to look |
|---|---|---|---|---|
| [`angelo_cnn/`](angelo_cnn/) | Angelo | From-scratch CNNs (3-block baseline, deeper, residual, pooling and class-weight variants) | Val ROC-AUC 0.59–0.62 (single 144-image validation split) | `README.md`, `Documentation.md` |
| [`malek_and_angelo/`](malek_and_angelo/) | Malek & Angelo | Data cleaning and QC report, Table 1, CNN design spec, RAD-DINO view classifier | 949 frontal images retained (51 lateral excluded) | `README.md`, `CNN_MODEL_DESIGN_SPEC.md` |
| [`malek_and_angelo/malek/`](malek_and_angelo/malek/) | Malek | From-scratch CNN and Enhanced Residual CXRNet (image only); frozen RAD-DINO linear probe; multimodal fusion of X-ray score with age, sex, BMI and SDI | Image only: ROC-AUC 0.60–0.65. Clinical only (LR): 0.69. Fusion: 0.71 (LR) to 0.91 (gradient boosting) | `report.md`, `*_comparison.csv` |
| [`rose_raddino/`](rose_raddino/) | Rose | Frozen RAD-DINO features + logistic regression; other encoders (DINOv2, BiomedCLIP, MedSigLIP) and partial fine-tuning | CV ROC-AUC 0.667–0.674; test 0.631–0.641 (own split, ~60 T2D) | `README.md`, each `attempt*/` folder |

## Lessons carried into the main model

- **A single small validation set is unreliable for choosing models.** The residual CNN's best validation ROC-AUC of 0.69 became 0.60 on test. The main model uses patient-grouped CV, a separate early-stopping split, and a test set scored once.
- **From-scratch CNNs on ~950 images underfit or memorize.** Chest-X-ray pretraining (TorchXRayVision) gave the clearest image-only gain under a common protocol.
- **Frozen foundation-model features are a strong, cheap baseline** (Rose's RAD-DINO results). Comparing them against the main model on the shared split is an open item.
- **Clinical variables carry substantial signal on their own** (Malek's clinical-only ROC-AUC 0.69). Any fusion model should be compared against both the image-only and the metadata-only models.
