# DenseNet-121 (TorchXRayVision) for Type 2 diabetes detection from chest X-rays

The team's best image-only model. It's a DenseNet-121 pretrained on chest X-rays
(TorchXRayVision `densenet121-res224-all`) and fine-tuned to predict T2D from a single
frontal chest radiograph.

Research use only. It is not a diagnostic tool.

## Results (948 cleaned frontal CXRs, image only)

| Evaluation | n (T2D) | ROC-AUC (95% CI) | AP | Sens / Spec @ 0.488 | F1 |
|---|---|---|---|---|---|
| 5-fold CV × 3 seeds (development) | 758 (252) | 0.624 (0.579–0.664) | 0.445 | 0.687 / 0.542 | 0.527 |
| **Held-out test, scored once** | 190 (55) | **0.703 (0.616–0.781)** | 0.496 | 0.727 / 0.519 | 0.500 |

For comparison, a CNN trained from scratch scored 0.584 ROC-AUC in cross-validation on the same split.

## Folder layout

```
angelo_densenet121/
├── densenet_t2d/                     # the code
│   ├── splits.py                     # patient-grouped, stratified test split + 5 CV folds
│   ├── data.py                       # loading, aspect-preserving resize/pad, train-only augmentation
│   ├── models.py                     # DenseNet-121 (TorchXRayVision) with a 1-output T2D head
│   ├── metrics.py                    # ROC-AUC, AP, sens/spec/F1, bootstrap CIs, paired AUC test
│   ├── train_cv.py                   # cross-validated training with early stopping
│   ├── compare.py                    # compare experiments on identical dev images
│   └── final_test.py                 # one-time scoring of the locked test set
├── configs/densenet121_xrv_final.yaml  # final hyperparameters
├── notebooks/DenseNet121_T2D_Final.ipynb  # Colab notebook with the final run and figures
└── splits/splits.csv                 # THE split. Every experiment must use this file.
```

## Model and training

- **Architecture:** DenseNet-121 feature layers from TorchXRayVision, global average pooling, dropout 0.3, then `Linear(1024, 1)` producing a logit.
- **Input:** 224×224 grayscale, aspect ratio kept by padding, scaled to [-1024, 1024] as TorchXRayVision expects.
- **Optimizer:** AdamW, with lr **3e-4** for the head and **3e-5** for the pretrained layers. Weight decay 1e-4, cosine schedule with 3 warm-up epochs, gradient clipping at 1.0.
- **Loss:** `BCEWithLogitsLoss` with `pos_weight` = negatives / positives in each fold (about 2.0).
- **Augmentation (training only):** rotation up to ±7°, shift up to 6%, scale up to ±8%, brightness and contrast up to ±15%. No horizontal flips.
- **Early stopping:** on validation loss of a patient-grouped inner split (15% of the training folds), patience 10, at most 40 epochs.
- **Threshold:** 0.488, chosen by Youden's index on out-of-fold dev predictions and frozen for the test set.

## How to run

Install the dependencies:

```bash
pip install -r requirements.txt
```

All commands below are run from `models/angelo_densenet121/`.

1. Put `splits/splits.csv` in place. It isn't regenerated, so everyone uses the same test patients.
2. Set `data.image_dir` in the config, or pass `--set data.image_dir=/path/to/images`.
3. Train and evaluate:

```bash
python -m densenet_t2d.train_cv --config configs/densenet121_xrv_final.yaml --set data.image_dir="/path/to/Image Folder"
python -m densenet_t2d.final_test --experiment densenet121_xrv_final     # only once, at the end
```

Results go to `results/experiments/<name>/`, which git ignores. Each experiment folder contains:
- `summary.json`
- `oof.csv`
- `history.csv` (per-epoch losses)
- `learning_curves.png`
- fold checkpoints

On Colab, use the notebook instead. It runs without any local files.

## Extending with patient metadata (age, gender, ethnicity, SDI)

The goal is to beat the image-only result honestly. Follow these rules:

1. **Keep `splits/splits.csv` fixed.** Train and select only on `split == "dev"`. Don't touch `split == "test"` until the final model is chosen.
2. **Fit any preprocessing inside each fold.** This includes scalers, imputation for missing SDI and encoders, which should be fit on the training folds only and never on all data.
3. **Compare against the image-only model on the same dev images:**

   ```bash
   python -m densenet_t2d.compare --baseline densenet121_xrv_final
   ```

   A gain counts only if the paired AUC-difference CI excludes 0.
4. **Run a metadata-only model too**, for example logistic regression on the same folds. If metadata alone performs as well as the fusion model, the X-ray isn't adding much.

Where to add things:

- **Late fusion (simplest).** Use the saved out-of-fold DenseNet probabilities (`oof.csv`) as one feature next to the metadata in a logistic regression or XGBoost model, trained fold by fold.
- **Embedding fusion.** Extract the 1,024-d pooled DenseNet features per image, from the fold model that didn't train on that image. Then train a tabular model on features plus metadata.
- **End-to-end fusion.** In `models.py`, add a model that concatenates the pooled image features with a small MLP over the metadata. In `data.py`, have the dataset also return the metadata tensor.

## Caveats

- The TorchXRayVision `all` weights were partly pretrained on Open-i, which these images appear to come from. Those weights never saw T2D labels. To check, swap in `densenet121-res224-chex`.
- With 55 test positives, the test AUC has a wide CI. Report the CV and test results together.
- Never commit images, clinical CSVs, or `results/`. The repo's data rules apply.

## References

- Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017.
- Cohen et al., *TorchXRayVision: A library of chest X-ray datasets and models*, MIDL 2022.
