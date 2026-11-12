# DenseNet-121 predictions (final image-only model)

Use these as the X-ray input for any fusion or metadata model. Don't retrain the DenseNet.

| File | Contents |
|---|---|
| `oof.csv` | 758 dev images. `prob` is out-of-fold: each score comes from a fold model that never saw that image. Columns: image_name, patient_id, label, fold, prob |
| `test_predictions.csv` | 190 locked test images, scored by the ensemble of fold models |
| `summary.json` | CV results: ROC-AUC 0.624, frozen threshold 0.488 |
| `test_report.json` | Test results: ROC-AUC 0.703 |
| `config.yaml` | Exact training settings |
| `history.csv` | Per-epoch training and validation losses |

## Rules
- Train and cross-validate only on `oof.csv`, using its `fold` column (identical to `splits/splits.csv`), so there's no leakage.
- Don't use `test_predictions.csv` for training, tuning or choosing a model. Use it once, for the final evaluation of a pre-specified model.

## Example
    python -m densenet_t2d.fusion_xgb --clinical "/path/Clinical_data_with_SDI.csv" \
        --image-exp predictions/densenet121_final --features full
