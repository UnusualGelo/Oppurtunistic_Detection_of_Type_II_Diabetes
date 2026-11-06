# RAD-DINO "all layers" image-only model (frozen RAD-DINO -> logistic regression)

What it is: logistic regression (L2, class_weight="balanced") on FROZEN RAD-DINO (microsoft/rad-dino) features from frontal chest X-rays.
RAD-DINO was not trained or changed. Each .joblib is a scikit-learn Pipeline (StandardScaler + logistic regression, best C chosen by 5-fold
stratified CV on the training images only). Seed 42, scikit-learn 1.6.1.

Features: for each image, the mean of the patch tokens taken at four depths of the network (layers 3, 6, 9 and 12, i.e. 25/50/75/100% of
depth; the last-layer one is the same patch-token average used in the earlier baseline), concatenated = 3072 numbers. Whole image used
(squashed to 518x518, no centre crop), ImageNet-style normalisation from the model's own processor. Extracted by extract_multi.py.
Output: predicted probability of T2D (call T2D at >= 0.5).

| File | Training images |
|---|---|
| models/model_run1_raddino_multilayer.joblib | 759 (all) |
| models/model_run2_raddino_multilayer.joblib | 716 (lowest-50 hazy images excluded) |

Scores (image_only_multilayer_results.csv). CV AUC = out-of-fold on training images; test = held-out test set, scored once, 0.5 cutoff for
accuracy/recall/precision/F1:
- Run 1: CV AUC 0.674, test AUC 0.641 [0.56, 0.722], accuracy 0.626, recall 0.565, precision 0.443, F1 0.496
- Run 2: CV AUC 0.671, test AUC 0.648 [0.563, 0.729], accuracy 0.634, recall 0.567, precision 0.453, F1 0.504
Earlier baseline for comparison (patch-token average, last layer only): Run 1 CV 0.667 / test 0.631.

predictions/ holds out-of-fold predictions for training images (oof_*) and test predictions (testpred_*).
raddino_multilayer_embeddings.npz holds the four 768-number layer features per image (same row order as master_table.csv).
To reproduce a score: python predict_example.py --model models/model_run1_raddino_multilayer.joblib --table master_table.csv --emb raddino_multilayer_embeddings.npz
To rebuild everything: python build_multilayer_package.py --root <folder with master_table.csv> --emb emb/raddino.npz

Honest summary: the improvement over the last-layer baseline is about +0.007 CV AUC, which is within noise (fold-to-fold spread is ~0.05 and the
test-set 95% CI is about +/-0.08). Four other pretrained encoders (DINOv2 base/large, BiomedCLIP, MedSigLIP) and partial fine-tuning of RAD-DINO
all landed at the same ~0.65-0.68 CV AUC. Research only; not for clinical use.
