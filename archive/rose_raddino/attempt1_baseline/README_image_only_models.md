# Rose's image-only models (RAD-DINO embeddings -> logistic regression)

What they are: logistic regression (L2, class_weight="balanced") trained on FROZEN RAD-DINO (microsoft/rad-dino) embeddings of
frontal chest X-rays. RAD-DINO itself was not trained or changed. Each .joblib file is a scikit-learn Pipeline (StandardScaler +
logistic regression, best C chosen by 5-fold stratified CV on the training images only). Seed 42, scikit-learn 1.9.1.
Input: ONE 768-number embedding per image. Output: predicted probability of T2D (call T2D at >= 0.5).

| File | Embedding used | Training images |
|---|---|---|
| models/model_run1_embedding_patchmean.joblib  (chosen model) | patch-token average | 759 (245 T2D) |
| models/model_run1_embedding_cls.joblib | CLS token | 759 (245 T2D) |
| models/model_run2_embedding_patchmean.joblib | patch-token average | 716 (228 T2D), lowest-50 hazy images excluded |
| models/model_run2_embedding_cls.joblib | CLS token | 716 (228 T2D), lowest-50 hazy images excluded |

Scores are in image_only_results.csv (CV AUC on training images; accuracy/recall/precision/F1/AUC on the held-out test set at a 0.5 cutoff).
Patch-token average was chosen because it had the higher CV AUC in both runs.

predictions/ holds out-of-fold predictions for the training images (oof_*) and test-set predictions (testpred_*).
To reproduce a score: python predict_example.py --model models/model_run1_embedding_patchmean.joblib --table master_table.csv --emb raddino_embeddings.npz
(needs raddino_embeddings.npz from Step 2). To retrain everything: python step3_baselines.py.

Limitations: research only. Test set has 60-62 T2D cases, so 95% CIs are about +/-0.08. Image-only test AUC is 0.60-0.63.
The clinical labels do not appear to be linked to the X-rays (embeddings cannot predict recorded age or sex).
