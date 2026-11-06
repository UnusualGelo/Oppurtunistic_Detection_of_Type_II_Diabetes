# Which script does what (attempt 1)
1. extract_raddino_embeddings.py  -> raddino_embeddings.npz (CLS + patch-mean, 768 each) and master_table.csv
2. step3_baselines.py             -> models/*.joblib, predictions/, image_only_results.csv
3. predict_example.py             -> reload a .joblib and reproduce its test AUC
Models were pickled with scikit-learn 1.9.1 (see requirements.txt).

Note: step3_baselines.py also trains metadata (age/sex/BMI) models, so it needs the FULL master_table with clinical columns
(not in this repo). The stripped master_table.csv here is enough for predict_example.py with the embedding models.
