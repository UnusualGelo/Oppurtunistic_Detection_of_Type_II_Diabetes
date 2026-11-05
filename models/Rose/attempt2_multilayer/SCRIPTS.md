# Which script does what (attempt 2)
1. extract_multi.py                -> emb/raddino.npz (pm_L3, pm_L6, pm_L9, pm_L12; run with --models raddino)
   (a copy of the result is raddino_multilayer_embeddings.npz in this folder; use --emb raddino_multilayer_embeddings.npz)
2. build_multilayer_package.py     -> models/*.joblib, predictions/, image_only_multilayer_results.csv, README_multilayer_model.md
3. predict_example.py              -> reload a .joblib and reproduce its test AUC
4. training_curves.py              -> loss / ROC / learning-curve figures (optional)
Models were pickled with scikit-learn 1.6.1, so use the pinned version in requirements.txt.
Note: scripts expect --root to hold master_table.csv; here it is in this folder, so use --root .
