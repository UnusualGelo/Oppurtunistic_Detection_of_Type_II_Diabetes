# Opportunistic detection of Type 2 diabetes from frontal chest X-rays (image only)

Frozen RAD-DINO features -> logistic regression. 

| Folder | What | CV AUC | Test AUC (95% CI) |
|---|---|---|---|
| attempt1_baseline/ | Last-layer patch-token average (768 numbers) | 0.667 | 0.631 [0.549, 0.714] |
| attempt2_multilayer/ | Patch-token averages from layers 3, 6, 9, 12 (3072 numbers) | 0.674 | 0.641 [0.560, 0.722] |
| experiments/ | Other encoders (DINOv2, BiomedCLIP, MedSigLIP) and partial fine-tuning | ~0.65-0.68 | not saved as models |

The gain from attempt 1 to 2 (+0.007 CV AUC) is within noise. Test set has ~60 T2D cases, so 95% CIs are about +/-0.08.
Each attempt folder is self-contained (its own README, models, scripts, master_table.csv, requirements.txt).
The trained models are the .joblib files in each models/ folder. They take RAD-DINO features, not raw images.

Note: master_table.csv here is stripped to image_name, T2D, split, exclude_lowest50 (no clinical columns).
The full table with age/sex/race/zip/BMI is deliberately NOT in the repo 
