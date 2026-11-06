# Experiments (no saved models)
- eval_embeddings.py: compares frozen embeddings from several encoders (needs emb/<model>.npz from attempt2_multilayer/extract_multi.py and master_table.csv).
- finetune_raddino.py: partial fine-tuning of the last K RAD-DINO blocks. Saves predictions only, no weights. Needs the X-ray images.
Result: all landed at ~0.65-0.68 CV AUC, same as the frozen baseline.
