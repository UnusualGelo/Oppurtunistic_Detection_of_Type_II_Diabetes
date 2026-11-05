"""
Extract frozen embeddings from several pretrained image models (run once, on a Colab GPU).

Colab:
  !pip install -q transformers open_clip_torch pandas pillow scikit-learn
  # MedSigLIP is gated: accept the terms at https://huggingface.co/google/medsiglip-448, then
  from huggingface_hub import login; login()          # paste a read token
  !python extract_multi.py --root /content/root --models raddino dinov2_base dinov2_large medsiglip biomedclip --out emb

--root = folder with master_table.csv and dataset/train, dataset/test (same as finetune_raddino.py).
Writes emb/<model>.npz with: ids + several feature matrices (keys listed at the end of each model).
Whole image is kept (squashed to a square, no centre crop). Features:
  dino-style models: CLS and mean-patch tokens from the last layer AND from 75% / 50% / 25% depth
                     (intermediate layers often carry more low-level / texture signal than the last one).
  medsiglip:         pooled image embedding, its mean patch token.
  biomedclip:        image embedding.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

SPECS = {
    "raddino":      dict(kind="dino",     id="microsoft/rad-dino",       size=518),
    "dinov2_base":  dict(kind="dino",     id="facebook/dinov2-base",     size=518),
    "dinov2_large": dict(kind="dino",     id="facebook/dinov2-large",    size=518),
    "medsiglip":    dict(kind="siglip",   id="google/medsiglip-448",     size=448),
    "biomedclip":   dict(kind="openclip", id="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", size=224),
}
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--models", nargs="+", default=list(SPECS))
ap.add_argument("--out", default="emb")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--size", type=int, default=0, help="override input size for every model (debug / speed)")
ap.add_argument("--ids", default="", help="debug: name=path_or_id,name2=... overrides model ids")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
for kv in filter(None, args.ids.split(",")):
    k, v = kv.split("="); SPECS[k]["id"] = v

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", dev, flush=True)
root = Path(args.root); out = Path(args.out); out.mkdir(exist_ok=True, parents=True)
m = pd.read_csv(root / "master_table.csv")
if args.limit: m = m.head(args.limit)
paths = [root / "dataset" / s / n for s, n in zip(m.split, m.image_name)]

def batches(size, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1); std = torch.tensor(std).view(1, 3, 1, 1)
    for i in range(0, len(paths), args.bs):
        ims = [np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)) for p in paths[i:i + args.bs]]
        x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float() / 255.0
        yield ((x - mean) / std).to(dev)

def run(name, spec):
    size = args.size or spec["size"]
    feats = {}
    def add(k, t): feats.setdefault(k, []).append(t.float().cpu().numpy())
    ac = lambda: torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda")
    with torch.no_grad():
        if spec["kind"] == "dino":
            from transformers import AutoImageProcessor, AutoModel
            pr = AutoImageProcessor.from_pretrained(spec["id"]); bb = AutoModel.from_pretrained(spec["id"]).to(dev).eval()
            for x in batches(size, pr.image_mean, pr.image_std):
                with ac(): o = bb(pixel_values=x, output_hidden_states=True)
                hs = o.hidden_states; L = len(hs) - 1
                for frac in (1, .75, .5, .25):
                    li = round(L * frac)
                    h = o.last_hidden_state if li == L else bb.layernorm(hs[li])
                    add(f"pm_L{li}", h[:, 1:].mean(1))
                    if frac == 1: add(f"cls_L{li}", h[:, 0])
        elif spec["kind"] == "siglip":
            from transformers import AutoImageProcessor, AutoModel
            pr = AutoImageProcessor.from_pretrained(spec["id"]); md = AutoModel.from_pretrained(spec["id"]).to(dev).eval()
            for x in batches(size, pr.image_mean, pr.image_std):
                with ac(): vo = md.vision_model(pixel_values=x)
                add("pooled", vo.pooler_output); add("mean_tokens", vo.last_hidden_state.mean(1))
        else:
            import open_clip
            md, _, pp = open_clip.create_model_and_transforms(spec["id"]); md = md.to(dev).eval()
            norm = [t for t in pp.transforms if t.__class__.__name__ == "Normalize"][0]
            for x in batches(size, list(norm.mean), list(norm.std)):
                with ac(): add("emb", md.encode_image(x))
    arrs = {k: np.concatenate(v) for k, v in feats.items()}
    np.savez(out / f"{name}.npz", ids=m.image_name.values, **arrs)
    print(f"{name}: saved {out / (name + '.npz')} keys={ {k: v.shape for k, v in arrs.items()} }", flush=True)

for name in args.models:
    try:
        run(name, SPECS[name])
    except Exception as e:                    # one gated / failing model should not kill the rest
        print(f"!! {name} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
