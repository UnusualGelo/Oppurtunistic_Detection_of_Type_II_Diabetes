#!/usr/bin/env python3
"""
Automated cleaning of a chest X-ray folder so only frontal (PA/AP) views remain.

Stages
  1. View classification   frontal vs lateral (view_classifier.py, trained on reviewed_labels.csv)
  2. Rotation check        frontal films saved sideways / upside-down are routed to review
  3. cleanX quality checks run on the frontal set:
        - exact duplicates         (cleanX find_duplicated_images_todf)
        - perceptual duplicates    (cleanX cv2_phash_for_dupes)
        - inverted images          (cleanX histogram_difference_for_inverts_todf)
        - hazy / low-edge images   (cleanX find_very_hazy)   -> reported only
  4. Outputs: <out>/frontal, <out>/excluded/{lateral,duplicate}, <out>/review/{uncertain,rotated,inverted}
              plus <out>/report.csv
"""
import argparse, glob, os, shutil, sys, tempfile
import cv2, numpy as np, pandas as pd
import view_classifier as vc

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "view_model.pkl")
LABELS = os.path.join(HERE, "reviewed_labels.csv")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def retrain(image_dir):
    lab = pd.read_csv(LABELS)
    X = np.array([vc.features(cv2.imread(os.path.join(image_dir, f), 0)) for f in lab.file])
    m = vc.train(X, (lab.view == "frontal").astype(int).values); vc.save(m, MODEL)
    print(f"retrained on {len(lab)} reviewed images -> {MODEL}")

def _polarity(img):
    g = cv2.resize(img, (128, 128)).astype(float)
    return np.r_[g[30:90, 22:46].ravel(), g[30:90, 82:106].ravel()].mean() - g[30:90, 58:70].mean()

def _phash(img):
    d = cv2.dct(cv2.resize(img, (32, 32)).astype(np.float32))[:8, :8]
    return int("".join("1" if v else "0" for v in (d > np.median(d[1:])).ravel()), 2)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--cleanx-path", default=os.path.join(HERE, "..", "cleanX"),
                    help="path to the cloned cleanX repo (or omit if pip-installed)")
    ap.add_argument("--retrain", action="store_true", help="rebuild model from reviewed_labels.csv")
    ap.add_argument("--p-frontal", type=float, default=0.5, help="threshold for frontal (default 0.5)")
    a = ap.parse_args()

    if os.path.isdir(a.cleanx_path): sys.path.insert(0, os.path.abspath(a.cleanx_path))
    from cleanX.image_work import image_functions as cx   # needs tesserocr installed

    if a.retrain or not os.path.exists(MODEL): retrain(a.input)
    model = vc.load(MODEL)

    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith(EXTS))
    imgs = {f: cv2.imread(os.path.join(a.input, f), cv2.IMREAD_GRAYSCALE) for f in files}
    bad = [f for f, i in imgs.items() if i is None]
    files = [f for f in files if f not in bad]
    F = np.array([vc.features(imgs[f]) for f in files])
    p = model.predict_proba(F)[:, 1]
    rep = pd.DataFrame({"file": files, "p_frontal": p.round(4),
                        "lr_symmetry": F[:, 0].round(3), "ud_symmetry": F[:, 4].round(3)})
    rep["decision"] = ""; rep["reason"] = ""

    # 1+2: view and rotation
    for i, r in rep.iterrows():
        if r.p_frontal < a.p_frontal:
            rep.loc[i, ["decision", "reason"]] = ["excluded/lateral", "classified as lateral view"]
        elif r.ud_symmetry > 0.8 and r.ud_symmetry > r.lr_symmetry + 0.4:
            rep.loc[i, ["decision", "reason"]] = ["review/rotated", "top-bottom symmetric: film likely rotated 90 degrees"]
        elif r.p_frontal < 0.8:
            rep.loc[i, ["decision", "reason"]] = ["review/uncertain", "low classifier confidence"]
        else:
            rep.loc[i, "decision"] = "frontal"

    # 3: cleanX checks on the frontal candidates (cleanX globs *.jpg only -> temp JPG copies)
    cand = rep[rep.decision == "frontal"].file.tolist()
    with tempfile.TemporaryDirectory() as tmp:
        for f in cand:
            cv2.imwrite(os.path.join(tmp, os.path.splitext(f)[0] + ".jpg"), imgs[f], [cv2.IMWRITE_JPEG_QUALITY, 100])
        stem = {os.path.splitext(f)[0]: f for f in cand}
        def orig(path): return stem.get(os.path.splitext(os.path.basename(str(path)))[0])
        # exact duplicates (byte-identical *originals*)
        seen = {}
        for f in cand:
            h = open(os.path.join(a.input, f), "rb").read(); h = hash(h)
            if h in seen: rep.loc[rep.file == f, ["decision", "reason"]] = ["excluded/duplicate", f"identical to {seen[h]}"]
            else: seen[h] = f
        # perceptual duplicates (cleanX; falls back to a DCT pHash if cv2.img_hash is unavailable)
        try:
            try:
                d = cx.cv2_phash_for_dupes(tmp)
            except AttributeError:
                d = pd.DataFrame({"hash": [_phash(imgs[f]) for f in cand]},
                                 index=[os.path.join(tmp, os.path.splitext(f)[0] + ".jpg") for f in cand])
                d = d[d.hash.duplicated(keep=False)]
            if len(d):
                d = d.reset_index(); d.columns = ["path", "hash"]; d["file"] = d.path.map(orig)
                for _, grp in d.groupby("hash"):
                    keep, *rest = sorted(grp.file.dropna())
                    for f in rest:
                        m = rep.file == f
                        if (rep.loc[m, "decision"] == "frontal").all():
                            rep.loc[m, ["decision", "reason"]] = ["excluded/duplicate", f"perceptual duplicate of {keep}"]
        except Exception as e: print("phash check skipped:", e)
        # inverted: cleanX's histogram heuristic alone flagged ~25% of normal films here (bright abdomen /
        # white borders), and a lung-vs-mediastinum polarity score alone overlaps for arms/collimation.
        # An image is routed to review/inverted only when BOTH indicators agree (thresholds uncalibrated:
        # this dataset has no known inverted examples, so treat hits as "please eyeball").
        inv = cx.histogram_difference_for_inverts_todf(tmp)
        inv_files = {orig(x) for x in inv[inv.iloc[:, 1].astype(str).str.contains("invert", case=False)].iloc[:, 0]}
        rep["cleanx_invert_suspect"] = rep.file.isin(inv_files)
        rep["lung_minus_mediastinum"] = [round(_polarity(imgs[f]), 1) for f in rep.file]
        m = (rep.decision == "frontal") & rep.cleanx_invert_suspect & (rep.lung_minus_mediastinum > 25)
        rep.loc[m, ["decision", "reason"]] = ["review/inverted", "lungs brighter than mediastinum: likely inverted"]
        # hazy (informational)
        hz = cx.find_very_hazy(tmp); hz = hz[hz.label_for_haze == "hazy"]
        rep["hazy_flag"] = rep.file.isin([orig(x) for x in hz[0]])

    # 4: write outputs
    for sub in ["frontal", "excluded/lateral", "excluded/duplicate", "review/uncertain", "review/rotated", "review/inverted"]:
        os.makedirs(os.path.join(a.output, sub), exist_ok=True)
    for _, r in rep.iterrows():
        shutil.copy2(os.path.join(a.input, r.file), os.path.join(a.output, r.decision, r.file))
    rep.to_csv(os.path.join(a.output, "report.csv"), index=False)
    print(rep.decision.value_counts().to_string()); print("unreadable:", bad)
    print("hazy flagged (kept, informational):", int(rep[rep.decision == 'frontal'].hazy_flag.sum()))

if __name__ == "__main__":
    main()
