# Frontal Chest X-ray Cleaner

Keeps only frontal (PA/AP) chest X-rays from a folder and flags blurry images for review. Uses [cleanX](https://github.com/drcandacemakedamoore/cleanX) for quality checks. Originals are copied, never modified.

## Files

- `clean_frontal_cxr.py`: main script. Sorts images into frontal, lateral, duplicate or review, and writes `report.csv`.
- `quality_filter.py`: optional script. Flags the blurriest / lowest-contrast frontal images for review.
- `view_classifier.py`: frontal-vs-lateral detector, imported by the main script (required).
- `reviewed_labels.csv`: training labels (807 images checked by eye).
- `view_model.pkl`: the trained model.
- `requirements.txt`: Python packages.

## Setup the enviroment:

```bash
# macOS 
python3 -m venv cxr-env 
source cxr-env/bin/activate
pip install numpy pandas matplotlib scikit-learn opencv-contrib-python tesserocr   # or: pip install -r requirements.txt
git clone https://github.com/drcandacemakedamoore/cleanX.git
cd frontal_cxr_cleaner
```

## Run Cleaner:

```bash
# 1. Keep only frontal images (example paths; use your own)
python clean_frontal_cxr.py --input "/Users/rose/Desktop/Image Folder" --output cleaned --cleanx-path ~/Desktop/cleanX

# 2. (Optional) flag blurry / low-contrast images
python quality_filter.py --input cleaned/frontal --output quality_checked
```

Results in `cleaned/`: `frontal/` (the images you want), `excluded/` (lateral, duplicate), `review/` (rotated, uncertain, inverted), and `report.csv` (decision, reason and `p_frontal` confidence per image).

`quality_checked/` holds `keep/`, `review_low_quality/` and a contact sheet. Check the flagged images and delete only the truly unusable ones.

## How it works

A random forest classifies each image using left-right symmetry about the spine (frontals are symmetric, laterals aren't). Sideways frontals go to `review/rotated`. Then cleanX checks for duplicates, inverted and hazy images, using temporary JPG copies because cleanX only reads `.jpg`.
