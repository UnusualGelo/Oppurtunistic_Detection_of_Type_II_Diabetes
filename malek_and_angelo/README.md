# Opportunistic Detection of Type 2 Diabetes: Data Cleaning, Quality Assurance, and Cohort Harmonization

**Authors:** Malek & Angelo  
**Course:** CS 396 — Artificial Intelligence in Healthcare  
**Repository:** [Opportunistic Detection of Type II Diabetes](https://github.com/UnusualGelo/Oppurtunistic_Detection_of_Type_II_Diabetes)  

---

## Executive Summary

Opportunistic screening leverages routine medical imaging acquired for standard clinical indications (such as chest radiography) to detect underdiagnosed systemic conditions like Type 2 Diabetes (T2D). Because thoracic radiographs capture cardiac silhouette morphology, mediastinal width, aortic elongation, pulmonary vascular redistribution, and subcutaneous adipose distribution, foundation vision models can extract latent metabolic biomarkers.

However, raw clinical archives and radiographic repositories present substantial data quality challenges, including non-standard anatomical projections (lateral vs. frontal views), motion blur, improper aspect ratios, sensor clipping, and missing social determinant indices. This report documents the rigorous quality control (QC), data cleaning, and clinical cohort harmonization conducted by **Malek & Angelo**, detailing every metric, decision boundary, clinical rationale, and reproducible procedure.

---

## Section 1: Overview of Repository & Pipeline Tools

### Pipeline Overview
- **Repository Module:** `malek_and_angelo/`
- **Focus:** Comprehensive automated quality control of chest radiographs, clinical metadata auditing, Social Deprivation Index (SDI) integration, baseline cohort stratification (Table 1), and vision foundation model preparation.
- **Data Protection Notice:** In strict accordance with **HIPAA Safe Harbor** guidelines, all patient health identifiers (PHI) and raw radiograph binary images are excluded from public version control. Quality control metrics, hashes, and aggregated demographic summaries are retained.

### High-Level Summary of Findings
| Metric / Cohort Category | Count / Value | Details |
| :--- | :--- | :--- |
| **Initial Raw Images Scanned** | 1,000 | Discovered across source directory |
| **Lateral Views Excluded** | 51 | Excluded; study requires frontal PA/AP projections |
| **Total Frontal Images Retained** | 949 | Base frontal cohort for permissive modeling |
| **Images Flagged for Severe Blur** | 266 | Laplacian variance $< 100$ |
| **Images Flagged for Aspect Ratio Distortion** | 1 | Aspect ratio $0.589$ (outside standard $0.70 - 1.40$ range) |
| **High-Confidence Strict QC Frontal Images** | 713 | Satisfies all sharpness, dynamic range, and view constraints |
| **Total Patients in Master Clinical Registry** | 1,000 | Demographic and biometric records |
| **Patients with Verified Complete SDI** | 960 | 40 records omitted due to unmatchable ZIP codes |
| **Clean Multimodal Aligned Cohort** | 713 | Complete overlap across strict frontal X-ray + complete clinical/SDI |

---

## Section 2: Six-Part Data Cleaning Report

### 1. Cleaning Process
The data cleaning and validation pipeline addressed two parallel, interconnected modalities:
1. **Clinical and Sociodemographic Data:**
   - Patient records were systematically audited for demographic consistency across age, biological sex, self-reported race, ethnicity, marital status, and body mass index (BMI).
   - Social Deprivation Index (SDI) metrics were linked using 5-digit patient residential ZIP codes.
   - ZIP code strings were standardized with zero-padding to resolve leading-zero truncation caused by numeric integer formatting.
   - Two clinical datasets were established: a complete master file ($N=1,000$) and an SDI-complete cohort ($N=960$).
   - Table 1 was constructed to summarize covariate distributions, standard deviations, and statistical significance testing between T2D and non-T2D cohorts.

2. **Chest Radiograph Image Quality Control:**
   - All 1,000 original PNG radiographs were validated for file integrity, SHA-256 consistency, and full 8-bit/16-bit decoding.
   - Images were classified into anatomical projections (frontal PA/AP vs. lateral). All 51 lateral images were segregated and excluded.
   - For every frontal image, objective image quality metrics were computed before any resizing or lossy transformations:
     - **Dynamic range clipping:** Proportion of pixels at extreme black ($\le 0.02$) and extreme white ($\ge 0.98$).
     - **Sharpness / Motion Blur:** Computed via the variance of the 2D Laplacian operator ($\sigma^2_{\Delta}$).
     - **Aspect Ratio:** Width-to-height ratio ($W/H$).
   - An authoritative audit manifest (`image_qc_manifest.csv`) was generated, recording exact metric values, individual pass/fail flags, and composite quality decisions.

---

### 2. Data Quality Issues Identified
- **Issue 1: Anatomic Projection Incompatibility (Lateral Views)**  
  51 radiographs were lateral chest views. Lateral views depict anatomical structures along the sagittal plane rather than the coronal plane, invalidating thoracic symmetry calculations and confounding cardiothoracic ratio measurements used in diabetes prediction.
- **Issue 2: Motion Artifact and Severe Blur**  
  266 frontal radiographs exhibited low high-frequency edge content (Laplacian variance $< 100$). These artifacts arise from patient respiratory motion, cardiac pulsation, or low diagnostic exposure, which blur fine soft-tissue and vascular boundaries.
- **Issue 3: Aspect Ratio Distortion**  
  1 image exhibited a heavily cropped aspect ratio ($0.589$), indicating partial lung field amputation or improper scanning crop.
- **Issue 4: Missing Social Determinants of Health (SDI)**  
  40 clinical records lacked valid SDI linkage because their ZIP codes could not be matched against census ZCTA tables or were unrecorded.
- **Issue 5: Geographic Leading Zero Truncation**  
  When ZIP codes were ingested as standard numeric integers, East Coast and Puerto Rico postal codes starting with zero (e.g., `01xxx`–`08xxx`) lost their leading zero, resulting in invalid 4-digit numbers that failed SDI matching.

---

### 3. Solutions Implemented
- **Segregation of Lateral Projections:**  
  All 51 lateral images were cataloged and excluded from downstream frontal modeling pipelines, leaving 949 verified frontal radiographs.
- **Multi-Tier Quality Manifesting:**  
  Rather than arbitrarily discarding all flagged images, the pipeline generated an authoritative manifest classifying images into:
  - *Base Frontal Cohort ($N=949$):* All frontal images, suitable for permissive data pipelines.
  - *Strict QC Cohort ($N=713$):* Frontal images strictly passing sharpness, dynamic range, and geometry criteria.
- **String Formatting & ZIP Code Sanitization:**  
  All ZIP codes were reformatted as 5-digit zero-padded strings (`zfill(5)`), recovering geographic matches.
- **Bifurcated Clinical Cohort Strategy:**  
  Both the full clinical dataset ($N=1,000$) and the complete SDI cohort ($N=960$) were preserved, allowing statistical evaluation of potential exclusion bias.
- **Table 1 Statistical Stratification:**  
  A clinical Table 1 was finalized to document baseline population features and quantify group differences using two-sample t-tests and Chi-squared tests.

---

### 4. Software, Libraries, and Tools
- **Language & Runtime:** Python 3.12 (CPython)
- **Data Engineering & Tabular Statistics:** `pandas`, `numpy`, `scipy.stats`
- **Computer Vision & Image Auditing:** OpenCV (`cv2`) for Laplacian kernel convolution, Pillow (`PIL`) for image decompression, checksum, and metadata decoding.
- **Deep Learning & Foundation Models:** PyTorch (`torch`, `torchvision`), Hugging Face `transformers` (`microsoft/rad-dino`), and Google Health CXR Foundation (`google/cxr-foundation`).
- **Version Control:** Git & GitHub (`Oppurtunistic_Detection_of_Type_II_Diabetes`).

---

### 5. Rationale & Clinical Principles
- **Anatomical Standardization:** Frontal projections (PA/AP) provide reproducible visualization of the cardiothoracic ratio, pulmonary vascular patterns, and mediastinal adiposity, which have established epidemiological associations with insulin resistance and metabolic syndrome.
- **Preservation of Raw Pixel Dynamics:** Measuring QC metrics on raw 8-bit/16-bit values prior to normalization prevents artificial contrast distortion from masking underexposure or saturation.
- **Leakage Prevention via Group Stratification:** Patients with multiple images must never appear in both training and test sets. Stratified group partitioning by `patient_id` ensures clinical validity.
- **Social Determinants of Health (SDI):** Type 2 Diabetes risk is strongly influenced by socioeconomic barriers, poverty, food insecurity, and neighborhood infrastructure. Retaining clean SDI metrics allows models to capture both biological and environmental risks.

---

### 6. Outcome
- **Clinical Cohort:** 1,000 total patients reviewed; 960 patients with verified SDI scores; Table 1 established.
- **Imaging Cohort:** 1,000 total radiographs audited; 51 lateral images isolated; 949 frontal radiographs retained; 713 confirmed as strictly meeting all high-sharpness and aspect ratio criteria.
- **Harmonized Dataset:** A clean multimodal dataset aligning patient clinical covariates, SDI indices, and frontal chest radiographs.

---

## Section 3: Table 1 — Baseline Characteristics by T2D Status

The following table summarizes the baseline clinical, demographic, and socioeconomic characteristics of the study cohort stratified by Type 2 Diabetes status ($N=1,000$):

| Characteristic | Overall ($N=1,000$) | No T2D ($N=683$) | T2D ($N=317$) | P-Value |
| :--- | :--- | :--- | :--- | :--- |
| **Age, years (mean $\pm$ SD)** | 53.6 $\pm$ 25.5 | 51.4 $\pm$ 25.8 | 58.4 $\pm$ 24.1 | **$< 0.001$** |
| **Sex, n (%)** | | | | **$< 0.001$** |
| &nbsp;&nbsp;&nbsp;&nbsp;Female | 595 (59.5%) | 453 (66.3%) | 142 (44.8%) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Male | 405 (40.5%) | 230 (33.7%) | 175 (55.2%) | |
| **Race, n (%)** | | | | 0.900 |
| &nbsp;&nbsp;&nbsp;&nbsp;Black or African American | 534 (53.4%) | 369 (54.0%) | 165 (52.1%) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Latino | 230 (23.0%) | 159 (23.3%) | 71 (22.4%) | |
| &nbsp;&nbsp;&nbsp;&nbsp;White | 140 (14.0%) | 95 (13.9%) | 45 (14.2%) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Asian | 33 (3.3%) | 20 (2.9%) | 13 (4.1%) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Native Hawaiian / Pacific Isl. | 30 (3.0%) | 18 (2.6%) | 12 (3.8%) | |
| &nbsp;&nbsp;&nbsp;&nbsp;American Indian / Alaska Native | 20 (2.0%) | 13 (1.9%) | 7 (2.2%) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Unknown / Not Reported | 13 (1.3%) | 9 (1.3%) | 4 (1.3%) | |
| **Ethnicity, n (%)** | | | | 0.299 |
| &nbsp;&nbsp;&nbsp;&nbsp;Hispanic | 231 (23.4%) | 161 (23.9%) | 70 (22.4%) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Non-Hispanic | 754 (76.5%) | 512 (76.1%) | 242 (77.3%) | |
| **Married, n (%)** | 500 (50.0%) | 326 (47.7%) | 174 (54.9%) | **0.041** |
| **BMI, $\text{kg/m}^2$ (mean $\pm$ SD)** | 30.8 $\pm$ 11.4 | 29.0 $\pm$ 11.1 | 34.5 $\pm$ 10.9 | **$< 0.001$** |
| **SDI Score (mean $\pm$ SD)** | 44.5 $\pm$ 31.0 | 44.7 $\pm$ 31.0 | 44.0 $\pm$ 30.9 | 0.717 |

*Values are mean (SD) for continuous variables and n (%) for categorical variables. P-values compare patients with and without Type 2 Diabetes via independent two-sample t-tests (continuous) and Pearson Chi-Square tests (categorical). Abbreviations: T2D, Type 2 Diabetes; BMI, Body Mass Index; SDI, Social Deprivation Index.*

### Key Clinical Observations from Table 1:
1. **Age:** Patients with T2D were significantly older ($58.4$ vs. $51.4$ years, $p < 0.001$).
2. **Sex:** Males represented a significantly higher proportion of the diabetic group ($55.2\%$ vs. $33.7\%$, $p < 0.001$).
3. **BMI:** Patients with T2D had significantly elevated BMI ($34.5$ vs. $29.0\text{ kg/m}^2$, $p < 0.001$), consistent with established metabolic pathophysiology.
4. **SDI:** Overall mean SDI was $44.5$, indicating a moderately deprived study catchment area, with similar distributions across diabetic and non-diabetic cohorts.

---

## Section 4: Quantitative Quality Control Criteria

| Quality Parameter | Mathematical Definition / Threshold | Failure Trigger | Clinical Consequence |
| :--- | :--- | :--- | :--- |
| **View Projection** | AI/Symmetry Classifier Confidence | Lateral classification ($p_{\text{frontal}} < 0.50$) | Excluded (non-comparable anatomy) |
| **Laplacian Sharpness** | $\text{Var}(\Delta I) = \frac{1}{N}\sum (\Delta I - \mu)^2$ | $\text{Var} < 100.0$ | Flagged for blur / motion artifact |
| **Black Level Clipping** | Fraction of pixels with intensity $\le 0.02$ | $> 20\%$ black pixels | Over-penetration or background crop issue |
| **White Level Saturation**| Fraction of pixels with intensity $\ge 0.98$ | $> 15\%$ white pixels | Under-penetration / detector saturation |
| **Aspect Ratio** | $\text{Aspect Ratio} = \frac{\text{Width}}{\text{Height}}$ | $\text{AR} < 0.70$ or $\text{AR} > 1.40$ | Distorted or truncated thoracic geometry |

---

## Section 5: Foundation Model Comparison (For Notes Tab)

To prepare for the modeling phase, we evaluated candidate vision architectures and feature extraction pipelines:

1. **Microsoft RAD-DINO (`microsoft/rad-dino`)**
   - *Architecture:* Vision Transformer (ViT-B/14) pre-trained via self-supervised DINOv2 on clinical chest radiographs.
   - *Output:* 768-dimensional latent representation vector.
   - *Strengths:* Captures global thoracic morphology and fine patch-level vascular features without requiring manual task labels. Standardized PyTorch implementation.
   - *Performance in Benchmark:* Vision-only AUROC $0.573 \pm 0.041$; Multimodal fusion with clinical covariates increases AUROC to $\approx 0.68 - 0.70$.

2. **Google CXR Foundation (`google/cxr-foundation`)**
   - *Architecture:* Contrastive ELIXR vision backbone pre-trained on hundreds of thousands of chest X-rays.
   - *Output:* 1376-dimensional embedding.
   - *Strengths:* Exceptional domain specificity trained directly on raw radiograph aspect ratios.
   - *Considerations:* Requires official TensorFlow serving client and accepted Hugging Face access agreement.

3. **Standard Transfer Learning Baselines (DenseNet-121 / ResNet-50)**
   - *Architecture:* Convolutional networks pre-trained on ImageNet or CheXpert.
   - *Strengths:* Low parameter count and fast training.
   - *Limitations:* Vulnerable to shortcut learning and overfitting compared to large self-supervised foundation backbones.

---

## Section 6: Instructions to Run Quality Control & Pipeline

### Prerequisites
Ensure Python 3.12 is installed. Create a clean virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn opencv-python pillow tqdm torch torchvision transformers
```

### Running Image QC and Feature Preparation
```bash
# 1. Run full quality control and preprocessing (reads local uncompressed image directory)
python run_qc_and_prep.py --input-dir "Image Folder" --output-dir "prepared_images" --no-install

# 2. View generated manifests
head -n 20 prepared_images/prepared_index.csv
```

### Reproducibility & Data Governance
- Raw DICOM/PNG files and un-anonymized clinical spreadsheets must remain in local secure storage.
- Never add or commit `.png`, `.jpg`, `.zip`, `.npy`, or patient-level ZIP files to git.
- All code, manifests, and aggregated summaries in this folder adhere to HIPAA and institutional IRB review standards.
