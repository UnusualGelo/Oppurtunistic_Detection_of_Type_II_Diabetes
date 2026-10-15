# Opportunistic Detection of Type II Diabetes

Opportunistic screening of Type 2 Diabetes (T2D) from chest radiographs and sociodemographic indicators.

---

## Repository Structure & Modules

- **[`frontal_cxr_cleaner/`](frontal_cxr_cleaner/)**  
  Automated frontal vs. lateral X-ray classifier and initial quality assurance using cleanX.

- **[`malek_and_angelo/`](malek_and_angelo/)**  
  Comprehensive Data Cleaning Report, Table 1 baseline clinical characteristics, quantitative image quality control metrics (blur, dynamic range, aspect ratio), and foundation model evaluation.

---

## Data Privacy & HIPAA Compliance
To comply with HIPAA Safe Harbor and healthcare privacy standards:
- No raw medical radiographs (`.png`, `.jpg`, `.dcm`), binary embeddings (`.npy`), or patient-level identifiable health information (PHI) are tracked in this repository.
- All exclusion rules are strictly maintained via `.gitignore`.
