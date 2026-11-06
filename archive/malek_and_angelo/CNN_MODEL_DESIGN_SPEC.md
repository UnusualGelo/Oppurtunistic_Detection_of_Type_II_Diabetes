# Custom CNN Architecture Specification & Filter Exploration Framework for Opportunistic Type 2 Diabetes Detection

**Project:** Opportunistic Detection of Type 2 Diabetes from Chest Radiographs  
**Authors:** Malek & Angelo  
**Course:** CS 396 — Artificial Intelligence in Healthcare  
**Module:** `malek_and_angelo/CNN_MODEL_DESIGN_SPEC.md`  

---

## 1. Objective & Scientific Rationale

### 1.1 Why Build a CNN from Scratch?
While pre-trained foundation models (e.g., Microsoft RAD-DINO, Google CXR-Foundation) provide powerful out-of-the-box latent representations, they operate as opaque high-dimensional feature extractors whose internal representations cannot be easily re-engineered or dissected. 

Constructing a dedicated Convolutional Neural Network (CNN) from scratch provides three fundamental advantages:
1. **Direct Filter Introspection:** We can directly visualize, analyze, and control the exact convolutional filter kernels ($3\times3, 5\times5, 7\times7$) as they learn primitive edges, vascular patterns, and gross cardiothoracic geometries.
2. **Domain-Specific Architectural Optimization:** Unlike ImageNet models designed for natural photographs, a custom radiograph CNN can be explicitly tuned to medical grayscale dynamic ranges and anatomical receptive fields.
3. **Controlled Hypothesis Testing:** We can systematically evaluate how varying filter sizes, channel capacities, and receptive fields directly impact the detection of subtle thoracic indicators of metabolic disease.

### 1.2 Anatomical Markers of T2D in Frontal Chest X-rays
Type 2 Diabetes mellitus induces systemic micro- and macro-vascular complications that manifest subtly across standard frontal thoracic radiographs:
- **Cardiomegaly & Left Ventricular Hypertrophy:** Elevated cardiothoracic ratio ($>0.50$) linked to diabetic cardiomyopathy and hypertension.
- **Mediastinal & Epicardial Adiposity:** Widening and density changes in the superior mediastinum and paracardial fat pads.
- **Aortic Elongation & Vascular Calcification:** Sclerosis and tortuosity of the aortic knob.
- **Soft Tissue Thickness:** Increased subcutaneous thoracic wall thickness correlating with visceral obesity and elevated BMI.

---

## 2. Convolutional Filter Exploration Framework

A central goal of this design is **systematic experimentation with convolutional filters**. Filters determine how the model processes local spatial relationships and constructs hierarchical representations.

```
Input Radiograph (224 x 224 x 1 or 3)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Shallow Micro-Filters (Edge & Ridge Detection)│  <-- Kernels: 3x3, 5x5, or 7x7
│ Captures rib margins, lung-pleural boundaries, edges   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Intermediate Mesoscopic Filters               │  <-- Multi-scale (Inception-style)
│ Captures vascular branching, parenchymal texture       │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Deep Macro-Filters (Global Anatomical Shapes) │  <-- Large Receptive Field
│ Captures cardiac silhouette, mediastinal contours      │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Global Pooling & Classification Head                   │  <-- GAP + Dense + Dropout
│ Outputs P(Type 2 Diabetes)                             │
└────────────────────────────────────────────────────────┘
```

### 2.1 Filter Variables to Benchmark
We will experiment across four primary filter design dimensions:

1. **Kernel Size ($k \times k$):**
   - **Small Kernels ($3\times3$):** Highly parameter-efficient, stacks multiple non-linearities (VGG-style). Good for fine edge detection and lung markings.
   - **Medium Kernels ($5\times5$):** Larger immediate receptive field per layer; better at capturing intermediate rib-spacing and aortic contours in fewer layers.
   - **Large Kernels ($7\times7$ in Layer 1):** Standard in initial radiograph stages (e.g., ResNet) to downsample background thoracic fields while capturing gross chest orientation.
   - **Factorized / Asymmetric Kernels ($1\times3$ then $3\times1$):** Reduces parameters while providing directional orientation sensitivity (vertical spinal alignment vs. horizontal ribs).

2. **Filter Depth Progression (Channel Capacity):**
   - *Progression A (Standard Exponential):* $32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$
   - *Progression B (Narrow & Deep):* $16 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 128$ (avoids overfitting on small datasets)
   - *Progression C (Wide & Shallow):* $64 \rightarrow 128 \rightarrow 256$ (fewer layers, maximum channel diversity)

3. **Receptive Field Matching:**
   - The theoretical receptive field (TRF) of the final convolutional layer must be large enough to encompass the entire cardiothoracic silhouette ($\ge 180\times180$ pixels on a $224\times224$ input).

4. **Multi-Scale Inception Filters:**
   - In parallel branches within the same block, apply $1\times1$, $3\times3$, and $5\times5$ filters simultaneously and concatenate them. This allows the network to process fine lung vascularity alongside broad heart borders in a single stage.

---

## 3. Candidate Model Architectures to Implement

We will construct three distinct custom CNN variants in PyTorch to compare filter behaviors:

### Architecture 1: `CustomCXRNet-Classic` (Sequential Hierarchical Baseline)
- **Structure:** 4 or 5 sequential convolutional blocks followed by Global Average Pooling (GAP).
- **Block Layout:** `Conv2D(k=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2D(k=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`.
- **Dropout:** $p = 0.3$ before final linear projection.
- **Purpose:** Establishes the performance baseline for homogeneous $3\times3$ filters.

### Architecture 2: `CustomCXRNet-MultiFilter` (Multi-Scale Parallel Kernels)
- **Structure:** Modular blocks featuring split-transform-merge branches.
  - Branch 1: $1\times1$ Conv (channel compression / intensity mapping)
  - Branch 2: $3\times3$ Conv (local tissue texture)
  - Branch 3: $5\times5$ Conv (coarse anatomical contours)
- **Concatenation:** Along the channel axis followed by $1\times1$ pointwise convolution.
- **Purpose:** Evaluates whether simultaneous multi-scale filter capture outperforms uniform filter stacks.

### Architecture 3: `CustomCXRNet-Residual` (Deep Skip-Connection Network)
- **Structure:** Residual building blocks (`x + ConvBlock(x)`) with identity shortcuts.
- **Purpose:** Prevents vanishing gradients, allowing us to test deeper stacks of small filters without signal degradation.

---

## 4. Filter Interpretability & Visualization ("The Whole Nine Yards")

A critical component of this project is verifying **what the learned filters actually "see"**:

1. **Filter Weight Visualization (Layer 1):**
   - Plot the raw learned 2D convolution weights of the first layer as grayscale/color image matrices.
   - Verify whether the filters converge to Gabor-like edge detectors, Gaussian blur filters, or frequency filters.
2. **Intermediate Feature Map Activations:**
   - For sample passing radiographs, extract and plot the 2D feature maps from Layer 1, Layer 2, and Layer 4.
   - Observe how low-level edge maps transform into semantic anatomical heatmaps.
3. **Grad-CAM (Gradient-weighted Class Activation Mapping):**
   - Compute gradients of the T2D score with respect to the final convolutional feature map.
   - Generate heatmaps overlaid on the original frontal chest X-ray to clinically validate that the model focuses on the heart, aorta, and mediastinum rather than uninformative peripheral artifacts or radiology markers.

---

## 5. Experimental Protocol & Safeguards

### 5.1 Dataset & Input Preparation
- **Cohort:** Clean frontal chest X-rays ($N=713$ strict QC cohort or $N=949$ base frontal cohort).
- **Input Dimensions:** Float32 tensors of shape `(1, 224, 224)` for single-channel grayscale or `(3, 224, 224)` for ImageNet-standardized inputs.
- **Contrast Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization, clip limit 2.0, $8\times8$ grid) to normalize bone-tissue boundaries.

### 5.2 Patient-Level Stratified Validation (Zero Leakage)
- **Validation Scheme:** 5-fold `StratifiedGroupKFold`.
- **Grouping Variable:** `patient_id` ensures that multiple radiographs from the same individual never leak across train and validation folds.
- **Stratification Variable:** T2D diagnosis binary label ($0$ = Non-Diabetic, $1$ = Diabetic).

### 5.3 Loss Function & Class Imbalance
- In this cohort, T2D prevalence is $\approx 31.7\%$ (317 T2D vs. 683 non-T2D).
- **Loss:** Weighted Binary Cross-Entropy with Logits (`pos_weight = (683 / 317) ≈ 2.15`) or Focal Loss ($\gamma = 2.0$) to prioritize hard diabetic examples.

### 5.4 Evaluation Metrics
- Primary: **AUROC** (Area Under the Receiver Operating Characteristic curve)
- Secondary: **AUPRC** (Area Under the Precision-Recall curve, essential for imbalanced medical data)
- Operating Point Metrics (at threshold maximizing Youden's Index $J = \text{Sensitivity} + \text{Specificity} - 1$):
  - Sensitivity / Recall
  - Specificity
  - Accuracy & F1-score

---

## 6. Implementation Roadmap

| Phase | Milestone | Deliverable |
| :--- | :--- | :--- |
| **Phase 1: Architecture Blueprint** | Write design specification and PyTorch module template | `CNN_MODEL_DESIGN_SPEC.md` & `custom_cnn_models.py` |
| **Phase 2: Filter Exploration** | Implement 3 filter configurations ($3\times3$, $5\times5$, and multi-scale) | `train_custom_cnn.py` with filter argument flags |
| **Phase 3: 5-Fold Evaluation** | Run patient-grouped cross-validation across all filter variants | `cnn_cv_results.csv` and ROC curve plots |
| **Phase 4: Interpretability** | Generate Grad-CAM heatmaps and filter weight visualizations | `visualize_filters.py` and diagnostic overlay plots |
| **Phase 5: Benchmark Comparison**| Compare custom CNN against RAD-DINO and tabular clinical models | Final documentation update in `malek_and_angelo/` |

---

## 7. Next Steps & Code Execution

With the documentation and architectural specifications formalized:
1. Implement the PyTorch model definitions (`custom_cnn_models.py`) with configurable filter parameters.
2. Build the training loop script with 5-fold `StratifiedGroupKFold` cross-validation.
3. Build the filter visualization and Grad-CAM script.
