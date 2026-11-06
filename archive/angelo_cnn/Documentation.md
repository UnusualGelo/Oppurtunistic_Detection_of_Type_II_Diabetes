# CNN Model Development Documentation
**Project:** Opportunistic Detection of Type 2 Diabetes Using Frontal Chest X-Rays  
**Framework:** PyTorch  
**Model:** Custom Convolutional Neural Network (CNN)

## 1. Model Overview

A Convolutional Neural Network (CNN) was developed from scratch using PyTorch to investigate whether frontal chest X-rays contain visual patterns associated with Type 2 diabetes (T2D).

The model performs binary classification, predicting whether a patient has T2D (Class 1) or does not have T2D (Class 0). It uses chest X-ray images exclusively, without patient metadata, providing an image-only baseline for comparison with other models.

## 2. Dataset Preparation

The dataset consisted of 1,000 frontal chest X-ray images in PNG format, matched with clinical records containing diabetes labels.

Data preparation was implemented in `dataset.py` and included checking missing values, validating diabetes labels, checking duplicate image filenames, and verifying image file availability.

Images were converted to grayscale, resized to 224 × 224 pixels, converted into PyTorch tensors, and normalized using a mean and standard deviation of 0.5.

### Dataset Distribution and Splits

Stratified sampling was used to preserve class proportions across the training, validation, and testing datasets. A random seed of 42 was applied for reproducibility.

| Dataset | Total | No T2D (0) | T2D (1) |
|---|---:|---:|---:|
| Training (70%) | 700 | 478 | 222 |
| Validation (15%) | 150 | 102 | 48 |
| Testing (15%) | 150 | 103 | 47 |
| **Total** | **1,000** | **683** | **317** |

The overall dataset contained 68.3% non-diabetic and 31.7% diabetic images. Due to this class imbalance, weighted binary cross-entropy was used during full-dataset training.

Patient-level independence between splits could not be confirmed because patient identifiers were unavailable in the working dataset.

## 3. CNN Architecture

The CNN consisted of three convolutional blocks for feature extraction, followed by fully connected layers for binary classification.

| Component | Configuration |
|---|---|
| Input | 224 × 224 grayscale image |
| Convolutional Block 1 | 32 filters, 3 × 3 kernel |
| Convolutional Block 2 | 64 filters, 3 × 3 kernel |
| Convolutional Block 3 | 128 filters, 3 × 3 kernel |
| Activation | ReLU |
| Normalization | Batch Normalization |
| Pooling | 2 × 2 Max Pooling |
| Global Pooling | Adaptive Average Pooling (1 × 1) |
| Fully Connected Layer | 64 neurons |
| Dropout | 0.4 |
| Output | 1 logit for binary classification |

Each convolutional block used Batch Normalization, ReLU activation, and max pooling. The extracted features were reduced to 128 values through adaptive average pooling.

The classification head used a fully connected layer with dropout to reduce overfitting. A sigmoid function converted the final output into a probability during evaluation, with 0.5 used as the classification threshold.

## 4. Training Configuration

The model was trained using PyTorch with Apple Metal Performance Shaders (MPS) GPU acceleration.

| Parameter | Setting |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Alternative learning rate | 0.0001 |
| Batch size | 16 |
| Maximum epochs | 30 |
| Loss function | Weighted BCEWithLogitsLoss |
| Positive class weight | 2.153 |
| Early stopping patience | 5 epochs |
| Checkpoint selection | Lowest validation loss |

The positive class weight was calculated using the training class distribution:

**Positive weight = 478 / 222 = 2.153**

This increased the penalty for errors involving diabetes-positive examples.

Early stopping monitored validation loss and restored the best-performing model weights.

## 5. Initial Model Performance

The initial CNN was trained with Batch Normalization and a learning rate of 0.001.

Training stopped after 13 epochs, with the lowest validation loss of 0.9382 occurring at epoch 8.

### Validation Results

The model was evaluated on 150 validation images.

| Metric | Result |
|---|---:|
| Accuracy | 56.67% |
| ROC-AUC | 0.577 |
| Precision (T2D) | 35.59% |
| Recall / Sensitivity (T2D) | 43.75% |
| Specificity | 62.75% |
| F1-score (T2D) | 39.25% |

### Confusion Matrix

| Actual / Predicted | No T2D | T2D |
|---|---:|---:|
| No T2D | 64 | 38 |
| T2D | 27 | 21 |

The model correctly identified 21 of 48 diabetes-positive cases and 64 of 102 diabetes-negative cases.

However, it missed 27 diabetes-positive cases and incorrectly classified 38 diabetes-negative cases as positive.

The ROC-AUC of 0.577 indicated weak discrimination between the two classes, only slightly above chance-level ranking (0.50).

Predicted probabilities ranged from 0.379 to 0.609, with a mean of 0.488, suggesting limited separation between positive and negative predictions.

## 6. Model Development Experiments

Three experiments were conducted on a small subset of 32 training images to evaluate the CNN's learning capacity and investigate training stability.

These experiments used unweighted binary cross-entropy and trained for 200 epochs. They were debugging experiments, not independent validation tests.

### Small-Subset Experiments

| Configuration | Best Training Accuracy | Final Accuracy | Final Loss |
|---|---:|---:|---:|
| BatchNorm, LR 0.001 | 100% | 71.88% | 0.0443 |
| BatchNorm, LR 0.0001 | 96.88% | 90.62% | 0.2744 |
| GroupNorm, LR 0.0001 | 84.38% | 84.38% | 0.4871 |

**Experiment 1 — Original CNN:** The model reached 100% accuracy multiple times, demonstrating that it could memorize a small subset of training images. However, accuracy fluctuated considerably.

**Experiment 2 — Reduced Learning Rate:** Lowering the learning rate to 0.0001 resulted in slower learning, with the model reaching 96.88% accuracy and finishing at 90.62%.

**Experiment 3 — Group Normalization:** Replacing Batch Normalization with Group Normalization did not improve the small-subset results under the tested configuration.

The experiments confirmed that the CNN could fit a small dataset, although performance varied with the training configuration.

## 7. Full-Dataset Learning Rate Comparison

Following the debugging experiments, the original BatchNorm architecture was restored and trained on the full dataset using a reduced learning rate of 0.0001.

| Metric | LR 0.001 | LR 0.0001 |
|---|---:|---:|
| Best validation loss | 0.9382 | 0.9292 |
| Best epoch | 8 | 18 |
| Epochs completed | 13 | 23 |

Reducing the learning rate improved the best validation loss from 0.9382 to 0.9292, a decrease of approximately 0.96%.

The model also trained for 10 additional epochs before early stopping.

However, updated classification metrics for this experiment have not been confirmed because the evaluation output appeared to use the original model checkpoint.

## 8. Results Summary and Limitations

The CNN successfully completed the image-processing, training, and classification pipeline.

The small-subset experiments demonstrated that the architecture had sufficient capacity to memorize training examples. However, the initial validation results indicated limited predictive performance.

The most important findings were:

- **ROC-AUC of 0.577**, indicating weak discrimination between diabetes-positive and diabetes-negative images.
- **Recall of 43.75%**, meaning more than half of the diabetes-positive validation cases were missed.
- **Precision of 35.59%**, indicating that many positive predictions were false positives.
- **Lower validation loss with LR 0.0001**, although improved classification performance was not yet confirmed.

The primary limitations were the relatively small dataset, class imbalance, lack of confirmed patient-level independence between splits, and absence of final held-out test results.

The reported classification metrics therefore represent initial validation performance rather than final clinical or test-set performance.