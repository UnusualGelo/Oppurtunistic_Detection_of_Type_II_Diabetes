# Custom CNN for Diabetes Detection Using Chest X-Rays

## Project Overview

This project focuses on developing a convolutional neural network (CNN) from scratch using PyTorch to investigate whether frontal chest X-ray images contain patterns associated with Type 2 diabetes.

The model will be trained using cleaned frontal chest X-rays and evaluated against other models developed by the research team.

## Project Structure

- `src/dataset.py` — Image loading and preprocessing
- `src/model.py` — CNN architecture
- `src/train.py` — Model training and validation
- `src/evaluate.py` — Model evaluation
- `notebooks/` — Exploratory analysis and experiments
- `configs/` — Training configurations
- `results/` — Performance metrics and visualizations

## Planned Workflow

1. Load and preprocess frontal chest X-ray images.
2. Match images with patient diabetes labels.
3. Build a custom CNN architecture using PyTorch.
4. Train and validate the model.
5. Evaluate performance using ROC-AUC, sensitivity, specificity, accuracy, and F1-score.
6. Compare performance with pretrained models developed by other team members.

## Status

Initial project setup. Model development has not yet started.