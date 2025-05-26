# UCF101 Action Classification Comparison and Hyperparameter Tuning: Deep Learning

## Overview

This repository contains the code, notebooks, and results for the COS80027 Machine Learning final project at Swinburne University of Technology, Semester 1 2025. This project focused exclusively on the deep learning approach, including extensive hyperparameter tuning for r3d_18 and mc3_18 models on a subset of the UCF-101 dataset.

## Repository Structure

```
├── README.md
├── development_DL.ipynb       # Deep learning implementation and hyperparameter tuning
├── final_report.pdf           # Supplementary materials and plots
├── train_main.csv             # 80% of clips for final training
├── test_main.csv              # 20% of clips for final evaluation
├── train_inner_fold{1..3}.csv # 3 inner folds: 80% train / 20% val splits
├── val_inner_fold{1..3}.csv
└── UCF-101/                   # Local video dataset directory (not in repo)
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ucf101-deep-learning-hyperparameter-tuning.git
   cd ucf101-deep-learning-hyperparameter-tuning
   ```
2. Install dependencies:
   ```bash
   conda create -n ucf101_env python=3.12
   conda activate ucf101_env
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   conda install scikit-learn pandas matplotlib opencv-python tqdm
   ```
3. Download and place the UCF-101 dataset under `UCF-101/` as described in the notebook.

## Usage

- **Deep Learning Notebook**: `development_DL.ipynb`
  - Splitting scripts generate CSV manifests.
  - Preprocessing, dataset class definitions, 3-fold CV hyperparameter tuning for `r3d_18` and `mc3_18`.
  - Final retraining on full train split and evaluation on test split.

## Results

- **Best Hyperparameters**
  - `r3d_18`: Learning Rate = 1e-4, Weight Decay = 1e-5
  - `mc3_18`: Learning Rate = 1e-3, Weight Decay = 0.0

- **Test Accuracies**
  - `r3d_18`: 96.91%
  - `mc3_18`: 96.65%
