# Computer Vision Equine Grimace Scale

A Computer Vision project for automatic horse pain assessment using facial expression recognition.

This project investigates whether deep learning can classify horse facial expressions into three pain levels using the Equine Grimace Scale (EGS).

The project follows a complete machine learning workflow including:

- Dataset preparation
- Data preprocessing
- Baseline CNN
- Data augmentation
- Transfer learning
- Model selection
- Hyperparameter tuning
- Ablation studies
- Explainability and Critical Analysis
- Final untouched test evaluation

---

# Project Objective

Pain assessment in horses is traditionally performed manually by veterinarians using facial expressions. This process is subjective, time-consuming, and requires trained experts.

The objective of this project is to develop an automated computer vision system capable of classifying horse facial images into:

- No Pain
- Moderate Pain
- Severe Pain

using deep convolutional neural networks.

---

# Dataset

This project uses the OpenFarm Horse Grimace dataset.

Each image is labelled according to the Equine Grimace Scale.

Classes:

| Label  | Class         |
|--------|---------------|
| 0      | No Pain       |
| 1      | Moderate Pain |
| 2      | Severe Pain   |

---

# Repository Structure

```text
computer-vision-equine-grimace-scale/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_data_pipeline.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_augmentation_experiments.ipynb
│   ├── 05_transfer_learning.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_ablation_studies.ipynb
│   ├── 09_explainability.ipynb
│   └── 10_final_evaluation.ipynb
│
├── reports/
│
├── results/
│
├── src/
│
├── requirements.txt
└── README.md
```

---

# Software Requirements

Python 3.11+

Recommended environment:

- PyTorch
- TorchVision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter Notebook

## Installation

Clone the repository

```bash
git clone https://github.com/mufibee/computer-vision-equine-grimace-scale.git
cd computer-vision-equine-grimace-scale
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

The notebooks should be executed sequentially.

## Phase 1 — Dataset Preparation

Run:

```
notebooks/01_dataset_preparation.ipynb
```

This notebook:

- downloads and prepares the dataset
- removes invalid images
- creates train/validation/test splits
- performs dataset analysis

Outputs:

- dataset statistics
- class distributions
- split CSV files

---

## Phase 2 — Data Pipeline

Run:

```
notebooks/02_data_pipeline.ipynb
```

This notebook:

- creates the custom HorseGrimaceDataset
- applies preprocessing
- creates PyTorch DataLoaders
- verifies image loading

---

## Phase 3 — Baseline CNN

Run:

```
notebooks/03_baseline_cnn.ipynb
```

This notebook:

- trains a CNN from scratch
- evaluates on the validation set
- saves:

- model checkpoints
- learning curves
- confusion matrix
- classification report

---

## Phase 4 — Data Augmentation

Run:

```
notebooks/04_data_augmentation.ipynb
```

This notebook compares training with and without augmentation.

---

## Phase 5 — Transfer Learning

Run:

```
notebooks/05_transfer_learning.ipynb
```

Experiments:

- Frozen ResNet18
- Fine-tuned ResNet18

The best transfer learning model is selected.

---

## Phase 6 — Model Selection

Run:

```
notebooks/06_model_selection.ipynb
```

Compares:

- baseline CNN
- augmented CNN
- transfer learning models

The best validation model is selected.

---

## Phase 7 — Hyperparameter Tuning

Run:

```
notebooks/07_hyperparameter_tuning.ipynb
```

Tunes:

- learning rate
- dropout
- weight decay

The best configuration becomes the canonical model stored in:

```
results/final_model/
```

---

## Phase 8 — Ablation Studies

Run:

```
notebooks/08_ablation_studies.ipynb
```

Compares:

- augmentation vs no augmentation
- class weights vs no class weights
- frozen vs fine-tuned transfer learning
- baseline CNN vs transfer learning

---

## Phase 9 — Explainability and Critical Analysis

Run:

```text
notebooks/09_explainability.ipynb
```

Generates qualitative visualizations and analyses of the final selected ResNet18 model, including:

- Learned first-layer convolutional filters
- Feature-map visualizations from early, middle, and deep convolutional layers
- Representative correctly classified examples from each pain category
- High-confidence misclassified examples
- Confidence distribution analysis
- Visual interpretation of the model's learned feature representations

---

## Phase 10 — Final Test Evaluation

Run:

```
notebooks/10_final_evaluation.ipynb
```

This notebook performs the first and only evaluation on the untouched test set.

Outputs include:

- Test accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Classification report
- Confusion matrix
- Normalized confusion matrix
- Prediction CSV
- Confidence analysis
- High-confidence mistakes
- Misclassified examples

No training or model selection is performed during this phase.

---

# Final Model

Architecture:

- ResNet18

Training:

- Fine-tuned transfer learning

Output classes:

- No Pain
- Moderate Pain
- Severe Pain

Selection metric:

- Validation Macro F1

---

# Results

The final evaluation reports:

- Test loss
- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Classification report
- Confusion matrix

The project also includes qualitative analysis through learned convolutional filters, feature-map visualizations, representative correctly classified examples, high-confidence misclassified examples, confidence analysis, and detailed error analysis to better understand the model's behaviour.

---

# Reproducibility

Random seeds are fixed to ensure reproducibility.

The final evaluation loads the canonical model stored in:

```
results/final_model/
```

without retraining or hyperparameter tuning.

---


American University of Sharjah

Computer Vision Course

Summer 2026
