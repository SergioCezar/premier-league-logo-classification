# Premier League Logo Classification ⚽

This project applies **deep learning and transfer learning** techniques to classify
**Premier League club logos** from images.

A Convolutional Neural Network (CNN) based on **ResNet-18** was fine-tuned to recognize
the logos of **20 Premier League teams**, achieving strong classification performance.

This work was developed as an academic machine learning project and is accompanied by
a scientific article.

---

## Problem Description

Given an input image containing a Premier League club logo, the model predicts
which club the logo belongs to among 20 possible teams.

---

## Model and Techniques

- **Architecture:** ResNet-18 (pretrained on ImageNet)
- **Technique:** Transfer Learning
- **Framework:** PyTorch
- **Data Augmentation:** Albumentations
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam

---

## Dataset

- 20 Premier League teams
- Images organized by class (one folder per team)
- Dataset stored externally (Google Drive)

> ⚠️ The dataset is not included in this repository due to size constraints.

---

## Training Pipeline

- Train / validation split (80% / 20%) with stratification
- Extensive data augmentation:
  - Rotation, perspective transform
  - Color jitter
  - Gaussian noise and blur
  - Coarse dropout
- Fine-tuning all network layers

---

## Evaluation Metrics

- Accuracy (train and validation)
- Confusion Matrix
- Precision, Recall, and F1-score per class
- Visualization of most frequent classification errors

---

## Results

- High validation accuracy across most teams
- Confusion matrix highlights visually similar logos
- Detailed metrics reported in the accompanying article

---

## Scientific Article

The full scientific report describing the methodology, experiments, and results
is available in:

report/scientific_article.pdf

## Technologies

- Python

- PyTorch

- Torchvision

- Albumentations

- OpenCV

- Matplotlib

- Scikit-learn


## License
This project is licensed under the MIT License.
