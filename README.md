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

The dataset was specifically designed to evaluate model robustness in Domain Adaptation—moving 
from controlled environments to noisy, real-world scenarios. 
It consists of 3,000 images distributed across 20 classes (Premier League clubs).
- Images organized by class (one folder per team)
- Dataset stored externally (Google Drive)

### Data Distribution
To ensure a fair evaluation and prevent class bias, the dataset is perfectly balanced:

- Classes: 20 teams.

- Samples per Class: 150 images.

- Total Dataset Size: 3,000 images.

### Image Profiles
Each class is composed of two distinct subsets to simulate different levels of difficulty:

#### 1. Synthetic:

High-contrast logos on neutral backgrounds.

Subjects subjected to geometric distortions (rotation, scaling, shearing) to teach the model geometric invariance.

#### 2. Real-World:

Apparel: Logos on player kits and training gear, featuring fabric deformations and wrinkles.

Fan Merchandise: Scarves, hats, and flags held by supporters.

Contextual Noise: Logos captured in stadium environments with variable lighting, motion blur, and partial occlusions.

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

## Technologies

- Python

- PyTorch

- Torchvision

- Albumentations

- OpenCV

- Matplotlib

- Scikit-learn

---

### Requirements

```bash
pip install -r requirements.txt
```

---

## Results

- High validation accuracy across most teams
- Confusion matrix highlights visually similar logos
- Detailed metrics reported in the accompanying article

---

## Scientific Article

The full scientific report describing the methodology, experiments, and results
is available in:

scientific_article/article.pdf

---

## License
This project is licensed under the MIT License.
