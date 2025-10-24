# Multi-Class Classification (Bird Species)

[![Licence: AGPL v3](https://img.shields.io/badge/Licence-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/darklorddad/Swinburne-COS30082-AML-Multi-Class-Classification-Bird-Species)

### Project Overview

This project tackles the challenge of fine-grained visual categorisation by classifying 200 different species of birds. The primary goal is to build and train a model that can accurately identify bird species from images, using deep learning techniques.

---

### Dataset

The project uses the **Caltech-UCSD Birds 200 (CUB-200)** dataset. This is a challenging image dataset containing photos of 200 bird species, primarily from North America.

- **Total Species:** 200
- **Total Images:** 6,033
- **Class Distribution:** The number of images per class is imbalanced, ranging from 20 to 39 images per species in the training set.

![Image](Caltech-UCSD-Birds-200-(CUB-200)/Class-distribution.png)

---

### Methodology

Several models were trained to find the best-performing architecture for this classification task. The approach uses transfer learning, fine-tuning pre-trained models that have demonstrated strong performance on general image classification tasks.

---

### Results

The performance of each model was evaluated on the test set. The SwinV2-Large model achieved the highest accuracy.

| Model | Accuracy | F1 (Macro) |
| :--- | :---: | :---: |
| SwinV2 Large | **89.18%** | **0.8875** |
| Swin Transformer | 88.85% | 0.8861 |
| ConvNeXt V2 Tiny | 86.72% | 0.8615 |
| SwinV2 Tiny | 83.77% | 0.8313 |
| FocalNet Base | 82.62% | 0.8157 |
| Swin Tiny (EuroSAT) | 80.00% | 0.7928 |
| Swin Tiny | 78.44% | 0.7758 |

---

### Evaluation

The model's performance is evaluated using two primary metrics:

-   **Top-1 Accuracy:** The overall classification accuracy across all test images.
    -   Formula: `Top-1 accuracy = (1/N) * Σ_k=1^N 1{argmax(y) == groundtruth}`
-   **Average Accuracy Per Class:** The average of accuracies for each individual bird species, providing insight into the model's performance on a per-class basis.
    -   Formula: `Ave = (1/C) * Σ_i=1^C T_i`

Where `N` is the total number of testing images, `C` is the total number of classes, `y` is the output probabilities, and `T_i` is the average accuracy for class `C_i`.

---

### Licence

This project is licensed under the **GNU Affero General Public License v3.0**. See the [LICENCE](LICENSE) file for full details.

Copyright (C) 2025 [darklorddad](https://github.com/darklorddad)
