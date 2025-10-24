# COS30082 Assignment: Bird Species Classification

**Name**: [Your Name]
**ID**: [Your Student ID]

**YouTube Presentation Link**: [Placeholder YouTube Link]
**Hugging Face Application Link**: [Placeholder Hugging Face Link]

---

### 1. Methodology

This section outlines the systematic approach taken to address the multi-class bird species classification problem using the Caltech-UCSD Birds 200 (CUB-200) dataset. The methodology encompasses data preparation, model selection, training strategy, and techniques for mitigating overfitting.

#### 1.1. Dataset and Pre-processing

The project utilises the **Caltech-UCSD Birds 200 (CUB-200) dataset**, which contains 6,033 images of 200 bird species. The provided data was split into a training set of 4,829 images and a test set.

An initial analysis of the dataset revealed a slight class imbalance, with the number of images per class ranging from 20 to 39. The imbalance ratio (max/min) was calculated to be 1.95:1. While not extreme, this imbalance was noted, and its potential impact was considered during evaluation by using macro-averaged metrics that treat each class equally.

The pre-processing pipeline was automated and consisted of several key steps:
1.  **Dataset Organisation**: The initial dataset, provided as ZIP archives and text-based annotation files, was reorganised into a standard image folder structure (`<class_name>/<image_file>`). This format is required by many modern training frameworks, including Hugging Face's `autotrain`.
2.  **Name Normalisation**: Class directory names and image filenames were converted to lowercase. This standardises the naming convention, preventing potential case-related errors and ensuring compatibility with various tools and libraries.
3.  **Data Splitting**: The organised dataset was further split into `train` and `validation` sets. A 20% validation split was used, ensuring that each class had a minimum of 5 images in both splits to allow for robust validation. This step is crucial for monitoring the model's generalisation performance during training.

#### 1.2. Model Architectures and Transfer Learning

The core strategy involved **transfer learning**, leveraging the power of large-scale, pre-trained vision models. Instead of training a model from scratch, which would be computationally expensive and prone to overfitting on a relatively small dataset like CUB-200, several state-of-the-art Vision Transformer (ViT) and ConvNeXt models were fine-tuned.

The following pre-trained models were selected for experimentation to compare their effectiveness:

| Model Name | Base Architecture | Pre-trained On |
| :--- | :--- | :--- |
| **Swin-Tiny-78** | `microsoft/swin-tiny-patch4-window7-224` | ImageNet-1K |
| **Swin-Tiny-Eurosat-80** | `nielsr/swin-tiny-patch4-window7-224-finetuned-eurosat` | EuroSAT (Satellite) |
| **Focalnet-Base-82** | `microsoft/focalnet-base` | ImageNet-22K |
| **SwinV2-Tiny-83** | `microsoft/swinv2-tiny-patch4-window16-256` | ImageNet-22K |
| **ConvNeXt-V2-Tiny-86** | `facebook/convnextv2-tiny-22k-224` | ImageNet-22K |
| **Swin-Transformer-88** | `XinWenMonash/swin_transformer` | ImageNet-1K |
| **SwinV2-Large-89** | `microsoft/swinv2-large-patch4-window12-192-22k` | ImageNet-22K |

These models, particularly those based on the **Swin Transformer** architecture, employ a hierarchical structure and shifted windows to capture both local and global features efficiently, making them highly effective for computer vision tasks. A classification head was added to each pre-trained model and fine-tuned on the CUB-200 dataset.

#### 1.3. Training Strategy and Hyperparameters

All models were trained using the Hugging Face `autotrain-advanced` tool, which provides a streamlined yet powerful interface for fine-tuning. The training process was configured with a consistent set of robust hyperparameters to ensure a fair comparison.

-   **Loss Function**: **Cross-Entropy Loss**, the standard for multi-class classification, was used.
-   **Optimizer**: The **AdamW** optimizer (`adamw_torch`) was chosen for its effectiveness in training transformer-based models.
-   **Learning Rate**: A learning rate of `5e-5` was used, with a linear or cosine warmup schedule over the first 10% of training steps to stabilise training.
-   **Batch Size**: A batch size of 32 was used, with gradient accumulation (3 or 4 steps) to simulate a larger effective batch size, aiding in stable gradient estimation.
-   **Regularisation**: To combat overfitting, **weight decay** (L2 regularisation) was set to `0.01`.
-   **Mixed Precision**: `fp16` or `bf16` mixed-precision training was enabled to accelerate training and reduce memory consumption without significant loss in performance.
-   **Early Stopping**: A crucial mechanism to prevent overfitting, training was configured to stop if the validation loss did not improve by at least `0.01` for 5 consecutive epochs.

The models were trained for a maximum of 100 epochs, but early stopping often terminated the process sooner, saving computational resources and yielding models with better generalisation.

---

### 2. Result and Discussion

This section presents a comparative analysis of the performance of the seven fine-tuned models and provides a justification for the selection of the best model.

#### 2.1. Model Performance Comparison

The models were evaluated on the validation set using the mandatory metrics of Top-1 Accuracy and Average Accuracy Per Class (represented by macro-averaged F1-score and Recall). The table below summarises the final validation metrics for each model.

| Model | Base Architecture | Top-1 Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) | Validation Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SwinV2-Large-89** | SwinV2-Large | **0.8918** | **0.8875** | **0.9082** | **0.8886** | **0.3713** |
| **Swin-Transformer-88** | Swin-Transformer | 0.8885 | 0.8861 | 0.9023 | 0.8880 | 0.3976 |
| **ConvNeXt-V2-Tiny-86**| ConvNeXt-V2-Tiny | 0.8672 | 0.8615 | 0.8762 | 0.8645 | 0.4875 |
| **SwinV2-Tiny-83** | SwinV2-Tiny | 0.8377 | 0.8313 | 0.8636 | 0.8356 | 0.5219 |
| **Focalnet-Base-82** | FocalNet-Base | 0.8262 | 0.8157 | 0.8477 | 0.8219 | 0.6603 |
| **Swin-Tiny-Eurosat-80**| Swin-Tiny (EuroSAT) | 0.8000 | 0.7928 | 0.8271 | 0.7960 | 0.7601 |
| **Swin-Tiny-78** | Swin-Tiny | 0.7844 | 0.7758 | 0.8028 | 0.7803 | 0.7969 |

*Metrics are reported on the validation set. Top-1 Accuracy is the primary evaluation metric. Macro-averaged Recall is equivalent to the Average Accuracy Per Class.*

#### 2.2. Analysis and Discussion

The results clearly demonstrate the superior performance of larger, more complex models pre-trained on extensive datasets.

-   **Best Performing Model**: The **`SwinV2-Large-89`** model achieved the highest Top-1 Accuracy of **89.18%**. Its leading performance is also reflected in its F1, Precision, and Recall scores, and it attained the lowest validation loss. This is attributable to its larger architecture (`Large` vs. `Tiny` or `Base`) and its pre-training on the comprehensive ImageNet-22K dataset, which provides a highly robust feature foundation for transfer learning.

-   **Close Contenders**: The generic `Swin-Transformer-88` model was a surprisingly strong performer, achieving 88.85% accuracy, nearly matching the `SwinV2-Large` model. This highlights the power of the Swin Transformer architecture itself. The `ConvNeXt-V2-Tiny-86` model also performed admirably, securing the third-highest accuracy at 86.72%, demonstrating the strength of modern convolutional architectures.

-   **Impact of Model Size and Pre-training**: A clear trend emerges where larger models (`SwinV2-Large`) outperform their smaller counterparts (`SwinV2-Tiny`). Furthermore, the domain of pre-training data is significant. The `Swin-Tiny-Eurosat-80` model, pre-trained on satellite imagery, performed worse than models pre-trained on general-purpose datasets like ImageNet, confirming that pre-training on a domain closer to the target task (general object recognition vs. satellite images) is more beneficial.

#### 2.3. Overfitting and Model Generalisation

Overfitting was a primary concern given the dataset's size. The training logs confirm that our mitigation strategies were effective. For instance, the training for `Model-Swin-Transformer-88` was halted by the early stopping callback at epoch 25, well before the 100-epoch limit. This occurred because the validation loss ceased to improve, preventing the model from memorising the training data and losing its ability to generalise.

A visual analysis of the training and validation loss curves for the top models would typically show the training loss continuing to decrease while the validation loss plateaus or begins to increase. The early stopping mechanism intervenes at this plateau, capturing the model at its point of optimal generalisation. The low final validation loss of the top models, especially `SwinV2-Large-89` (0.3713), indicates that they generalised well to unseen data.

#### 2.4. Justification for the Best Model

The **`Model-SwinV2-Large-89`** is unequivocally the best model from this study. The justification is threefold:

1.  **Superior Quantitative Performance**: It achieved the highest Top-1 Accuracy (89.18%) and the best scores across all macro-averaged metrics, indicating it is not only the most accurate overall but also performs most consistently across all 200 bird species.
2.  **Robust Generalisation**: It recorded the lowest validation loss, suggesting it is the least overfitted model and the most likely to perform well on a true, unseen test set.
3.  **Architectural Advantage**: As a large-scale Swin Transformer (v2), it benefits from the latest architectural improvements and a vast pre-training regimen, giving it a significant advantage in feature extraction and representation.

While computationally more intensive, the performance gains justify its selection as the final, recommended model for this bird classification task.
