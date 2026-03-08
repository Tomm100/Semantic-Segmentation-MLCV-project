### Dataset

- **Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Content**: 5,863 chest X-ray images, organized into train, validation, and test splits
- **Classes**: NORMAL (1,341 train) and PNEUMONIA (3,875 train) — naturally imbalanced (~1:3 ratio)
- **Subset strategy**: 20% of train split is added to validation split, while the remaining 80% is used for training. Test split remains unchanged.

### Architecture

| Component | Architecture | Details |
|---|---|---|
| **Classifier** | ResNet-18 (pre-trained on ImageNet) | Final fully-connected layer replaced with a binary classification head |
| **GAN** | WGAN-GP | Conditional on class label; 5 transposed conv layers; outputs 64×64 grayscale images |

### Training Setup

#### Phase 1 — Classifier Baseline
- **Method**: Fine-tuning of the pre-trained ResNet-18

#### Phase 2 — GAN Training
- **Method**: Training WGAN-GP on the train split

#### Phase 3 — Classifier with Augmented Data
- **Method**: Fine-tuning of the pre-trained ResNet-18 on the train split with augmented data

### Evaluation Metrics

| Metric |
|---|
| **Accuracy** |
| **Precision** |
| **Recall** |
| **F1-Score** |

A **Confusion Matrix** will also be used as a diagnostic tool to provide a detailed breakdown of classification errors per class.