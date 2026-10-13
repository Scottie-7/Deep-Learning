## Overview & Execution Commands

This assignment implements and evaluates three architectures on the Oxford-IIIT Pet dataset: a regular CNN, ResNet-18, and a small U-Net.

### Setup and Execution Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download and verify dataset
python data.py

# Train models
python train.py

# Evaluate models and generate figures
python evaluate.py
```

## Hardware & Environment


- **Hardware:** NVIDIA H100 NVL GPU


- **Environment:** Conda (`env_csci4425`, Python 3.12, PyTorch 2.5.1)


## Architectures & Training Settings


- **Random Seed:** 42


- **Batch Size:** 16


- **Training Epochs:** 10


- **Optimizer:** Adam (Learning Rate: 0.001)


- **Loss Function:** `nn.CrossEntropyLoss()`


- **Regular CNN:** 3 `Conv2d` blocks (16, 32, 64 channels), `MaxPool2d`, `AdaptiveAvgPool2d`, `Linear(64, 2)`.


- **ResNet-18:** `torchvision.models.resnet18(weights=None)` with modified final layer (`nn.Linear` for 2 classes).


- **Small U-Net:** Encoder-decoder architecture with skip connections (`DoubleConv`, `MaxPool2d`, `ConvTranspose2d`).


## Results & Analysis

### Dataset Information & Samples


- **Training Samples:** 3,680 (1,188 Cat, 2,492 Dog)


- **Testing Samples:** 3,669 (1,183 Cat, 2,486 Dog)


- **Original Image Dimensions:** Variable (e.g., 394 x 500), resized to 128 x 128 for training/evaluation.


- **Processed Batch Shapes:**

  - `images`: `torch.Size([16, 3, 128, 128])`
  - `labels`: `torch.Size([16])`
  - `masks`: `torch.Size([16, 128, 128])`

### Classification Accuracy Table

| Model                                  | Test Accuracy (%) |
|----------------------------------------|-------------------|
| Majority-Class Baseline (Always Dog)   | 67.76%            |
| Regular CNN                            | 68.66%            |
| ResNet-18                              | 80.84%            |

### U-Net Pixel Accuracy


- **U-Net Pixel Accuracy:** 78.74%


### Training Losses & Visualizations


- **Observation:** Training losses for all three models steadily decreased across the 10 epochs. ResNet-18 showed rapid convergence, while the regular CNN and U-Net exhibited stable, gradual loss reduction.


### Segmentation Examples & Analysis


- **Shape Capture:** The small U-Net successfully captures the general shape and localization of the pets.


- **Insights:** Segmentation reveals precise pixel-level boundaries and semantic masking, whereas classification only yields a global cat/dog label.


## GenAI Acknowledgment

Generative AI tools were utilized to assist with structuring boilerplate PyTorch modules, formatting evaluation loops, and organizing documentation according to assignment guidelines. All code was reviewed, tested, and verified on the NMSU Simurgh server. (Google Gemini & Claude)