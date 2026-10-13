# Assignment: Image Classification and Segmentation with CNNs

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
