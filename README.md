# Computer Vision Competition: Fine-Grained Classification 

## Fine-Grained Classification in Computer Vision

...

## Repository Overview

This repository contains code for experiments conducted, methods utilized, and utilities. Here's a brief overview of the main components:

```
├── example.ipynb
├── experiments
│   ├── EfficientNetB5_FGVCAircraft_SAM
│   ┆   ├── config.json
│   ┆   └── events.out.tfevents.1717092624.lab4G24P6.1885
│   ┆
├── methods
│   ├── CMAL
│   ├── PIM
│   └── SAM
├── requirements.txt
├── train
│   ├── test.py
│   └── train.py
└── utility
    ├── gradcam
    ┆
```

### Experiments

The `experiments` directory contains configurations (`config.json`) to reproduce results and TensorBoard log files (`events.out.tfevents`) for visualizing them.

### Methods

The `methods` directory includes optimization methods and additional FGCV blocks to stuck on top of several backbones:

- **SAM (Sharpness-Aware Minimization)**: Implementation of the SAM method from the paper ["Sharpness-Aware Minimization for Efficiently Improving Generalization"](https://arxiv.org/pdf/2010.01412) by [davda54](https://github.com/davda54/sam/tree/main). SAM simultaneously minimizes loss value and loss sharpness. In particular, it seeks parameters that lie in neighborhoods having uniformly low loss. SAM improves model generalization and yields [SoTA performance for several datasets](https://paperswithcode.com/paper/sharpness-aware-minimization-for-efficiently-1). Additionally, it provides robustness to label noise on par with that provided by SoTA procedures that specifically target learning with noisy labels.

<br>

<p align="center">
  <img src="Images/Resnet_SAM_loss_landscape.png" width="512"/>  
</p>

<p align="center">
  <sub><em>ResNet loss landscape at the end of training with and without SAM. Sharpness-aware updates lead to a significantly wider minimum, which then leads to better generalization properties.</em></sub>
</p>

<br> 

- **PIM**: (brief description, paper, images)

- **CMAL**: (brief description, paper, images)
  
### Training

The `train` folder contains essential scripts for training and testing:

- `train.py`: Instantiate a `Trainer` class, which manages training, testing, logging, etc.
- `test.py`: Script for testing trained models.

### Utility

The `utility` directory comprises various utility scripts and the implementation of Grad-CAM. Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique to visualize and understand the decisions made by a Convolutional Neural Network (CNN) by highlighting the regions of the input image that are important for predictions. The Grad-CAM code is adapted from the official repository available [here](https://github.com/jacobgil/pytorch-grad-cam).
