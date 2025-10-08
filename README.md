# Medical_Image_Segmentation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive medical image segmentation system for multi-organ segmentation (liver, lungs, brain) using state-of-the-art deep learning models with interactive 3D visualization and quantitative evaluation metrics.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

This project implements an advanced medical image segmentation pipeline that enables healthcare professionals and researchers to accurately segment and visualize three critical organs (liver, lungs, and brain) from medical imaging data. The system integrates three powerful deep learning architectures.

### Key Highlights
- **Multi-Model Approach**: Leverages MedSAM, DeepLabV3+, and U-Net for robust segmentation
- **Multi-Organ Support**: Segments liver, lungs, and brain with distinct color-coded outputs
- **Comprehensive Evaluation**: Implements Dice coefficient, IoU, and Hausdorff distance metrics
- **Interactive 3D Visualization**: Real-time 3D rendering with customizable parameters

## ✨ Features

### Segmentation Models
- **MedSAM** (Medical Segment Anything Model): Foundation model for medical image segmentation
- **DeepLabV3+**: State-of-the-art semantic segmentation with atrous convolution
- **U-Net**: Classic architecture optimized for biomedical image segmentation

### Supported Organs
|  Organ  |
|---------|
|  Liver  |
|  Lungs  |
|  Brain  |

### Evaluation Metrics
- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **Intersection over Union (IoU)**: Quantifies segmentation accuracy
- **Hausdorff Distance**: Evaluates boundary accuracy in millimeters

### Visualization Features
- Real-time 3D rendering
- 3 Colored organ parts
- Multiple viewing angles and rotation
- Export capabilities for visualizations

## 🎬 Demo

### Liver Segmentation Results
![Liver Segmentation Results](https://github.com/user-attachments/assets/168034e7-0886-47d6-9c92-40f077453770)


### Liver 3D Visualization
![Liver 3D Visualization](https://github.com/user-attachments/assets/f379c106-0f09-475b-90e9-3d5121fc7ecc)

### Brain Segmentation Results
![Brain Segmentation Results](https://github.com/user-attachments/assets/3c1ef8ca-591f-45c5-a100-0589f859e869)

### Brain 3D Visualization
![Brain 3D Visualization](https://github.com/user-attachments/assets/bc143bbc-2e66-47f6-8052-f82643f7a906)

## 🏗️ Architecture

```
┌───────────────────────────────────────────────┐
│              Medical Image Input              │
└─────────────────────┬─────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌────────────────────┐       ┌────────────────────┐
│  - Model Selection │       │  - Organ Selection │
│  - MedSAM          │       │  - Liver           │
│  - DeepLabV3+      │       │  - Lungs           │
│  - U-Net           │       │  - Brain           │
└────────┬───────────┘       └────────┬───────────┘
         │                            │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │  Segmentation Pipeline  │
         │  - Preprocessing        │
         │  - Model Inference      │
         │  - Post-processing      │
         └────────┬────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
        ▼                    ▼
┌──────────────┐    ┌─────────────────-─┐
│  Evaluation  │    │  3D Visualization │
│  - IoU       │    │                   │
│  - Dice      │    │  - Part changing  │
│  - Hausdorff │    │                   │
└──────────────┘    └─────────────────-─┘
```

## Requirements

### Hardware
- A mouse or a trackpad for navigation and interaction
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 5GB+ free disk space

### Software
- Python 3.8 or higher
- Required libraries:
  - Numpy
  - torch
  - Matplotlib
  - nibabel
  - scipy
  

## 💾 Installation

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/RadwaHa/Medical_Image_Segmentation.git
```

2. **Navigate to the project directory**
```bash
cd Medical_Image_Segmentation
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🤖 Models

### MedSAM (Medical Segment Anything Model)
- **Architecture**: Transformer-based foundation model
- **Parameters**: ~90M
- **Input Size**: 1024x1024
- **Training Data**: Large-scale medical imaging datasets
- **Strengths**: Zero-shot capability, robust to domain shift

### DeepLabV3+
- **Architecture**: Atrous Spatial Pyramid Pooling (ASPP) with encoder-decoder
- **Backbone**: ResNet-101
- **Parameters**: ~43M
- **Input Size**: 512x512
- **Strengths**: Multi-scale context, sharp boundaries

### U-Net
- **Architecture**: Classic encoder-decoder with skip connections
- **Depth**: 5 levels
- **Parameters**: ~31M
- **Input Size**: 256x256
- **Strengths**: Efficient, excellent for limited data

## 📊 Evaluation Metrics

### Dice Coefficient (F1 Score)
```
Dice = 2 × |X ∩ Y| / (|X| + |Y|)
```
- Range: [0, 1], higher is better
- Measures overlap between prediction and ground truth

### Intersection over Union (IoU / Jaccard Index)
```
IoU = |X ∩ Y| / |X ∪ Y|
```
- Range: [0, 1], higher is better
- Quantifies segmentation accuracy

### Hausdorff Distance
```
HD = max(h(X, Y), h(Y, X))
where h(X, Y) = max_{x∈X} min_{y∈Y} ||x - y||
```
- Unit: millimeters, lower is better
- Measures maximum boundary error

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Coding Standards
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 🙏 Acknowledgments

- **MedSAM Team** for the foundation model
- **PyTorch Team** for the deep learning framework
- **VTK Community** for 3D visualization tools
- **Medical Imaging Dataset Providers** for training data

## 📧 Contact

**Project Maintainer**: [Radwa Hamdy]
- Email: radwahamdy922@gmail.com
- GitHub: [@RadwaHa](https://github.com/RadwaHa)
- LinkedIn: [Radwa Hamdy](https://linkedin.com/in/radwa-hamdy1)

**Project Maintainer**: [Habiba Ibrahem]
- Email: habeba.ibrahem2016@gmail.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/habeba-zaki)

- **Project Maintainer**: [Your Name]
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

- **Project Maintainer**: [Your Name]
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{medical_organ_segmentation,
  author = {Radwa Hamdy},
  title = {Medical Organ Segmentation System},
  year = {2024},
  url = {https://github.com/RadwaHa/medical-organ-segmentation}
}

---

**⭐ If you find this project helpful, please consider giving it a star!**

*Last Updated: October 2025*
