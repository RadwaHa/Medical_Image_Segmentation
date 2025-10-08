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
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
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
|            Organ            |
|-----------------------------|
|  Liver  |  Lungs  |  Brain  |

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

## 💾 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 5GB+ free disk space

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/medical-organ-segmentation.git
cd medical-organ-segmentation
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n medseg python=3.8
conda activate medseg
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models**
```bash
# Script to download model checkpoints
python scripts/download_models.py
```

5. **Verify installation**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 🚀 Usage

### Basic Usage

```bash
# Launch GUI application
python main.py
```

### Command Line Interface

```bash
# Segment single image
python src/segmentation/segment.py \
    --input path/to/image.nii.gz \
    --model unet \
    --organ liver \
    --output results/

# Batch processing
python src/segmentation/batch_segment.py \
    --input_dir data/images/ \
    --model medsam \
    --organ brain

# Evaluate predictions
python src/evaluation/evaluate.py \
    --predictions results/predictions/ \
    --ground_truth data/ground_truth/ \
    --metrics dice iou hausdorff
```

### Programmatic Usage

```python
from src.models import MedSAM, DeepLabV3Plus, UNet
from src.segmentation import Segmentor
from src.evaluation import compute_metrics
from src.visualization import Visualizer3D

# Initialize model
model = UNet(in_channels=1, out_channels=3)
model.load_checkpoint('models/unet_checkpoint.pth')

# Perform segmentation
segmentor = Segmentor(model)
segmentation = segmentor.segment(image, organ='liver')

# Evaluate
metrics = compute_metrics(segmentation, ground_truth)
print(f"Dice: {metrics['dice']:.4f}")

# Visualize
visualizer = Visualizer3D()
visualizer.add_organ(segmentation, color=(255, 0, 0), opacity=0.8)
visualizer.show()
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

## 📁 Project Structure

```
medical-organ-segmentation/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License file
│
├── main.py                     # Main application entry point
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/                 # Model implementations
│   │   ├── __init__.py
│   │   ├── medsam.py
│   │   ├── deeplabv3.py
│   │   └── unet.py
│   ├── segmentation/           # Segmentation pipeline
│   │   ├── __init__.py
│   │   ├── segmentor.py
│   │   └── preprocessing.py
│   ├── evaluation/             # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── visualization/          # 3D visualization
│   │   ├── __init__.py
│   │   └── visualizer_3d.py
│   └── gui/                    # GUI components
│       ├── __init__.py
│       └── main_window.py
│
├── data/                       # Data directory
│   ├── sample/                 # Sample images
│   └── README.md
│
├── models/                     # Trained model checkpoints
│   ├── medsam_checkpoint.pth
│   ├── deeplabv3_checkpoint.pth
│   └── unet_checkpoint.pth
│
├── results/                    # Output directory
│   └── .gitkeep
│
├── docs/                       # Documentation
│   ├── images/                 # Screenshots
│   ├── videos/                 # Demo videos
│   └── user_guide.md
│
├── tests/                      # Unit tests
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_visualization.py
│
└── scripts/                    # Utility scripts
    ├── download_models.py
    └── prepare_data.py
```

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
