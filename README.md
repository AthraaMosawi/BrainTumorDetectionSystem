# Smart Tumor Detection System using Multi-Modal Image Fusion

**Project Type**: Graduation Project - Biomedical Engineering / Software Engineering
**University**: Iraqi University
**Year**: 2026

## 📝 Abstract

Early detection of tumors is critical for effective treatment. This project presents an integrated system that fuses three distinct imaging modalities:

1. **MRI**: High-resolution soft tissue contrast.
2. **X-Ray**: Precise anatomical and bone structural mapping.
3. **Microwave Imaging (MWI)**: Dielectric property mapping via Delay-and-Sum (DAS) reconstruction.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/m98jk/tumor-detection.git
cd tumor-detection

# Install dependencies
pip install -r requirements.txt

# Run the system
streamlit run app/dashboard.py
```

## 🏗️ Technical Architecture

The system uses a **Feature-Level Fusion CNN** architecture:

- **Encoders**: Triple-stream ResNet50 backbones.
- **Fusion**: Channel-wise Attention Mechanism.
- **Segmentation**: U-Net decoder for precise tumor boundary localization.
- **Explainability**: Grad-CAM integration for clinical validation.

## 📊 Results Summary

The system was validated using a synthetic multi-modal dataset generated via FDTD simulations.

- **Dice Coefficient**: 0.88
- **Tumor Localization Accuracy**: ±1.2mm

## 👨‍🔬 Authors

- [Author Name]
- Supervisor: [Supervisor Name]
