# Smart Tumor Detection System Using X-ray, MRI, and Microwave Images

**Project Type**: Graduation Project – Biomedical Engineering / Software Engineering  
**University**: Iraqi University  
**Year**: 2026  

---

## 📝 Abstract

Early detection of tumors is critical for effective treatment. This project presents an integrated system that fuses three distinct imaging modalities:

1. **MRI** – High-resolution soft tissue contrast (T2-weighted).
2. **X-Ray** – Precise anatomical and bone structural mapping.
3. **Microwave Imaging (MWI)** – Dielectric property mapping via Gaussian diffusion reconstruction.

The system employs a **hybrid decision engine** combining a trained Keras deep learning model with classical computer vision (CLAHE preprocessing, thresholding, contour detection) for robust, explainable tumor localization.

---

## 🏗️ Technical Architecture

```
Image Acquisition
    ├── MRI  (T2 soft tissue)
    ├── X-Ray (anatomical)
    └── MWI   (derived / uploaded)
         ↓
Pre-processing  →  Denoise + CLAHE Enhancement
         ↓
Feature Analysis
    ├── Deep Learning: Keras CNN (brainTumor.keras)
    └── Computer Vision: Thresholding + Contour Detection
         ↓
Hybrid Decision Fusion  →  ML weight + CV vote score
         ↓
Output: Classification · Confidence · Localization · Heatmaps
```

---

## 🛠️ Installation & Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd BrainTumorDetectionSystem

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit dashboard
streamlit run app/dashboard.py
```

---

## 🖼️ Sample Images

Pre-loaded sample images are located in the `sample/` folder:

| File | Modality |
|------|----------|
| `mri1.jpg`, `mri3.jpg`, `mri5.jpg`, `mre2.jpg`, `mrei4.jpg` | MRI |
| `xray.jpg` | X-Ray |

In the sidebar, select **"Load from Sample Folder"** and choose an MRI and an X-Ray image — the MWI map will be auto-derived. Then press **Run Multi-Modal Fusion Detection**.

---

## 📁 Project Structure

```
BrainTumorDetectionSystem/
├── app/
│   ├── dashboard.py          # Main Streamlit application
│   └── brainTumor.keras      # Pre-trained Keras model
├── processing/
│   └── mwi_reconstruction.py # MWI DAS algorithm + Image processor
├── data/
│   └── synthetic_generator.py # Synthetic dataset generator (methodology)
├── sample/                   # Real test images (MRI + X-Ray)
├── requirements.txt
└── README.md
```

---

## 👨‍🔬 Authors

- [Author Name]  
- Supervisor: [Supervisor Name]

---

> ⚠️ **Disclaimer**: This system is intended for research and educational purposes only. It is not a clinical diagnostic device.
