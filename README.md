# Adaptive SNR-Aware Zero-DCE for Domain-Generalizable Low-Light Image Enhancement

This project proposes an An improved implementation of Zero-DCE with domain generalization, adaptive loss functions, and perceptual quality improvements for low-light image enhancement (LLIE).
Unlike supervised approaches, this method does **not require paired low/normal-light datasets**, making it more practical for real-world scenarios.

## 📌 Overview
Low-light image enhancement (LLIE) is a critical task in computer vision, especially for applications such as surveillance, autonomous systems, and mobile photography.
This project extends the original Zero-DCE architecture with several training and loss improvements targeting better PSNR, SSIM, MAE, and BRISQUE scores on mixed indoor/outdoor low-light datasets.
Key improvements over the original Zero-DCE:

Mixup-based Domain Generalization — cross-domain image blending in the dataloader to improve generalization across different lighting domains
Adaptive Exposure Loss — dynamic exposure weight based on per-batch input brightness, replacing a fixed weight
Local Patch-wise SNR Loss — spatially-aware noise regularization that focuses on the darkest, noisiest regions
Warmup + Cosine LR Scheduling — prevents early LR collapse and training plateau
BRISQUE evaluation via piq — cleaner no-reference quality metri
---

## 🗂️ Project Structure
```
├── model.py              # Zero-DCE network (enhance_net_nopool)
├── dataloader.py         # Dataset loader with Mixup-DG augmentation
├── lowlight_train.py     # Training script with all improved losses
├── lowlight_test.py      # Evaluation script with BRISQUE, PSNR, SSIM, MAE
├── Myloss.py             # All loss functions (L_TV, L_spa, L_color, L_exp, L_per, L_SSIM)
├── find_exposure_target.py  # Diagnostic script to calibrate L_exp to your dataset
├── data/
│   └── train_data/
│       └── Train_Mix/    # Training images (subfolders supported)
└── snapshots/            # Saved model checkpoints
```
## 📦 Training Data

Note: The dataset is not included in this repository due to its size.
Download it here: [Google Drive](https://drive.google.com/drive/folders/1Crg60dNMr9VxVW7r13pPTZkyvLUklf22?usp=sharing)

After downloading, place the data in the following structure:
```
data/
└── train_data/
        └── (your downloaded folders here)
```
The dataloader recursively collects all .png, .jpg, and .jpeg files from subfolders automatically — no code changes needed.

---
## 🧠 Methodology

### 🔹 Zero-DCE (Zero-Reference Deep Curve Estimation)

* Learns pixel-wise curve mapping for illumination adjustment
* No ground truth required (zero-reference learning)
* Lightweight and efficient network
* Iterative curve estimation for adaptive enhancement

### 🔹 SNR-Aware Enhancement

* Incorporates signal-to-noise ratio into the enhancement process
* Helps distinguish between noise and meaningful signal
* Improves enhancement quality in extremely low-light environments
* Enhances generalization across datasets (domain shift)

### 🔹 Combined Approach

The proposed pipeline:

1. Input low-light image
2. Estimate SNR map
3. Apply adaptive curve estimation (Zero-DCE)
4. Produce enhanced output image
---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/Azuna218/ZeroDCE_Skripsi.git
cd ZeroDCE_Skripsi
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🧪 Testing

Update model_path in lowlight_test.py to point to your snapshots checkpoint:
pythonmodel_path = 'snapshots/yourfolder/yourepoch.pth'

Then run: 
```bash
python lowlight_test.py
```

---


### 🔸 Training

Train the model from scratch:

```bash
python lowlight_train.py
```

---

## 📊 Evaluation Metrics

This project uses the following quantitative metrics:

* **MAE (Mean Absolute Error)**
* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**
* **BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)**


## ⚠️ Important Notes

* Dataset is not included due to size limitations, it can be downloaded from the dataset link above.
* Pretrained model weights (`.pth`) are included in snapshots, use **Final Version** for the latest pretrained weight.
* Ensure input and ground truth images have **matching name** before computing metrics.

---

## 🔬 Research Contribution

* Combines Zero-DCE with SNR-aware modeling
* Improves robustness under extreme low-light conditions
* Enhances domain generalization capability
* Maintains lightweight and efficient architecture

---

## 👤 Author

**Francesco Sebastian**
**Rudy Hartanto**
Computer Science – Bina Nusantara University

---

## ⭐ Acknowledgements

This work is developed as part of undergraduate thesis research in low-light image enhancement.
Original Zero-DCE by Li et al.:
Chongyi Li, Chunle Guo, Linghao Han, Jun Jiang, Ming-Ming Cheng, Jinwei Gu, Chen Change Loy. "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement." CVPR 2020.
Original repository: https://github.com/Li-Chongyi/Zero-DCE
---
