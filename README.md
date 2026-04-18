# Adaptive SNR-Aware Zero-DCE for Domain-Generalizable Low-Light Image Enhancement

## 📌 Overview

Low-light image enhancement (LLIE) is a critical task in computer vision, especially for applications such as surveillance, autonomous systems, and mobile photography.

This project proposes an enhanced version of **Zero-Reference Deep Curve Estimation (Zero-DCE)** by integrating an **SNR-aware (Signal-to-Noise Ratio)** mechanism to improve robustness under extremely dark conditions and across different domains.

Unlike supervised approaches, this method does **not require paired low/normal-light datasets**, making it more practical for real-world scenarios.

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

## 🏗️ Project Structure

```
ZeroDCE_Skripsi/
│
├── model/                  # Model architecture (Zero-DCE + SNR modules)
├── dataloader/             # Data loading and preprocessing
├── utils/                  # Utility functions (metrics, helpers)
├── lowlight_train.py       # Training script
├── lowlight_test.py        # Inference / evaluation script
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

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

### 🔸 Inference (Testing)

Run enhancement on low-light images:

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

> ⚠️ Note:
> All images are resized to a common resolution before evaluation to ensure consistent metric computation.

---

## 🖼️ Results

### Example Outputs

| Input (Low-Light) | Enhanced Output |
| ----------------- | --------------- |
| (add image)       | (add image)     |

💡 Tip: Upload images to your repo and replace `(add image)` with:

```markdown
![input](path/to/image.png)
```

---

## ⚠️ Important Notes

* Dataset is not included due to size limitations
* Pretrained model weights (`.pth`) are not included
* Ensure input and ground truth images have **matching dimensions** before computing metrics
* Resize operations may slightly affect evaluation results

---

## 🔬 Research Contribution

* Combines Zero-DCE with SNR-aware modeling
* Improves robustness under extreme low-light conditions
* Enhances domain generalization capability
* Maintains lightweight and efficient architecture

---

## 📚 References

* Guo et al., *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*
* SNR-aware Low-Light Image Enhancement (CVPR-based methods)
* Retinex-based image enhancement theory

---

## 👤 Author

**Francesco Sebastian**
**Rudy Hartanto**
Computer Science – Bina Nusantara University

---

## 📄 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgements

This work is developed as part of undergraduate thesis research in low-light image enhancement.

---

## 🚀 Future Work

* Improve noise modeling using diffusion-based approaches
* Integrate frequency-domain enhancement (Fourier-based LLIE)
* Benchmark against state-of-the-art models (Diffusion LLIE, SNR-Net)

---
