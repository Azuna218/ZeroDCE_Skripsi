"""
Run this BEFORE training to find the optimal exposure target
for your specific test set. This tells you what mean brightness
your ground truth images actually have, so L_exp targets the right value.

Usage:
    python find_exposure_target.py
"""

import os
import glob
import numpy as np
from PIL import Image


def analyze_brightness(folder, label):
    images = glob.glob(os.path.join(folder, "**", "*.*"), recursive=True)
    images = [x for x in images if x.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not images:
        print(f"[{label}] No images found in: {folder}")
        return None

    means = []
    for path in images:
        img = np.asarray(Image.open(path).convert('RGB')) / 255.0
        means.append(np.mean(img))

    arr = np.array(means)
    print(f"\n[{label}] ({len(images)} images)")
    print(f"  Mean brightness : {arr.mean():.4f}  ← use this as L_exp target")
    print(f"  Std             : {arr.std():.4f}")
    print(f"  Min             : {arr.min():.4f}")
    print(f"  Max             : {arr.max():.4f}")
    print(f"  Median          : {np.median(arr):.4f}")
    return arr.mean()


if __name__ == "__main__":

    # ── Edit these paths to match your folder structure ──
    LOW_FOLDER  = "data/test_data/"       # your low-light test images
    HIGH_FOLDER = "High/"                  # your ground truth test images
    TRAIN_FOLDER = "data/train_data/Train_Mix"  # your training images
    # ─────────────────────────────────────────────────────

    low_mean   = analyze_brightness(LOW_FOLDER,   "Test  LOW ")
    high_mean  = analyze_brightness(HIGH_FOLDER,  "Test  HIGH (ground truth)")
    train_mean = analyze_brightness(TRAIN_FOLDER, "Train LOW ")

    print("\n" + "=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)

    if high_mean is not None:
        print(f"Set L_exp target to: {high_mean:.2f}  (mean of your GT images)")
        print(f"In lowlight_train.py:")
        print(f"  L_exp_strict  = Myloss.L_exp(16, {high_mean:.2f})")
        print(f"  L_exp_relaxed = Myloss.L_exp(16, {max(0.4, high_mean - 0.1):.2f})")
    else:
        print("Could not find ground truth folder.")
        print("Check HIGH_FOLDER path in this script.")