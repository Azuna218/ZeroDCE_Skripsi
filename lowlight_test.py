import os
import time
import glob
import sys

import torch
import torchvision
import numpy as np
import cv2
from PIL import Image

import model

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Fix for BRISQUE dependency (svmutil)
from libsvm import svmutil
sys.modules['svmutil'] = svmutil
from brisque import BRISQUE

# Initialize BRISQUE model
brisque_model = BRISQUE()


def calculate_snr(image):
    """
    Compute Signal-to-Noise Ratio (SNR) for an image.

    Args:
        image (numpy.ndarray): Input image in range [0, 1]

    Returns:
        float: SNR value
    """
    gray = np.mean(image, axis=2)
    return np.mean(gray) / (np.std(gray) + 1e-8)


def calculate_brisque(image):
    """
    Compute BRISQUE score (no-reference image quality metric).

    Args:
        image (numpy.ndarray): Input image in range [0, 1]

    Returns:
        float: BRISQUE score (lower is better)
    """
    img_uint8 = (image * 255).astype(np.uint8)
    return brisque_model.score(img_uint8)


def calculate_mae(img1, img2):
    """
    Compute Mean Absolute Error (MAE) between two images.

    Args:
        img1, img2 (numpy.ndarray): Images in range [0, 1]

    Returns:
        float: MAE value
    """
    return np.mean(np.abs(img1 - img2))


def calculate_ssim(img1, img2):
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        img1, img2 (numpy.ndarray): Images in range [0, 1]

    Returns:
        float: SSIM value
    """
    g1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return ssim(g1, g2)


def calculate_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1, img2 (numpy.ndarray): Images in range [0, 1]

    Returns:
        float: PSNR value
    """
    return psnr(img1, img2, data_range=1.0)

def align_shapes(img1, img2):
    if img1.shape != img2.shape:
        h, w = img1.shape[:2]
        img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
        img2_pil = img2_pil.resize((w, h), Image.Resampling.LANCZOS)
        img2 = np.asarray(img2_pil) / 255.0
    return img1, img2


def process_image(image_path, model_net,
                  brisque_list, mae_list, ssim_list, psnr_list):
    """
    Run inference on a single image and compute evaluation metrics.

    Args:
        image_path (str): Path to input image
        model_net (torch.nn.Module): Trained model
        metric lists: Lists to accumulate results
    """

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.asarray(image) / 255.0

    snr_before = calculate_snr(image_np)

    # Convert to tensor
    input_tensor = torch.from_numpy(image_np).float()
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).cuda()

    # Inference
    start_time = time.time()
    _, enhanced_image, _ = model_net(input_tensor)

    # Convert output to numpy
    enhanced_np = enhanced_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    enhanced_np = np.clip(enhanced_np, 0, 1)

    snr_after = calculate_snr(enhanced_np)

    # Compute BRISQUE (no ground truth required)
    brisque_score = calculate_brisque(enhanced_np)
    brisque_list.append(brisque_score)

    # Attempt to load ground truth
    filename = os.path.basename(image_path)
    gt_path = os.path.join("data/High", filename)

    if os.path.exists(gt_path):
        gt_img = Image.open(gt_path).convert('RGB')
        gt_np = np.asarray(gt_img) / 255.0

        mae_score = calculate_mae(enhanced_np, gt_np)
        ssim_score = calculate_ssim(enhanced_np, gt_np)
        psnr_score = calculate_psnr(enhanced_np, gt_np)

        mae_list.append(mae_score)
        ssim_list.append(ssim_score)
        psnr_list.append(psnr_score)

        print(f"MAE: {mae_score:.4f} | SSIM: {ssim_score:.4f} | PSNR: {psnr_score:.2f}")
    else:
        print("Ground truth not found")

    # Print per-image results
    print(f"Processing: {image_path}")
    print(f"Inference time: {time.time() - start_time:.4f} seconds")
    print(f"SNR before: {snr_before:.4f} | after: {snr_after:.4f}")
    print(f"BRISQUE: {brisque_score:.2f}")
    print("-" * 40)

    # Save output image
    output_tensor = torch.from_numpy(
        enhanced_np.transpose(2, 0, 1)
    ).unsqueeze(0).float().cuda()

    output_path = image_path.replace('test_data', 'MixupResult50')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchvision.utils.save_image(output_tensor, output_path)


if __name__ == '__main__':
    with torch.no_grad():

        model_path = 'snapshots/MixupDGv2/Epoch50.pth'

        # Load model
        DCE_net = model.enhance_net_nopool().cuda()
        DCE_net.load_state_dict(torch.load(model_path, weights_only=True))
        DCE_net.eval()

        input_folder = 'data/test_data/'

        # Metric accumulators
        brisque_list = []
        mae_list = []
        ssim_list = []
        psnr_list = []

        # Recursively collect image files
        test_images = glob.glob(os.path.join(input_folder, "**", "*.*"), recursive=True)
        test_images = [
            x for x in test_images
            if x.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print("Total test images:", len(test_images))

        for image_path in test_images:
            process_image(
                image_path,
                DCE_net,
                brisque_list,
                mae_list,
                ssim_list,
                psnr_list
            )

        # Final aggregated results
        print("\n===== FINAL RESULTS =====")

        if brisque_list:
            print("Average BRISQUE:", np.mean(brisque_list))

        if mae_list:
            print("Average MAE:", np.mean(mae_list))

        if ssim_list:
            print("Average SSIM:", np.mean(ssim_list))

        if psnr_list:
            print("Average PSNR:", np.mean(psnr_list))