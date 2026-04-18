import os
import argparse

import torch
import torch.nn as nn
import torch.optim
import torchvision

import dataloader
import model
import Myloss


def weights_init(m):
    """
    Initialize model weights.

    Convolution layers: Normal(0, 0.02)
    BatchNorm layers: Normal(1, 0.02), bias = 0
    """
    classname = m.__class__.__name__

    if 'Conv' in classname:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)

    elif 'BatchNorm' in classname:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def compute_local_snr_loss(image, patch_size=16):
    """
    Compute patch-wise LOCAL Signal-to-Noise Ratio loss.

    Unlike a global SNR estimate, this function divides the image into
    non-overlapping patches of size (patch_size x patch_size) and computes
    SNR independently per patch. This allows the model to:
        - Identify spatially localized noisy dark regions
        - Apply stronger regularization to high-noise patches
        - Leave cleaner regions relatively unaffected

    This directly supports the paper's claim of "local noise estimation"
    and aligns with the SNR-Aware approach of Xu et al. (CVPR 2022).

    Mathematical formulation per patch p:
        signal_p = mean(gray_p)
        noise_p  = std(gray_p)
        SNR_p    = signal_p / (noise_p + eps)
        loss_p   = 1 / (SNR_p + eps)

    Minimizing mean(loss_p) across all patches = maximizing mean patch SNR.
    Darker, noisier patches (low signal, high std) will have the highest
    individual loss_p values and therefore contribute most to the gradient,
    naturally focusing the model's attention on problematic regions.

    Args:
        image (torch.Tensor): Enhanced image tensor [B, C, H, W], range [0, 1]
        patch_size (int): Size of each square patch. Default 16 matches
                          the patch size used in L_exp for consistency.

    Returns:
        torch.Tensor: Scalar SNR loss value
    """

    # Convert RGB to grayscale by averaging channels → [B, 1, H, W]
    gray = torch.mean(image, dim=1, keepdim=True)

    # Use unfold to extract non-overlapping patches
    # After unfold: [B, 1, num_h, num_w, patch_size, patch_size]
    patches = gray.unfold(2, patch_size, patch_size) \
                  .unfold(3, patch_size, patch_size)

    # Compute per-patch statistics along the last two dimensions (spatial)
    signal = patches.mean(dim=(-2, -1))   # [B, 1, num_h, num_w]
    noise  = patches.std(dim=(-2, -1))    # [B, 1, num_h, num_w]

    # Per-patch SNR map
    snr_map = signal / (noise + 1e-8)     # [B, 1, num_h, num_w]

    # Loss = mean of inverse SNR across all patches and batch items
    # High-noise patches (low SNR) dominate → model focuses on them
    loss = torch.mean(1.0 / (snr_map + 1e-8))

    return loss


def compute_exposure_weight(mean_light):
    """
    Adaptive exposure weighting based on input brightness.

    Dynamically adjusts the exposure loss weight per batch so that
    very dark images receive stronger correction while already-bright
    images are treated more conservatively. This is the core of the
    "Adaptive" claim in the paper title.

    Args:
        mean_light (float): Average brightness of input image batch, [0, 1]

    Returns:
        float: Exposure loss weight
    """
    if mean_light < 0.3:
        return 10.0   # very dark — aggressive correction
    elif mean_light < 0.5:
        return 7.0    # moderately dark — standard correction
    else:
        return 3.0    # already relatively bright — conservative correction


def train(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Model initialization
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.apply(weights_init)

    if config.load_pretrain:
        DCE_net.load_state_dict(
            torch.load(config.pretrain_dir, weights_only=True)
        )

    # Dataset with Mixup-DG enabled
    # mixup_prob: probability per sample of applying cross-image Mixup
    # mixup_alpha: Beta distribution parameter (0.4 keeps blends realistic)
    train_dataset = dataloader.lowlight_loader(
        config.lowlight_images_path,
        mixup_prob=config.mixup_prob,
        mixup_alpha=config.mixup_alpha
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Loss functions
    L_color = Myloss.L_color()
    L_spa   = Myloss.L_spa()
    L_exp   = Myloss.L_exp(16, 0.54)
    L_TV    = Myloss.L_TV()

    # Optimizer
    optimizer = torch.optim.Adam(
        DCE_net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler — reduces LR by 50% every 25 epochs
    # This allows aggressive learning early and fine refinement later,
    # addressing the observed convergence plateau around epoch 50.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.5
    )

    DCE_net.train()

    for epoch in range(config.num_epochs):

        epoch_loss = 0.0

        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()

            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            # Compute base losses
            loss_tv  = 700 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhanced_image))

            # Adaptive exposure loss — weight changes per batch
            mean_light = torch.mean(img_lowlight).item()
            exp_weight = compute_exposure_weight(mean_light)
            loss_exp   = exp_weight * torch.mean(L_exp(enhanced_image))

            # Local patch-wise SNR loss (replaces global SNR)
            # patch_size=16 matches L_exp for spatial consistency
            loss_snr = config.snr_weight * compute_local_snr_loss(
                enhanced_image, patch_size=16
            )

            # Total loss
            loss = loss_tv + loss_spa + loss_col + loss_exp + loss_snr

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                DCE_net.parameters(),
                config.grad_clip_norm
            )

            optimizer.step()

            epoch_loss += loss.item()

            if (iteration + 1) % config.display_iter == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch: {epoch} | Iter: {iteration + 1} | "
                    f"Loss: {loss.item():.4f} | "
                    f"SNR_loss: {loss_snr.item():.4f} | "
                    f"Exp_w: {exp_weight:.1f} | "
                    f"LR: {current_lr:.6f}"
                )

        # Step LR scheduler at end of each epoch
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(f"=== Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} ===")

        # Save model checkpoint after each epoch
        torch.save(
            DCE_net.state_dict(),
            os.path.join(config.snapshots_folder, f"Epoch{epoch}.pth")
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--lowlight_images_path', type=str,
                        default="data/Train_Mix/")

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    # Training schedule
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--display_iter', type=int, default=10)

    # Snapshots
    parser.add_argument('--snapshots_folder', type=str,
                        default="snapshots/MixupDGv2/")
    parser.add_argument('--load_pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_dir', type=str,
                        default="snapshots/MixupDG/Epoch4.pth")

    # Mixup-DG parameters
    parser.add_argument('--mixup_prob', type=float, default=0.5,
                        help='Probability of applying Mixup per sample (0=disabled)')
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                        help='Beta distribution alpha for Mixup lambda sampling')

    # Local SNR loss weight
    parser.add_argument('--snr_weight', type=float, default=0.20,
                        help='Weight for local patch-wise SNR loss')

    config = parser.parse_args()

    os.makedirs(config.snapshots_folder, exist_ok=True)

    train(config)
