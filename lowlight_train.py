import os
import argparse
import math

import torch
import torch.nn as nn
import torch.optim
import torchvision

import dataloader
import model
import Myloss


def weights_init(m):
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
    gray    = torch.mean(image, dim=1, keepdim=True)
    patches = gray.unfold(2, patch_size, patch_size) \
                  .unfold(3, patch_size, patch_size)
    signal  = patches.mean(dim=(-2, -1))
    noise   = patches.std(dim=(-2, -1))
    noise   = torch.clamp(noise, min=0.01)
    snr_map = signal / noise
    snr_map = torch.clamp(snr_map, min=0.1, max=100.0)
    return torch.mean(1.0 / snr_map)


def compute_exposure_weight(mean_light):
    """
    Tuned-down weights — previous values (10/7/3) were over-brightening
    images, which hurt BRISQUE and caused unnatural output.
    """
    if mean_light < 0.15:
        return 8.0
    elif mean_light < 0.3:
        return 5.0
    elif mean_light < 0.5:
        return 3.0
    else:
        return 1.5


def get_scheduler(optimizer, num_epochs, warmup_epochs=5):
    """
    Linear warmup + cosine decay with 0.1 floor.
    Floor prevents LR dying too early — fixes plateau seen at epoch 14-15.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.apply(weights_init)

    if config.load_pretrain:
        state_dict = torch.load(config.pretrain_dir, weights_only=True)
        missing, unexpected = DCE_net.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Loss functions
    # Run find_exposure_target.py first, then set --exp_strict and
    # --exp_relaxed to the mean brightness of your GT images.
    # ------------------------------------------------------------------ #
    L_color       = Myloss.L_color()
    L_spa         = Myloss.L_spa()
    L_exp_strict  = Myloss.L_exp(16, config.exp_strict).cuda()
    L_exp_relaxed = Myloss.L_exp(16, config.exp_relaxed).cuda()
    L_TV          = Myloss.L_TV()
    L_per         = Myloss.perception_loss().cuda()

    # ------------------------------------------------------------------ #
    # Optimizer + scheduler
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(
        DCE_net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    scheduler = get_scheduler(optimizer, config.num_epochs, warmup_epochs=5)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    DCE_net.train()

    best_epoch = 0
    best_loss  = float('inf')

    for epoch in range(config.num_epochs):

        epoch_loss = 0.0

        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()

            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            loss_tv  = 400  * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 12    * torch.mean(L_color(enhanced_image))
            loss_per = 1  * torch.mean(L_per(enhanced_image))

            mean_light = torch.mean(img_lowlight).item()
            exp_weight = compute_exposure_weight(mean_light)

            if mean_light < 0.3:
                loss_exp = exp_weight * torch.mean(L_exp_strict(enhanced_image))
            else:
                loss_exp = exp_weight * torch.mean(L_exp_relaxed(enhanced_image))

            loss_snr = config.snr_weight * compute_local_snr_loss(
                enhanced_image, patch_size=16
            )

            loss = loss_tv + loss_spa + loss_col + loss_exp + loss_snr + loss_per

            if not torch.isfinite(loss):
                print(f"WARNING: Non-finite loss at iter {iteration + 1}, skipping")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()

            if (iteration + 1) % config.display_iter == 0:
                print(
                    f"Epoch: {epoch} | Iter: {iteration + 1} | "
                    f"Loss: {loss.item():.4f} | "
                    f"TV: {loss_tv.item():.4f} | "
                    f"SNR: {loss_snr.item():.4f} | "
                    f"Per: {loss_per.item():.4f} | "
                    f"Exp_w: {exp_weight:.1f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(
            f"=== Epoch {epoch} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} ==="
        )

        # Save best model by loss
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_epoch = epoch
            torch.save(
                DCE_net.state_dict(),
                os.path.join(config.snapshots_folder, "best.pth")
            )
            print(f"  ↑ New best at epoch {epoch} (loss={best_loss:.4f}) → best.pth")

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(
                DCE_net.state_dict(),
                os.path.join(config.snapshots_folder, f"Epoch{epoch + 1}.pth")
            )

    print(f"\nDone. Best epoch: {best_epoch} | Best loss: {best_loss:.4f}")
    print(f"→ Use snapshots/TargetV1/best.pth for testing")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--lowlight_images_path', type=str,
                        default="data/train_data/")

    # Optimizer
    parser.add_argument('--lr',             type=float, default=0.0001)
    parser.add_argument('--weight_decay',   type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    # Training
    parser.add_argument('--num_epochs',       type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--num_workers',      type=int, default=3)
    parser.add_argument('--display_iter',     type=int, default=10)

    # Snapshots
    parser.add_argument('--snapshots_folder', type=str,
                        default="snapshots/FixedV2/")
    parser.add_argument('--load_pretrain',    type=bool, default=False)
    parser.add_argument('--pretrain_dir',     type=str,
                        default="snapshots/FixedV1/Epoch100.pth")

    # Mixup
    parser.add_argument('--mixup_prob',  type=float, default=0.2)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)

    # Loss weights
    parser.add_argument('--snr_weight',   type=float, default=0.10)

    # Exposure targets — calibrate with find_exposure_target.py first
    parser.add_argument('--exp_strict',  type=float, default=0.58,
                        help='L_exp target for very dark images')
    parser.add_argument('--exp_relaxed', type=float, default=0.48,
                        help='L_exp target for moderately dark images')

    config = parser.parse_args()
    os.makedirs(config.snapshots_folder, exist_ok=True)
    train(config)