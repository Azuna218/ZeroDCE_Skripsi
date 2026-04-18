import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random

# Set random seed for reproducibility
random.seed(1143)


def populate_train_list(lowlight_images_path):
    """
    Recursively collect all image file paths from the given directory.

    Args:
        lowlight_images_path (str): Root directory containing training images.

    Returns:
        list: Shuffled list of valid image file paths.
    """

    image_list = glob.glob(
        os.path.join(lowlight_images_path, "**", "*.*"),
        recursive=True
    )

    # Filter only supported image formats
    image_list = [
        x for x in image_list
        if x.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    random.shuffle(image_list)

    return image_list


def mixup_images(img1, img2, alpha=0.4):
    """
    Apply Mixup interpolation between two images.

    Mixup creates a convex combination of two images using a
    Beta-distributed blending coefficient lambda. This forces
    the model to learn domain-invariant features by training
    on synthetic in-between samples from different lighting domains.

    Reference: FIXED - Frustratingly Easy Domain Generalization with Mixup
               (Lu et al., 2024) — arXiv:2211.05228

    Args:
        img1 (numpy.ndarray): First image, shape (H, W, C), range [0, 1]
        img2 (numpy.ndarray): Second image, shape (H, W, C), range [0, 1]
        alpha (float): Beta distribution concentration parameter.
                       Lower alpha (e.g. 0.4) keeps lambda closer to 0 or 1,
                       meaning most mixed images still resemble one parent.
                       This avoids creating unrealistically blended images.

    Returns:
        numpy.ndarray: Mixed image in range [0, 1]
        float: Lambda blending coefficient used
    """
    lam = np.random.beta(alpha, alpha)

    # Ensure lambda stays in a sensible range — avoid 50/50 blends
    # that produce visually ambiguous images that confuse the loss functions
    lam = max(lam, 1 - lam)  # always take the stronger side (lam >= 0.5)

    mixed = lam * img1 + (1 - lam) * img2
    mixed = np.clip(mixed, 0, 1)

    return mixed, lam


class lowlight_loader(data.Dataset):
    """
    PyTorch Dataset for loading low-light images with Mixup-based
    Domain Generalization.

    This dataset:
    - Loads images recursively from subfolders
    - Resizes images to a fixed resolution
    - Applies random brightness scaling augmentation (Lighting Domain Randomization)
    - Applies Mixup interpolation between pairs of images (Mixup-DG)
    - Converts images to PyTorch tensors

    Domain Generalization Strategy:
        Two complementary mechanisms are applied:
        1. Lighting Domain Randomization: randomly darkens images to simulate
           different lighting domains (scale in [0.2, 0.6]).
        2. Mixup-DG: interpolates between two randomly selected training images
           from potentially different lighting domains, forcing the model to
           learn features that generalize across domain boundaries rather than
           memorizing domain-specific patterns.
    """

    def __init__(self, lowlight_images_path, mixup_prob=0.3, mixup_alpha=0.2):
        """
        Initialize dataset.

        Args:
            lowlight_images_path (str): Path to training images.
            mixup_prob (float): Probability of applying Mixup to a sample.
                                Default 0.5 — applies to half the training batch.
            mixup_alpha (float): Beta distribution parameter for Mixup lambda.
                                 Lower = blends lean more toward one image.
        """
        self.data_list = populate_train_list(lowlight_images_path)
        self.size = 256
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

        print("Total training examples:", len(self.data_list))
        print(f"Mixup-DG enabled — prob={mixup_prob}, alpha={mixup_alpha}")

    def _load_and_preprocess(self, path):
        """
        Load a single image, resize, normalize, and apply brightness augmentation.

        Args:
            path (str): Path to image file.

        Returns:
            numpy.ndarray: Preprocessed image in range [0, 1], shape (H, W, C)
        """
        image = Image.open(path).convert('RGB')
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        image = np.asarray(image) / 255.0

        # Lighting Domain Randomization:
        # Randomly darken image to simulate different lighting domains.
        # 50% probability, scale uniformly sampled from [0.2, 0.6].
        if random.random() < 0.5:
            scale = random.uniform(0.2, 0.6)
            image = np.clip(image * scale, 0, 1)

        return image

    def __getitem__(self, index):
        """
        Retrieve a single training sample.

        If Mixup is applied, a second image is randomly selected from the
        dataset and the two images are blended using a Beta-distributed
        coefficient. This creates synthetic cross-domain samples.

        Args:
            index (int): Index of the primary image.

        Returns:
            torch.Tensor: Image tensor of shape (C, H, W), range [0, 1]
        """

        # Load primary image
        image = self._load_and_preprocess(self.data_list[index])

        # Mixup-based Domain Generalization
        # Apply with probability mixup_prob
        if random.random() < self.mixup_prob:

            # Randomly select a DIFFERENT image from the dataset
            # This is the key DG mechanism: the two images may come from
            # different subdatasets (LOL, ExDark, etc.) representing different
            # lighting domains. Mixing them forces domain-invariant learning.
            mix_index = random.randint(0, len(self.data_list) - 1)

            # Avoid mixing image with itself
            while mix_index == index:
                mix_index = random.randint(0, len(self.data_list) - 1)

            image2 = self._load_and_preprocess(self.data_list[mix_index])

            # Blend the two images
            image, _ = mixup_images(image, image2, alpha=self.mixup_alpha)

        # Convert to tensor: (H, W, C) → (C, H, W)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)

        return image

    def __len__(self):
        """
        Return total number of samples.
        """
        return len(self.data_list)
