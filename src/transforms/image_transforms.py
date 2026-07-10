"""
Image preprocessing and augmentation pipelines.

Training images receive mild random augmentation.
Validation and test images receive deterministic preprocessing only.
"""

from typing import Tuple

from torchvision import transforms


# Standard ImageNet normalization values.
IMAGENET_MEAN: Tuple[float, float, float] = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD: Tuple[float, float, float] = (
    0.229,
    0.224,
    0.225,
)


def get_train_transforms(
    image_size: int = 224,
) -> transforms.Compose:
    """
    Return transformations used for the training dataset.

    Augmentations are deliberately mild so that pain-related facial
    features are not strongly distorted.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(
                degrees=7,
            ),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                shear=3,
            ),

            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.05,
                hue=0.02,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_evaluation_transforms(
    image_size: int = 224,
) -> transforms.Compose:
    """
    Return deterministic transformations for validation and testing.

    No random augmentation is used during evaluation.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )