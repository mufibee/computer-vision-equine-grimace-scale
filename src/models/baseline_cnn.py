"""Baseline convolutional neural network for horse grimace classification."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """
    Lightweight CNN for three-class horse grimace classification.

    Expected input shape:
        (batch_size, 3, 224, 224)

    Output shape:
        (batch_size, num_classes)

    The output contains raw logits. Softmax is not applied because
    nn.CrossEntropyLoss expects unnormalized logits.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 × 224 × 224 -> 32 × 112 × 112
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 × 112 × 112 -> 64 × 56 × 56
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 × 56 × 56 -> 128 × 28 × 28
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 × 28 × 28 -> 256 × 28 × 28
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Convert every feature map to one value.
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(
                in_features=256,
                out_features=num_classes,
            ),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            images: Image batch with shape (B, 3, 224, 224).

        Returns:
            Raw class logits with shape (B, num_classes).
        """
        features = self.features(images)
        logits = self.classifier(features)

        return logits


if __name__ == "__main__":
    # Simple standalone shape test.
    model = BaselineCNN(num_classes=3)

    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_outputs = model(dummy_images)

    print(f"Input shape:  {dummy_images.shape}")
    print(f"Output shape: {dummy_outputs.shape}")