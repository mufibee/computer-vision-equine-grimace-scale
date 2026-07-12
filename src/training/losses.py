"""Loss-function utilities for model training."""

from typing import Optional, Sequence

import torch
from torch import nn


def create_classification_loss(
    class_weights: Optional[Sequence[float] | torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create a weighted cross-entropy loss function.

    Args:
        class_weights:
            Optional class weights ordered by class index.
            Example: [weight_class_0, weight_class_1, weight_class_2].

        device:
            Device on which the class-weight tensor should be stored.

    Returns:
        Configured CrossEntropyLoss instance.
    """
    if class_weights is None:
        return nn.CrossEntropyLoss()

    weights_tensor = torch.as_tensor(
        class_weights,
        dtype=torch.float32,
    )

    if device is not None:
        weights_tensor = weights_tensor.to(device)

    return nn.CrossEntropyLoss(weight=weights_tensor)