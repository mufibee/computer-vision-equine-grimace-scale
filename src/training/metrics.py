"""Metric utilities for multiclass classification."""

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def calculate_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, object]:
    """
    Calculate classification metrics from predicted and true labels.

    Args:
        predictions:
            Tensor containing predicted class indices with shape (N,).

        targets:
            Tensor containing true class indices with shape (N,).

        num_classes:
            Total number of classes.

    Returns:
        Dictionary containing:
            - accuracy
            - macro_precision
            - macro_recall
            - macro_f1
            - weighted_precision
            - weighted_recall
            - weighted_f1
            - per_class_accuracy
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    accuracy = accuracy_score(
        targets_np,
        predictions_np,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            targets_np,
            predictions_np,
            average="macro",
            labels=list(range(num_classes)),
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            targets_np,
            predictions_np,
            average="weighted",
            labels=list(range(num_classes)),
            zero_division=0,
        )
    )

    per_class_accuracy = calculate_per_class_accuracy(
        predictions=predictions_np,
        targets=targets_np,
        num_classes=num_classes,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class_accuracy": per_class_accuracy,
    }


def calculate_per_class_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> List[float]:
    """
    Calculate accuracy separately for each class.

    Per-class accuracy is the proportion of samples belonging to a
    particular class that were classified correctly.

    Args:
        predictions:
            NumPy array of predicted class indices.

        targets:
            NumPy array of true class indices.

        num_classes:
            Total number of classes.

    Returns:
        List containing one accuracy value per class.
    """
    class_accuracies = []

    for class_index in range(num_classes):
        class_mask = targets == class_index
        class_count = class_mask.sum()

        if class_count == 0:
            class_accuracy = 0.0
        else:
            correct_predictions = (
                predictions[class_mask] == targets[class_mask]
            ).sum()

            class_accuracy = correct_predictions / class_count

        class_accuracies.append(float(class_accuracy))

    return class_accuracies