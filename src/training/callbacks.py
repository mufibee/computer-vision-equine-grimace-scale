"""Callbacks for early stopping and model checkpointing."""

from pathlib import Path
from typing import Optional

import torch
from torch import nn


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Args:
        patience:
            Number of consecutive epochs without improvement allowed.

        min_delta:
            Minimum decrease in validation loss required to count
            as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, validation_loss: float) -> bool:
        """
        Update the early-stopping state.

        Returns:
            True when training should stop.
        """
        improved = validation_loss < (
            self.best_loss - self.min_delta
        )

        if improved:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """
    Save the model with the lowest validation loss.

    Args:
        checkpoint_path:
            Path where the checkpoint will be saved.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_loss = float("inf")

    def step(
        self,
        validation_loss: float,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        extra_state: Optional[dict] = None,
    ) -> bool:
        """
        Save a checkpoint when validation loss improves.

        Returns:
            True if a new checkpoint was saved.
        """
        if validation_loss >= self.best_loss:
            return False

        self.best_loss = validation_loss

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "validation_loss": validation_loss,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = (
                optimizer.state_dict()
            )

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if extra_state is not None:
            checkpoint.update(extra_state)

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

        return True