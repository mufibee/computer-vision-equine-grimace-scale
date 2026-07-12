"""Reusable training framework for multiclass image classification."""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint
from .metrics import calculate_classification_metrics


class Trainer:
    """Train, validate, and evaluate a multiclass image classifier."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_classes: int,
        scheduler: Optional[object] = None,
        checkpoint_path: str | Path = (
            "results/baseline/best_model.pth"
        ),
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.scheduler = scheduler

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )

        self.checkpoint = ModelCheckpoint(
            checkpoint_path=checkpoint_path,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_precision": [],
            "val_macro_recall": [],
            "val_macro_f1": [],
            "val_weighted_f1": [],
            "learning_rate": [],
        }

    def _extract_batch(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract image and label tensors from a dataset batch.

        Supports:
            - dictionary batches from HorseGrimaceDataset
            - tuple/list batches from TensorDataset
        """
        if isinstance(batch, dict):
            images = batch["image"]
            targets = batch["pain_score"]
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:
            raise TypeError(
                "Unsupported batch format. Expected a dictionary with "
                "'image' and 'label' keys, or a tuple/list containing "
                "images and labels."
            )

        images = images.to(
            self.device,
            non_blocking=True,
        )

        targets = targets.to(
            self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        return images, targets

    def train_one_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """Train the model for one epoch."""

        self.model.train()

        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(
            train_loader,
            desc="Training",
            leave=False,
        )

        for batch in progress_bar:
            images, targets = self._extract_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0)

            running_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == targets
            ).sum().item()

            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
            )

        if total_samples == 0:
            raise RuntimeError("The training DataLoader is empty.")

        epoch_loss = running_loss / total_samples
        epoch_accuracy = total_correct / total_samples

        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
        }

    @torch.no_grad()
    def validate_one_epoch(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, object]:
        """Evaluate the model on the validation set."""

        self.model.eval()

        running_loss = 0.0
        total_samples = 0

        all_predictions = []
        all_targets = []

        progress_bar = tqdm(
            val_loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress_bar:
            images, targets = self._extract_batch(batch)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            batch_size = targets.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)

            all_predictions.append(
                predictions.detach().cpu()
            )

            all_targets.append(
                targets.detach().cpu()
            )

        if total_samples == 0:
            raise RuntimeError("The validation DataLoader is empty.")

        predictions_tensor = torch.cat(all_predictions)
        targets_tensor = torch.cat(all_targets)

        metrics = calculate_classification_metrics(
            predictions=predictions_tensor,
            targets=targets_tensor,
            num_classes=self.num_classes,
        )

        metrics["loss"] = running_loss / total_samples

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs."""

        if num_epochs <= 0:
            raise ValueError("num_epochs must be greater than zero.")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(
                train_loader,
            )

            val_metrics = self.validate_one_epoch(
                val_loader,
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(
                train_metrics["loss"]
            )

            self.history["train_accuracy"].append(
                train_metrics["accuracy"]
            )

            self.history["val_loss"].append(
                val_metrics["loss"]
            )

            self.history["val_accuracy"].append(
                val_metrics["accuracy"]
            )

            self.history["val_macro_precision"].append(
                val_metrics["macro_precision"]
            )

            self.history["val_macro_recall"].append(
                val_metrics["macro_recall"]
            )

            self.history["val_macro_f1"].append(
                val_metrics["macro_f1"]
            )

            self.history["val_weighted_f1"].append(
                val_metrics["weighted_f1"]
            )

            self.history["learning_rate"].append(
                current_lr
            )

            checkpoint_saved = self.checkpoint.step(
                validation_loss=val_metrics["loss"],
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                extra_state={
                    "history": self.history,
                    "num_classes": self.num_classes,
                },
            )

            self._step_scheduler(
                validation_loss=val_metrics["loss"],
            )

            print(
                f"Epoch {epoch:03d}/{num_epochs:03d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Macro F1: {val_metrics['macro_f1']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            if checkpoint_saved:
                print("Saved new best checkpoint.")

            should_stop = self.early_stopping.step(
                val_metrics["loss"]
            )

            if should_stop:
                print(
                    "Early stopping triggered at "
                    f"epoch {epoch}."
                )
                break

        return self.history

    @torch.no_grad()
    def predict(
        self,
        data_loader: DataLoader,
    ) -> Dict[str, object]:
        """
        Evaluate a dataset and return predictions, targets,
        probabilities, metadata, and metrics.
        """

        self.model.eval()

        running_loss = 0.0
        total_samples = 0

        all_predictions = []
        all_targets = []
        all_probabilities = []

        all_horse_ids = []
        all_face_regions = []
        all_image_paths = []

        progress_bar = tqdm(
            data_loader,
            desc="Evaluation",
            leave=False,
        )

        for batch in progress_bar:
            images, targets = self._extract_batch(batch)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = logits.argmax(dim=1)

            batch_size = targets.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.append(
                predictions.detach().cpu()
            )

            all_targets.append(
                targets.detach().cpu()
            )

            all_probabilities.append(
                probabilities.detach().cpu()
            )

            if isinstance(batch, dict):
                all_horse_ids.extend(
                    list(batch.get("horse_id", []))
                )

                all_face_regions.extend(
                    list(batch.get("face_region", []))
                )

                all_image_paths.extend(
                    list(batch.get("image_path", []))
                )

        if total_samples == 0:
            raise RuntimeError("The evaluation DataLoader is empty.")

        predictions_tensor = torch.cat(all_predictions)
        targets_tensor = torch.cat(all_targets)
        probabilities_tensor = torch.cat(all_probabilities)

        metrics = calculate_classification_metrics(
            predictions=predictions_tensor,
            targets=targets_tensor,
            num_classes=self.num_classes,
        )

        metrics["loss"] = running_loss / total_samples

        return {
            "metrics": metrics,
            "predictions": predictions_tensor,
            "targets": targets_tensor,
            "probabilities": probabilities_tensor,
            "horse_ids": all_horse_ids,
            "face_regions": all_face_regions,
            "image_paths": all_image_paths,
        }

    def _step_scheduler(
        self,
        validation_loss: float,
    ) -> None:
        """Advance the learning-rate scheduler."""

        if self.scheduler is None:
            return

        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            self.scheduler.step(validation_loss)
        else:
            self.scheduler.step()