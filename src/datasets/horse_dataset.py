"""
Custom PyTorch Dataset for the Equine Grimace Scale project.

The dataset reads image information from a CSV file and returns:
    - transformed image tensor
    - pain-score label
    - horse ID
    - facial region
    - original image path
"""

from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HorseGrimaceDataset(Dataset):
    """
    PyTorch Dataset for horse facial-region pain classification.

    Parameters
    ----------
    csv_file : str or pathlib.Path
        Path to a train, validation, or test CSV file.

    image_root : str or pathlib.Path, optional
        Root directory containing the images. This is joined with the
        image path stored in the CSV when that path is relative.

    transform : callable, optional
        Image transformation pipeline.

    image_column : str
        Name of the CSV column containing the image path.

    label_column : str
        Name of the CSV column containing the pain-score label.

    horse_id_column : str
        Name of the CSV column containing the horse ID.

    face_region_column : str
        Name of the CSV column containing the facial region.

    verify_images : bool
        If True, verify during initialization that every image exists.
    """

    VALID_LABELS = {0, 1, 2}

    def __init__(
        self,
        csv_file: Union[str, Path],
        image_root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        image_column: str = "image_path",
        label_column: str = "label",
        horse_id_column: str = "horse_id",
        face_region_column: str = "face_region",
        verify_images: bool = False,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform

        self.image_column = image_column
        self.label_column = label_column
        self.horse_id_column = horse_id_column
        self.face_region_column = face_region_column

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CSV file was not found: {self.csv_file.resolve()}"
            )

        self.data = pd.read_csv(self.csv_file)

        self._validate_columns()
        self._validate_labels()

        if verify_images:
            self._verify_all_images_exist()

    def _validate_columns(self) -> None:
        """Check that all required columns are present in the CSV."""

        required_columns = {
            self.image_column,
            self.label_column,
            self.horse_id_column,
            self.face_region_column,
        }

        missing_columns = required_columns.difference(self.data.columns)

        if missing_columns:
            raise ValueError(
                "The CSV is missing the following required columns: "
                f"{sorted(missing_columns)}\n"
                f"Available columns: {list(self.data.columns)}"
            )

    def _validate_labels(self) -> None:
        """Check that labels are integers and belong to {0, 1, 2}."""

        if self.data[self.label_column].isna().any():
            raise ValueError(
                f"Missing values were found in '{self.label_column}'."
            )

        try:
            self.data[self.label_column] = (
                self.data[self.label_column].astype(int)
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Labels in '{self.label_column}' must be integers."
            ) from error

        found_labels = set(self.data[self.label_column].unique())

        invalid_labels = found_labels.difference(self.VALID_LABELS)

        if invalid_labels:
            raise ValueError(
                f"Invalid labels found: {sorted(invalid_labels)}. "
                f"Expected only {sorted(self.VALID_LABELS)}."
            )

    def _resolve_image_path(self, stored_path: str) -> Path:
        """
        Convert the image path stored in the CSV into a usable path.

        Absolute paths are used directly. Relative paths are interpreted
        relative to image_root when image_root is provided.
        """

        image_path = Path(str(stored_path))

        if image_path.is_absolute():
            return image_path

        if self.image_root is not None:
            return self.image_root / image_path

        return image_path

    def _verify_all_images_exist(self) -> None:
        """Verify that every image listed in the CSV exists."""

        missing_images = []

        for stored_path in self.data[self.image_column]:
            image_path = self._resolve_image_path(stored_path)

            if not image_path.exists():
                missing_images.append(str(image_path))

        if missing_images:
            preview = "\n".join(missing_images[:10])

            raise FileNotFoundError(
                f"{len(missing_images)} image files could not be found.\n"
                f"First missing paths:\n{preview}"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """
        Load and return one dataset sample.

        Returns
        -------
        dict
            Dictionary containing the image, label, metadata, and path.
        """

        row = self.data.iloc[index]

        image_path = self._resolve_image_path(row[self.image_column])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found for dataset index {index}: "
                f"{image_path.resolve()}"
            )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

        except Exception as error:
            raise RuntimeError(
                f"Could not load image at index {index}: {image_path}"
            ) from error

        sample = {
            "image": image,
            "label": int(row[self.label_column]),
            "horse_id": str(row[self.horse_id_column]),
            "face_region": str(row[self.face_region_column]),
            "image_path": str(image_path),
        }

        return sample