from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

from app.labels import LabelStore

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images_in_folder(folder: Path) -> list[Path]:
    return [p for p in folder.glob("**/*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def assign_image_to_label(
    image_path: Path, labels: LabelStore, target_label_id: str, dataset_root: Path
) -> Path:
    if target_label_id not in {label.id for label in labels.labels}:
        raise ValueError(f"Unknown label id {target_label_id}")
    destination_dir = dataset_root / target_label_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / image_path.name
    counter = 1
    while destination_path.exists():
        destination_path = destination_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
        counter += 1
    shutil.copy2(image_path, destination_path)
    LOGGER.info("Assigned %s to label %s", image_path, target_label_id)
    return destination_path


def ensure_dataset_structure(dataset_root: Path, labels: LabelStore) -> None:
    labels.ensure_image_dirs(dataset_root)


def iter_dataset_images(dataset_root: Path, labels: LabelStore) -> Iterable[Path]:
    for label in labels.labels:
        folder = dataset_root / label.id
        if not folder.exists():
            continue
        for img in folder.glob("**/*"):
            if img.suffix.lower() in IMAGE_EXTENSIONS:
                yield img
