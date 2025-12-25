from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from app.labels import LabelStore
from ml.data import create_dataloaders, preprocess_image

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    label: str
    probability: float


class LegoModelManager:
    """Wraps model creation, prediction and training."""

    def __init__(self, labels: LabelStore, model_path: Path) -> None:
        self.labels = labels
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.idx_to_class: dict[int, str] = {}
        self._ensure_model()

    # Model helpers
    def _build_model(self, num_classes: int) -> nn.Module:
        # MobileNetV3-Small offers a good trade-off for CPU-bound inference.
        # Users might be offline, so we gracefully fall back to random weights
        # if pretrained weights cannot be fetched from the internet.
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = models.mobilenet_v3_small(weights=weights)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Could not load pretrained MobileNetV3-Small weights; falling back to random init: %s",
                exc,
            )
            model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    def _ensure_model(self) -> None:
        num_classes = max(1, len(self.labels.labels))
        self.model = self._build_model(num_classes)
        self.idx_to_class = {i: lbl.id for i, lbl in enumerate(self.labels.labels)}
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["state_dict"])
                self.idx_to_class = checkpoint.get("idx_to_class", self.idx_to_class)
                LOGGER.info("Loaded model from %s", self.model_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to load model from %s: %s", self.model_path, exc)
        self.model.to(self.device)
        self.model.eval()

    # Prediction
    def predict_topk(self, image_path: Path, topk: int = 5) -> List[Tuple[str, float]]:
        if self.model is None:
            self._ensure_model()
        assert self.model is not None
        with torch.no_grad():
            tensor = preprocess_image(image_path).unsqueeze(0).to(self.device)
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
            probs, indices = probs.topk(min(topk, probs.shape[1]))
            results = []
            for prob, idx in zip(probs[0].cpu(), indices[0].cpu()):
                label_id = self.idx_to_class.get(int(idx), f"class_{int(idx)}")
                label_obj = self.labels.get_label(label_id)
                name = f"{label_id} - {label_obj.name if label_obj else 'Unknown'}"
                results.append((name, float(prob)))
            return results

    # Training
    def train(
        self, progress_cb: Callable[[int], None], log_cb: Callable[[str], None], epochs: int = 5
    ) -> str:
        dataset_root = Path(__file__).resolve().parent.parent / "data" / "images"
        batch_size = 16
        while batch_size >= 1:
            try:
                train_loader, val_loader, idx_to_class = create_dataloaders(dataset_root, batch_size)
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    batch_size = batch_size // 2
                    log_cb(f"Reducing batch size to {batch_size} due to OOM")
                    continue
                raise
        else:
            raise RuntimeError("Could not allocate training batch")

        self.idx_to_class = idx_to_class
        num_classes = len(idx_to_class)
        self.model = self._build_model(num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, (images, targets) in enumerate(train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_batches = len(train_loader)
                progress = int(((epoch * total_batches) + (batch_idx + 1)) / (epochs * total_batches) * 100)
                progress_cb(progress)
            log_cb(f"Epoch {epoch+1}/{epochs} - loss: {running_loss/ max(1, len(train_loader)):.4f}")

            if val_loader:
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, targets in val_loader:
                        images, targets = images.to(self.device), targets.to(self.device)
                        outputs = self.model(images)
                        preds = outputs.argmax(dim=1)
                        correct += (preds == targets).sum().item()
                        total += targets.size(0)
                acc = correct / total if total else 0.0
                log_cb(f"Validation accuracy: {acc:.2%}")

        self._save_checkpoint()
        return f"Training finished. Model saved to {self.model_path}"

    def _save_checkpoint(self) -> None:
        assert self.model is not None
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "idx_to_class": self.idx_to_class,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.model_path)
