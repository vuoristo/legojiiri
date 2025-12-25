from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

LOGGER = logging.getLogger(__name__)


@dataclass
class Label:
    id: str
    name: str


class LabelStore:
    """Simple helper to manage labels.json and keep id->name mapping."""

    def __init__(self, labels_path: Path) -> None:
        self.labels_path = labels_path
        self.labels: List[Label] = []
        self._by_id: Dict[str, Label] = {}
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self) -> None:
        if not self.labels_path.exists():
            LOGGER.warning("No labels.json found at %s, creating empty file", self.labels_path)
            self.labels = []
            self._by_id = {}
            self._save_internal()
            return

        try:
            data = json.loads(self.labels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOGGER.error("labels.json is invalid: %s", exc)
            data = []
        self.labels = [Label(**entry) for entry in data]
        self._by_id = {lbl.id: lbl for lbl in self.labels}

    def _save_internal(self) -> None:
        payload = [label.__dict__ for label in self.labels]
        self.labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_label(self, label_id: str, name: str) -> None:
        if label_id in self._by_id:
            raise ValueError(f"Label {label_id} already exists")
        new_label = Label(label_id, name)
        self.labels.append(new_label)
        self._by_id[label_id] = new_label
        self._save_internal()

    def update_label(self, label_id: str, name: str) -> None:
        if label_id not in self._by_id:
            raise ValueError(f"Label {label_id} not found")
        self._by_id[label_id].name = name
        self._save_internal()

    def get_label(self, label_id: str) -> Label | None:
        return self._by_id.get(label_id)

    def to_choices(self) -> List[str]:
        return [f"{lbl.id} - {lbl.name}" for lbl in self.labels]

    def ensure_image_dirs(self, images_root: Path) -> None:
        for lbl in self.labels:
            (images_root / lbl.id).mkdir(parents=True, exist_ok=True)
