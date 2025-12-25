from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

from PySide6.QtCore import QObject, QThread, Signal

from ml.model import InferenceResult, LegoModelManager

LOGGER = logging.getLogger(__name__)


class PredictWorker(QThread):
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, model: LegoModelManager, image_path: Path) -> None:
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self) -> None:  # noqa: D401
        try:
            results = self.model.predict_topk(self.image_path, topk=5)
            self.finished.emit(results)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Prediction failed")
            self.failed.emit(str(exc))


class TrainWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        trainer: Callable[[Callable[[int], None], Callable[[str], None]], str],
    ) -> None:
        super().__init__()
        self.trainer = trainer

    def run(self) -> None:
        try:
            message = self.trainer(self._emit_progress, self._emit_log)
            self.finished.emit(message)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Training failed")
            self.failed.emit(str(exc))

    def _emit_progress(self, value: int) -> None:
        self.progress.emit(value)

    def _emit_log(self, message: str) -> None:
        self.log.emit(message)
