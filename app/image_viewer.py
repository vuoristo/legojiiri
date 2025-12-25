from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel


class ImageViewer(QLabel):
    """Simple QLabel-based viewer that scales image to fit."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self._pixmap: Optional[QPixmap] = None
        self.setStyleSheet("background-color: #202020; color: white;")

    def load_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.setText("Failed to load image")
            self._pixmap = None
            return
        self._pixmap = pixmap
        self._update_pixmap()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
