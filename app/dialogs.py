from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QMessageBox,
)


class NewLabelDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create New Label")
        self.id_edit = QLineEdit()
        self.name_edit = QLineEdit()
        layout = QFormLayout(self)
        layout.addRow("Part ID", self.id_edit)
        layout.addRow("Display name", self.name_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def values(self) -> tuple[str, str]:
        return self.id_edit.text().strip(), self.name_edit.text().strip()


class ModelPathDialog(QDialog):
    def __init__(self, model_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.path_edit = QLineEdit(str(model_path))
        layout = QFormLayout(self)
        layout.addRow("Model file", self.path_edit)
        browse = QDialogButtonBox(QDialogButtonBox.Open)
        browse.button(QDialogButtonBox.Open).setText("Browse…")
        browse.accepted.connect(self._browse)
        layout.addWidget(browse)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Choose model file", self.path_edit.text(), "PyTorch (*.pt *.pth)"
        )
        if file_name:
            self.path_edit.setText(file_name)

    @property
    def selected_path(self) -> Path:
        return Path(self.path_edit.text()).expanduser()


class Dialogs:
    @staticmethod
    def error(parent, title: str, message: str) -> None:
        QMessageBox.critical(parent, title, message)
