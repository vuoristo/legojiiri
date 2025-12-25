from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAction,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QLabel,
    QComboBox,
    QMessageBox,
)

from app.dataset import assign_image_to_label, ensure_dataset_structure, find_images_in_folder
from app.dialogs import Dialogs, ModelPathDialog, NewLabelDialog
from app.image_viewer import ImageViewer
from app.labels import LabelStore
from app.workers import PredictWorker, TrainWorker
from ml.model import LegoModelManager

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LABELS_FILE = DATA_DIR / "labels.json"
IMAGES_ROOT = DATA_DIR / "images"
DEFAULT_MODEL_PATH = DATA_DIR / "model.pt"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LEGOJiiRi - LEGO Part Labeler")
        self.resize(1200, 700)

        self.labels = LabelStore(LABELS_FILE)
        ensure_dataset_structure(IMAGES_ROOT, self.labels)

        self.model_manager = LegoModelManager(self.labels, DEFAULT_MODEL_PATH)

        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self._on_file_selected)

        self.image_viewer = ImageViewer()

        self.prediction_label = QLabel("Predictions will appear here")
        self.prediction_label.setWordWrap(True)

        self.label_dropdown = QComboBox()
        self._refresh_label_dropdown()
        self.assign_button = QPushButton("Assign label")
        self.assign_button.clicked.connect(self._on_assign_label)

        self.train_button = QPushButton("Train model")
        self.train_button.clicked.connect(self._on_train)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Top-5 predictions"))
        right_layout.addWidget(self.prediction_label)
        right_layout.addWidget(QLabel("Assign label"))
        right_layout.addWidget(self.label_dropdown)
        right_layout.addWidget(self.assign_button)
        right_layout.addWidget(self.train_button)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(QLabel("Training log"))
        right_layout.addWidget(self.train_log)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter()
        splitter.addWidget(self.file_list)
        splitter.addWidget(self.image_viewer)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.addWidget(splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        self._create_toolbar()

        self.current_predict_worker: PredictWorker | None = None
        self.current_train_worker: TrainWorker | None = None

    # Toolbar setup
    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_files_action = QAction("Open Images…", self)
        open_files_action.triggered.connect(self._on_open_images)
        toolbar.addAction(open_files_action)

        open_folder_action = QAction("Open Folder…", self)
        open_folder_action.triggered.connect(self._on_open_folder)
        toolbar.addAction(open_folder_action)

        predict_action = QAction("Predict", self)
        predict_action.triggered.connect(self._on_predict)
        toolbar.addAction(predict_action)

        train_action = QAction("Train", self)
        train_action.triggered.connect(self._on_train)
        toolbar.addAction(train_action)

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._on_settings)
        toolbar.addAction(settings_action)

        new_label_action = QAction("New Label", self)
        new_label_action.triggered.connect(self._on_new_label)
        toolbar.addAction(new_label_action)

    # File handling
    def _add_files(self, paths: List[Path]) -> None:
        for path in paths:
            if any(self.file_list.item(i).data(Qt.UserRole) == path for i in range(self.file_list.count())):
                continue
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, path)
            self.file_list.addItem(item)

    def _on_open_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select LEGO part images",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        self._add_files([Path(f) for f in files])

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose folder containing images")
        if folder:
            paths = find_images_in_folder(Path(folder))
            self._add_files(paths)

    def _on_file_selected(self) -> None:
        items = self.file_list.selectedItems()
        if not items:
            return
        path: Path = items[0].data(Qt.UserRole)
        self.image_viewer.load_image(path)
        self.prediction_label.setText("Ready for prediction…")

    def _on_predict(self) -> None:
        items = self.file_list.selectedItems()
        if not items:
            Dialogs.error(self, "No image", "Select an image to run prediction.")
            return
        path: Path = items[0].data(Qt.UserRole)
        self.prediction_label.setText("Running…")
        self.current_predict_worker = PredictWorker(self.model_manager, path)
        self.current_predict_worker.finished.connect(self._on_prediction_ready)
        self.current_predict_worker.failed.connect(self._on_prediction_failed)
        self.current_predict_worker.start()

    def _on_prediction_ready(self, results):
        lines = [f"{name}: {prob:.2%}" for name, prob in results]
        self.prediction_label.setText("\n".join(lines))

    def _on_prediction_failed(self, message: str):
        Dialogs.error(self, "Prediction failed", message)
        self.prediction_label.setText("Prediction failed")

    def _refresh_label_dropdown(self) -> None:
        self.label_dropdown.clear()
        for label in self.labels.labels:
            self.label_dropdown.addItem(f"{label.id} - {label.name}", userData=label.id)

    def _on_assign_label(self) -> None:
        items = self.file_list.selectedItems()
        if not items:
            Dialogs.error(self, "No image", "Select an image to assign a label.")
            return
        if self.label_dropdown.currentIndex() < 0:
            Dialogs.error(self, "No label", "Create a label first.")
            return
        path: Path = items[0].data(Qt.UserRole)
        label_id = self.label_dropdown.currentData()
        assigned_path = assign_image_to_label(path, self.labels, label_id, IMAGES_ROOT)
        QMessageBox.information(self, "Saved", f"Image copied to dataset: {assigned_path}")

    def _on_train(self) -> None:
        if self.current_train_worker and self.current_train_worker.isRunning():
            Dialogs.error(self, "Busy", "Training already running")
            return

        def trainer(progress_cb, log_cb):
            return self.model_manager.train(progress_cb, log_cb)

        self.progress.setValue(0)
        self.train_log.clear()
        self.current_train_worker = TrainWorker(trainer)
        self.current_train_worker.progress.connect(self.progress.setValue)
        self.current_train_worker.log.connect(self._append_log)
        self.current_train_worker.finished.connect(self._on_train_finished)
        self.current_train_worker.failed.connect(self._on_train_failed)
        self.current_train_worker.start()

    def _append_log(self, message: str) -> None:
        self.train_log.append(message)

    def _on_train_finished(self, message: str) -> None:
        QMessageBox.information(self, "Training complete", message)
        self.progress.setValue(100)

    def _on_train_failed(self, message: str) -> None:
        Dialogs.error(self, "Training failed", message)

    def _on_settings(self) -> None:
        dialog = ModelPathDialog(self.model_manager.model_path, self)
        if dialog.exec() == dialog.Accepted:
            self.model_manager.model_path = dialog.selected_path
            QMessageBox.information(self, "Settings saved", f"Model path set to {dialog.selected_path}")

    def _refresh_labels(self) -> None:
        self.labels.load()
        ensure_dataset_structure(IMAGES_ROOT, self.labels)
        self._refresh_label_dropdown()

    # menu events
    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key_N and event.modifiers() & Qt.ControlModifier:
            self._on_new_label()
        super().keyPressEvent(event)

    def _on_new_label(self):
        dialog = NewLabelDialog(self)
        if dialog.exec() == dialog.Accepted:
            label_id, name = dialog.values
            if not label_id or not name:
                Dialogs.error(self, "Invalid", "Fill in both id and name")
                return
            try:
                self.labels.add_label(label_id, name)
            except ValueError as exc:
                Dialogs.error(self, "Duplicate", str(exc))
                return
            self._refresh_label_dropdown()
            ensure_dataset_structure(IMAGES_ROOT, self.labels)

