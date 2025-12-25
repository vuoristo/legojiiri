# LEGOJiiRi - Offline LEGO Part Labeler

LEGOJiiRi is a standalone Windows desktop application for labeling LEGO parts from photos. It uses a lightweight MobileNetV3 transfer learning pipeline that runs on CPU by default and optionally leverages GPU when present.

## Features
- PySide6 desktop UI with file browser, image preview, predictions, and label assignment.
- Top-5 classification with confidences using a pretrained MobileNetV3-Small backbone.
- Label manager (ID + display name) and dataset folder maintenance (`data/images/<class_id>/...`).
- One-click training/fine-tuning on your labeled dataset with CPU-friendly defaults and progress logs.
- Model save/load path configurable from Settings; checkpoint stored as PyTorch `.pt`.
- Designed for offline use—no cloud services or telemetry.

## Project layout
- `app/`: GUI entrypoint and widgets.
- `ml/`: Training and inference utilities.
- `data/`: Labels (`labels.json`) and image folders (`images/<class_id>/`).
- `build_windows.ps1`: PyInstaller helper for Windows packaging.

## Getting started
1. Install Python 3.11+ and Git.
2. Clone the repository and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   python -m app
   ```

### Controls
- Toolbar: **Open Images**, **Open Folder**, **Predict**, **Train**, **Settings**, **New Label**.
- Left pane: list of queued images. Center: zoom-to-fit preview. Right: top-5 predictions, label dropdown, training log.
- Assign label copies the current image into `data/images/<label_id>/` (creating subfolders as needed).

## Model usage
- Initial model: MobileNetV3-Small with ImageNet weights; classifier head adjusted to your label count.
- Prediction: Center-crop + resize to 224, ImageNet normalization, top-5 softmax probabilities.
- Training defaults: batch size 16 (auto-reduces on OOM), 5 epochs, Adam optimizer, optional validation split.
- Checkpoints include class mapping so predictions remain stable between sessions.

## Dataset format
```
data/
  labels.json          # [{"id":"3001","name":"Brick 2x4"}, ...]
  images/
    3001/...
    3003/...
```
Use **Assign label** or drag files into the corresponding folder to grow the dataset.

## Packaging for Windows (PyInstaller)
1. Ensure dependencies are installed in the active virtual environment.
2. Run the helper script:
   ```powershell
   ./build_windows.ps1
   ```
3. The executable will be placed under `dist/app/` (single-folder mode). Copy the `data/` directory alongside the executable for runtime labels/model.

## Troubleshooting
- If training reports out-of-memory on CPU, the app will automatically retry with a smaller batch size.
- Missing labels? Use **New Label** (Ctrl+N) to add IDs and names; folders are created automatically.

## How to take good photos for best accuracy
- Use even lighting to avoid harsh shadows and overexposure.
- Place the LEGO part on a plain, high-contrast background.
- Keep one part per image with the piece roughly centered and occupying most of the frame.
- Maintain consistent distance/scale across photos when possible.

## How to expand to object detection later
- Replace the classification head with an object detector (e.g., YOLOv8) trained on bounding boxes.
- Collect annotations in COCO or YOLO format instead of class folders.
- Swap the inference code to load an ONNX-exported detector and draw bounding boxes in the preview widget.

## Contributing
Issues and PRs are welcome. Please keep the app offline-only and lightweight.
