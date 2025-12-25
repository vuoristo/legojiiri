## Prompt for codex

You are Codex acting as a senior Windows desktop + applied ML engineer. Build a complete Python project for a standalone Windows app that labels LEGO parts from photos, designed to run on a typical laptop offline.

High-level goals
- A Windows desktop GUI app where the user can:
  1) Load one or more images (JPG/PNG) of LEGO parts.
  2) See the predicted label (top-5 with confidences).
  3) Correct the label if wrong (for dataset building).
  4) Manage a list of part labels (IDs + display names).
  5) Optionally fine-tune a small CNN locally on the user’s labeled images.
  6) Save and load the trained model.
- The app should work without a GPU. If GPU is present it can use it but must not require it.
- The deliverable must run on Windows 10/11.

Technical requirements
- Language: Python 3.11+
- GUI: PySide6 (Qt) with a clean layout:
  - Left pane: file list / dataset browser
  - Center: image preview with zoom-to-fit
  - Right pane: predictions + “Assign label” controls
  - Top toolbar: Open Images, Open Folder, Predict, Train, Settings
- ML: start with transfer learning image classification:
  - Use torchvision pretrained MobileNetV3-Small (or ResNet18 if simpler).
  - Input size 224x224.
  - Support training on CPU with reasonable defaults:
    - batch_size 16 (auto-reduce if OOM)
    - epochs 5 by default
    - early stopping optional
- Inference:
  - Use PyTorch eager OR export to ONNX and run with onnxruntime (choose one path and implement fully).
  - Provide top-5 predictions with softmax probabilities.
- Dataset format:
  - Store images under `data/images/<class_id>/...`
  - Maintain `data/labels.json` with entries like:
    [
      {"id":"3001","name":"Brick 2x4"},
      {"id":"3003","name":"Brick 2x2"}
    ]
  - When the user assigns a label in the UI, copy (or move) the image into the correct class folder and update metadata if needed.
- Project quality:
  - Provide a `README.md` with setup steps, usage, and packaging instructions.
  - Provide `requirements.txt`.
  - Provide sensible logging and error handling.
  - Keep code modular:
    - `app/` for GUI
    - `ml/` for training/inference
    - `data/` for labels and images
- Packaging:
  - Include instructions for PyInstaller to build a single-folder distribution.
  - Provide a `build_windows.ps1` script that builds the executable.

Features to include in the GUI
1) “Open Images…”: select multiple files, add to a queue/list.
2) “Predict”: runs inference on the currently selected image and shows top-5.
3) “Assign label”: dropdown of known labels + a “Create new label” dialog (id + name).
4) “Train”: trains/fine-tunes the model on current dataset; show progress bar and log output.
5) “Model management”: load model, save model, show model path in Settings.
6) Basic pre-processing: center-crop/resize to 224, normalize using ImageNet stats.

Non-goals (do not implement)
- Cloud services, web backends, accounts, telemetry.
- Heavy AutoML.
- Complex 3D reconstruction.

Deliverables
- Generate the complete repository tree with all source files.
- Ensure code is runnable after `pip install -r requirements.txt` and `python -m app`.
- Provide clear comments where choices matter (e.g., why this backbone, how to add more augmentations).
- Provide minimal but functional UI polish.

Also include a short section in README:
- “How to take good photos for best accuracy” (lighting, plain background, one piece per image, consistent scale).
- “How to expand to object detection later” (briefly outline switching to YOLO, not required to implement).

Start by outputting the repository structure, then each file’s content in code blocks with correct filenames as headings.
