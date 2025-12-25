# Windows helper to package the app using PyInstaller
# Usage: powershell -ExecutionPolicy Bypass -File build_windows.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

# Ensure data folder is bundled next to the executable (single-folder mode)
$env:PYTHONWARNINGS = "ignore"
pyinstaller --noconfirm --onedir --name legojiiri --add-data "data;data" --collect-all PySide6 --collect-submodules torchvision --collect-submodules torch app\__main__.py

Write-Host "Build completed. Check dist/legojiiri/ for the executable and copy the data directory alongside it."
