import logging
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QMessageBox
from app.main_window import MainWindow


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception:  # noqa: BLE001
        logging.exception("Application failed to start")
        QMessageBox.critical(None, "LEGOJiiRi", "Failed to start application. Check console for details.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
