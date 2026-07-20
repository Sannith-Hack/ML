"""
HECAR Framework — Main Entry Point
===================================
Hybrid ECG PDF-Based Arrhythmia Classification and Cardiovascular Risk Prediction

Run:
    python main.py
    or double-click run.bat
"""

import sys
import logging
from pathlib import Path

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Pre-load and lock local config module into sys.modules so OpenCV (cv2/config.py) never hijacks it
import config as _hecar_config
sys.modules["config"] = _hecar_config

# ── Bootstrap logging before any imports ────────────────────────────────────
from config import LOG_LEVEL, LOG_FORMAT, APP_TITLE

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "data" / "outputs" / "hecar.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Verify critical dependencies are importable before launching GUI."""
    required = {
        "customtkinter": "customtkinter",
        "pdfplumber": "pdfplumber",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "tensorflow": "tensorflow",
        "xgboost": "xgboost",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "shap": "shap",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error("Missing dependencies: %s", ", ".join(missing))
        logger.error("Run:  pip install -r requirements.txt")
        return False

    logger.info("All core dependencies verified.")
    return True


def main() -> None:
    """Launch the HECAR GUI application."""
    logger.info("=" * 60)
    logger.info("  %s", APP_TITLE)
    logger.info("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    try:
        from modules.gui.app import HECARApp
    except ImportError as exc:
        logger.critical("Failed to import GUI module: %s", exc)
        sys.exit(1)

    try:
        app = HECARApp()
        logger.info("Application started successfully.")
        app.run()
    except KeyboardInterrupt:
        logger.info("Application closed by user.")
    except Exception as exc:
        logger.exception("Unexpected error in main application: %s", exc)
        sys.exit(1)
    finally:
        logger.info("HECAR Framework shut down.")


if __name__ == "__main__":
    main()
