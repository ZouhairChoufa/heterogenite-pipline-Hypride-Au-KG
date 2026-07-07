"""
utils/logger.py
===============
Logger centralisé pour l'ensemble du pipeline PFE.
- Console  : niveau INFO  (lisible en temps réel dans Lightning.ai)
- Fichier  : niveau DEBUG (traceback complet, rotation 5 MB × 3 backups)

Usage dans chaque module :
    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "pipeline.log"

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger nommé, configuré avec deux handlers.
    Idempotent : les handlers ne sont ajoutés qu'une seule fois.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Handler 1 — Console (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_FORMATTER)

    # Handler 2 — Fichier rotatif (DEBUG+ avec traceback)
    fh = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FORMATTER)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
