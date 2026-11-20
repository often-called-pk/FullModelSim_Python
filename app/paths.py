"""Frozen-aware path resolution for the desktop GUI (Windows and Linux).

Bundled (PyInstaller) read-only resources live under sys._MEIPASS; in a dev
checkout they live in the repo root (app/paths.py -> app/ -> repo root).
Solve output (Results/Plots) defaults to a user-writable Documents folder.
"""
import os
import sys


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def resource_root():
    if is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(*parts):
    return os.path.join(resource_root(), *parts)


ORG_NAME = "FullModelSim"
APP_NAME = "FullModelSim"


def default_output_dir():
    """Qt Documents folder + FullModelSim. Home folder when Qt gives none."""
    docs = ""
    try:
        from PySide6.QtCore import QStandardPaths
        docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    except Exception:
        pass
    return os.path.join(docs or os.path.expanduser("~"), "FullModelSim")


def settings():
    """Per-user QSettings (last output folder, Coin-HSL folder). The env
    FULLMODELSIM_SETTINGS names an ini file to use instead (the tests set it)."""
    from PySide6.QtCore import QSettings
    path = os.environ.get("FULLMODELSIM_SETTINGS")
    if path:
        return QSettings(path, QSettings.IniFormat)
    return QSettings(ORG_NAME, APP_NAME)


def user_presets_dir():
    """Writable directory for user-saved setup presets (created on demand by
    the caller). The bundled app/presets/ is read-only when frozen."""
    return os.path.join(default_output_dir(), "presets")


def user_cars_dir():
    """Writable directory for the .car files that the GUI writes (edited cars and saved cars)."""
    return os.path.join(user_presets_dir(), "cars")
