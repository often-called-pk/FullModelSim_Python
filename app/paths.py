"""Frozen-aware path resolution for the windows-app GUI.

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


def default_output_dir():
    return os.path.join(os.path.expanduser("~"), "Documents", "FullModelSim")


def user_presets_dir():
    """Writable directory for user-saved setup presets (created on demand by
    the caller). The bundled app/presets/ is read-only when frozen."""
    return os.path.join(default_output_dir(), "presets")
