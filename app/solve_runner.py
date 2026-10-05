"""Spawn the headless solve as a subprocess and stream its merged stdout.
In a frozen app the exe re-invokes itself (`<exe> --headless cfg`); in dev it
runs `python headless_solve.py cfg`.
"""
import os
import sys

from PySide6.QtCore import QObject, QProcess, Signal

from app.paths import is_frozen, resource_root


def solve_command(cfg_path):
    if is_frozen():
        return [sys.executable, "--headless", cfg_path]
    return [sys.executable, os.path.join(resource_root(), "headless_solve.py"), cfg_path]


class SolveRunner(QObject):
    output = Signal(str)
    finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None

    def start(self, cfg_path):
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(lambda code, _status: self.finished.emit(int(code)))
        argv = solve_command(cfg_path)
        self._proc.start(argv[0], argv[1:])

    def _on_output(self):
        text = bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.output.emit(text)

    def cancel(self):
        if self._proc is not None and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()
