"""app/widgets.py — small reusable Qt widgets for the parameter editor."""
from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QWidget, QToolButton, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
)

from app.vp_params import fmt_sci


class ScientificField(QWidget):
    """A line-edit numeric field that preserves full float precision (including
    tiny/exponent values such as -9.1492e-11), exposing the QDoubleSpinBox-like
    API the Setup tab relies on: value()/setValue()/valueChanged."""
    valueChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edit = QLineEdit(self)
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.ScientificNotation)
        self._edit.setValidator(validator)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._edit)
        self._edit.editingFinished.connect(self._emit)

    def _emit(self):
        try:
            self.valueChanged.emit(self.value())
        except ValueError:
            pass

    def value(self):
        return float(self._edit.text())

    def setValue(self, x):
        self._edit.setText(fmt_sci(x))

    def set_changed(self, changed):
        self._edit.setStyleSheet("font-weight: bold; color: #b30000;" if changed else "")

    def setToolTip(self, text):
        super().setToolTip(text)
        self._edit.setToolTip(text)


class CollapsibleSection(QWidget):
    """A titled section whose body collapses. Header shows an arrow, the title,
    and a '(N changed)' badge; an inline 'Reset' button emits reset_requested."""
    reset_requested = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title = title
        self._changed = 0

        self._btn = QToolButton(self)
        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.setStyleSheet("QToolButton { border: none; font-weight: bold; }")

        self._reset = QToolButton(self)
        self._reset.setText("Reset")
        self._reset.setStyleSheet("QToolButton { border: none; color: #555; }")
        self._reset.clicked.connect(self.reset_requested)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._btn)
        header.addStretch(1)
        header.addWidget(self._reset)

        self._body = QWidget(self)
        self._form = QFormLayout(self._body)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(header)
        lay.addWidget(self._body)

        self._btn.toggled.connect(self._on_toggled)
        self._refresh()

    def _on_toggled(self, on):
        self._body.setVisible(on)
        self._refresh()

    def _refresh(self):
        arrow = "▾" if self._btn.isChecked() else "▸"
        badge = f"   ({self._changed} changed)" if self._changed else ""
        self._btn.setText(f"{arrow}  {self._title}{badge}")

    def addRow(self, label, field):
        self._form.addRow(label, field)

    def set_expanded(self, on):
        self._btn.setChecked(on)

    def set_changed_count(self, n):
        self._changed = n
        self._refresh()
