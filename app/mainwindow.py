"""Main window: Main / Setup / Output / Advanced tabs, Run/Cancel, live log,
results summary. Non-visual logic (collect_runconfig, conflict rule) is exposed
as methods for testing.
"""
import os
import tempfile
import webbrowser

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QFormLayout, QVBoxLayout, QHBoxLayout,
    QComboBox, QDoubleSpinBox, QCheckBox, QPushButton, QPlainTextEdit, QLabel,
    QLineEdit, QFileDialog, QListWidget,
)

from app.runconfig import RunConfig, TIER1_FIELDS
from app.solve_runner import SolveRunner
from app import results, paths

CIRCUITS = ["Sturn", "Straight", "Hairpin", "Circle", "ZigZag", "ZigZagMirror",
            "VirtualTrack", "BCN", "BCN_S1", "BCN_S2", "BCN_S3", "Jarama", "Spa",
            "BCNAssetto"]
AEROS = ["Static", "Active_RW", "Active", "AALB"]
TYRES = ["CombinedSlip", "PureSlip"]
SOLVERS = ["ma57", "ma97", "ma27", "mumps"]


def _spin(value, lo, hi, step=1.0, decimals=4):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(value)
    return s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FullModelSim — MLTP")
        self._runner = SolveRunner(self)
        self._runner.output.connect(self._append_log)
        self._runner.finished.connect(self._on_finished)
        self._defaults = RunConfig()

        tabs = QTabWidget()
        tabs.addTab(self._build_main_tab(), "Main")
        tabs.addTab(self._build_setup_tab(), "Setup")
        tabs.addTab(self._build_output_tab(), "Output")
        tabs.addTab(self._build_advanced_tab(), "Advanced")

        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._runner.cancel)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.cancel_btn)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addWidget(tabs)
        lay.addLayout(buttons)
        lay.addWidget(QLabel("Solver log:"))
        lay.addWidget(self.log)
        self.setCentralWidget(central)

    # ---- tab builders -----------------------------------------------------
    def _build_main_tab(self):
        w = QWidget(); form = QFormLayout(w)
        self.circuit = QComboBox(); self.circuit.addItems(CIRCUITS)
        self.aero = QComboBox(); self.aero.addItems(AEROS)
        self.atd = QCheckBox("ATD on"); self.atd.setChecked(True)
        self.em4 = QCheckBox("4 Motors on")
        self.em4.toggled.connect(self._on_em4_toggled)
        self.tyre = QComboBox(); self.tyre.addItems(TYRES)
        self.vi = _spin(60.0, 0.0, 150.0, 1.0, 2)
        self.ni_free = QCheckBox("free"); self.ni_free.setChecked(True)
        self.ni = _spin(0.0, -10.0, 10.0, 0.1, 3)
        self.ni.setEnabled(False)
        self.ni_free.toggled.connect(lambda f: self.ni.setEnabled(not f))
        ni_row = QWidget(); nilay = QHBoxLayout(ni_row); nilay.setContentsMargins(0, 0, 0, 0)
        nilay.addWidget(self.ni); nilay.addWidget(self.ni_free)
        form.addRow("Circuit", self.circuit)
        form.addRow("Aero config", self.aero)
        form.addRow("ATD", self.atd)
        form.addRow("4 Motors", self.em4)
        form.addRow("Tyre model", self.tyre)
        form.addRow("Initial speed [m/s]", self.vi)
        form.addRow("Initial lateral n [m]", ni_row)
        return w

    def _build_setup_tab(self):
        w = QWidget(); form = QFormLayout(w)
        d = self._defaults
        self.brkB = _spin(d.brkB, 0.0, 1.0, 0.01)
        self.Tdist = _spin(d.Tdist, 0.0, 1.0, 0.01)
        self.ksD = _spin(d.ksD, 0.0, 1.0, 0.01)
        self.alpha_FL = _spin(d.alpha_FL, 0.0, 10.0, 0.5, 2)
        self.alpha_FR = _spin(d.alpha_FR, 0.0, 10.0, 0.5, 2)
        self.alpha_RW = _spin(d.alpha_RW, 0.0, 30.0, 0.5, 2)
        self.alpha_TW = _spin(d.alpha_TW, -12.0, 12.0, 0.5, 2)
        form.addRow("Brake bias (front)", self.brkB)
        form.addRow("Torque dist (rear)", self.Tdist)
        form.addRow("Roll stiff (rear)", self.ksD)
        form.addRow("Front wing L [deg]", self.alpha_FL)
        form.addRow("Front wing R [deg]", self.alpha_FR)
        form.addRow("Rear wing [deg]", self.alpha_RW)
        form.addRow("Rear wing tilt [deg]", self.alpha_TW)
        return w

    def _build_output_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        form = QFormLayout()
        self.output_dir = QLineEdit(self._defaults.output_dir)
        browse = QPushButton("Browse…"); browse.clicked.connect(self._pick_output)
        odir = QWidget(); ol = QHBoxLayout(odir); ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(self.output_dir); ol.addWidget(browse)
        self.save_cb = QCheckBox("Save .mat"); self.save_cb.setChecked(True)
        self.plot_cb = QCheckBox("Generate plots"); self.plot_cb.setChecked(True)
        form.addRow("Output folder", odir)
        form.addRow("", self.save_cb)
        form.addRow("", self.plot_cb)
        lay.addLayout(form)
        self.summary = QLabel("No results yet.")
        self.plot_list = QListWidget()
        self.plot_list.itemDoubleClicked.connect(
            lambda it: webbrowser.open(it.data(256)))
        lay.addWidget(self.summary)
        lay.addWidget(QLabel("Plots (double-click to open):"))
        lay.addWidget(self.plot_list)
        return w

    def _build_advanced_tab(self):
        w = QWidget(); form = QFormLayout(w)
        self.solver = QComboBox(); self.solver.addItems(SOLVERS)
        self.warm_start = QLineEdit(); self.warm_start.setPlaceholderText("(auto warm start)")
        ws_btn = QPushButton("…"); ws_btn.clicked.connect(self._pick_warm)
        ws = QWidget(); wl = QHBoxLayout(ws); wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(self.warm_start); wl.addWidget(ws_btn)
        self.expert = QLineEdit(); self.expert.setPlaceholderText("(no expert config)")
        ex_btn = QPushButton("…"); ex_btn.clicked.connect(self._pick_expert)
        ex = QWidget(); el = QHBoxLayout(ex); el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(self.expert); el.addWidget(ex_btn)
        form.addRow("Linear solver", self.solver)
        form.addRow("Warm start .mat", ws)
        form.addRow("Expert config .json", ex)
        return w

    # ---- conflict rule ----------------------------------------------------
    def _on_em4_toggled(self, on):
        if on:
            self.atd.setChecked(False)
        self.atd.setEnabled(not on)

    def set_em4(self, on):           # test hook
        self.em4.setChecked(on)

    def atd_enabled(self):           # test hook
        return self.atd.isEnabled()

    # ---- collect / run ----------------------------------------------------
    def collect_runconfig(self):
        return RunConfig(
            circuit=self.circuit.currentText(),
            AeroConfig=self.aero.currentText(),
            ATD="On" if self.atd.isChecked() else "Off",
            Electric_4Motors="On" if self.em4.isChecked() else "Off",
            TyreModel=self.tyre.currentText(),
            vi=self.vi.value(),
            ni=None if self.ni_free.isChecked() else self.ni.value(),
            linear_solver=self.solver.currentText(),
            warm_start=self.warm_start.text() or None,
            save=self.save_cb.isChecked(),
            plot=self.plot_cb.isChecked(),
            output_dir=self.output_dir.text(),
            brkB=self.brkB.value(), Tdist=self.Tdist.value(), ksD=self.ksD.value(),
            alpha_FL=self.alpha_FL.value(), alpha_FR=self.alpha_FR.value(),
            alpha_RW=self.alpha_RW.value(), alpha_TW=self.alpha_TW.value(),
            expert_config=self.expert.text() or None,
        )

    def _on_run(self):
        rc = self.collect_runconfig()
        try:
            os.makedirs(rc.output_dir, exist_ok=True)
            cfg_path = os.path.join(tempfile.gettempdir(), "fms_solve_cfg.json")
            rc.write_cfg(cfg_path)
        except Exception as exc:
            self._append_log(f"[error] {exc}\n")
            return
        self._active_rc = rc
        self.log.clear()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._runner.start(cfg_path)

    def _on_finished(self, code):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if code == 0:
            self._show_results(self._active_rc)
        else:
            self._append_log(f"\n[solve exited with code {code}]\n")

    def _show_results(self, rc):
        mat = results.result_mat_path(rc.output_dir, rc.circuit, rc.AeroConfig,
                                      rc.ATD, rc.Electric_4Motors)
        try:
            summ = results.parse_summary(mat)
            e = summ["energy_kWh"]
            self.summary.setText(
                f"Lap time: {summ['lap_time_s']:.3f} s"
                + (f"   |   Energy: {e:.3f} kWh" if e is not None else ""))
        except Exception as exc:
            self.summary.setText(f"Results unreadable: {exc}")
        pdir = results.plot_dir(rc.output_dir, rc.circuit, rc.AeroConfig,
                                rc.ATD, rc.Electric_4Motors)
        self.plot_list.clear()
        for p in results.list_plots(pdir):
            from PySide6.QtWidgets import QListWidgetItem
            it = QListWidgetItem(os.path.basename(p))
            it.setData(256, p)
            self.plot_list.addItem(it)

    # ---- helpers ----------------------------------------------------------
    def _append_log(self, text):
        # Fix: brief had self.log.textCursor().End which is an instance with no .End attr.
        # QTextCursor.End is a class-level enum; use it directly.
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.output_dir.text())
        if d:
            self.output_dir.setText(d)

    def _pick_warm(self):
        f, _ = QFileDialog.getOpenFileName(self, "Warm start .mat", "", "MAT (*.mat)")
        if f:
            self.warm_start.setText(f)

    def _pick_expert(self):
        f, _ = QFileDialog.getOpenFileName(self, "Expert config", "", "JSON (*.json)")
        if f:
            self.expert.setText(f)
            self._load_expert_tier1(f)

    def _load_expert_tier1(self, path):
        import json
        try:
            data = json.load(open(path))
        except Exception as exc:
            self._append_log(f"[expert config error] {exc}\n")
            return
        for k in TIER1_FIELDS:
            if k in data and hasattr(self, k):
                getattr(self, k).setValue(float(data[k]))
