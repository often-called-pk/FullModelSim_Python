"""Main window: Main / Setup / Output / Advanced tabs, Run/Cancel, live log,
results summary. Non-visual logic (collect_runconfig, conflict rule) is exposed
as methods for testing.
"""
import os
import json
import tempfile
import webbrowser

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QFormLayout, QVBoxLayout, QHBoxLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QPlainTextEdit,
    QLabel, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QScrollArea,
    QGroupBox,
)

from app.runconfig import RunConfig
from app.solve_runner import SolveRunner
from app import results, paths
from app.vp_params import PARAM_GROUPS, meta_for, all_vp_defaults
from app.widgets import CollapsibleSection, ScientificField
from app.paths import user_presets_dir

CIRCUITS = ["Sturn", "Straight", "Hairpin", "Circle", "ZigZag", "ZigZagMirror",
            "VirtualTrack", "BCN", "BCN_S1", "BCN_S2", "BCN_S3", "Jarama", "Spa",
            "BCNAssetto"]
AEROS = ["Static", "Active_RW", "Active", "AALB"]
TYRES = ["CombinedSlip", "PureSlip"]
SOLVERS = ["ma57", "ma97", "ma27", "mumps"]
_PATH_ROLE = 256  # Qt.ItemDataRole.UserRole — stores the plot's filesystem path on the list item


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
        self._active_rc = None

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
        self.vp_spins = {}            # key -> QDoubleSpinBox | ScientificField
        self._sections = {}           # title -> CollapsibleSection
        self._section_keys = {}       # title -> [keys]
        defaults = all_vp_defaults()

        container = QWidget()
        vlay = QVBoxLayout(container)

        bar = QHBoxLayout()
        reset_all = QPushButton("Reset all to defaults")
        reset_all.clicked.connect(self._reset_all_params)
        save_btn = QPushButton("Save preset…"); save_btn.clicked.connect(self._save_preset)
        load_btn = QPushButton("Load preset…"); load_btn.clicked.connect(self._load_preset)
        bar.addWidget(reset_all); bar.addWidget(save_btn); bar.addWidget(load_btn)
        bar.addStretch(1)
        vlay.addLayout(bar)

        for title, keys in PARAM_GROUPS:
            sec = CollapsibleSection(title)
            self._sections[title] = sec
            self._section_keys[title] = list(keys)
            for key in keys:
                m = meta_for(key)
                field = self._field_for(key)
                field.setValue(float(defaults[key]))
                field.valueChanged.connect(lambda _v=None, k=key: self._on_param_changed(k))
                self.vp_spins[key] = field
                lbl = QLabel(m.label); lbl.setToolTip(m.tooltip or m.label)
                sec.addRow(lbl, field)
            sec.set_expanded(title == "Balance & Aero")
            sec.reset_requested.connect(lambda t=title: self._reset_section(t))
            vlay.addWidget(sec)

        vlay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    def _field_for(self, key):
        m = meta_for(key)
        if m.kind == "sci":
            field = ScientificField()
        else:
            field = QDoubleSpinBox()
            field.setRange(m.lo, m.hi)
            field.setSingleStep(m.step)
            field.setDecimals(m.decimals)
            if m.unit:
                field.setSuffix(f" {m.unit}")
        field.setToolTip(m.tooltip or m.label)
        return field

    # ---- changed-from-default highlighting --------------------------------
    def _on_param_changed(self, key):
        defaults = all_vp_defaults()
        self._set_field_changed_style(self.vp_spins[key], self._is_changed(key, defaults))
        for title, keys in self._section_keys.items():
            if key in keys:
                self._update_section_badge(title, defaults)
                break

    def _is_changed(self, key, defaults):
        try:
            return float(self.vp_spins[key].value()) != float(defaults[key])
        except ValueError:
            return True

    def _set_field_changed_style(self, field, changed):
        if isinstance(field, ScientificField):
            field.set_changed(changed)
        else:
            field.setStyleSheet("font-weight: bold; color: #b30000;" if changed else "")

    def _update_section_badge(self, title, defaults=None):
        if defaults is None:
            defaults = all_vp_defaults()
        n = sum(1 for k in self._section_keys[title] if self._is_changed(k, defaults))
        self._sections[title].set_changed_count(n)

    def _refresh_all_changed(self):
        defaults = all_vp_defaults()
        for k in self.vp_spins:
            self._set_field_changed_style(self.vp_spins[k], self._is_changed(k, defaults))
        for title in self._section_keys:
            self._update_section_badge(title, defaults)

    # ---- reset / preset ---------------------------------------------------
    def _reset_all_params(self):
        defaults = all_vp_defaults()
        for k, field in self.vp_spins.items():
            field.setValue(float(defaults[k]))
        self._refresh_all_changed()

    def _reset_section(self, title):
        defaults = all_vp_defaults()
        for k in self._section_keys[title]:
            self.vp_spins[k].setValue(float(defaults[k]))
        self._refresh_all_changed()

    def _save_preset(self):
        os.makedirs(user_presets_dir(), exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save preset", os.path.join(user_presets_dir(), "setup.json"),
            "JSON (*.json)")
        if not path:
            return
        data = {k: float(self.vp_spins[k].value()) for k in self.vp_spins}
        try:
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
            self._append_log(f"[preset saved] {path}\n")
        except Exception as exc:
            self._append_log(f"[preset save error] {exc}\n")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load preset", user_presets_dir(), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:
            self._append_log(f"[preset load error] {exc}\n")
            return
        self._apply_param_dict(data)

    def _apply_param_dict(self, data):
        applied, ignored = 0, 0
        for k, v in data.items():
            if k in self.vp_spins:
                try:
                    self.vp_spins[k].setValue(float(v))
                    applied += 1
                except (TypeError, ValueError):
                    ignored += 1
            else:
                ignored += 1
        self._refresh_all_changed()
        msg = f"[preset] applied {applied} values"
        if ignored:
            msg += f", ignored {ignored} unknown/invalid keys"
        self._append_log(msg + "\n")

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
            lambda it: webbrowser.open(it.data(_PATH_ROLE)))
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

        grp = QGroupBox("Solver & Collocation")
        gform = QFormLayout(grp)
        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 100000); self.max_iter.setValue(6000)
        self.opt_ds = QDoubleSpinBox(); self.opt_ds.setRange(1.0, 500.0)
        self.opt_ds.setSingleStep(1.0); self.opt_ds.setDecimals(2)
        self.opt_ds.setValue(30.0); self.opt_ds.setSuffix(" m")
        self.opt_d = QSpinBox(); self.opt_d.setRange(1, 6); self.opt_d.setValue(3)
        self.opt_e = ScientificField(); self.opt_e.setValue(1e-2)
        self.tol = ScientificField(); self.tol.setValue(1e-4)
        gform.addRow("Max iterations", self.max_iter)
        gform.addRow("Collocation step", self.opt_ds)
        gform.addRow("Polynomial degree", self.opt_d)
        gform.addRow("Path-constraint slack", self.opt_e)
        gform.addRow("IPOPT tolerance", self.tol)
        form.addRow(grp)
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
        defaults = all_vp_defaults()
        vp = {}
        for k, field in self.vp_spins.items():
            try:
                vp[k] = float(field.value())
            except (TypeError, ValueError):
                vp[k] = float(defaults[k])
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
            expert_config=self.expert.text() or None,
            max_iter=self.max_iter.value(),
            OPT_ds=self.opt_ds.value(),
            OPT_d=self.opt_d.value(),
            OPT_e=float(self.opt_e.value()),
            tol=float(self.tol.value()),
            vp=vp,
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
            it = QListWidgetItem(os.path.basename(p))
            it.setData(_PATH_ROLE, p)
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
            self._load_expert_into_widgets(f)

    def _load_expert_into_widgets(self, path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:
            self._append_log(f"[expert config error] {exc}\n")
            return
        self._apply_param_dict(data)
