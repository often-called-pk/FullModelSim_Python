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
from app import results, paths, car_edit
from app.vp_params import PARAM_GROUPS, meta_for, all_vp_defaults
from app.widgets import CollapsibleSection, ExactField, ScientificField
from app.paths import settings
from functions import chassissim
from headless_solve import build_solve_kwargs
from userOpts import car_refusal

CIRCUITS = ["Sturn", "Straight", "Hairpin", "Circle", "ZigZag", "ZigZagMirror",
            "VirtualTrack", "BCN", "BCN_S1", "BCN_S2", "BCN_S3", "Jarama", "Spa",
            "NBR"]
AEROS = ["Static", "Active_RW", "Active", "AALB"]
TYRES = ["CombinedSlip", "PureSlip"]
SOLVERS = ["ma57", "ma97", "ma27", "mumps"]
TYRE_SETS = ["MF205", "CopyB"]              # vehParams tyre_set names, default first
MESHES = ["auto", "uniform", "curvature"]   # userOpts mesh choices, default first
# F2 combos: RunConfig field, label, items (the first item is the RunConfig default), tooltip
OPTION_COMBOS = [
    ("mode", "Run", ["nlp", "qss"],
     "nlp = the optimal-control solve. qss = a quasi-steady lap ESTIMATE, not an optimum. It takes about 1 s, "
     "uses no IPOPT and follows the centreline. On Sturn and BCN its lap is 4% to 11% above the NLP lap. It "
     "saves a .mat file and no plots. It ignores the warm start, ladder, homotopy, refine, continue_from and "
     "the IPOPT options."),
    ("physics", "Physics", ["matlab", "corrected"],
     "matlab = the MATLAB model, bit for bit (default). corrected = the opt-in changes A1 to A6 (mass "
     "bookkeeping, damper, frame, aero load path, 4-motor speed, toe). The result file name gets the "
     "suffix _physCorrected."),
    ("start_state", "Start state", ["auto", "matlab", "rolling", "periodic"],
     "auto = periodic for a ChassisSim car on a ChassisSim circuit that joins at the line, else matlab "
     "(default). matlab = the MATLAB start box (a car also holds its wheel speeds at rolling speed). "
     "rolling = the wheels and the chassis start at the coast equilibrium. periodic = free start, and the "
     "end state equals the start state (closed circuits and ChassisSim circuits)."),
]
# RunConfig fields that the "More options" JSON box may set (they have no widget). refine is not one of them:
# an accepted refinement pass saves as <stem>_meshAdaptive, a file that the Output tab does not look for.
MORE_KEYS = ("n_bounds", "ladder", "ipopt_overrides", "Steering", "CamberGain", "pt_overrides",
             "between_knots", "collocation", "state_scaling", "atd_tbr", "continue_from")
MORE_TIP = (
    "More options as a JSON object. Keys: n_bounds (half-width in m, or [lo, hi]), ladder (auto, legacy, qss7, "
    "qss23), ipopt_overrides (IPOPT options), Steering (NA, Ackermann), CamberGain (Off, Linear, Table), "
    "pt_overrides (the A5 keys gear, Tmax, Pmax, Preg, Ppack: physics = corrected only), between_knots (matlab, "
    "linear, linear+power), collocation (legendre, radau), state_scaling (matlab, typical), atd_tbr (torque-bias "
    "ratio, 1 or more: ATD on only), continue_from (auto = start from the stored default result).")
HOMOTOPY_TIP = (
    "Solve three times, with the tyre friction scaled by 1.2, then 1.1, then 1.0. Each step starts from the "
    "result before it. Use it when a cold solve fails. Choosing 4 Motors ticks this box, because a cold "
    "4-motor solve can fail. Clearing 4 Motors unticks it. Untick it to solve once, for example from a warm "
    "start .mat. MLTP refuses continue_from together with this box.")
CAR_4EM = "4EM (FullModelSim)"
CAR_TIP = ("Select the car of the solve. 4EM (FullModelSim) is the built-in car with four motors. A ChassisSim "
           ".car file selects the car of that file. The solve of a ChassisSim car uses corrected physics, one "
           "engine and static aero. The list shows the cars of the ChassisSim folder (Advanced tab) and of the "
           "Models folder beside it. It also shows the saved cars of <Documents>/FullModelSim/presets/cars. "
           "Browse adds a .car file from a different folder.")
CS_DIR_TIP = ("Select the ChassisSim install folder. The ChassisSim circuits and cars come from this folder. "
              "If the field is empty, the program uses the CHASSISSIM_DIR environment variable.")
CAR_LOCK_TIP = ("A ChassisSim car sets this value, because the car model needs it. "
                "Select the car 4EM (FullModelSim) to change the value.")
_PATH_ROLE = 256  # Qt.ItemDataRole.UserRole, stores the plot's filesystem path on the list item

TYRE_SET_TIP = (
    "MF205 = MATLAB-run set (default); CopyB = legacy shipped set with pKy4=0 "
    "(zero cornering stiffness), kept to reproduce old results. The Setup tab's "
    "Pacejka defaults, Reset and changed-value highlighting follow this choice.")
MESH_TIP = (
    "Collocation knot placement. auto = curvature-weighted knots for laps >= 2000 m, "
    "uniform otherwise. uniform = equally spaced knots; curvature = always "
    "curvature-weighted (same number of knots).")

# Setup rows the full model ignores (vehModel.py, powertrain torque split). ATD on:
# the wheel-torque split comes from the four ATD inputs, so the fixed brake bias and
# drive-torque distribution are both unused. 4 Motors on: each wheel gets its own motor
# torque plus T_brake * brkB, so only the Tdist drive split is unused. ATD_INERT_KEYS is
# the superset, i.e. every row _update_inert_params manages.
ATD_INERT_KEYS = ("brkB", "Tdist")
EM4_INERT_KEYS = ("Tdist",)
ATD_INERT_TIP = (
    "Inactive while ATD is on: the full model takes the wheel-torque split from the "
    "four ATD inputs, so this value is ignored (only the 7-state warm-start model "
    "still reads it).")
EM4_INERT_TIP = (
    "Inactive while 4 Motors is on: the full model drives each wheel from its own "
    "motor torque, so this value is ignored (only the 7-state warm-start model "
    "still reads it).")


PT_TITLE = "Powertrain (physics corrected)"
PT_PHYSICS_TIP = "The powertrain keys need physics = corrected."
# key, label, unit, lo, hi, decimals, unit factor to the model (kW -> W), limit row (True: a check box switches it on,
# False: sent when changed), start value of a limit row, tooltip
PT_ROWS = [
    ("gear", "Gear ratio", "", 0.1, 100.0, 4, 1.0, False, None,
     "Final drive ratio. The default is OMmax x Rw / Vmax of Powertrain.py, with the old Rw 0.3142857."),
    ("Tmax", "Max torque per motor", "N m", 1.0, 5000.0, 2, 1.0, False, None, "Torque rating of one motor."),
    ("Pmax", "Max power per motor", "kW", 1.0, 5000.0, 2, 1000.0, False, None, "Power rating of one motor."),
    ("Preg", "Regen power per motor", "kW", 0.0, 5000.0, 2, 1000.0, True, 50.0,
     "Off: the motor does not regenerate. On: the motor regenerates up to this power. 0 kW has the same effect as Off."),
    ("Ppack", "Pack power", "kW", 1.0, 5000.0, 2, 1000.0, True, 300.0,
     "Limit of the sum of the motor powers. Off: the model has no such limit."),
]


HSL_TIP = ("Select the folder that holds the Coin-HSL library (libhsl.dll or libhsl.so). "
           "The ma* solvers use it. The COINHSL_DIR environment variable has priority. "
           "If the folder has no library, the solve uses MUMPS.")


def _spin(value, lo, hi, step=1.0, decimals=4):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(value)
    return s


def _tipped_label(text, tip):
    lbl = QLabel(text)
    lbl.setToolTip(tip)
    return lbl


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FullModelSim, MLTP")
        self._runner = SolveRunner(self)
        self._runner.output.connect(self._append_log)
        self._runner.finished.connect(self._on_finished)
        self._defaults = RunConfig()
        self._active_rc = None
        self._cfg_path = None

        # Build order matters: the Setup table seeds its defaults from the Tyre set
        # combo (Advanced tab) and its brkB/Tdist rows follow the ATD and 4 Motors
        # boxes (Main tab). The tabs are still shown in the order Main, Setup,
        # Output, Advanced.
        main_tab = self._build_main_tab()
        advanced_tab = self._build_advanced_tab()
        setup_tab = self._build_setup_tab()
        output_tab = self._build_output_tab()
        tabs = QTabWidget()
        tabs.addTab(main_tab, "Main")
        tabs.addTab(setup_tab, "Setup")
        tabs.addTab(output_tab, "Output")
        tabs.addTab(advanced_tab, "Advanced")

        self.atd.toggled.connect(self._update_inert_params)
        self.em4.toggled.connect(self._update_inert_params)
        self.opt_combos["physics"].currentTextChanged.connect(self._update_pt)
        self._update_pt()
        self._update_inert_params()          # ATD starts on, so brkB/Tdist start inert

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

        self._cs_logged = None
        self._refresh_chassissim(initial=True)      # after self.log: a folder that does not list logs one line
        self.car.currentIndexChanged.connect(self._on_car_changed)
        self._on_car_changed()

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
        self.car = QComboBox(); self.car.setToolTip(CAR_TIP)
        car_btn = QPushButton("Browse..."); car_btn.clicked.connect(self._pick_car)
        car_row = QWidget(); cl = QHBoxLayout(car_row); cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.car, 1); cl.addWidget(car_btn)
        self._extra_cars = []         # .car files added by Browse or Load preset
        self._pre_car = None          # the controls before a car locked them
        form.addRow("Circuit", self.circuit)
        form.addRow(_tipped_label("Car", CAR_TIP), car_row)
        form.addRow("Aero config", self.aero)
        form.addRow("ATD", self.atd)
        form.addRow("4 Motors", self.em4)
        form.addRow("Tyre model", self.tyre)
        form.addRow("Initial speed [m/s]", self.vi)
        form.addRow("Initial lateral n [m]", ni_row)
        return w

    def _build_setup_tab(self):
        self.vp_spins = {}            # key -> QDoubleSpinBox | ScientificField
        self.vp_labels = {}           # key -> QLabel (row caption)
        self._sections = {}           # title -> CollapsibleSection
        self._car_edit = None         # car_edit.CarEdit of the selected car, None for the 4EM car
        self._car_path = None         # the car that the car sections show
        self._car_fields = {}         # (cid, slot) -> widget of the car sections
        self._car_secs = {}           # group title -> CollapsibleSection of the car
        self._section_keys = {}       # title -> [keys]
        defaults = self._vp_defaults()
        self._seeded_tyre_set = self._tyre_set()    # tyre set the table defaults were last seeded for
        self.tyre_set.currentTextChanged.connect(self._on_tyre_set_changed)

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
                self.vp_labels[key] = lbl
                sec.addRow(lbl, field)
            sec.set_expanded(title == "Balance & Aero")
            sec.reset_requested.connect(lambda t=title: self._reset_section(t))
            vlay.addWidget(sec)

        self._build_pt_section(vlay)

        self._car_box = QWidget()      # the car sections: shown instead of the 4EM sections with a car
        QVBoxLayout(self._car_box).setContentsMargins(0, 0, 0, 0)
        vlay.addWidget(self._car_box)
        vlay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        return scroll

    def _build_pt_section(self, vlay):
        """The A5 powertrain keys of the 4EM car. They are not vp primaries, so they stay out of PARAM_GROUPS."""
        from Powertrain import Powertrain
        from functions.context import Ctx
        ctx = Ctx()
        Powertrain(ctx)
        base = {"gear": ctx.vp.gear, "Tmax": ctx.pt.Tmax, "Pmax": ctx.pt.Pmax / 1000.0}
        self.pt_spins, self.pt_checks, self.pt_defaults, self.pt_labels = {}, {}, {}, {}
        sec = self._pt_sec = CollapsibleSection(PT_TITLE)
        for key, label, unit, lo, hi, dec, _f, limit, start, tip in PT_ROWS:
            sp = _spin(base.get(key, start), lo, hi, 0.1 if key == "gear" else 1.0, dec)
            if unit:
                sp.setSuffix(f" {unit}")
            sp.valueChanged.connect(lambda _v=None: self._update_pt_changed())
            self.pt_spins[key] = sp
            self.pt_defaults[key] = sp.value()
            lbl = self.pt_labels[key] = QLabel(label)      # _update_pt sets the tooltips
            if limit:
                cb = self.pt_checks[key] = QCheckBox("on")
                sp.setEnabled(False)                       # the value counts only while the box is on
                cb.toggled.connect(sp.setEnabled)
                cb.toggled.connect(lambda _v=None: self._update_pt_changed())
                row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0)
                rl.addWidget(cb); rl.addWidget(sp)
                sec.addRow(lbl, row)
            else:
                sec.addRow(lbl, sp)
        sec.set_expanded(False)
        sec.reset_requested.connect(self._reset_pt)
        vlay.addWidget(sec)

    def _pt_changed(self, key):
        if key in self.pt_checks:
            return self.pt_checks[key].isChecked()
        return self.pt_spins[key].value() != self.pt_defaults[key]

    def _update_pt_changed(self):
        n = 0
        for key, sp in self.pt_spins.items():
            ch = self._pt_changed(key)
            n += ch
            self._set_field_changed_style(sp, ch)
        self._pt_sec.set_changed_count(n)

    def _reset_pt(self):
        for key, sp in self.pt_spins.items():
            sp.setValue(self.pt_defaults[key])      # a limit row keeps its start value, its check box goes off
            if key in self.pt_checks:
                self.pt_checks[key].setChecked(False)
        self._update_pt_changed()

    def _update_pt(self, _x=None):
        """The group needs corrected physics and the 4EM car."""
        car = bool(self.car.currentData())
        ok = self.opt_combos["physics"].currentText() == "corrected"
        self._pt_sec.setVisible(not car)
        self._pt_sec.setEnabled(ok)
        self._pt_sec.setToolTip("" if ok else PT_PHYSICS_TIP)
        for r in PT_ROWS:
            tip = r[-1] if ok else f"{r[-1]} {PT_PHYSICS_TIP}"
            for w in (self.pt_spins[r[0]], self.pt_labels[r[0]]):
                w.setToolTip(tip)

    def _pt_values(self):
        """The changed and switched-on values of the group in model units [W, N m]. The round removes the float
        noise of kW x 1000."""
        return {r[0]: round(self.pt_spins[r[0]].value() * r[6], 6) for r in PT_ROWS if self._pt_changed(r[0])}

    def pt_overrides(self):
        """The group as pt_overrides. None if nothing is sent: no change, a ChassisSim car or matlab physics."""
        if self.car.currentData() or self.opt_combos["physics"].currentText() != "corrected":
            return None
        return self._pt_values() or None

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

    # ---- tyre set: the table defaults follow the Tyre set combo ------------
    def _tyre_set(self):
        return self.tyre_set.currentText()

    def _vp_defaults(self):
        """Full vp default dict of the currently selected tyre set."""
        return all_vp_defaults(self._tyre_set())

    def _on_tyre_set_changed(self, new_set):
        """Re-seed the Setup table for the newly selected tyre set. A field still at
        the previous set's default takes the new set's default; a field the user
        edited (differs from the previous default) keeps its value and is then
        flagged against the new defaults like any other edit."""
        old = all_vp_defaults(self._seeded_tyre_set)
        new = all_vp_defaults(new_set)
        for k, field in self.vp_spins.items():
            if old[k] == new[k]:
                continue                     # not tyre-set dependent
            try:
                at_old_default = float(field.value()) == float(old[k])
            except ValueError:               # half-typed text: treat as an edit, leave it
                continue
            if at_old_default:
                field.setValue(float(new[k]))
        self._seeded_tyre_set = new_set
        self._refresh_all_changed()

    # ---- changed-from-default highlighting --------------------------------
    def _on_param_changed(self, key):
        defaults = self._vp_defaults()
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
            defaults = self._vp_defaults()
        n = sum(1 for k in self._section_keys[title] if self._is_changed(k, defaults))
        self._sections[title].set_changed_count(n)

    def _refresh_all_changed(self):
        defaults = self._vp_defaults()
        for k in self.vp_spins:
            self._set_field_changed_style(self.vp_spins[k], self._is_changed(k, defaults))
        for title in self._section_keys:
            self._update_section_badge(title, defaults)

    # ---- reset / preset ---------------------------------------------------
    def _reset_all_params(self):
        if self.car.currentData():
            return self._reset_car()
        defaults = self._vp_defaults()
        for k, field in self.vp_spins.items():
            field.setValue(float(defaults[k]))
        self._refresh_all_changed()
        self._reset_pt()

    def _reset_section(self, title):
        defaults = self._vp_defaults()
        for k in self._section_keys[title]:
            self.vp_spins[k].setValue(float(defaults[k]))
        self._refresh_all_changed()

    def _save_preset(self):
        if self.car.currentData():
            return self._save_car()
        os.makedirs(paths.user_presets_dir(), exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save preset", os.path.join(paths.user_presets_dir(), "setup.json"),
            "JSON (*.json)")
        if not path:
            return
        data = {k: float(self.vp_spins[k].value()) for k in self.vp_spins}
        pt = self._pt_values()
        if pt:
            data["pt_overrides"] = pt
        try:
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
            self._append_log(f"[preset saved] {path}\n")
        except Exception as exc:
            self._append_log(f"[preset save error] {exc}\n")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load preset", paths.user_presets_dir(), "Presets (*.json *.car)")
        if not path:
            return
        if path.lower().endswith(".car"):
            if os.path.abspath(path) == self.car.currentData():     # the selected car: read the file again
                self._car_path = None
                self._show_car_setup()
            else:
                self._select_car(path)
            self._append_log(f"[car selected] {path}\n")
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:
            self._append_log(f"[preset load error] {exc}\n")
            return
        self._reset_pt()        # a preset is a full snapshot: a file without the key "pt_overrides" gives the default group
        self._apply_param_dict(data)

    def _apply_param_dict(self, data):
        applied, ignored = 0, 0
        data = dict(data)
        pt = data.pop("pt_overrides", None)
        if isinstance(pt, dict):
            rows = {r[0]: r for r in PT_ROWS}
            self._reset_pt()                    # the preset decides the group: a key that it lacks is off
            for k, v in pt.items():
                try:
                    self.pt_spins[k].setValue(float(v) / rows[k][6])
                    if k in self.pt_checks:
                        self.pt_checks[k].setChecked(True)
                    applied += 1
                except (KeyError, TypeError, ValueError):
                    ignored += 1
        elif pt is not None:
            ignored += 1
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
        if self.car.currentData():
            self._append_log("[preset] The selected ChassisSim car does not read these values. "
                             "They apply to the 4EM car.\n")

    def _build_output_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        form = QFormLayout()
        self.output_dir = QLineEdit(self._saved("output_dir") or self._defaults.output_dir)
        browse = QPushButton("Browse..."); browse.clicked.connect(self._pick_output)
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
        self.tyre_set = QComboBox(); self.tyre_set.addItems(TYRE_SETS)
        self.tyre_set.setToolTip(TYRE_SET_TIP)
        self.warm_start = QLineEdit(); self.warm_start.setPlaceholderText("(auto warm start)")
        ws_btn = QPushButton("Browse..."); ws_btn.clicked.connect(self._pick_warm)
        ws = QWidget(); wl = QHBoxLayout(ws); wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(self.warm_start); wl.addWidget(ws_btn)
        self.expert = QLineEdit(); self.expert.setPlaceholderText("(no expert config)")
        ex_btn = QPushButton("Browse..."); ex_btn.clicked.connect(self._pick_expert)
        ex = QWidget(); el = QHBoxLayout(ex); el.setContentsMargins(0, 0, 0, 0)
        el.addWidget(self.expert); el.addWidget(ex_btn)
        self.hsl_dir = QLineEdit(self._saved("hsl_dir"))
        self.hsl_dir.setPlaceholderText("(not set: COINHSL_DIR, else MUMPS)")
        self.hsl_dir.setToolTip(HSL_TIP)
        hs_btn = QPushButton("Browse..."); hs_btn.clicked.connect(self._pick_hsl)
        hs = QWidget(); hl = QHBoxLayout(hs); hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.hsl_dir); hl.addWidget(hs_btn)
        self.chassissim_dir = QLineEdit(self._saved("chassissim_dir"))
        self.chassissim_dir.setPlaceholderText("(not set: CHASSISSIM_DIR)")
        self.chassissim_dir.setToolTip(CS_DIR_TIP)
        self.chassissim_dir.editingFinished.connect(self._refresh_chassissim)
        cs_btn = QPushButton("Browse..."); cs_btn.clicked.connect(self._pick_chassissim)
        cs = QWidget(); csl = QHBoxLayout(cs); csl.setContentsMargins(0, 0, 0, 0)
        csl.addWidget(self.chassissim_dir); csl.addWidget(cs_btn)
        form.addRow("Linear solver", self.solver)
        form.addRow(_tipped_label("ChassisSim folder", CS_DIR_TIP), cs)
        form.addRow(_tipped_label("Coin-HSL folder", HSL_TIP), hs)
        form.addRow(_tipped_label("Tyre set", TYRE_SET_TIP), self.tyre_set)
        form.addRow("Warm start .mat", ws)
        form.addRow("Expert config .json", ex)

        grp = QGroupBox("Solver & Collocation")
        gform = QFormLayout(grp)
        self.max_iter = QSpinBox(); self.max_iter.setRange(1, 100000); self.max_iter.setValue(6000)
        self.opt_ds = QDoubleSpinBox(); self.opt_ds.setRange(1.0, 500.0)
        self.opt_ds.setSingleStep(1.0); self.opt_ds.setDecimals(2)
        self.opt_ds.setValue(30.0); self.opt_ds.setSuffix(" m")
        self.mesh = QComboBox(); self.mesh.addItems(MESHES)
        self.mesh.setToolTip(MESH_TIP)
        self.opt_d = QSpinBox(); self.opt_d.setRange(1, 6); self.opt_d.setValue(3)
        self.opt_e = ScientificField(); self.opt_e.setValue(1e-2)
        self.tol = ScientificField(); self.tol.setValue(1e-4)
        gform.addRow("Max iterations", self.max_iter)
        gform.addRow("Collocation step", self.opt_ds)
        gform.addRow(_tipped_label("Mesh", MESH_TIP), self.mesh)
        gform.addRow("Polynomial degree", self.opt_d)
        gform.addRow("Path-constraint slack", self.opt_e)
        gform.addRow("IPOPT tolerance", self.tol)
        form.addRow(grp)

        # F2 options. No ampersand in the title: Qt reads it as a mnemonic key.
        self.opt_combos = {}
        grp2 = QGroupBox("Run, model and start")
        f2 = QFormLayout(grp2)
        for key, label, items, tip in OPTION_COMBOS:
            cb = QComboBox(); cb.addItems(items); cb.setToolTip(tip)
            self.opt_combos[key] = cb
            f2.addRow(_tipped_label(label, tip), cb)
        self.homotopy = QCheckBox("friction homotopy (1.2, 1.1, 1.0)")
        self.homotopy.setToolTip(HOMOTOPY_TIP)
        self.more_json = QPlainTextEdit()
        self.more_json.setMaximumHeight(60)
        self.more_json.setPlaceholderText('(none)  e.g. {"n_bounds": 4.5, "ipopt_overrides": {"mu_strategy": "monotone"}}')
        self.more_json.setToolTip(MORE_TIP)
        f2.addRow(_tipped_label("Homotopy", HOMOTOPY_TIP), self.homotopy)
        f2.addRow(_tipped_label("More options (JSON)", MORE_TIP), self.more_json)
        form.addRow(grp2)
        return w

    # ---- conflict rule ----------------------------------------------------
    def _on_em4_toggled(self, on):
        if on:
            self.atd.setChecked(False)
        self.atd.setEnabled(not on)
        # F2-V1: the homotopy converged for 4 Motors on Sturn and BCN, and the cold Sturn solve hit max_iter.
        # This sets the default only. The user can untick the box.
        self.homotopy.setChecked(on)

    def set_em4(self, on):           # test hook
        self.em4.setChecked(on)

    def atd_enabled(self):           # test hook
        return self.atd.isEnabled()

    def inert_vp_keys(self):
        """Setup parameters the full model ignores for the current config. 4 Motors
        wins over ATD (userOpts forces ATD off when both are on) and leaves only
        Tdist inert; otherwise ATD on makes both brkB and Tdist inert."""
        if self.em4.isChecked():
            return set(EM4_INERT_KEYS)
        return set(ATD_INERT_KEYS) if self.atd.isChecked() else set()

    def _update_inert_params(self, _checked=None):
        """Grey out (and explain, via tooltip) the Setup rows that are inert."""
        inert = self.inert_vp_keys()
        why = EM4_INERT_TIP if self.em4.isChecked() else ATD_INERT_TIP
        for key in ATD_INERT_KEYS:
            is_inert = key in inert
            m = meta_for(key)
            base = m.tooltip or m.label
            tip = f"{base}. {why}" if is_inert else base
            for w in (self.vp_spins[key], self.vp_labels[key]):
                w.setEnabled(not is_inert)
                w.setToolTip(tip)

    # ---- collect / run ----------------------------------------------------
    def _more_options(self):
        """The "More options" JSON box as RunConfig fields. A ValueError names the problem."""
        text = self.more_json.toPlainText().strip()
        if not text:
            return {}
        try:
            d = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"More options are not valid JSON: {exc}") from None
        if not isinstance(d, dict):
            raise ValueError('More options must be a JSON object, for example {"n_bounds": 4.5}')
        unknown = sorted(set(d) - set(MORE_KEYS))
        if unknown:
            raise ValueError(f"More options: unknown key(s) {unknown}. Allowed keys: {list(MORE_KEYS)}")
        return d

    def collect_runconfig(self):
        tyre_set = self._tyre_set()
        defaults = all_vp_defaults(tyre_set)
        vp = {}
        for k, field in self.vp_spins.items():
            try:        # a car takes its setup from its file, so its run gets the defaults (= no vp_overrides)
                vp[k] = float(defaults[k] if self.car.currentData() else field.value())
            except (TypeError, ValueError):
                vp[k] = float(defaults[k])
        opts = {k: cb.currentText() for k, cb in self.opt_combos.items()}
        more = self._more_options()
        pt = self.pt_overrides()
        if pt:
            if "pt_overrides" in more:
                raise ValueError("Set the powertrain keys in one place: the Powertrain group or More options.")
            more["pt_overrides"] = pt
        return RunConfig(
            circuit=self.circuit.currentText(),
            AeroConfig=self.aero.currentText(),
            ATD="On" if self.atd.isChecked() else "Off",
            Electric_4Motors="On" if self.em4.isChecked() else "Off",
            TyreModel=self.tyre.currentText(),
            vi=self.vi.value(),
            ni=None if self.ni_free.isChecked() else self.ni.value(),
            linear_solver=self.solver.currentText(),
            hsl_dir=self.hsl_dir.text().strip() or None,
            chassissim_dir=self.chassissim_dir.text().strip() or None,
            car=self.car_ref(),
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
            mesh=self.mesh.currentText(),
            tyre_set=tyre_set,
            homotopy=True if self.homotopy.isChecked() else None,
            **opts,
            **more,
            vp=vp,
        )

    def _on_run(self):
        try:
            rc = self.collect_runconfig()           # a bad "More options" box raises here: log it, start nothing
            if self._car_edit and self._car_edit.edits:      # the run uses a new file that holds the edits
                err = self._car_edit.validate(rc.chassissim_dir)
                if err:
                    raise ValueError(err)
                rc.car = chassissim.PREFIX + self._car_edit.materialize(paths.user_cars_dir())
            why = rc.car and car_refusal(build_solve_kwargs(rc._cfg(), ""))    # the texts of userOpts, before a subprocess
            if why:
                raise ValueError(why)
            rc.output_dir = os.path.abspath(rc.output_dir.strip() or paths.default_output_dir())
            self.output_dir.setText(rc.output_dir)  # an empty field shows the default folder that the run uses
            res_dir = os.path.join(rc.output_dir, "Results")
            os.makedirs(res_dir, exist_ok=True)
            with tempfile.TemporaryFile(dir=res_dir):       # a read-only folder raises here, before the solve
                pass
            self._save("output_dir", rc.output_dir)
            self._save("hsl_dir", rc.hsl_dir or "")
            self._save("chassissim_dir", rc.chassissim_dir or "")
            self._save("car", self.car.currentData() or "")
            fd, cfg_path = tempfile.mkstemp(prefix="fms_solve_cfg_", suffix=".json")
            os.close(fd)
            self._cfg_path = cfg_path
            rc.write_cfg(cfg_path)
        except Exception as exc:
            self._remove_cfg()
            self._log_error(exc)
            return
        self._active_rc = rc
        self.log.clear()
        self.summary.setText("No results yet.")      # a failed run must not leave the last lap on show
        self.plot_list.clear()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._runner.start(cfg_path)

    def _remove_cfg(self):
        if self._cfg_path:
            try:
                os.remove(self._cfg_path)
            except OSError:
                pass
            self._cfg_path = None

    def closeEvent(self, event):
        self._runner.cancel()
        self._remove_cfg()
        super().closeEvent(event)

    def _on_finished(self, code):
        self._remove_cfg()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if code == 0:
            self._show_results(self._active_rc)
        else:
            self._append_log(f"\n[solve exited with code {code}]\n")

    def _show_results(self, rc):
        pdir = ""
        try:
            mat, pdir = results.result_paths(rc)
            summ = results.parse_summary(mat)
            e = summ["energy_kWh"]
            self.summary.setText(
                ("QSS estimate (not an optimum): " if rc.mode == "qss" else "Lap time: ")
                + f"{summ['lap_time_s']:.3f} s"
                + (f"   |   Energy: {e:.3f} kWh" if e is not None else ""))
        except Exception as exc:
            self._append_log(f"[error] {exc}\n")
            self.summary.setText(f"Results unreadable: {exc}")
        self.plot_list.clear()
        for p in results.list_plots(pdir):
            it = QListWidgetItem(os.path.basename(p))
            it.setData(_PATH_ROLE, p)
            self.plot_list.addItem(it)

    # ---- helpers ----------------------------------------------------------
    def _log_error(self, exc):
        for line in str(exc).splitlines() or [""]:
            self._append_log(f"[error] {line}\n")

    def _append_log(self, text):
        # Fix: brief had self.log.textCursor().End which is an instance with no .End attr.
        # QTextCursor.End is a class-level enum; use it directly.
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)

    def _saved(self, key):
        return str(settings().value(key, "") or "")

    def _save(self, key, value):
        settings().setValue(key, value)

    # ---- ChassisSim circuits and cars --------------------------------------
    def _cs_dir(self):
        """The ChassisSim folder: the Advanced field, else the environment. None when neither is set."""
        return self.chassissim_dir.text().strip() or os.environ.get("CHASSISSIM_DIR") or None

    def car_ref(self):
        """'chassissim:<absolute path>' of the selected car, None for the 4EM car."""
        path = self.car.currentData()
        return chassissim.PREFIX + path if path else None

    def _refresh_chassissim(self, initial=False):
        """Rebuild the circuit and car combos from the ChassisSim folder. The selections stay. A selection that
        left its list goes back to the first item, with one log line. A listing error gives one log line."""
        cs = self._cs_dir()
        names, cars, cs_cars = [], [], []
        try:
            names = chassissim.list_circuits(cs) if cs and os.path.isdir(cs) else []
        except Exception as exc:
            self._append_log(f"[error] The GUI cannot list the circuits of the ChassisSim folder '{cs}': "
                             f"{str(exc).splitlines()[0]}\n")
        try:        # apart from the circuits: a bad Circ_prop.ini does not hide the cars
            cs_cars = chassissim.list_cars(cs)
            cars = list(dict.fromkeys(cs_cars + chassissim.list_cars(None, paths.user_cars_dir())))
        except Exception as exc:
            self._append_log(f"[error] The GUI cannot list the cars of the ChassisSim folder '{cs}': "
                             f"{str(exc).splitlines()[0]}\n")
        if cs and cs != self._cs_logged:      # one line for each new folder, also for a folder that lists nothing
            self._append_log(f"[ChassisSim] {cs}: {len(names)} circuits, {len(cs_cars)} cars.\n")
        self._cs_logged = cs
        want = self.circuit.currentText()
        self.circuit.blockSignals(True)
        self.circuit.clear()
        self.circuit.addItems(CIRCUITS + [chassissim.PREFIX + n for n in names])
        i = self.circuit.findText(want)
        self.circuit.setCurrentIndex(max(i, 0))
        self.circuit.blockSignals(False)
        if i < 0 and want:
            self._append_log(f"[circuit] '{want}' is not in the list now. The circuit is {CIRCUITS[0]}.\n")
        want_car = self.car.currentData() or (self._saved("car") if initial else "")
        if initial and want_car and want_car not in cars and os.path.isfile(want_car):
            self._extra_cars.append(want_car)       # a saved car from a different folder stays a browsed car
        cars = list(dict.fromkeys(cars + sorted(self._extra_cars, key=os.path.basename)))
        base = [os.path.basename(f) for f in cars]
        self.car.blockSignals(True)
        self.car.clear()
        self.car.addItem(CAR_4EM, None)
        for f, b in zip(cars, base):       # a file name that repeats gets its folder name
            self.car.addItem(b + (f" ({os.path.basename(os.path.dirname(f))})" if base.count(b) > 1 else ""), f)
            self.car.setItemData(self.car.count() - 1, f, 3)    # Qt.ToolTipRole
        i = self.car.findData(want_car) if want_car else 0
        self.car.setCurrentIndex(max(i, 0))
        self.car.blockSignals(False)
        if i < 0:
            self._append_log(f"[car] '{want_car}' is not in the list now. The car is {CAR_4EM}.\n")
        if not initial:
            self._on_car_changed()

    def _pick_chassissim(self):
        d = QFileDialog.getExistingDirectory(self, "ChassisSim folder", self.chassissim_dir.text())
        if d:
            self.chassissim_dir.setText(d)
            self._refresh_chassissim()

    def _pick_car(self):
        f, _ = QFileDialog.getOpenFileName(self, "ChassisSim car", self._cs_dir() or "", "ChassisSim car (*.car)")
        if f:
            self._select_car(f)

    def _select_car(self, path):
        path = os.path.abspath(path)
        if self.car.findData(path) < 0:
            self._extra_cars.append(path)
            self._refresh_chassissim()
        self.car.setCurrentIndex(self.car.findData(path))

    def _on_car_changed(self, _i=None):
        """A ChassisSim car sets and locks the controls that its model needs. The 4EM car restores them."""
        start = self.opt_combos["start_state"]
        rolling = start.model().item(2)     # items: auto, matlab, rolling, periodic
        # widget, value with a car, tooltip without a car
        locked = [(self.opt_combos["physics"], "corrected", OPTION_COMBOS[1][3]), (self.aero, "Static", ""),
                  (self.em4, False, ""), (self.atd, False, ""), (self.homotopy, False, HOMOTOPY_TIP),
                  (self.tyre, TYRES[0], ""), (self.tyre_set, TYRE_SETS[0], TYRE_SET_TIP)]

        def value(w):
            return w.isChecked() if isinstance(w, QCheckBox) else w.currentText()

        def put(w, v):
            w.setChecked(v) if isinstance(w, QCheckBox) else w.setCurrentText(v)

        car = bool(self.car.currentData())
        if car and self._pre_car is None:
            self._pre_car = [value(w) for w, _, _ in locked] + [start.currentText() == "rolling"]
            for w, v, _ in locked:
                put(w, v)
                w.setEnabled(False)
                w.setToolTip(CAR_LOCK_TIP)
            if self._pre_car[-1]:
                start.setCurrentText("auto")
            rolling.setEnabled(False)
            rolling.setToolTip(CAR_LOCK_TIP)
        elif not car and self._pre_car is not None:
            pre, self._pre_car = self._pre_car, None
            for w, _, tip in locked:
                w.setEnabled(True)
                w.setToolTip(tip)
            for (w, _, _), v in zip(locked, pre):       # after the loop above: 4 Motors disables ATD again
                if not (w is self.atd and self.em4.isChecked()):
                    put(w, v)
            rolling.setEnabled(True)
            rolling.setToolTip("")
            if pre[-1] and start.currentText() == "auto":       # only the lock moved it. A later choice stays
                start.setCurrentText("rolling")
        for sec in self._sections.values():     # the 4EM values stay in the table for the 4EM car
            sec.setVisible(not car)
        self._update_inert_params()
        self._update_pt()
        self._show_car_setup()

    # ---- the car sections of the Setup tab --------------------------------
    def _show_car_setup(self):
        """Build the car sections for the selected car. A new car drops the edits of the old one."""
        path = self.car.currentData()
        if path == self._car_path:
            return
        if self._car_edit and self._car_edit.edits:
            self._append_log(f"[car] The GUI drops the setup edits of the car {self._car_edit.base}.\n")
        self._car_edit, self._car_path, self._car_fields, self._car_secs = None, path, {}, {}
        lay = self._car_box.layout()
        while lay.count():
            lay.takeAt(0).widget().deleteLater()
        if not path:
            return
        try:
            ce = car_edit.CarEdit(path)
        except Exception as exc:
            self._append_log(f"[error] {os.path.basename(path)}: {exc}\n")
            lay.addWidget(QLabel("The GUI cannot edit this car file. The run uses the file as it is."))
            return
        self._car_edit = ce
        rows = ce.rows()
        for title in car_edit.GROUPS:
            keys = [k for k, f, _ in rows if f.group == title]
            if not keys:
                continue
            sec = CollapsibleSection(title)
            for key, f, val in (r for r in rows if r[1].group == title):
                if f.kind == "i":
                    w = QSpinBox()
                    w.setRange(int(f.lo), int(f.hi))
                    w.setValue(int(val))
                else:
                    w = ExactField()
                    w.setValue(val)
                w.setToolTip(f"{f.label} [{f.unit}], cid {key[0]}" + ("\n" + ce.tip(key) if ce.tip(key) else ""))
                w.valueChanged.connect(lambda _v=None, k=key: self._on_car_param_changed(k))
                self._car_fields[key] = w
                sec.addRow(QLabel(f"{f.label} [{f.unit}]"), w)
            sec.set_expanded(title == car_edit.GROUPS[0])
            sec.reset_requested.connect(lambda t=title: self._reset_car(t))
            self._car_secs[title] = (sec, keys)
            lay.addWidget(sec)

    def _on_car_param_changed(self, key):
        w = self._car_fields[key]
        try:
            self._car_edit.set(key, w.value())
        except ValueError:          # half-typed text
            return
        self._set_field_changed_style(w, key in self._car_edit.edits)
        sec, keys = self._car_secs[car_edit.FIELDS[key].group]
        sec.set_changed_count(sum(k in self._car_edit.edits for k in keys))

    def _reset_car(self, title=None):
        """Put the car fields of one group, or of all groups, back to the file values."""
        for t, (_, keys) in self._car_secs.items():
            if title in (None, t):
                for k in keys:
                    self._car_fields[k].setValue(self._car_edit.file_value(k))

    def _save_car(self):
        """Save the edited car as a new .car file. A file that exists is never overwritten."""
        ce = self._car_edit
        if ce is None:
            return self._append_log("[error] The GUI cannot edit this car file, so it cannot save the file.\n")
        folder = paths.user_cars_dir()
        os.makedirs(folder, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save car", os.path.join(folder, ce.base + "_edit.car"), "ChassisSim car (*.car)",
            options=QFileDialog.DontConfirmOverwrite)
        if not path:
            return
        path = os.path.abspath(path if path.lower().endswith(".car") else path + ".car")
        if os.path.exists(path):
            return self._append_log(f"[error] The file {path} exists. Choose a new name.\n")
        if any(car_edit.in_chassissim(path, cs) for cs in (self._cs_dir(), os.environ.get("CHASSISSIM_DIR"))):
            return self._append_log("[error] Do not save into the ChassisSim folders. Choose another folder.\n")
        try:
            ce.save(path, self._cs_dir())
        except Exception as exc:
            return self._log_error(exc)
        self._append_log(f"[car saved] {path}\n")
        self._refresh_chassissim()

    def _pick_hsl(self):
        d = QFileDialog.getExistingDirectory(self, "Coin-HSL folder", self.hsl_dir.text())
        if d:
            self.hsl_dir.setText(d)

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
