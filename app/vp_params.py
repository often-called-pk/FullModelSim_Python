"""app/vp_params.py — presentation registry for the Setup-tab vehicle-parameter
editor. Pure Python (no Qt). Parameter *values* come from vehParams; this module
only describes how each parameter is labelled, ranged, and edited.
"""
from dataclasses import dataclass

from vehParams import default_primaries, _default_mf, PRIMARY_KEYS, MF_KEYS


@dataclass
class ParamMeta:
    label: str
    unit: str = ""
    tooltip: str = ""
    lo: float = -1.0e6
    hi: float = 1.0e6
    step: float = 0.1
    decimals: int = 6
    kind: str = "spin"          # "spin" -> QDoubleSpinBox, "sci" -> ScientificField


def all_vp_defaults():
    """Full adjustable parameter set (primaries + Pacejka mf) as a flat dict."""
    return {**default_primaries(), **vars(_default_mf())}


def fmt_sci(x):
    """Format a float for a ScientificField; round-trips exactly via float()."""
    return "%.12g" % float(x)


_PACEJKA_LONG = [
    "pCx1", "pDx1", "pDx2", "pDx3", "pEx1", "pEx2", "pEx3", "pEx4",
    "pKx1", "pKx2", "pKx3", "pHx1", "pHx2", "pVx1", "pVx2",
    "rBx1", "rBx2", "rBx3", "rCx1", "rEx1", "rEx2", "rHx1",
]
_PACEJKA_LAT = [
    "pCy1", "pDy1", "pDy2", "pDy3", "pEy1", "pEy2", "pEy3", "pEy4", "pEy5",
    "pKy1", "pKy2", "pKy3", "pKy4", "pKy5", "pKy6", "pKy7", "pHy1", "pHy2",
    "pVy1", "pVy2", "pVy3", "pVy4", "rBy1", "rBy2", "rBy3", "rBy4",
    "rCy1", "rEy1", "rEy2", "rHy1", "rHy2",
    "rVy1", "rVy2", "rVy3", "rVy4", "rVy5", "rVy6",
]

PARAM_GROUPS = [
    ("Balance & Aero", ["brkB", "Tdist", "ksD",
                        "alpha_FL", "alpha_FR", "alpha_RW", "alpha_TW"]),
    ("Masses", ["mb", "md", "muf", "mur"]),
    ("Dimensions", ["A", "t", "l", "wB", "hcg", "huf", "hur", "hw",
                    "hRCf", "hRCr", "hride"]),
    ("Inertias", ["I_z", "I_y", "I_x"]),
    ("Tyre & Wheel", ["Rw", "Jw", "f", "kt", "Fz0", "Fz0_shift"]),
    ("Suspension stiffness", ["k_fl", "k_fr", "k_rl", "k_rr"]),
    ("Suspension damping", ["zeta_fl", "zeta_fr", "zeta_rl", "zeta_rr"]),
    ("Brakes", ["Tbrake_max"]),
    ("Camber & Toe", ["gamma_fl", "gamma_rl", "toe_front", "toe_rear"]),
    ("Aero & Environment", ["rho", "g", "Cd", "Cl"]),
    ("Numerical", ["eps_x", "eps_y", "eps_K"]),
    ("Pacejka 5.2 — longitudinal", _PACEJKA_LONG),
    ("Pacejka 5.2 — lateral", _PACEJKA_LAT),
]

META = {
    # Balance & Aero
    "brkB": ParamMeta("Brake bias (front)", "", "Front brake-torque fraction", 0.0, 1.0, 0.01, 4),
    "Tdist": ParamMeta("Torque dist (rear)", "", "Rear drive-torque fraction", 0.0, 1.0, 0.01, 4),
    "ksD": ParamMeta("Roll stiffness (rear)", "", "Rear roll-stiffness distribution", 0.0, 1.0, 0.01, 4),
    "alpha_FL": ParamMeta("Front wing L", "deg", "Front wing angle of attack, left", 0.0, 10.0, 0.5, 2),
    "alpha_FR": ParamMeta("Front wing R", "deg", "Front wing angle of attack, right", 0.0, 10.0, 0.5, 2),
    "alpha_RW": ParamMeta("Rear wing", "deg", "Rear wing angle of attack", 0.0, 30.0, 0.5, 2),
    "alpha_TW": ParamMeta("Rear wing tilt", "deg", "Rear wing tilt trim", -12.0, 12.0, 0.5, 2),
    # Masses
    "mb": ParamMeta("Body mass", "kg", "Sprung body mass", 0.0, 5000.0, 10.0, 1),
    "md": ParamMeta("Driver mass", "kg", "Driver mass", 0.0, 200.0, 5.0, 1),
    "muf": ParamMeta("Unsprung mass front", "kg", "Front unsprung mass", 0.0, 300.0, 5.0, 1),
    "mur": ParamMeta("Unsprung mass rear", "kg", "Rear unsprung mass", 0.0, 300.0, 5.0, 1),
    # Dimensions
    "A": ParamMeta("Frontal area", "m^2", "Reference frontal area", 0.5, 5.0, 0.05, 3),
    "t": ParamMeta("Track width", "m", "Axle track width", 0.5, 3.0, 0.05, 3),
    "l": ParamMeta("Wheelbase", "m", "Wheelbase", 1.0, 5.0, 0.05, 3),
    "wB": ParamMeta("Weight balance (rear)", "", "Rear weight fraction", 0.0, 1.0, 0.01, 4),
    "hcg": ParamMeta("CoG height", "m", "Centre-of-gravity height", 0.0, 1.5, 0.01, 4),
    "huf": ParamMeta("Unsprung CoG h front", "m", "Front unsprung CoG height", 0.0, 1.0, 0.001, 7),
    "hur": ParamMeta("Unsprung CoG h rear", "m", "Rear unsprung CoG height", 0.0, 1.0, 0.001, 7),
    "hw": ParamMeta("Wing height", "m", "Aero application height", 0.0, 2.0, 0.01, 4),
    "hRCf": ParamMeta("Roll centre h front", "m", "Front roll-centre height", -0.5, 1.0, 0.01, 4),
    "hRCr": ParamMeta("Roll centre h rear", "m", "Rear roll-centre height", -0.5, 1.0, 0.01, 4),
    "hride": ParamMeta("Ride height", "m", "Static ride height", 0.0, 0.5, 0.001, 4),
    # Inertias
    "I_z": ParamMeta("Yaw inertia", "kg*m^2", "Yaw moment of inertia", 0.0, 10000.0, 50.0, 1),
    "I_y": ParamMeta("Pitch inertia", "kg*m^2", "Pitch moment of inertia", 0.0, 10000.0, 50.0, 1),
    "I_x": ParamMeta("Roll inertia", "kg*m^2", "Roll moment of inertia", 0.0, 10000.0, 50.0, 1),
    # Tyre & Wheel
    "Rw": ParamMeta("Wheel radius", "m", "Loaded wheel radius", 0.1, 0.6, 0.005, 4),
    "Jw": ParamMeta("Wheel inertia", "kg*m^2", "Rotational inertia per wheel", 0.0, 20.0, 0.1, 3),
    "f": ParamMeta("Rolling resistance", "", "Rolling-resistance coefficient", 0.0, 0.1, 0.001, 4),
    "kt": ParamMeta("Tyre vert. stiffness", "N/m", "Tyre vertical stiffness", 0.0, 1.0e6, 1000.0, 1),
    "Fz0": ParamMeta("Nominal tyre load", "N", "Pacejka nominal vertical load", 0.0, 20000.0, 50.0, 1),
    "Fz0_shift": ParamMeta("Nominal load shift", "", "Scale on nominal load Fz0", 0.0, 5.0, 0.05, 4),
    # Suspension stiffness
    "k_fl": ParamMeta("Spring rate FL", "N/m", "Front-left spring rate", 0.0, 300000.0, 1000.0, 1),
    "k_fr": ParamMeta("Spring rate FR", "N/m", "Front-right spring rate", 0.0, 300000.0, 1000.0, 1),
    "k_rl": ParamMeta("Spring rate RL", "N/m", "Rear-left spring rate", 0.0, 300000.0, 1000.0, 1),
    "k_rr": ParamMeta("Spring rate RR", "N/m", "Rear-right spring rate", 0.0, 300000.0, 1000.0, 1),
    # Suspension damping
    "zeta_fl": ParamMeta("Damping ratio FL", "", "Front-left damping ratio", 0.0, 2.0, 0.05, 3),
    "zeta_fr": ParamMeta("Damping ratio FR", "", "Front-right damping ratio", 0.0, 2.0, 0.05, 3),
    "zeta_rl": ParamMeta("Damping ratio RL", "", "Rear-left damping ratio", 0.0, 2.0, 0.05, 3),
    "zeta_rr": ParamMeta("Damping ratio RR", "", "Rear-right damping ratio", 0.0, 2.0, 0.05, 3),
    # Brakes
    "Tbrake_max": ParamMeta("Max brake torque", "N*m", "Maximum total brake torque", 0.0, 20000.0, 100.0, 1),
    # Camber & Toe
    "gamma_fl": ParamMeta("Camber front (L)", "deg", "Front camber, left (mirrored to right)", -10.0, 10.0, 0.1, 3),
    "gamma_rl": ParamMeta("Camber rear (L)", "deg", "Rear camber, left (mirrored to right)", -10.0, 10.0, 0.1, 3),
    "toe_front": ParamMeta("Toe front", "deg", "Front toe angle", -10.0, 10.0, 0.1, 3),
    "toe_rear": ParamMeta("Toe rear", "deg", "Rear toe angle", -10.0, 10.0, 0.1, 3),
    # Aero & Environment
    "rho": ParamMeta("Air density", "kg/m^3", "Ambient air density", 0.5, 2.0, 0.001, 4),
    "g": ParamMeta("Gravity", "m/s^2", "Gravitational acceleration", 0.0, 20.0, 0.01, 4),
    "Cd": ParamMeta("Drag coefficient", "", "OVERWRITTEN by Data/DATA_AA.mat when present", 0.0, 5.0, 0.01, 4),
    "Cl": ParamMeta("Lift coefficient", "", "OVERWRITTEN by Data/DATA_AA.mat when present", 0.0, 5.0, 0.01, 4),
    # Numerical (scientific notation; tiny values)
    "eps_x": ParamMeta("eps_x", "", "Longitudinal-slip smoothing epsilon", 1.0e-12, 1.0, 0.0, 12, kind="sci"),
    "eps_y": ParamMeta("eps_y", "", "Lateral-slip smoothing epsilon", 1.0e-12, 1.0, 0.0, 12, kind="sci"),
    "eps_K": ParamMeta("eps_K", "", "Curvature smoothing epsilon", 1.0e-12, 1.0, 0.0, 12, kind="sci"),
}


def meta_for(key):
    """Annotated metadata, or a generic scientific-notation fallback (used by
    every Pacejka mf coefficient — wide range, no rounding)."""
    if key in META:
        return META[key]
    return ParamMeta(label=key, unit="", tooltip=key,
                     lo=-1.0e6, hi=1.0e6, step=0.0, decimals=12, kind="sci")
