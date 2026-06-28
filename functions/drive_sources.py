"""Drive-source abstraction (MPI-free, CasADi-free).

Single source of truth for the powertrain topology, the control-vector
ordering, and the path-constraint name ordering. Consumed by userOpts.py,
vehModel.py and MLTP.py so the control order is defined exactly once.

Topology strings: 'single' | 'single_atd' | 'four_motor' | 'hybrid'.
"""
from dataclasses import dataclass, field


@dataclass
class DriveSource:
    name: str            # 'motor'|'fl'|'fr'|'r'|'ice' (forms channel/constraint names)
    type: str            # 'emotor' | 'ice'
    node: str            # wheels it feeds: 'fl'|'fr'|'rl'|'rr'|'front'|'rear'|'all'
    gear: float          # per-source ratio (motor spins gear x wheel)
    Tmax: float          # torque scale / cap (Nm)
    Pmax: float          # power cap (W)      — emotor flat cap
    OMmax: float         # speed cap (rad/s)  — emotor flat cap / ice redline
    ctrl_key: str        # control-channel name it owns
    curve: tuple = None  # ICE rpm->torque poly coeffs (high->low order); None for emotor
    regen: bool = False  # False -> torque in [0,Tmax]; True -> [-Tmax,Tmax]
    eff: float = 0.90    # post-solve energy efficiency


@dataclass
class SplitPolicy:
    node: str            # multi-wheel node it distributes: 'front'|'rear'|'all'
    kind: str            # 'fixed' | 'atd' | 'tv'
    keys: list = field(default_factory=list)   # fraction-control channel names


_NODE_WHEELS = {
    "all":   ["fl", "fr", "rl", "rr"],
    "front": ["fl", "fr"],
    "rear":  ["rl", "rr"],
    "fl": ["fl"], "fr": ["fr"], "rl": ["rl"], "rr": ["rr"],
}


def node_wheels(node):
    """Wheels driven by a node, in canonical order."""
    return list(_NODE_WHEELS[node])


def aero_keys(ActAero):
    """Active-aero control channels for a given ActAero level (naming quirk preserved)."""
    if ActAero == 1:
        return ["RW"]
    if ActAero == 2:
        return ["FW", "RW"]
    if ActAero == 3:
        return ["FW", "FW", "RW", "TW"]
    return []


def control_keys(sources, splits, ActAero):
    """THE control-vector ordering: sources -> brake -> split fractions -> aero -> delta."""
    keys = [s.ctrl_key for s in sources]
    keys.append("T_brake")
    for sp in splits:
        keys.extend(sp.keys)
    keys.extend(aero_keys(ActAero))
    keys.append("delta")
    return keys


def _topology(EM4, ATD, Hybrid):
    """Map the resolved flags to a topology string. Hybrid wins."""
    if Hybrid == 1:
        return "hybrid"
    if EM4 == 1:
        return "four_motor"
    if ATD == 1:
        return "single_atd"
    return "single"


def build_topology(topology, pt, vp, ice_gear=1.0):
    """Return (sources, splits) for a topology.

    Legacy topologies reuse the shared scalars pt.Pmax/Tmax/OMmax as-is (so
    'single' caps at pt.Pmax as an aggregate and each 'four_motor' source caps
    at pt.Pmax per motor — the existing dual semantics, preserved). Only
    'hybrid' uses explicit per-source ratings/gears (Zenvo Aurora constants).
    """
    if topology in ("single", "single_atd"):
        src = [DriveSource("motor", "emotor", "all", vp.gear,
                           pt.Tmax, pt.Pmax, pt.OMmax, "T_motor", eff=pt.eff)]
        if topology == "single_atd":
            splits = [SplitPolicy("all", "atd", ["ATD", "ATD", "ATD", "ATD"])]
        else:
            splits = [SplitPolicy("all", "fixed", [])]
        return src, splits

    if topology == "four_motor":
        src = [DriveSource(w, "emotor", w, vp.gear, pt.Tmax, pt.Pmax, pt.OMmax,
                           f"T_motor_{w}", eff=pt.eff)
               for w in ("fl", "fr", "rl", "rr")]
        return src, []

    if topology == "hybrid":
        # ----- Zenvo Aurora source list (in-code constants, spec sec.10) -----
        OM_EM = 25000 * 3.141592653589793 / 30.0       # 25000 rpm -> rad/s (2618)
        EM_P, EM_T, EM_EFF = 150e3, 143.0, 0.90        # per front/rear e-motor
        rear_em_gear = 1.5 * ice_gear * 3.5
        ice_drive_gear = ice_gear * 3.5
        # PLACEHOLDER ICE map: flat peak-torque cap until the engine-map image
        # is digitized. Replace `ice_curve` with real coeffs (high->low order)
        # and set ice_peak / ICE_OM_REDLINE / ICE_EFF from the supplied sheet.
        ice_peak = 700.0                               # Nm (hypercar placeholder)
        ice_curve = (ice_peak,)                        # degree-0 poly => flat cap
        ICE_OM_REDLINE = 8000 * 3.141592653589793 / 30.0   # 8000 rpm crank (placeholder)
        ICE_EFF = 0.35                                  # thermal eff (placeholder)
        src = [
            DriveSource("fl", "emotor", "fl", 6.0, EM_T, EM_P, OM_EM, "T_motor_fl", eff=EM_EFF),
            DriveSource("fr", "emotor", "fr", 6.0, EM_T, EM_P, OM_EM, "T_motor_fr", eff=EM_EFF),
            DriveSource("r",  "emotor", "rear", rear_em_gear, EM_T, EM_P, OM_EM, "T_motor_r", eff=EM_EFF),
            DriveSource("ice", "ice", "rear", ice_drive_gear, ice_peak, ice_peak,
                        ICE_OM_REDLINE, "T_ice_r", curve=ice_curve, eff=ICE_EFF),
        ]
        splits = [SplitPolicy("rear", "tv", ["split_R"])]
        return src, splits

    raise ValueError(f"Unknown topology '{topology}'.")


def path_constraint_names(sources, splits):
    """Path-constraint row names: 4 rho_lim first, then powertrain rows, then ATD_eq.

    Grouped by constraint type so the legacy single/four_motor names + order are
    reproduced byte-for-byte. The single aggregate emotor (node 'all') uses the
    unsuffixed legacy names motor_power/motor_rpm/BrTh_1.
    """
    names = ["rho_lim_fl", "rho_lim_fr", "rho_lim_rl", "rho_lim_rr"]
    emotors = [s for s in sources if s.type == "emotor"]
    ices = [s for s in sources if s.type == "ice"]
    aggregate = len(emotors) == 1 and emotors[0].node == "all" and not ices
    if aggregate:
        names += ["motor_power", "motor_rpm", "BrTh_1"]
    else:
        names += [f"motor_power_{s.name}" for s in emotors]
        names += [f"motor_rpm_{s.name}" for s in emotors]
        names += [f"ice_curve_{s.name}" for s in ices]
        names += [f"ice_rpm_{s.name}" for s in ices]
        names += [f"BrTh_{s.name}" for s in (emotors + ices)]
    for sp in splits:
        if sp.kind == "atd":
            names.append("ATD_eq")
    return names


def source_signal_keys(source):
    """(power_key, speed_key) for the saved vehicle dict / plot prefixes."""
    if source.type == "ice":
        return "P_ice_r", "Om_ice_r"
    if source.node == "all":
        return "P_motor", "Om_motor"          # legacy single aggregate
    return f"P_motor_{source.name}", f"Om_motor_{source.name}"
