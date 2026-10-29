"""RunConfig: the GUI's editable state, serialisable to the solve cfg.json.

`to_dict`/`from_dict` persist the full GUI state. `write_cfg` emits the JSON the
headless solve consumes: run-config fields, the five top-level solver/collocation
options, the four speed/fidelity options (mesh, mesh_opts, tyre_set, screening;
mesh and tyre_set have Advanced-tab combos, mesh_opts and screening have no GUI
widget yet and are settable from a saved RunConfig / cfg.json), and a single
merged `vp_overrides` dict (vp diff-from-default). An expert config file is written
into a default-seeded vp at construction; given an explicit vp it only contributes
keys absent from vp.

The F2 options are top-level cfg fields too: mode, physics, start_state, n_bounds, Steering,
CamberGain, between_knots, collocation, state_scaling, atd_tbr, ladder, homotopy, refine,
continue_from and ipopt_overrides. Each default is the MLTP or userOpts default. So a cfg without
them makes the same call as before. `pt_overrides` (the A5 powertrain keys) is not a cfg key.
`write_cfg` adds it to `vp_overrides`, where vehParams reads it.
"""
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Union

from app.paths import default_output_dir
from app.vp_params import all_vp_defaults
from userOpts import setup_knobs, knob_tag
from vehParams import PRIMARY_KEYS, MF_KEYS, PT_KEYS


@dataclass
class RunConfig:
    circuit: str = "Sturn"
    AeroConfig: str = "Static"
    ATD: str = "On"
    Electric_4Motors: str = "Off"
    TyreModel: str = "CombinedSlip"
    vi: float = 60.0
    ni: Optional[float] = None
    linear_solver: str = "ma57"
    warm_start: Optional[str] = None
    save: bool = True
    plot: bool = True
    output_dir: str = field(default_factory=default_output_dir)
    expert_config: Optional[str] = None
    # solver / collocation options (top-level cfg fields, NOT vp_overrides)
    max_iter: int = 6000
    OPT_ds: float = 30.0
    OPT_d: int = 3
    OPT_e: float = 1e-2
    tol: float = 1e-4
    # speed / fidelity options (top-level cfg fields, forwarded to userOpts)
    mesh: str = "auto"                      # 'auto' | 'uniform' | 'curvature' knots; auto = curvature if track >= 2000 m
    mesh_opts: Optional[dict] = None        # kwargs for functions.mesh.curvature_mesh
    tyre_set: str = "MF205"                 # 'MF205' | 'CopyB' (vehParams tyre_set)
    screening: bool = False                 # loose IPOPT tolerances (userOpts SCREENING_IPOPT)
    # F2 options (top-level cfg fields). headless_solve forwards a value only when it differs from the default.
    mode: str = "nlp"                       # 'nlp' (MLTP) | 'qss' (MLTP_screen: a lap ESTIMATE, no NLP)
    physics: str = "matlab"                 # 'matlab' | 'corrected' (the A1-A6 changes)
    start_state: str = "matlab"             # 'matlab' | 'rolling' | 'periodic' (closed circuits)
    n_bounds: Optional[Union[float, list]] = None   # None (model +-4 m) | half-width W [m] | [lo, hi]
    Steering: str = "NA"                    # 'NA' | 'Ackermann'
    CamberGain: str = "Off"                 # 'Off' | 'Linear' | 'Table'
    between_knots: str = "matlab"           # 'matlab' | 'linear' | 'linear+power' (B3)
    collocation: str = "legendre"           # 'legendre' | 'radau' (B4)
    state_scaling: str = "matlab"           # 'matlab' | 'typical' (B7)
    atd_tbr: Optional[float] = None         # None | torque-bias ratio >= 1 (C4)
    ladder: str = "auto"                    # 'auto' (= legacy) | 'legacy' | 'qss7' | 'qss23'
    homotopy: Optional[Union[bool, list]] = None    # None | True | friction scales that end in 1.0
    refine: Optional[Union[bool, dict]] = None      # None | True | refine options (functions.refine)
    continue_from: Optional[str] = None     # None | 'auto' (B5: start from the stored default result)
    ipopt_overrides: Optional[dict] = None  # any IPOPT options, merged last
    pt_overrides: Optional[dict] = None     # A5 keys gear Tmax Pmax Preg Ppack (physics='corrected' only)
    # full vehicle-parameter set (primaries + Pacejka mf), seeded from the tyre_set defaults
    vp: Optional[dict] = None

    def __post_init__(self):
        if self.vp is None:
            self.vp = all_vp_defaults(self.tyre_set)
            if self.expert_config:
                with open(self.expert_config) as fh:
                    expert = json.load(fh)
                unknown = set(expert) - PRIMARY_KEYS - MF_KEYS
                if unknown:
                    raise ValueError(f"Unknown vehParams override keys: {sorted(unknown)}")
                self.vp.update(expert)

    def vp_overrides(self):
        ov = {}
        if self.expert_config:
            with open(self.expert_config) as fh:
                expert = json.load(fh)
            ov.update({k: v for k, v in expert.items() if k not in self.vp})   # expert fills only keys absent from vp
        defaults = all_vp_defaults(self.tyre_set)
        _MISSING = object()
        for k, v in self.vp.items():                 # a key held by vp is decided by vp alone
            if v != defaults.get(k, _MISSING):
                ov[k] = v
        unknown = set(ov) - PRIMARY_KEYS - MF_KEYS
        if unknown:
            raise ValueError(f"Unknown vehParams override keys: {sorted(unknown)}")
        return ov

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def _cfg(self):
        d = asdict(self)
        d.pop("expert_config", None)
        d.pop("vp", None)
        pt = d.pop("pt_overrides", None) or {}
        unknown = set(pt) - PT_KEYS
        if unknown:
            raise ValueError(f"Unknown powertrain override keys: {sorted(unknown)}. Allowed: {sorted(PT_KEYS)}")
        d["vp_overrides"] = {**self.vp_overrides(), **pt}
        return d

    def setup_tag(self):
        """The tag that MLTP puts at the end of the result stem for this run. It comes from the same function
        (userOpts.setup_knobs) of the same kwargs (headless_solve.build_solve_kwargs). "" for the default setup.
        In mode 'qss', build_solve_kwargs drops TyreModel (headless_solve.NLP_ONLY), so TyreModel is no knob there."""
        from headless_solve import build_solve_kwargs
        return knob_tag(setup_knobs(build_solve_kwargs(self._cfg(), "")))

    def write_cfg(self, path):
        cfg = self._cfg()
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return cfg
