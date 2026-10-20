"""RunConfig: the GUI's editable state, serialisable to the solve cfg.json.

`to_dict`/`from_dict` persist the full GUI state. `write_cfg` emits the JSON the
headless solve consumes: run-config fields, the five top-level solver/collocation
options, the four speed/fidelity options (mesh, mesh_opts, tyre_set, screening;
mesh and tyre_set have Advanced-tab combos, mesh_opts and screening have no GUI
widget yet and are settable from a saved RunConfig / cfg.json), and a single
merged `vp_overrides` dict (vp diff-from-default). An expert config file is written
into a default-seeded vp at construction; given an explicit vp it only contributes
keys absent from vp.
"""
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

from app.paths import default_output_dir
from app.vp_params import all_vp_defaults
from vehParams import PRIMARY_KEYS, MF_KEYS


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
        d["vp_overrides"] = self.vp_overrides()
        return d

    def write_cfg(self, path):
        cfg = self._cfg()
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return cfg
