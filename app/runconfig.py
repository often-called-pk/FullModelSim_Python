"""RunConfig: the GUI's editable state, serialisable to the solve cfg.json.

`to_dict`/`from_dict` persist the full GUI state. `write_cfg` emits the JSON the
headless solve consumes: run-config fields, the five top-level solver/collocation
options, and a single merged `vp_overrides` dict (vp diff-from-default + any
expert config file).
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
    # full vehicle-parameter set (primaries + Pacejka mf), seeded from defaults
    vp: dict = field(default_factory=all_vp_defaults)

    def vp_overrides(self):
        ov = {}
        if self.expert_config:
            with open(self.expert_config) as fh:
                ov.update(json.load(fh))            # expert is the base layer
        defaults = all_vp_defaults()
        for k, v in self.vp.items():                 # GUI diff overrides expert
            if v != defaults[k]:
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
