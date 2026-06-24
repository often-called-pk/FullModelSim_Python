"""RunConfig: the GUI's editable state, serialisable to the solve cfg.json.

`to_dict`/`from_dict` persist the full GUI state. `write_cfg` emits the JSON
the headless solve consumes: run-config fields plus a single merged
`vp_overrides` dict (Tier-1 tunables + any expert config file).
"""
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

from app.paths import default_output_dir
from vehParams import PRIMARY_KEYS, MF_KEYS

TIER1_FIELDS = ("brkB", "Tdist", "ksD",
                "alpha_FL", "alpha_FR", "alpha_RW", "alpha_TW")


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
    # Tier-1 tunables (vehParams primaries)
    brkB: float = 0.6766
    Tdist: float = 0.7281
    ksD: float = 0.4620
    alpha_FL: float = 10.0
    alpha_FR: float = 10.0
    alpha_RW: float = 8.0
    alpha_TW: float = 0.0
    expert_config: Optional[str] = None

    def vp_overrides(self):
        ov = {}
        if self.expert_config:
            with open(self.expert_config) as fh:
                ov.update(json.load(fh))
        for k in TIER1_FIELDS:
            ov[k] = getattr(self, k)
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
        for k in TIER1_FIELDS:
            d.pop(k, None)
        d["vp_overrides"] = self.vp_overrides()
        return d

    def write_cfg(self, path):
        cfg = self._cfg()
        with open(path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        return cfg
