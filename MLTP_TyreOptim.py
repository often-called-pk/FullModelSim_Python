"""Port of MLTP_TyreOptim.m - co-optimise tyre nominal-load-shift Fz0_shift with the racing line.

Fz0_shift scales the load fed into the Pacejka formula (fz_shift = fz / Fz0_shift
at every wheel), so optimising it tunes the tyre model's operating point.

Reuses the shared solve core from MLTP_paramOptim.py; only the promoted parameter
set differs.
"""

import numpy as np

from MLTP_paramOptim import optimise_design


def MLTP_TyreOptim(circuit="Sturn", vi=60.0, ni=np.nan,
                   Fz0_shift_bounds=(0.5, 1.5),
                   AeroConfig="Static", ATD="On", Electric_4Motors="Off",
                   warm_start=None, save=True, results_dir="Results", **useropts_kwargs):
    """Co-optimise the tyre nominal-load-shift ``Fz0_shift`` with the lap.
    ``Fz0_shift_bounds`` sets the (lower, upper) bound on the shift fraction."""
    lb, ub = Fz0_shift_bounds
    params = [("Fz0_shift", float(lb), float(ub))]
    return optimise_design(params, "TyreOptim", circuit=circuit, vi=vi, ni=ni,
                           warm_start=warm_start, AeroConfig=AeroConfig, ATD=ATD,
                           Electric_4Motors=Electric_4Motors, save=save,
                           results_dir=results_dir, **useropts_kwargs)


if __name__ == "__main__":
    MLTP_TyreOptim(circuit="Sturn", vi=60.0)
