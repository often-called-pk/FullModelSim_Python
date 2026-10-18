"""MLTP_TyreOptim.py - direct port of MLTP_TyreOptim.m

Co-optimises the tyre nominal-load-shift parameter ``Fz0_shift`` with the racing
line. vehParams.m documents this parameter as the "fraction shift in nominal
wheel load - optimisation parameter in TyreOptim script". It acts in two places.
It divides the load fed into the Pacejka formula (``fz_shift = fz / Fz0_shift``
at every wheel). As in MLTP_TyreOptim.m:128 (``vp.Fz0 = 4905*vp.Fz0_shift``
before vehModel), it also scales the nominal load ``Fz0``.

So the shift S enters the cornering stiffness Kya twice. The load ratio is
``fz/S`` over ``pKy2*Fz0``, with ``Fz0 = 4905*S``. This is vehModel.m:770-773
(``vehModel._kya``), ported as is. MLTP_paramOptim and setup_sweep keep Fz0 fixed.

The QSS mirror of a TyreOptim setup needs no code:
``MLTP_screen(vp_overrides={"Fz0_shift": S, "Fz0": 4905*S})``.

It reuses the shared solve core from MLTP_paramOptim.py
(``optimise_design(Fz0_follows_shift=True)``). Only the promoted parameter and
the Fz0 coupling differ.
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
                           results_dir=results_dir, Fz0_follows_shift=True, **useropts_kwargs)


if __name__ == "__main__":
    MLTP_TyreOptim(circuit="Sturn", vi=60.0)
