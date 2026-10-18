"""MLTP_TyreOptim nominal-load coupling (item C6). Plain script: no solve, no IPOPT.
Run from the repo root:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_tyreoptim_fz0.py

MLTP_TyreOptim.m:128 sets vp.Fz0 = 4905*vp.Fz0_shift before vehModel. optimise_design with
Fz0_follows_shift=True (passed only by MLTP_TyreOptim) does the same with the promoted
symbol, so at P = S the TyreOptim model is the plain model built with vp_overrides
{Fz0: 4905*S, Fz0_shift: S}. MLTP_paramOptim keeps Fz0 fixed, as MLTP_paramOptim.m does.

  1. TyreOptim model at P = S vs the plain model (matlab S 0.5 / 0.8 / 1.5, corrected S 0.5):
     dx and the 20 tyre signals at 3 states, max rel diff <= 1e-12 (floor 1)
  2. S = 0.5, state 1: mu_fl_x and mu_fl_y follow the coupled dfz = (Fz - 2452.5) / 2452.5
  3. S = 1: TyreOptim model == plain default model, max rel diff == 0.0
  4. MLTP_paramOptim promoting Fz0_shift: no coupling (mu_fl_y uses Fz0 = 4905), and the
     coupled and uncoupled mu_fl_y differ by more than 0.1 at S = 0.5

A stand-in replaces build_and_solve_nlp (it records the model and P, then aborts), and
MLTP_initial is stubbed. Needs casadi and Data/DATA_AA.mat (SKIP and exit 0 otherwise).
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import sys
import warnings
from types import SimpleNamespace

DATA = os.path.join(_bootstrap.ROOT, "Data")
if not os.path.exists(os.path.join(DATA, "DATA_AA.mat")):       # pragma: no cover
    print("SKIP test_tyreoptim_fz0: Data/DATA_AA.mat missing")
    sys.exit(0)

import casadi as ca
import numpy as np

from functions.context import Ctx
from userOpts import userOpts
from vehModel import vehModel
import MLTP_paramOptim as po_mod
import MLTP_TyreOptim as to_mod


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *a, **kw):
    """fn(*a, **kw) with stdout swallowed and warnings ignored."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


KW = dict(circuit="Sturn", vi=60.0, AeroConfig="Static", ATD="On", Electric_4Motors="Off",
          mesh="uniform")
W = ("fl", "fr", "rl", "rr")
SIG = ([f"fx_{w}" for w in W] + [f"fy_{w}" for w in W] + [f"fz_{w}" for w in W]
       + [f"mu_{w}_x" for w in W] + [f"mu_{w}_y" for w in W])
INIT = SimpleNamespace(x_opt=np.vstack([np.linspace(30, 50, 10), np.zeros((4, 10)),
                                        np.full((2, 10), 100.0)]),
                       u_opt=np.vstack([np.full(10, 300.0), np.zeros((2, 10))]))


class _Captured(Exception):
    pass


def captured(module, fn, **kw):
    """module.fn(save=False, **kw) up to build_and_solve_nlp: the model m and the P symbol.
    optimise_design looks build_and_solve_nlp up in MLTP_paramOptim, so the stand-in goes there."""
    seen = {}

    def stand_in(ca_, m, *args, param=None, **kwargs):
        seen.update(m=m, P=param["sym"])
        raise _Captured()

    real = (po_mod.build_and_solve_nlp, po_mod.MLTP_initial)
    po_mod.build_and_solve_nlp = stand_in
    po_mod.MLTP_initial = lambda *a, **k: SimpleNamespace(data=SimpleNamespace(init=INIT))
    try:
        quiet(getattr(module, fn), save=False, **kw)
    except _Captured:
        pass
    finally:
        po_mod.build_and_solve_nlp, po_mod.MLTP_initial = real
    return seen


def outputs_fn(m, extra=()):
    """dx and the 20 tyre signals as one casadi Function of (x, u, pv, extra...)."""
    return ca.Function("T", [m.x, m.u, m.pv, *extra], [m.dx] + [getattr(m, n) for n in SIG])


def plain(physics="matlab", ov=None):
    c = Ctx()
    quiet(userOpts, c, physics=physics, vp_overrides=ov, **KW)
    vehModel(c)
    return c


def states(c):
    """Three normalised states: 30 / 45 / 38 m/s, wheel slip -6..+8 %, side slip, yaw rate,
    tyre loads 0.7-1.4 x static, coast inputs (ATD 0.25 each), a small steer, k = 0.005."""
    m, vp = c.m23, c.vp
    xs, us = np.asarray(m.x_s, float), np.asarray(m.u_s, float)
    out = []
    for vx, vy, r, slip, load in ((30.0, 0.5, 0.2, (0.03, -0.06, 0.08, 0.05), (1.4, 0.7, 1.3, 0.8)),
                                  (45.0, -1.0, -0.3, (-0.04, 0.02, 0.06, -0.02), (0.8, 1.3, 0.9, 1.2)),
                                  (38.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))):
        X = np.zeros(m.nx)
        X[0], X[1], X[2] = vx, vy, r
        X[5:9] = vx / vp.Rw * (1 + np.asarray(slip))
        X[19:23] = np.asarray(load) * np.array([vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr])
        U = np.zeros(m.nu)
        U[2:6] = 0.25
        U[-1] = 0.02
        out.append((X / xs, U / us, 0.005))
    return out


def rel(o1, o2):
    """Max relative difference (floor 1) over two lists of Function outputs."""
    return max(float(np.max(np.abs(np.asarray(p, float) - np.asarray(q, float))
                            / np.maximum(np.abs(np.asarray(q, float)), 1.0)))
               for p, q in zip(o1, o2))


print("1. TyreOptim model at P = S == plain model with Fz0 = 4905*S, Fz0_shift = S")
for physics, S_list in (("matlab", (0.5, 0.8, 1.5)), ("corrected", (0.5,))):
    a = captured(to_mod, "MLTP_TyreOptim", physics=physics, **KW)
    F = outputs_fn(a["m"], [a["P"]])
    for S in S_list:
        c = plain(physics, {"Fz0": 4905.0 * S, "Fz0_shift": S})
        G = outputs_fn(c.m23)
        d = max(rel(F(x, u, k, S), G(x, u, k)) for x, u, k in states(c))
        ok(f"{physics}, S={S}: dx and the 20 tyre signals, max rel diff {d:.1e} <= 1e-12",
           d <= 1e-12)

print("2. the coupling at S = 0.5: dfz = (Fz - 2452.5) / 2452.5 in mu_x and mu_y")
c0 = plain()
mf = c0.mf
a = captured(to_mod, "MLTP_TyreOptim", **KW)
x, u, k = states(c0)[0]
H = ca.Function("H", [a["m"].x, a["m"].u, a["m"].pv, a["P"]],
                [a["m"].fz_fl, a["m"].mu_fl_x, a["m"].mu_fl_y])
fz, mux, muy = [float(v) for v in H(x, u, k, 0.5)]
dfz = (fz - 2452.5) / 2452.5
ok(f"S=0.5, Fz_fl={fz:.0f} N: mu_fl_x = pDx1 + pDx2*dfz, mu_fl_y = pDy1 + pDy2*dfz (rel 1e-12)",
   abs(mux - (mf.pDx1 + mf.pDx2 * dfz)) <= 1e-12 * abs(mux)
   and abs(muy - (mf.pDy1 + mf.pDy2 * dfz)) <= 1e-12 * abs(muy))

print("3. S = 1: the TyreOptim model is the plain MLTP model (full+duals warm start stays exact)")
F1 = outputs_fn(a["m"], [a["P"]])
G0 = outputs_fn(c0.m23)
d = max(rel(F1(x_, u_, k_, 1.0), G0(x_, u_, k_)) for x_, u_, k_ in states(c0))
ok(f"max rel diff {d:.1e} == 0", d == 0.0)

print("4. no coupling outside MLTP_TyreOptim (MLTP_paramOptim promoting Fz0_shift)")
b = captured(po_mod, "MLTP_paramOptim", params=[("Fz0_shift", 0.5, 1.5)], **KW)
Hb = ca.Function("Hb", [b["m"].x, b["m"].u, b["m"].pv, b["P"]], [b["m"].fz_fl, b["m"].mu_fl_y])
fzb, muyb = [float(v) for v in Hb(x, u, k, 0.5)]
ok("MLTP_paramOptim promoting Fz0_shift: mu_fl_y = pDy1 + pDy2*(Fz - 4905)/4905 at S=0.5",
   abs(muyb - (mf.pDy1 + mf.pDy2 * (fzb - 4905.0) / 4905.0)) <= 1e-12 * abs(muyb))
ok(f"coupled vs uncoupled mu_fl_y differ at S=0.5 ({muy:.4f} vs {muyb:.4f})",
   abs(muy - muyb) > 0.1)

print("\nALL TyreOptim Fz0 TESTS PASSED")
