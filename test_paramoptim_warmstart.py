"""MLTP_paramOptim.optimise_design warm-started from a full 23-state result (plain script,
no pytest). Run from the repo root:

    venv\\Scripts\\python.exe test_paramoptim_warmstart.py

optimise_design appends nP static design parameters P to the decision vector and adds no
constraint, so a plain MLTP result of the same NLP minus that P block seeds it exactly:
w0 = [w_opt; P0] (P0 = the current vp values), lam_x0 = [lam_x; 0], lam_g0 = lam_g.

  1. functions.warmstart.extend_full_start on a fake record: the extension, refusals
  2. functions.warmstart.plan_design_warm_start: full+duals / full-primal / full-interp /
     cold (structure compared without the P block; a co-optimisation source interpolates)
  3. what optimise_design hands to build_and_solve_nlp for every warm_start form (a
     stand-in captures the call, nothing is solved; MLTP_initial is stubbed): result file,
     MLTP ctx, data namespace / dict, duals off, another N, another tyre set (falls back to
     the 7-state init), the unchanged 7-state paths, a bad source, MLTP_TyreOptim
  4. one real solve capped at max_iter=5 from a plausible source: the transcription takes
     the extended vectors and the mode lands in ctx.elapsed, data["nlp"] and the summary line

Sections 1-2 need numpy / scipy only; 3-4 need casadi and Data/DATA_AA.mat (SKIP otherwise).
"""
import contextlib
import inspect
import io
import os
import sys
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)                                   # optimise_design reads Circuits/ and Data/ relatively

import functions.warmstart as W
from functions.importfile import load_solution


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def raises(exc, fn, *args, **kw):
    """The message of the ``exc`` that fn(...) raises ('raised' if empty), else None."""
    try:
        fn(*args, **kw)
    except exc as e:
        return str(e) or "raised"
    return None


def identical(g1, g2):
    return set(g1) == set(g2) and all(np.array_equal(g1[k], g2[k]) for k in g1)


TMP = tempfile.mkdtemp(prefix="po_ws_test_")
KEYS = ["T_motor", "T_brake", "ATD", "ATD", "ATD", "ATD", "delta"]   # Static / ATD On / EM4 Off
N, d, nh, nP = 18, 3, 4, 6
st = W.nlp_structure(23, 7, 0, N, d, nh)                  # a plain MLTP solve (n_param = 0)
st_P = W.nlp_structure(23, 7, 0, N, d, nh, n_param=nP)    # the design problem (6 parameters)
ok("design NLP = plain NLP + nP variables, same constraints (Sturn: n_w 1812 + 6, n_g 1904)",
   st["n_w"] == 1812 and st_P["n_w"] == 1818 and st["n_g"] == st_P["n_g"] == 1904
   and st_P["n_param"] == nP)

rs = np.random.RandomState(1)
w, lam_g, lam_x = rs.rand(st["n_w"]), rs.randn(st["n_g"]), rs.randn(st["n_w"])
rec = W.nlp_record({"w_opt": w, "lam_g": lam_g, "lam_x": lam_x, "structure": st,
                    "sym_type": "SX", "linear_solver": "ma57"},
                   {"iter_count": 179, "return_status": "Solve_Succeeded"},
                   np.linspace(1.0, 3.0, 23), np.linspace(1.0, 2.0, 7), "cold")
P0 = np.array([0.6766, 0.7281, 10.0, 10.0, 8.0, 0.0])         # Sturn default vp values

# =============================================================================
print("1. extend_full_start: [w_opt; P0], [lam_x; 0], lam_g")
w0, lx0, lg0 = W.extend_full_start(rec, nP, P0)
nw = st["n_w"]
ok("w0 = [w_opt; P0] with n_w + nP entries",
   w0.shape == (nw + nP,) and np.array_equal(w0[:nw], w) and np.array_equal(w0[nw:], P0))
ok("lam_x0 = [lam_x; zeros(nP)]: the bound multipliers of P start at 0",
   lx0.shape == (nw + nP,) and np.array_equal(lx0[:nw], lam_x) and not np.any(lx0[nw:]))
ok("lam_g0 = lam_g: P adds no constraint, n_g unchanged",
   np.array_equal(lg0, lam_g) and lg0.size == st_P["n_g"])
ok("the record itself is not modified", rec["w_opt"].size == nw and rec["lam_x"].size == nw)
ok("P0 as a list gives the same vectors", np.array_equal(W.extend_full_start(rec, nP, list(P0))[0], w0))
w0n, lxn, lgn = W.extend_full_start(dict(rec, lam_x=None, lam_g=None), nP, P0)
ok("record without multipliers -> lam_x0 / lam_g0 None, w0 as before",
   lxn is None and lgn is None and np.array_equal(w0n, w0))
w00, lx00, _ = W.extend_full_start(rec, 0, [])
ok("nP = 0 -> the saved vectors unchanged", np.array_equal(w00, w) and np.array_equal(lx00, lam_x))
ok("len(P0) != nP raises ValueError", raises(ValueError, W.extend_full_start, rec, nP, P0[:5]))
rec_po = dict(rec, w_opt=np.concatenate([w, P0]), lam_x=np.concatenate([lam_x, np.zeros(nP)]),
              structure=st_P)                             # a co-optimisation result's record
ok("a record that already ends with design parameters (n_param > 0) is refused",
   "n_param" in (raises(ValueError, W.extend_full_start, rec_po, nP, P0) or ""))
ok("w_opt length != structure.n_w raises ValueError",
   raises(ValueError, W.extend_full_start, dict(rec, w_opt=w[:-1]), nP, P0))
p_rec = os.path.join(TMP, "rec.mat")
sio.savemat(p_rec, {"data": {"x_opt": np.zeros((23, N + 1)), "nlp": rec}}, do_compression=True)
w0f, lxf, lgf = W.extend_full_start(load_solution(p_rec).nlp, nP, P0)
ok("a record reloaded from .mat (namespaces, squeezed ints) extends identically",
   np.array_equal(w0f, w0) and np.array_equal(lxf, lx0) and np.array_equal(lgf, lg0))

# =============================================================================
print("2. plan_design_warm_start: the structure check without the P block")
s_full = np.linspace(0.0, 540.0, N * (d + 1) + 1)
xs, us = np.linspace(1.0, 3.0, 23), np.linspace(1.0, 2.0, 7)
exp = dict(st_P, input_keys=KEYS, s_full=s_full, x_s=xs, u_s=us, tyre_set="MF205")
src = SimpleNamespace(x_opt=np.zeros((23, N + 1)), s_full=s_full, OPT_d=d, input_keys=KEYS,
                      tyre_set="MF205", nlp=rec)
warm, mode, note = W.plan_design_warm_start(src, exp, P0)
ok("plain source = the design NLP minus the P block -> 'full+duals'",
   mode == "full+duals" and set(warm) == {"x0", "lam_g0", "lam_x0", "ipopt"})
ok("... warm = extend_full_start(...) + the IPOPT warm-start recipe",
   np.array_equal(warm["x0"], w0) and np.array_equal(warm["lam_x0"], lx0)
   and np.array_equal(warm["lam_g0"], lg0) and warm["ipopt"] == W.warm_start_ipopt_opts())
ok("... the note says the design parameters were appended", "design parameter" in note)
ok("plan_full_warm_start alone would refuse it (n_w / n_param differ with the P block)",
   W.plan_full_warm_start(src, exp)[1] == "full-interp")
warm, mode, _ = W.plan_design_warm_start(src, exp, P0, ipopt_overrides={"mu_init": 1e-4, "tol": 1e-6})
ok("per-call ipopt_overrides win over the recipe",
   mode == "full+duals" and warm["ipopt"]["mu_init"] == 1e-4 and warm["ipopt"]["tol"] == 1e-6
   and warm["ipopt"]["warm_start_init_point"] == "yes")
warm, mode, _ = W.plan_design_warm_start(src, exp, P0, use_duals=False)
ok("use_duals=False -> 'full-primal': x0 = [w_opt; P0] only, no recipe",
   mode == "full-primal" and set(warm) == {"x0"} and np.array_equal(warm["x0"], w0))
src_sw = SimpleNamespace(**dict(vars(src), nlp=dict(rec, x_s=xs * 1.05), vp_overrides={"mb": 300.0}))
warm, mode, note = W.plan_design_warm_start(src_sw, exp, P0)
ok("a source solved with another setup / scaling (the sweep case) still re-injects",
   mode == "full+duals" and np.array_equal(warm["lam_g0"], lam_g) and "x_s" in note)
for label, e in (("another N", dict(exp, **W.nlp_structure(23, 7, 0, N + 1, d, nh, n_param=nP))),
                 ("another OPT_d", dict(exp, **W.nlp_structure(23, 7, 0, N, 2, nh, n_param=nP))),
                 ("another path-row count (n_g)", dict(exp, **W.nlp_structure(23, 7, 0, N, d, nh + 4, n_param=nP))),
                 ("other input_keys (config)", dict(exp, input_keys=["T_motor", "T_brake", "RW", "delta"])),
                 ("another collocation grid (mesh)", dict(exp, s_full=s_full + 0.5))):
    for duals in (True, False):
        warm, mode, note = W.plan_design_warm_start(src, e, P0, use_duals=duals)
        ok(f"{label}, use_duals={duals} -> 'full-interp', no warm vectors",
           mode == "full-interp" and warm is None and "differs" in note)
warm, mode, note = W.plan_design_warm_start(SimpleNamespace(**dict(vars(src), nlp=rec_po)), exp, P0)
ok("a co-optimisation source (its own 6 parameters) -> 'full-interp', said in the note",
   mode == "full-interp" and warm is None and "co-optimisation" in note)
src_nonlp = SimpleNamespace(**{k: v for k, v in vars(src).items() if k != "nlp"})
ok("a full result without data.nlp -> 'full-interp'",
   W.plan_design_warm_start(src_nonlp, exp, P0)[:2] == (None, "full-interp"))
src_cb = SimpleNamespace(**dict(vars(src), tyre_set="CopyB"))
src_pre = SimpleNamespace(**{k: v for k, v in vars(src).items() if k != "tyre_set"})
ok("another tyre set -> 'cold' (with duals, primal-only, and for a pre-flip source)",
   W.plan_design_warm_start(src_cb, exp, P0)[:2] == (None, "cold")
   and W.plan_design_warm_start(src_cb, exp, P0, use_duals=False)[:2] == (None, "cold")
   and W.plan_design_warm_start(src_pre, exp, P0)[:2] == (None, "cold"))
ok("expected n_param != len(P0) raises ValueError",
   raises(ValueError, W.plan_design_warm_start, src, exp, P0[:3]))
warm, mode, note = W.plan_design_warm_start(SimpleNamespace(**dict(vars(src), nlp=dict(rec, w_opt=w[:-3]))),
                                            exp, P0)
ok("a damaged record (w_opt shorter than its n_w) degrades to 'full-interp' instead of raising",
   mode == "full-interp" and warm is None and "not usable" in note)
p_src = os.path.join(TMP, "src.mat")
sio.savemat(p_src, {"data": {"x_opt": np.zeros((23, N + 1)), "s_full": s_full, "OPT_d": d,
                             "input_keys": KEYS, "tyre_set": "MF205", "nlp": rec}}, do_compression=True)
warm, mode, _ = W.plan_design_warm_start(load_solution(p_src), exp, P0)
ok("a source reloaded from .mat (padded keys, squeezed ints) -> 'full+duals', same vectors",
   mode == "full+duals" and np.array_equal(warm["x0"], w0) and np.array_equal(warm["lam_g0"], lam_g)
   and np.array_equal(warm["lam_x0"], lx0))

# =============================================================================
try:
    import casadi as ca                                                      # noqa: F401
except ImportError as exc:                                                   # pragma: no cover
    print(f"  [SKIP] sections 3-4: casadi not importable ({exc})")
    print("\nALL paramOptim warm-start TESTS PASSED")
    sys.exit(0)
if not os.path.exists(os.path.join(HERE, "Data", "DATA_AA.mat")):            # pragma: no cover
    print("  [SKIP] sections 3-4: Data/DATA_AA.mat missing")
    print("\nALL paramOptim warm-start TESTS PASSED")
    sys.exit(0)

from functions.context import Ctx
from functions.transcription import build_and_solve_nlp, discretise, unpack_solution, reconstruct_x_full
from userOpts import userOpts
from vehModel import vehModel
import MLTP_paramOptim as po_mod
import MLTP_TyreOptim as to_mod
from MLTP import build_path_constraints, warmstart_guesses, warmstart_guesses_full

print("3. what optimise_design hands to build_and_solve_nlp (captured, nothing solved)")
DEFAULT = ["brkB", "Tdist", "alpha_FL", "alpha_FR", "alpha_RW", "alpha_TW"]
KW = dict(circuit="Sturn", vi=60.0, AeroConfig="Static", ATD="On", Electric_4Motors="Off",
          mesh="uniform")
ctx = Ctx()
userOpts(ctx, **KW)
P0s = np.array([float(getattr(ctx.vp, f)) for f in DEFAULT])    # where optimise_design starts P
vehModel(ctx)
m = ctx.m23
disc = discretise(ctx.track, ctx.OPT_ds, ctx.OPT_d, mesh=ctx.mesh, mesh_opts=ctx.mesh_opts)
Ns, ds = disc["N"], ctx.OPT_d
nhs = len(build_path_constraints(ca, m, ctx.pt)[0])
st3 = W.nlp_structure(m.nx, m.nu, m.ny, Ns, ds, nhs)
nw3 = st3["n_w"]
rs = np.random.RandomState(0)
w3 = rs.uniform(0.1, 0.9, nw3)
lg3, lx3 = rs.randn(st3["n_g"]), rs.randn(nw3)
x3, u3, _, xc3 = unpack_solution(w3, m.nx, m.nu, m.ny, Ns, ds, m.x_s, m.u_s, None)
rec3 = W.nlp_record(dict(w_opt=w3, lam_g=lg3, lam_x=lx3, structure=st3, sym_type="SX",
                         linear_solver="ma57"),
                    {"iter_count": 179, "return_status": "Solve_Succeeded"}, m.x_s, m.u_s, "cold")
src3 = SimpleNamespace(x_opt=x3, u_opt=u3, x_full=reconstruct_x_full(x3, xc3, m.nx, Ns, ds),
                       s_full=disc["s_full"], N=Ns, OPT_d=ds, input_keys=list(ctx.input_keys),
                       tyre_set=ctx.tyre_set, nlp=rec3)
ok(f"fake plain MLTP result of the Sturn default NLP: N = {Ns}, n_w = {nw3}, n_g = {st3['n_g']}, "
   f"P0 = the vp values {', '.join(f'{k}={v:g}' for k, v in zip(DEFAULT, P0s))}",
   x3.shape == (23, Ns + 1) and st3["n_param"] == 0 and P0s.size == 6)

N0 = 9                                              # a synthetic 7-state init held in memory
X7 = np.vstack([np.linspace(30, 50, N0 + 1), np.zeros(N0 + 1), np.zeros(N0 + 1), np.zeros(N0 + 1),
                np.zeros(N0 + 1), np.full(N0 + 1, 100.0), np.full(N0 + 1, 100.0)])
U7 = np.vstack([np.full(N0 + 1, 300.0), np.zeros(N0 + 1), np.zeros(N0 + 1)])
INIT = SimpleNamespace(x_opt=X7, u_opt=U7)
g_init = warmstart_guesses(ctx, m, X7, U7, disc["s_knot"])
_SIG = inspect.signature(build_and_solve_nlp)


class _Captured(Exception):
    pass


def captured(module, fn, **kw):
    """module.fn(**kw) with build_and_solve_nlp replaced by a stand-in that records its bound
    arguments and aborts, and MLTP_initial by a stub returning INIT (a fallback to the 7-state
    init solves nothing). Returns (arguments, stdout, kwargs of each MLTP_initial call)."""
    seen, init_calls = {}, []

    def stand_in(*args, **kwargs):
        seen.update(_SIG.bind(*args, **kwargs).arguments)
        raise _Captured()

    def fake_initial(*args, **kwargs):
        init_calls.append(kwargs)
        return SimpleNamespace(data=SimpleNamespace(init=INIT))

    real = (po_mod.build_and_solve_nlp, po_mod.MLTP_initial)
    po_mod.build_and_solve_nlp, po_mod.MLTP_initial = stand_in, fake_initial
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(module, fn)(**kw)
    except _Captured:
        pass
    finally:
        po_mod.build_and_solve_nlp, po_mod.MLTP_initial = real
    return seen, buf.getvalue(), init_calls


a, out, calls = captured(po_mod, "MLTP_paramOptim", warm_start=src3, save=False, **KW)
wa = a["warm"]
ok("full result (data namespace) -> warm = {x0, lam_g0, lam_x0, ipopt}, reported as 'full+duals'",
   set(wa) == {"x0", "lam_g0", "lam_x0", "ipopt"} and "-> full+duals" in out)
ok("... x0 = [w_opt; P0] with P0 = the current vp values (= param['x0'])",
   wa["x0"].size == nw3 + 6 and np.array_equal(wa["x0"][:nw3], w3) and np.array_equal(wa["x0"][nw3:], P0s)
   and np.array_equal(a["param"]["x0"], P0s))
ok("... lam_x0 = [lam_x; 0 x 6], lam_g0 = lam_g, the IPOPT warm-start recipe",
   np.array_equal(wa["lam_x0"][:nw3], lx3) and not np.any(wa["lam_x0"][nw3:])
   and np.array_equal(wa["lam_g0"], lg3) and wa["ipopt"] == W.warm_start_ipopt_opts())
ok("... guesses = warmstart_guesses_full (the transcription's fallback w0); no 7-state init solved",
   identical(a["guesses"], warmstart_guesses_full(ctx, m, src3, disc)) and calls == [])
ok("... 6 design parameters, the same NLP inputs as a cold call (path rows, rate bounds)",
   a["param"]["sym"].numel() == 6 and a["h_eq"].size1_out(0) == nhs
   and np.array_equal(a["duk_ub"], m.duk_ub))
c_ctx = Ctx()
c_ctx.data = src3
for label, ws in (("an MLTP ctx (ctx.data)", c_ctx), ("a data dict", dict(vars(src3)))):
    a2, out2, _ = captured(po_mod, "MLTP_paramOptim", warm_start=ws, save=False, **KW)
    ok(f"{label} -> the same warm vectors ('full+duals')",
       "-> full+duals" in out2 and all(np.array_equal(a2["warm"][k], wa[k]) for k in ("x0", "lam_g0", "lam_x0")))
p3 = os.path.join(TMP, "Sturn_Static_ATDOn_EM4Off.mat")
sio.savemat(p3, {"data": dict(vars(src3))}, do_compression=True)
a3, out3, _ = captured(po_mod, "MLTP_paramOptim", warm_start=p3, save=False, **KW)
ok("a result .mat path -> 'full+duals' with the same vectors as reloaded",
   "-> full+duals" in out3 and all(np.array_equal(a3["warm"][k], wa[k]) for k in ("x0", "lam_g0", "lam_x0")))
a4, out4, _ = captured(po_mod, "MLTP_paramOptim", warm_start=src3, warm_start_duals=False, save=False, **KW)
ok("warm_start_duals=False -> 'full-primal': warm = {x0: [w_opt; P0]} only",
   set(a4["warm"]) == {"x0"} and np.array_equal(a4["warm"]["x0"], wa["x0"]) and "-> full-primal" in out4)
a5, out5, calls5 = captured(po_mod, "MLTP_paramOptim", warm_start=src3, OPT_ds=45, save=False, **KW)
ok(f"OPT_ds=45 (N {Ns} -> {a5['disc']['N']}) -> 'full-interp': no warm vectors, s-interpolated guesses",
   a5["warm"] is None and "-> full-interp" in out5 and a5["disc"]["N"] != Ns and calls5 == []
   and a5["guesses"]["x0"].shape == (23, a5["disc"]["N"] + 1))
src_cb3 = SimpleNamespace(**dict(vars(src3), tyre_set="CopyB"))
a6, out6, calls6 = captured(po_mod, "MLTP_paramOptim", warm_start=src_cb3, save=False, **KW)
ok("a source from another tyre set -> ignored: the 7-state init is solved once and interpolated",
   a6["warm"] is None and "warm start ignored" in out6 and len(calls6) == 1
   and calls6[0].get("save") is False and identical(a6["guesses"], g_init))

# the 7-state paths are unchanged
a7, out7, calls7 = captured(po_mod, "MLTP_paramOptim", warm_start={"init": INIT}, save=False, **KW)
ok("7-state init in memory -> no warm vectors, warmstart_guesses of the init, nothing solved",
   a7["warm"] is None and calls7 == [] and identical(a7["guesses"], g_init))
a8, _, calls8 = captured(po_mod, "MLTP_paramOptim", warm_start=None, save=False, **KW)
ok("warm_start=None -> the 7-state init is solved once and interpolated",
   a8["warm"] is None and len(calls8) == 1 and identical(a8["guesses"], g_init))
p7 = os.path.join(TMP, "init_Sturn.mat")
sio.savemat(p7, {"data": {"init": {"x_opt": X7, "u_opt": U7}}})
a9, _, calls9 = captured(po_mod, "MLTP_paramOptim", warm_start=p7, save=False, **KW)
ok("an init .mat path -> the same guesses, nothing solved",
   a9["warm"] is None and calls9 == [] and identical(a9["guesses"], g_init))
bad = SimpleNamespace(data=SimpleNamespace(x_opt=np.zeros((5, 4))))
with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    msg = raises(ValueError, po_mod.MLTP_paramOptim, warm_start=bad, save=False, **KW)
ok("a source that is neither a 7-state init nor a 23-state result raises ValueError",
   msg is not None and "7-state" in msg and "23-state" in msg)

# MLTP_TyreOptim shares optimise_design
fz0 = float(ctx.vp.Fz0_shift)
a10, out10, _ = captured(to_mod, "MLTP_TyreOptim", warm_start=src3, save=False, **KW)
ok("MLTP_TyreOptim gets the same start: x0 = [w_opt; Fz0_shift], lam_x0 = [lam_x; 0]",
   np.array_equal(a10["warm"]["x0"], np.concatenate([w3, [fz0]]))
   and np.array_equal(a10["warm"]["lam_x0"], np.concatenate([lx3, [0.0]])) and "-> full+duals" in out10)
a11, out11, _ = captured(to_mod, "MLTP_TyreOptim", warm_start=src3, warm_start_duals=False, save=False, **KW)
ok("MLTP_TyreOptim forwards warm_start_duals=False -> 'full-primal'",
   set(a11["warm"]) == {"x0"} and "-> full-primal" in out11)

# =============================================================================
print("4. one real solve capped at max_iter=5 (a plausible source: the cold-start guesses, zero multipliers)")
wp = np.concatenate([g_init["x0"].reshape(-1, order="F"), g_init["u0"].reshape(-1, order="F"),
                     g_init["xc0"].reshape(-1, order="F")])
xp, up, _, xcp = unpack_solution(wp, m.nx, m.nu, m.ny, Ns, ds, m.x_s, m.u_s, None)
recp = W.nlp_record(dict(w_opt=wp, lam_g=np.zeros(st3["n_g"]), lam_x=np.zeros(nw3), structure=st3,
                         sym_type="SX", linear_solver="ma57"),
                    {"iter_count": 0, "return_status": "Solve_Succeeded"}, m.x_s, m.u_s, "cold")
srcp = SimpleNamespace(x_opt=xp, u_opt=up, x_full=reconstruct_x_full(xp, xcp, m.nx, Ns, ds),
                       s_full=disc["s_full"], N=Ns, OPT_d=ds, input_keys=list(ctx.input_keys),
                       tyre_set=ctx.tyre_set, nlp=recp)
ok("the plausible source packs back to the cold-start w0 (n_w entries)", wp.size == nw3)
buf = io.StringIO()
with contextlib.redirect_stdout(buf), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    c5 = po_mod.MLTP_paramOptim(warm_start=srcp, save=False, max_iter=5,
                                ipopt_overrides={"print_level": 0, "sb": "yes"}, **KW)
out = buf.getvalue()
ok("the transcription took the extended primal and the duals",
   "primal x0 yes, duals yes" in out)
ok("mode 'full+duals' in ctx.elapsed, data['nlp'] and the summary line",
   c5.elapsed["warm_start"] == "full+duals" and c5.elapsed["duals"] is True
   and c5.data.nlp["warm_start"] == "full+duals" and "warm start=full+duals  duals=yes" in out)
ok("IPOPT capped at 5 iterations; ctx.elapsed carries init / solve / ipopt_iters",
   0 <= c5.elapsed["ipopt_iters"] <= 5 and c5.data.nlp["ipopt_iters"] == c5.elapsed["ipopt_iters"]
   and {"init", "solve", "ipopt_iters", "warm_start", "duals"} <= set(c5.elapsed))
stc = c5.data.nlp["structure"]
ok("saved structure: n_param = 6, n_w = plain n_w + 6, n_g unchanged",
   stc["n_param"] == 6 and stc["n_w"] == nw3 + 6 and stc["n_g"] == st3["n_g"])
ok("optimal_params lists the 6 promoted fields, finite",
   list(c5.data.optimal_params) == DEFAULT and all(np.isfinite(v) for v in c5.data.optimal_params.values()))

print("\nALL paramOptim warm-start TESTS PASSED")
