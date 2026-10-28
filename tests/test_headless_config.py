"""build_solve_kwargs maps a cfg dict -> MLTP kwargs (absolute resource/output
dirs, ni-null -> nan, overrides passed through). F2 options: forwarded only when
set and not default, the qss mode, and the forwarded names are real arguments.
Sections 1-4 build no solver. Section 5 spies on userOpts inside MLTP and
MLTP_screen (imports casadi, builds nothing). Section 6 runs headless_solve.main
with a stand-in MLTP. Section 7 runs the real QSS screen (numpy only) and prints
SKIP without Data/DATA_AA.mat."""
import os, sys, math, json, ast, inspect, warnings, contextlib, io, types, tempfile
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
from headless_solve import build_solve_kwargs

ROOT = _bootstrap.ROOT

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

cfg = {
    "circuit": "Sturn", "AeroConfig": "Static", "ATD": "On",
    "Electric_4Motors": "Off", "TyreModel": "CombinedSlip",
    "vi": 60.0, "ni": None, "linear_solver": "mumps",
    "save": True, "plot": True, "output_dir": "/out",
    "vp_overrides": {"mb": 2000.0},
}
kw = build_solve_kwargs(cfg, "/res")
ok("circuit mapped", kw["circuit"] == "Sturn")
ok("ni null -> nan", isinstance(kw["ni"], float) and math.isnan(kw["ni"]))
ok("results_dir under output", kw["results_dir"] == os.path.join("/out", "Results"))
ok("plots_dir under output", kw["plots_dir"] == os.path.join("/out", "Plots"))
ok("circuits_dir under resource", kw["circuits_dir"] == os.path.join("/res", "Circuits"))
ok("data_dir under resource", kw["data_dir"] == os.path.join("/res", "Data"))
ok("linear_solver passed", kw["linear_solver"] == "mumps")
ok("vp_overrides passed", kw["vp_overrides"] == {"mb": 2000.0})

cfg2 = dict(cfg, ni=0.3)
ok("numeric ni preserved", build_solve_kwargs(cfg2, "/res")["ni"] == 0.3)

# solver/collocation forwarding - defaults when absent
ok("max_iter default", kw["max_iter"] == 6000 and isinstance(kw["max_iter"], int))
ok("OPT_ds default", kw["OPT_ds"] == 30.0 and isinstance(kw["OPT_ds"], float))
ok("OPT_d default", kw["OPT_d"] == 3 and isinstance(kw["OPT_d"], int))
ok("OPT_e default", kw["OPT_e"] == 1e-2)
ok("tol default", kw["tol"] == 1e-4)

cfg3 = dict(cfg, max_iter=3000, OPT_ds=20, OPT_d=4, OPT_e=5e-3, tol=1e-6)
kw3 = build_solve_kwargs(cfg3, "/res")
ok("max_iter forwarded", kw3["max_iter"] == 3000)
ok("OPT_ds forwarded", kw3["OPT_ds"] == 20.0)
ok("OPT_d forwarded", kw3["OPT_d"] == 4)
ok("OPT_e forwarded", kw3["OPT_e"] == 5e-3)
ok("tol forwarded", kw3["tol"] == 1e-6)

# speed / fidelity options (mesh, mesh_opts, tyre_set, screening) - userOpts defaults when absent
ok("mesh default auto", kw["mesh"] == "auto")
ok("mesh_opts default None", kw["mesh_opts"] is None)
ok("tyre_set default MF205", kw["tyre_set"] == "MF205")
ok("screening default False", kw["screening"] is False)

mo = {"a": 2.0, "ds_max": 60.0}
cfg4 = dict(cfg, mesh="curvature", mesh_opts=mo, tyre_set="CopyB", screening=True)
kw4 = build_solve_kwargs(cfg4, "/res")
ok("mesh forwarded", kw4["mesh"] == "curvature")
ok("mesh_opts forwarded (copy)", kw4["mesh_opts"] == mo and kw4["mesh_opts"] is not mo)
ok("tyre_set forwarded", kw4["tyre_set"] == "CopyB")
ok("screening forwarded", kw4["screening"] is True)
ok("empty / null mesh_opts -> None",
   build_solve_kwargs(dict(cfg, mesh_opts={}), "/res")["mesh_opts"] is None
   and build_solve_kwargs(dict(cfg, mesh_opts=None), "/res")["mesh_opts"] is None)

# every forwarded key is a real userOpts / MLTP argument (userOpts only, no casadi / MLTP import)
import inspect
from userOpts import userOpts
_uo_args = set(inspect.signature(userOpts).parameters)
ok("mesh / mesh_opts / tyre_set / screening are userOpts arguments",
   {"mesh", "mesh_opts", "tyre_set", "screening"} <= _uo_args)


def fn_node(path, name):
    with open(os.path.join(ROOT, path)) as fh:
        tree = ast.parse(fh.read())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def fn_args(path, name):
    f = fn_node(path, name)
    return {a.arg for a in f.args.args + f.args.kwonlyargs}


def sig_defaults(path, name):
    """Plain literal defaults of a module-level function, read from the source (no import, no casadi)."""
    f = fn_node(path, name)
    pos = f.args.args[len(f.args.args) - len(f.args.defaults):]
    out = {}
    for a, d in list(zip(pos, f.args.defaults)) + list(zip(f.args.kwonlyargs, f.args.kw_defaults)):
        if d is None:
            continue
        try:
            out[a.arg] = ast.literal_eval(d)
        except ValueError:
            pass
    return out


def same(a, b):
    """dict equality that treats nan (ni = None gives float nan) as equal to nan."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# --- 2. F2 options: forwarded when set and not default, never when absent, default or null -------------
print("2. F2 options")
from functions.ladder import resolve_ladder, homotopy_schedule
from functions.refine import refine_options
from headless_solve import FORWARD, run_mode

FORWARD_DEFAULTS = {"ladder": "auto", "homotopy": None, "refine": None, "continue_from": None,
                    "ipopt_overrides": None, "physics": "matlab", "start_state": "matlab", "n_bounds": None,
                    "Steering": "NA", "CamberGain": "Off", "between_knots": "matlab",
                    "collocation": "legendre", "state_scaling": "matlab", "atd_tbr": None}
ok("FORWARD holds the 14 F2 options at these defaults", FORWARD == FORWARD_DEFAULTS)

OLD_KEYS = {"circuit", "vi", "ni", "AeroConfig", "ATD", "Electric_4Motors", "TyreModel", "linear_solver",
            "save", "plot", "results_dir", "plots_dir", "circuits_dir", "data_dir", "vp_overrides",
            "max_iter", "OPT_ds", "OPT_d", "OPT_e", "tol", "mesh", "mesh_opts", "tyre_set", "screening"}
ok("a cfg without the F2 keys makes the call it made before F2 (24 keys, none of FORWARD)",
   set(kw) == OLD_KEYS and not (set(FORWARD) & set(kw)))
ok("every F2 key at its default adds no kwarg", same(build_solve_kwargs(dict(cfg, **FORWARD_DEFAULTS), "/res"), kw))
ok("every F2 key null adds no kwarg", same(build_solve_kwargs(dict(cfg, **{k: None for k in FORWARD}), "/res"), kw))
with warnings.catch_warnings(record=True) as seen_warn:
    warnings.simplefilter("always")
    extra = build_solve_kwargs(dict(cfg, bogus_key=1), "/res")
ok("an unknown top-level key adds no kwarg and no warning", same(extra, kw) and not seen_warn)

F2 = dict(ladder="qss23", homotopy=True, refine={"passes": 1, "tol": 5e-3},
          ipopt_overrides={"mu_strategy": "monotone"}, physics="corrected", start_state="rolling",
          n_bounds=4.5, Steering="Ackermann", CamberGain="Table", between_knots="linear+power",
          collocation="radau", state_scaling="typical", atd_tbr=2.0)
CONT = {"continue_from": "auto"}
kwf = build_solve_kwargs(dict(cfg, **F2), "/res")
ok("every changed F2 option is forwarded as given", all(kwf[k] == v for k, v in F2.items()))
ok("... and nothing else moves", same({k: v for k, v in kwf.items() if k not in F2}, kw))
ok("each option forwards alone (continue_from too)",
   all(same(build_solve_kwargs(dict(cfg, **{k: v}), "/res"), dict(kw, **{k: v})) for k, v in {**F2, **CONT}.items()))
ok("homotopy False (explicit off) and a schedule list are forwarded",
   build_solve_kwargs(dict(cfg, homotopy=False), "/res")["homotopy"] is False
   and build_solve_kwargs(dict(cfg, homotopy=[1.3, 1.15, 1.0]), "/res")["homotopy"] == [1.3, 1.15, 1.0])
ok("n_bounds as a [lo, hi] pair is forwarded",
   build_solve_kwargs(dict(cfg, n_bounds=[-1.5, 2.5]), "/res")["n_bounds"] == [-1.5, 2.5])

# --- 3. QSS mode ---------------------------------------------------------------------------------------
print("3. mode = qss")
ok("mode default nlp, qss accepted", run_mode(cfg) == "nlp" and run_mode(dict(cfg, mode="qss")) == "qss")
ok("a bad mode raises ValueError (run_mode and build_solve_kwargs)",
   raises(lambda: run_mode(dict(cfg, mode="QSS"))) and raises(lambda: build_solve_kwargs(dict(cfg, mode="QSS"), "/res")))
kwn = build_solve_kwargs(dict(cfg, warm_start="/w.mat", **F2, **CONT), "/res")
ok("nlp mode keeps warm_start and every F2 option",
   kwn["warm_start"] == "/w.mat" and all(kwn[k] == v for k, v in {**F2, **CONT}.items()))
kq = build_solve_kwargs(dict(cfg, mode="qss", warm_start="/w.mat", **F2, **CONT), "/res")
ok("qss keeps results_dir, save, circuits_dir and the userOpts options",
   kq["results_dir"] == os.path.join("/out", "Results") and kq["save"] is True
   and kq["circuits_dir"] == os.path.join("/res", "Circuits")
   and all(kq[k] == v for k, v in F2.items() if k in _uo_args))

# --- 4. the forwarded names are real arguments (AST of the entry points, userOpts signature) -----------
print("4. forwarded names are real arguments")
mltp_ok = fn_args("MLTP.py", "MLTP") | _uo_args
screen_ok = fn_args("MLTP_screen.py", "MLTP_screen") | _uo_args
NLP_ONLY_KEYS = {"ladder", "homotopy", "refine", "continue_from", "warm_start", "plots_dir"}
ok("every key of the full NLP call is an MLTP or userOpts argument", set(kwn) <= mltp_ok)
ok("every key of the qss call is an MLTP_screen or userOpts argument", set(kq) <= screen_ok)
ok("that check can fail: a made-up key is no argument, and no NLP-only key is an MLTP_screen argument",
   "bogus_key" not in mltp_ok and not (NLP_ONLY_KEYS & screen_ok))
ok("the qss call drops every NLP-only key", not (NLP_ONLY_KEYS & set(kq)))
mltp_def = sig_defaults("MLTP.py", "MLTP")
uo_def = {k: p.default for k, p in inspect.signature(userOpts).parameters.items()}
ok("FORWARD defaults equal the MLTP / userOpts signature defaults",
   all(mltp_def.get(k, uo_def.get(k, "missing")) == d for k, d in FORWARD.items()))
ok("MLTP's own checks accept the forwarded ladder, homotopy and refine",
   resolve_ladder(kwf["ladder"])[0] == "qss23" and homotopy_schedule(kwf["homotopy"]) == (1.2, 1.1, 1.0)
   and refine_options(kwf["refine"], kwf["OPT_ds"])["passes"] == 1)

from functions.context import Ctx
uo = {k: v for k, v in kwf.items() if k in _uo_args}
uo["vp_overrides"] = {"Tmax": 400.0, "Pmax": 2.0e5, "Preg": 5.0e4, "Ppack": 3.0e5, "gear": 8.0}   # A5 keys, corrected only
uo.update(circuits_dir=os.path.join(ROOT, "Circuits"), data_dir=os.path.join(ROOT, "Data"))
c = userOpts(Ctx(), **uo)
ok("userOpts takes the forwarded options (physics, start state, band, switches, IPOPT, A5 keys)",
   c.physics == "corrected" and c.start_state == "rolling" and c.n_tag == "hw4p5" and c.Steering == "Ackermann"
   and c.CamberGain == "Table" and c.opts["ipopt"]["mu_strategy"] == "monotone"
   and (c.pt.Tmax, c.pt.Pmax, c.pt.Preg, c.pt.Ppack, c.vp.gear) == (400.0, 2.0e5, 5.0e4, 3.0e5, 8.0))
ok("A5 keys under physics='matlab' still raise ValueError (the refusal stays in vehParams)",
   raises(lambda: userOpts(Ctx(), **dict(uo, physics="matlab"))))

# --- 5. what MLTP and MLTP_screen hand to userOpts (spy: no model build, no solve) ---------------------
print("5. the entry points hand the options to userOpts")
USEROPTS_F2 = [k for k in F2 if k not in ("ladder", "homotopy", "refine")]
try:
    import casadi  # noqa: F401
    import MLTP as mltp_mod
    import MLTP_screen as screen_mod
except ImportError as exc:
    print(f"  [SKIP] section 5: casadi not importable ({exc})")
else:
    class _Stop(Exception):
        pass

    def spy_userOpts(entry_module, fn_name, kwargs):
        """Run fn_name(**kwargs) with userOpts replaced by a recorder. Returns the keywords userOpts got."""
        seen = {}

        def spy(ctx, **k):
            seen.update(k)
            raise _Stop

        real = entry_module.userOpts
        entry_module.userOpts = spy
        try:
            getattr(entry_module, fn_name)(**kwargs)
        except _Stop:
            pass
        finally:
            entry_module.userOpts = real
        return seen

    seen = spy_userOpts(mltp_mod, "MLTP", kwn)
    ok("MLTP hands the userOpts options on and keeps ladder, homotopy, refine and continue_from to itself",
       all(seen.get(k) == F2[k] for k in USEROPTS_F2)
       and not ({"ladder", "homotopy", "refine", "continue_from"} & set(seen)))
    seen_q = spy_userOpts(screen_mod, "MLTP_screen", kq)
    ok("MLTP_screen hands its userOpts options on and no NLP-only key",
       all(seen_q.get(k) == F2[k] for k in USEROPTS_F2)
       and not (NLP_ONLY_KEYS & set(seen_q)) and seen_q.get("circuit") == "Sturn")

# --- 6. headless_solve.main: nlp mode calls MLTP once (stand-in); a bad mode calls nothing ----------------
print("6. headless_solve.main dispatch")
import headless_solve

stand_in = types.ModuleType("MLTP")
calls = []
stand_in.MLTP = lambda **k: calls.append(k)
saved_mltp = sys.modules.get("MLTP")
with tempfile.TemporaryDirectory(prefix="headless_nlp_") as tmp0:
    path0 = os.path.join(tmp0, "cfg.json")
    try:
        sys.modules["MLTP"] = stand_in
        for name, c0 in (("an old cfg (no F2 key)", dict(cfg, output_dir=tmp0)),
                         ("a cfg with the F2 options", dict(cfg, output_dir=tmp0, **F2, **CONT))):
            calls.clear()
            with open(path0, "w") as fh:
                json.dump(c0, fh)
            with contextlib.redirect_stdout(io.StringIO()):
                code0 = headless_solve.main([path0])
            ok(f"nlp mode, {name}: MLTP called once with build_solve_kwargs, exit 0",
               code0 == 0 and len(calls) == 1 and same(calls[0], build_solve_kwargs(c0, headless_solve.resource_root())))
        calls.clear()
        with open(path0, "w") as fh:
            json.dump(dict(cfg, output_dir=tmp0, mode="QSS"), fh)
        with contextlib.redirect_stdout(io.StringIO()):
            code_bad = headless_solve.main([path0])
        ok("a bad mode exits 1 and calls no solver", code_bad == 1 and calls == [])
    finally:
        if saved_mltp is None:
            sys.modules.pop("MLTP", None)
        else:
            sys.modules["MLTP"] = saved_mltp

# --- 7. the real qss run through headless_solve.main (numpy only, no IPOPT) ---------------------------
print("7. headless_solve.main with mode = qss")
if not os.path.isfile(os.path.join(ROOT, "Data", "DATA_AA.mat")):
    print("\nSKIP section 7: Data/DATA_AA.mat missing (the QSS envelope needs ctx.aero)")
    sys.exit(0)

import scipy.io as sio
with tempfile.TemporaryDirectory(prefix="headless_qss_") as tmp:
    cfg_q = dict(cfg, output_dir=tmp, mode="qss", vp_overrides={}, save=True, plot=False,
                 warm_start="/not/read.mat", physics="corrected", ladder="qss23", homotopy=True)
    path = os.path.join(tmp, "cfg.json")
    with open(path, "w") as fh:
        json.dump(cfg_q, fh)
    with contextlib.redirect_stdout(io.StringIO()):
        code = headless_solve.main([path])
    res_dir = os.path.join(tmp, "Results")
    qss_name = "Sturn_Static_ATDOn_EM4Off_physCorrected_qss.mat"
    ok("main returns 0 and the qss run writes only <stem>_physCorrected_qss.mat",
       code == 0 and os.path.isdir(res_dir) and os.listdir(res_dir) == [qss_name])
    d = sio.loadmat(os.path.join(res_dir, qss_name), squeeze_me=True, struct_as_record=False)["data"]
    ok("the file is the QSS screen: fidelity qss, physics corrected, lap time 15 to 40 s",
       str(d.fidelity) == "qss" and str(d.physics) == "corrected" and 15.0 < float(d.lap_time) < 40.0)
    ok("no Plots folder from the qss mode", not os.path.isdir(os.path.join(tmp, "Plots")))
    with open(path, "w") as fh:
        json.dump(dict(cfg_q, mode="QSS"), fh)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code_bad = headless_solve.main([path])
    ok("a bad mode exits 1 with the reason in the log", code_bad == 1 and "mode must be" in buf.getvalue())

print("\nALL headless config TESTS PASSED")
