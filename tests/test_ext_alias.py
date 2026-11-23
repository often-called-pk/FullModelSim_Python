"""Neutral 'ext:' spellings (functions/chassissim.py alias helpers) for the ChassisSim names.

'ext:<x>' is the same as 'chassissim:<x>' for a circuit and a car, ext_dir is the same as chassissim_dir,
and the env EXT_SIM_DIR is the same as CHASSISSIM_DIR (it wins when both differ, with one warning).
Synthetic install in a temp folder, no vendor files, no real solve. Checks:
  1. alias(): 'ext:Foo' -> 'chassissim:Foo', other values unchanged,
  2. alias_dir / alias_kwargs: ext_dir folds into chassissim_dir, equal values pass, different values raise,
  3. env_dir: EXT_SIM_DIR alone, CHASSISSIM_DIR alone, equal (no warning), different (EXT_SIM_DIR wins, one warning),
  4. userOpts with circuit='ext:' + ext_dir gives the same track as 'chassissim:' + chassissim_dir,
  5. MLTP with car='ext:' and a missing car file raises the same error as 'chassissim:' (a full car build
     needs a ~200 line car builder, so the car path itself is checked by its refusal only),
  6. resolve_start_state gives the same answer for both spellings,
  7. the entry points fold the spelling (each check fails when its entry alias is missing): MLTP_screen and a
     fake-solver MLTP save under one file name, setup_knobs gives one car_file, setup_sweep gives one folder and
     one fingerprint, optimise_design saves a stem without ':', and result_paths, run_stage and the cfg key
     ext_dir follow.

    python tests/test_ext_alias.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import tempfile
import warnings

import numpy as np

import functions.transcription as T
from functions import chassissim as cs
from functions.context import Ctx
from MLTP import MLTP
from MLTP_screen import MLTP_screen
from userOpts import resolve_start_state, setup_knobs, userOpts

PASS = 0
CIRC = "SynthRing"


def ok(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    PASS += 1


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def raises_msg(exc, fn, *a, **kw):
    """The message of the exc that fn raises, or None."""
    try:
        quiet(fn, *a, **kw)
    except exc as e:
        return str(e)
    return None


TMP = tempfile.TemporaryDirectory()
INSTALL = os.path.join(TMP.name, "ChassisSim")
os.makedirs(os.path.join(INSTALL, "Circuits"))
# one circuit, 600 m, one lap of turning (k(0) == k(L), so it joins at the line)
S = np.linspace(0.0, 600.0, 121)
K = (2.0 * np.pi / 600.0) * (1.0 + 0.5 * np.sin(2.0 * np.pi * S / 600.0))
with open(os.path.join(INSTALL, "Circ_prop.ini"), "w", encoding="latin-1", newline="") as f:
    f.write("[Total Circ prop]\r\nTotal circuits = 1\r\n\r\n"
            f"[Circuit0]\r\nCircuit name = {CIRC}\r\nCircuit file name = tr_synth.txt\r\nBase Grip factor = 1.0\r\n"
            "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n")
with open(os.path.join(INSTALL, "Circuits", "tr_synth.txt"), "w", encoding="latin-1", newline="") as f:
    f.write("\r\n".join(f"{float(k)!r}   {float(s)!r}" for k, s in zip(K, S)) + "\r\n")

MISSING_CAR = os.path.join(TMP.name, "no_such_car.car")


def fake(ca_, nlp, opts, info=None):
    """Stand-in for functions.transcription._make_solver: returns the start point, no iterations."""
    def solver(**k):
        return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
    return solver


T._make_solver = fake

# ---- 1. alias() --------------------------------------------------------------------------------------------
ok("alias('ext:Foo') == 'chassissim:Foo'", cs.alias("ext:Foo") == "chassissim:Foo")
ok("alias('Sturn') unchanged", cs.alias("Sturn") == "Sturn")
ok("alias('chassissim:Foo') unchanged", cs.alias("chassissim:Foo") == "chassissim:Foo")
ok("alias(None) is None", cs.alias(None) is None)

# ---- 2. alias_dir / alias_kwargs ---------------------------------------------------------------------------
ok("alias_dir(ext_dir='/a') == '/a'", cs.alias_dir(ext_dir="/a") == "/a")
ok("alias_dir(chassissim_dir='/a') == '/a'", cs.alias_dir(chassissim_dir="/a") == "/a")
ok("alias_dir with equal values (same path, other spelling) gives the value",
   cs.alias_dir("/tmp/x", "/tmp/x/") == "/tmp/x/")
msg = raises_msg(ValueError, cs.alias_dir, "/one", "/two")
ok(f"alias_dir with different values raises, naming both: {msg!r}",
   msg is not None and "/one" in msg and "/two" in msg)
kw = cs.alias_kwargs({"circuit": "x", "car": "ext:/c.car", "ext_dir": "/d"})
ok("alias_kwargs: car 'ext:' -> 'chassissim:', ext_dir folded into chassissim_dir, key removed",
   kw["car"] == "chassissim:/c.car" and kw["chassissim_dir"] == "/d" and "ext_dir" not in kw)

# ---- 3. env_dir --------------------------------------------------------------------------------------------
SAVED = {k: os.environ.get(k) for k in ("EXT_SIM_DIR", "CHASSISSIM_DIR")}


def set_env(ext, old):
    for k, v in (("EXT_SIM_DIR", ext), ("CHASSISSIM_DIR", old)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


try:
    set_env("/e", None)
    ok("env_dir: EXT_SIM_DIR alone", cs.env_dir() == "/e")
    set_env(None, "/c")
    ok("env_dir: CHASSISSIM_DIR alone", cs.env_dir() == "/c")
    set_env("/same", "/same")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        got = cs.env_dir()
    ok("env_dir: equal values, no warning", got == "/same" and not seen)
    set_env("/e", "/c")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        got = cs.env_dir()
    ours = [str(w.message) for w in seen if "EXT_SIM_DIR" in str(w.message)]
    ok("env_dir: different values, EXT_SIM_DIR wins", got == "/e")
    ok(f"env_dir: different values, exactly one warning naming both: {ours}",
       len(seen) == 1 and "/e" in ours[0] and "/c" in ours[0])
finally:
    set_env(SAVED["EXT_SIM_DIR"], SAVED["CHASSISSIM_DIR"])

# ---- 4. userOpts: track from the two spellings -------------------------------------------------------------
ctx_ext = Ctx()
quiet(userOpts, ctx_ext, circuit=f"ext:{CIRC}", ext_dir=INSTALL)
ctx_cs = Ctx()
quiet(userOpts, ctx_cs, circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL)
ok("userOpts ext: track s equal to chassissim: track s",
   np.array_equal(ctx_ext.track.s, ctx_cs.track.s))
ok("userOpts ext: track k equal to chassissim: track k",
   np.array_equal(ctx_ext.track.k, ctx_cs.track.k))

# ---- 5. MLTP: car spelling (missing file path only) -------------------------------------------------------------
err = {}
for pre in ("ext:", "chassissim:"):
    try:
        quiet(MLTP, circuit=f"{pre}{CIRC}", car=f"{pre}{MISSING_CAR}", chassissim_dir=INSTALL,
              physics="corrected", save=False, plot=False, linear_solver="mumps")
        err[pre] = None
    except Exception as e:      # the missing file raises from the car read (FileNotFoundError), same for both
        err[pre] = (type(e).__name__, str(e))
ok(f"MLTP with a missing car: both spellings raise the same error: {err['ext:'][0]}",
   err["ext:"] is not None and err["ext:"] == err["chassissim:"])

# ---- 6. resolve_start_state --------------------------------------------------------------------------------
car = os.path.join(TMP.name, "any.car")
a = resolve_start_state(None, f"ext:{CIRC}", f"ext:{car}", INSTALL)
b = resolve_start_state(None, f"chassissim:{CIRC}", f"chassissim:{car}", INSTALL)
ok(f"resolve_start_state: ext: and chassissim: give the same start state ({a!r})", a == b == "periodic")

# ---- 7. entry points: one name, one tag, one NLP for the two spellings -------------------------------------
import setup_sweep as SS
from app.results import result_paths
from app.runconfig import RunConfig
from functions import correlate
from headless_solve import build_solve_kwargs
from MLTP_paramOptim import MLTP_paramOptim

RES = os.path.join(TMP.name, "Results")
EXT = dict(circuit=f"ext:{CIRC}", ext_dir=INSTALL)
OLD = dict(circuit=f"chassissim:{CIRC}", chassissim_dir=INSTALL)

qe = quiet(MLTP_screen, save=True, results_dir=RES, verbose=False, **EXT)
qc = quiet(MLTP_screen, save=True, results_dir=RES, verbose=False, **OLD)
ok(f"MLTP_screen: both spellings write {os.path.basename(qc.out_path)}",
   qe.out_path == qc.out_path and os.path.basename(qc.out_path) == f"cs_{CIRC}_Static_ATDOn_EM4Off_qss.mat"
   and qe.data.lap_time == qc.data.lap_time)

me = quiet(MLTP, save=True, plot=False, results_dir=RES, linear_solver="mumps", **EXT)
mc = quiet(MLTP, save=True, plot=False, results_dir=RES, linear_solver="mumps", **OLD)
ne, nc = me.data.nlp, mc.data.nlp
ok(f"MLTP (fake solver): both spellings write {os.path.basename(mc.out_path)}",
   me.out_path == mc.out_path and os.path.basename(mc.out_path) == f"cs_{CIRC}_Static_ATDOn_EM4Off.mat")
ok("MLTP (fake solver): the same NLP structure and start point",
   set(ne["structure"]) == set(nc["structure"])
   and all(np.array_equal(ne["structure"][k], nc["structure"][k]) for k in nc["structure"])
   and np.array_equal(ne["w_opt"], nc["w_opt"]))
ok("MLTP: data.circuit holds the chassissim: spelling for both", me.data.circuit == mc.data.circuit)

CAR = os.path.join(TMP.name, "any.car")
with open(CAR, "wb") as f:
    f.write(b"not a car, the knob hashes the bytes only")
ok("setup_knobs: car 'ext:' and 'chassissim:' give one car_file",
   setup_knobs({"car": "ext:" + CAR}) == setup_knobs({"car": "chassissim:" + CAR})
   and setup_knobs({"car": "ext:" + CAR})["car_file"].startswith(os.path.abspath(CAR) + "#"))

SW = dict(confirm=False, top_k=1, n_probes=1, results_root=os.path.join(TMP.name, "sweeps"), verbose=False,
          plot=False)
se = quiet(SS.setup_sweep, [("mb", 1730, 1910)], 4, **SW, **EXT)
sc = quiet(SS.setup_sweep, [("mb", 1730, 1910)], 4, **SW, **OLD)
ok(f"setup_sweep with ext: and ext_dir runs, folder {se.name} has no ':'",
   ":" not in se.name and se.name.startswith(f"cs_{CIRC}_") and os.path.isdir(se.out_dir))
ok("setup_sweep: one fingerprint and one folder for the two spellings (a resume works across them)",
   se.fingerprint == sc.fingerprint and se.out_dir == sc.out_dir)

po = quiet(MLTP_paramOptim, save=True, results_dir=RES, linear_solver="mumps", **EXT)
ok(f"optimise_design (fake solver) with ext: saves {os.path.basename(po.out_path)}",
   os.path.basename(po.out_path) == f"cs_{CIRC}_paramOptim.mat" and not [n for n in os.listdir(RES) if ":" in n]
   and po.data.circuit == f"chassissim:{CIRC}")

rp = [result_paths(RunConfig(circuit=c, car=pre + CAR, chassissim_dir=INSTALL, physics="corrected",
                             output_dir=TMP.name)) for pre, c in (("ext:", EXT["circuit"]), ("chassissim:", OLD["circuit"]))]
ok(f"result_paths: one file for the two spellings ({os.path.basename(rp[0][0])})",
   rp[0] == rp[1] and ":" not in os.path.basename(rp[0][0]) and "_car_any" in rp[0][0])

LED = os.path.join(TMP.name, "ledger.json")
r1 = quiet(correlate.run_stage, "grip", None, ledger=LED, lap=(60.0, 59.0), **EXT)
r2 = quiet(correlate.run_stage, "grip", None, ledger=LED, lap=(60.0, 59.0), **OLD)
ok("run_stage: one ledger takes both spellings (same circuit and useropts in the record)",
   r1["circuit"] == r2["circuit"] == f"chassissim:{CIRC}" and r1["useropts"] == r2["useropts"])

cfg = dict(circuit=EXT["circuit"], vi=60.0, AeroConfig="Static", ATD="On", Electric_4Motors="Off",
           TyreModel="CombinedSlip", output_dir=TMP.name, ext_dir=INSTALL)
ok("build_solve_kwargs keeps the cfg key ext_dir", build_solve_kwargs(cfg, "/res").get("ext_dir") == INSTALL)

TMP.cleanup()
print(f"test_ext_alias: {PASS} checks passed")
