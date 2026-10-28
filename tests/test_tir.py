"""Tyre files and MF 6.1 pressure terms (C5).

This file tests the .tir reader, tyre_set='file:<path>' and the pressure terms ppx1..ppx4 and ppy1..ppy5.
It uses INFLPRES and NOMPRES. The checks run in the full model and in the QSS screen.

Plain script, no pytest. Each check prints PASS or FAIL. The first FAIL stops the file. Run from the repo root:

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_tir.py

Every tyre file here is synthetic. The test writes each file at run time, from the in-repo MF205 coefficients.
The files go into a temporary directory and carry made-up pressure terms. No licensed tyre data enters the repo.

  1. reader: sections, comments, quotes, D exponent, table lines, lower case keys, CRLF
  2. vehParams with a file: round trip of MF205, FNOMIN, overrides win, optional keys, result_stem,
     errors and warnings (LMUV is neutral at 0), no .tir tracked by git
  3. mf_pressure: plain floats at zero pressure terms, the MF 6.1 factors at +10 % pressure
  4. full model, no solve: same graph for the round trip and for pp terms at dpi 0, a graph change at
     +10 % pressure, mu_x, mu_y, Kxk and Kya follow their factors, SVx and SVy do not move, camber
     stiffness, INFLPRES as a casadi symbol
  5. QSS screen: the lap at +10 % pressure equals the lap with LMUX and LMUY at the mu factors, the
     stiffness terms do not move it, two same-name files in two folders get different setup tags.
     setup_knobs: an override equal to the file value is no knob, a relative and an absolute spelling
     give one tag. A setup sweep gets a new fingerprint when the tyre file changes.
  6. parity export: no pressure key while INFLPRES == NOMPRES and every ppx and ppy term is 0
  7. MLTP with a fake IPOPT: continue_from='auto' on a tyre file starts from the default result of
     that file

Sections 4 to 7 need casadi and Data/DATA_AA.mat. Without DATA_AA they print SKIP and exit 0.
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

from functions.context import Ctx
from functions.importfile import read_tir, result_stem
from vehParams import mf_pressure, vehParams


def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


def quiet(fn, *args, **kwargs):
    """fn(*args, **kwargs) with stdout swallowed and warnings ignored."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def warned(fn, *args, **kwargs):
    """Return the messages of the warnings that fn(*args, **kwargs) raises."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            fn(*args, **kwargs)
    return [str(x.message) for x in w]


def value_error(fn, *args, **kwargs):
    """Return the message of the ValueError that fn raises, or None. Other exceptions propagate."""
    try:
        quiet(fn, *args, **kwargs)
    except ValueError as exc:
        return str(exc)
    return None


SECTIONS = (
    ("SCALING_COEFFICIENTS", ("LFZO", "LMUX", "LKX", "LMUY", "LKY")),
    ("LONGITUDINAL_COEFFICIENTS", ("pCx1", "pDx1", "pDx2", "pDx3", "pEx1", "pEx2", "pEx3", "pEx4",
                                   "pKx1", "pKx2", "pKx3", "pHx1", "pHx2", "pVx1", "pVx2",
                                   "ppx1", "ppx2", "ppx3", "ppx4", "rBx1", "rBx2", "rBx3", "rCx1", "rEx1", "rEx2",
                                   "rHx1")),
    ("LATERAL_COEFFICIENTS", ("pCy1", "pDy1", "pDy2", "pDy3", "pEy1", "pEy2", "pEy3", "pEy4", "pEy5",
                              "pKy1", "pKy2", "pKy3", "pKy4", "pKy5", "pKy6", "pKy7", "pHy1", "pHy2",
                              "pVy1", "pVy2", "pVy3", "pVy4", "ppy1", "ppy2", "ppy3", "ppy4", "ppy5",
                              "rBy1", "rBy2", "rBy3", "rBy4", "rCy1", "rEy1", "rEy2", "rHy1", "rHy2",
                              "rVy1", "rVy2", "rVy3", "rVy4", "rVy5", "rVy6")),
)
OPTIONAL = ("LMUX", "LMUY", "LFZO", "LKX", "LKY",
            "ppx1", "ppx2", "ppx3", "ppx4", "ppy1", "ppy2", "ppy3", "ppy4", "ppy5")
PP = dict(ppx1=-0.30, ppx2=0.10, ppx3=-0.15, ppx4=0.05, ppy1=0.40, ppy2=-0.20, ppy3=-0.12, ppy4=0.04, ppy5=0.15)
NOM = 220000.0
P1 = 1.1 * NOM                                  # +10 % inflation pressure
TOL_S = 1e-9                                    # s. The QSS fold may round the last bit differently.


def write_tir(path, mf, fz0=4905.0, nompres=NOM, inflpres=NOM, omit=(), extra=()):
    """Write a TIR-style file from a dict of mf values.

    The file has sections and KEY = value lines. It has '$' and '!' comments, quoted strings, a D exponent
    (rHy1) and a table section without '=' lines. A value of None for fz0, nompres or inflpres leaves that
    line out.
    """
    lines = ["[MDI_HEADER]", "FILE_TYPE                ='tir'", "FILE_VERSION             =3.0",
             "! SYNTHETIC test tyre, written from the in-repo MF205 coefficients. Not a measured tyre.",
             "$-----units", "[UNITS]", "LENGTH                   ='meter'", "FORCE                    ='newton'",
             "ANGLE                    ='radians'", "[MODEL]", "FITTYP                   =61   $Magic Formula 6.1",
             "TYRESIDE                 ='Left'", "PRESMIN                  = 150000.0",
             "PRESMAX                  = 300000.0",
             "[VERTICAL]"]
    lines += [] if fz0 is None else [f"FNOMIN                   = {fz0!r}   $nominal load"]
    lines += ["[OPERATING_CONDITIONS]"]
    lines += [] if inflpres is None else [f"INFLPRES                 = {inflpres!r}"]
    lines += [] if nompres is None else [f"NOMPRES                  = {nompres!r}"]
    for sec, keys in SECTIONS:
        lines.append(f"[{sec}]")
        for k in keys:
            if k not in omit:
                txt = repr(float(mf[k]))
                lines.append(f"{k.upper():24s} = {txt.replace('e', 'D') if k == 'rHy1' else txt}   $ {k}")
    lines += list(extra) + ["[SHAPE]", "$ table, no '=' on these lines", " 0.5  1.0  0.0"]
    with open(path, "w", encoding="ascii", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


c0 = Ctx()
quiet(vehParams, c0)
MF0 = dict(vars(c0.mf))                         # the MF205 default set
tmp = tempfile.TemporaryDirectory()


def T(name):
    return os.path.join(tmp.name, name)


# =============================================================================
print("1. reader")
p = T("reader.tir")
with open(p, "w", encoding="ascii", newline="") as fh:
    fh.write("[MDI_HEADER]\r\nFILE_TYPE='tir'\r\n! a comment = with an equals sign\r\n$ another\r\n\r\n[MODEL]\r\n"
             "pcx1   =  1.5D-1   $ lower case key, D exponent\r\nKEY2=-2.5e+3 ! trailing comment\r\n"
             "TEXT = 'Left handed'\r\n[SHAPE]\r\n 1.0 2.0 3.0\r\nLAST = 7\r\n")
r = read_tir(p)
ok("numbers are floats, names are upper case, CRLF is fine",
   r["PCX1"] == 0.15 and r["KEY2"] == -2500.0 and r["LAST"] == 7.0 and isinstance(r["PCX1"], float))
ok("quoted text stays a string and comments are cut", r["FILE_TYPE"] == "tir" and r["TEXT"] == "Left handed")
ok("section lines, comment lines and table lines add no key", set(r) == {"FILE_TYPE", "PCX1", "KEY2", "TEXT", "LAST"})

# =============================================================================
print("2. vehParams with a file")
f_round = write_tir(T("round.tir"), MF0)
c1 = Ctx()
quiet(vehParams, c1, tyre_set="file:" + f_round)
ok("MF205 -> .tir -> vehParams: mf equal to the default, bit for bit", vars(c1.mf) == MF0)
ok("ctx.tyre_set keeps the string and no mf override is set",
   c1.tyre_set == "file:" + f_round and c1.mf_overrides == [])
f_fz = write_tir(T("fz.tir"), MF0, fz0=3200.0)
c2 = Ctx()
quiet(vehParams, c2, tyre_set="file:" + f_fz)
ok("FNOMIN sets vp.Fz0", c2.vp.Fz0 == 3200.0)
c2 = Ctx()
quiet(vehParams, c2, tyre_set="file:" + f_fz, vp_overrides={"Fz0": 4000.0, "pDy1": 1.5})
ok("vp_overrides win over the file (Fz0, pDy1), and pDy1 is listed in mf_overrides",
   c2.vp.Fz0 == 4000.0 and c2.mf.pDy1 == 1.5 and c2.mf_overrides == ["pDy1"])
for name, kw in (("INFLPRES", dict(nompres=None, inflpres=190000.0)),
                 ("NOMPRES", dict(inflpres=None, nompres=190000.0))):
    c2 = Ctx()
    quiet(vehParams, c2, tyre_set="file:" + write_tir(T(f"one_{name}.tir"), MF0, **kw))
    ok(f"a file with {name} only (190000 Pa) sets both pressures",
       (c2.mf.NOMPRES, c2.mf.INFLPRES) == (190000.0, 190000.0))
c2 = Ctx()
quiet(vehParams, c2, tyre_set="file:" + write_tir(T("opt.tir"), MF0, omit=OPTIONAL, nompres=None, inflpres=None))
ok("a file without the optional keys (scales, pp terms, pressures) loads with the defaults",
   c2.mf.LMUX == 1.0 and c2.mf.ppy1 == 0.0 and c2.mf.NOMPRES == c2.mf.INFLPRES == NOM)
ok("result_stem: a file gives _tir_<base name>, other sets keep their stem",
   result_stem("Sturn", "x", "file:/a/b\\syn tyre.v2.tir") == "Sturn_x_tir_syn_tyre_v2"
   and result_stem("Sturn", "x", "CopyB") == "Sturn_x_CopyB" and result_stem("Sturn", "x") == "Sturn_x")

msg = value_error(vehParams, Ctx(), tyre_set="file:" + write_tir(T("miss.tir"), MF0, omit=("pKy1", "rVy6")))
ok("missing coefficients raise and are named", msg is not None and "pKy1" in msg and "rVy6" in msg)
msg = value_error(vehParams, Ctx(), tyre_set="file:" + write_tir(T("nan.tir"), MF0, extra=["[X]", "PKX1 = abc"]))
ok("a coefficient that is not a number raises and is named", msg is not None and "pKx1" in msg)
msg = value_error(vehParams, Ctx(), tyre_set="file:" + write_tir(T("nofz.tir"), MF0, fz0=None))
ok("a missing FNOMIN raises", msg is not None and "FNOMIN" in msg)
msg = value_error(vehParams, Ctx(), tyre_set="file:" + write_tir(T("iso.tir"), {**MF0, "pKy1": -MF0["pKy1"]}))
ok("pKy1 < 0 (ISO signs) raises and names pKy1", msg is not None and "pKy1" in msg)
c3 = Ctx()
quiet(vehParams, c3, tyre_set="file:" + T("iso.tir"), vp_overrides={"pKy1": MF0["pKy1"]})
ok("a vp_overrides pKy1 fixes the same file", c3.mf.pKy1 == MF0["pKy1"])
msg = value_error(vehParams, Ctx(), tyre_set="file:" + write_tir(T("deg.tir"), MF0, extra=["[U]", "ANGLE = 'degrees'"]))
ok("a unit that is not SI raises and names the unit", msg is not None and "degrees" in msg)
ok("nonzero PDX3 and LCY (no model line reads them) give a warning that names both",
   any("PDX3" in m and "LCY" in m for m in warned(
       vehParams, Ctx(), tyre_set="file:" + write_tir(T("extra.tir"), MF0, extra=["[S]", "LCY = 1.05", "PDX3 = 0.2"]))))
ok("LMUV 0 (no friction decay with slip speed) gives no 'not modelled' warning, LMUV 1 gives one that names LMUV",
   not any("not modelled" in m for m in warned(
       vehParams, Ctx(), tyre_set="file:" + write_tir(T("lmuv0.tir"), MF0, extra=["[S]", "LMUV = 0.0"])))
   and any("LMUV" in m for m in warned(
       vehParams, Ctx(), tyre_set="file:" + write_tir(T("lmuv1.tir"), MF0, extra=["[S]", "LMUV = 1.0"]))))
msg = value_error(vehParams, Ctx(), tyre_set="bogus")
ok("an unknown tyre_set raises and names tyre_set", msg is not None and "tyre_set" in msg)
try:
    quiet(vehParams, Ctx(), tyre_set="file:" + T("none.tir"))
    gone = False
except (OSError, ValueError):
    gone = True
ok("a file that does not exist raises", gone)
msg = value_error(vehParams, Ctx(), vp_overrides={"INFLPRES": 2.2})
ok("INFLPRES in bar, NOMPRES in Pa: raises and names INFLPRES", msg is not None and "INFLPRES" in msg)
ok("an INFLPRES override with every pp term 0 warns (no effect)",
   any("INFLPRES" in m for m in warned(vehParams, Ctx(), vp_overrides={"INFLPRES": 240000.0})))
f_pp = write_tir(T("pp.tir"), {**MF0, **PP})
ok("INFLPRES above the file's PRESMAX warns and names PRESMAX",
   any("PRESMAX" in m for m in warned(vehParams, Ctx(), tyre_set="file:" + f_pp, vp_overrides={"INFLPRES": 350000.0})))
msg = value_error(vehParams, Ctx(), vp_overrides={"ppy2": -3.0, "INFLPRES": 330000.0})
ok("a pressure factor <= 0 raises (ppy2 -3 at +50 %)", msg is not None and "factor" in msg)

try:
    git = subprocess.run(["git", "ls-files", "*.tir"], capture_output=True, text=True, cwd=_bootstrap.ROOT)
except OSError:
    git = None
if git is not None and git.returncode == 0:
    ok("no .tir file is tracked by git: licensed tyre data stays out of the repo", git.stdout.split() == [])
else:
    print("  SKIP tracked-file check: not a git checkout")

# =============================================================================
print("3. mf_pressure")
f0 = mf_pressure(c0.mf)
ok("defaults: six plain floats equal to 1.0", len(f0) == 6 and all(type(x) is float and x == 1.0 for x in f0))
c4 = Ctx()
quiet(vehParams, c4, vp_overrides={"INFLPRES": 242000.0})
ok("pp terms 0 and dpi 0.1: still plain 1.0", all(type(x) is float and x == 1.0 for x in mf_pressure(c4.mf)))
c4 = Ctx()
quiet(vehParams, c4, tyre_set="file:" + f_pp, vp_overrides={"INFLPRES": P1})
DPI = (P1 - NOM) / NOM
WANT = (1 + PP["ppx3"] * DPI + PP["ppx4"] * DPI * DPI,        # mu_x
        1 + PP["ppx1"] * DPI + PP["ppx2"] * DPI * DPI,        # Kxk
        1 + PP["ppy3"] * DPI + PP["ppy4"] * DPI * DPI,        # mu_y
        1 + PP["ppy1"] * DPI,                                 # Kya
        1 + PP["ppy2"] * DPI,                                 # Kya load peak
        1 + PP["ppy5"] * DPI)                                 # camber stiffness
ok("+10 %: the factors follow MF 6.1 (mu_x 0.9855, Kxk 0.971, mu_y 0.9884 with these made-up terms)",
   np.allclose(mf_pressure(c4.mf), WANT, rtol=0, atol=1e-15)
   and abs(WANT[0] - 0.9855) < 1e-12 and abs(WANT[1] - 0.971) < 1e-12 and abs(WANT[2] - 0.9884) < 1e-12)

if not os.path.exists(os.path.join(_bootstrap.ROOT, "Data", "DATA_AA.mat")):
    print("SKIP sections 4-7 (test_tir): Data/DATA_AA.mat missing")
    sys.exit(0)

# model imports come after the DATA_AA guard
import casadi as ca  # noqa: E402
import functions.transcription as TR  # noqa: E402
import MLTP as mltp_mod  # noqa: E402
from MLTP_screen import MLTP_screen  # noqa: E402
import setup_sweep as SS  # noqa: E402
from setup_sweep import nlp_signature  # noqa: E402
from userOpts import knob_tag, setup_knobs, userOpts  # noqa: E402
from vehModel import vehModel  # noqa: E402
import validation.py_export as py_export  # noqa: E402

W = ("fl", "fr", "rl", "rr")
ONE = (1.0, 1.0, 1.0, 1.0)
LOADS_B = (1.4, 0.7, 1.3, 0.8)
HY0 = {"pHy1": 0.0, "pHy2": 0.0}                # No lateral shift. The slope at zero slip is then Kya.
OPS = (dict(vx=30.0, vy=4.2, r=0.2, kap=(0.08, -0.06, 0.04, -0.03), load=LOADS_B, steer=0.02),
       dict(vx=45.0, vy=-3.0, r=-0.3, kap=(-0.04, 0.02, 0.06, -0.06), load=(0.8, 1.3, 0.9, 1.2), steer=0.0),
       dict(vx=38.0, vy=-5.5, r=0.1, kap=(0.02, -0.02, 0.05, -0.05), load=ONE, steer=-0.02))


def build(tyre_file=None, ov=None):
    """Return the full 23-state model. Nothing is solved."""
    c = Ctx()
    quiet(userOpts, c, circuit="Sturn", tyre_set=("file:" + tyre_file) if tyre_file else "MF205", vp_overrides=ov)
    quiet(vehModel, c)
    return c


def state(c, vx, vy=0.0, r=0.0, kap=(0.0, 0.0, 0.0, 0.0), load=ONE, steer=0.0):
    """Normalised (x, u, pv) at one operating point. Wheel speeds come from the slip ratios, tyre deflections from
    the load factors."""
    m, vp = c.m23, c.vp
    X = np.zeros(m.nx)
    X[0], X[1], X[2] = vx, vy, r
    X[5:9] = vx / vp.Rw * (1.0 + np.asarray(kap, float))
    X[19:23] = np.asarray(load, float) * np.array([vp.xti_fl, vp.xti_fr, vp.xti_rl, vp.xti_rr])
    U = np.zeros(m.nu)
    U[2:6] = 0.25
    U[-1] = steer
    return X / np.asarray(m.x_s, float), U / np.asarray(m.u_s, float), 0.005


def values(c, names, states):
    """Model outputs (names) at each (x, u, pv) of states."""
    m = c.m23
    F = ca.Function("F", [m.x, m.u, m.pv], [getattr(m, n) for n in names])
    return np.array([[float(v) for v in F(*s)] for s in states])


def slopes(c, kind, h=1e-6):
    """Central-difference slope at zero slip of the four tyre forces, and the four normal loads at LOADS_B.

    kind "x": fx over kappa. kind "y": fy over alpha.
    """
    m = c.m23
    F = ca.Function("F", [m.x, m.u, m.pv], [getattr(m, f"f{kind}_{w}") for w in W])
    Fz = ca.Function("Fz", [m.x, m.u, m.pv], [getattr(m, f"fz_{w}") for w in W])

    def f(s):
        st = (state(c, 30.0, kap=(s,) * 4, load=LOADS_B) if kind == "x"
              else state(c, 30.0, vy=-30.0 * np.tan(s), load=LOADS_B))
        return np.array([float(v) for v in F(*st)])
    return (f(h) - f(-h)) / (2 * h), np.array([float(v) for v in Fz(*state(c, 30.0, load=LOADS_B))])


def kya_ratio(c, fz, ppy1, ppy2):
    """Return the MF 6.1 ratio Kya(p) / Kya(p0) at the normal loads fz. The camber is zero."""
    mf, Fz0 = c.mf, c.vp.Fz0

    def kya(dpi):
        return (1 + ppy1 * dpi) * np.sin(mf.pKy4 * np.arctan(fz / (mf.pKy2 * Fz0 * (1 + ppy2 * dpi))))
    return kya(DPI) / kya(0.0)


def nodes(c):
    """Return the SX node count of the 23-state dynamics."""
    return ca.Function("g", [c.m23.x, c.m23.u, c.m23.pv], [c.m23.dx]).n_nodes()


# =============================================================================
print("4. full model, no solve")
cd = build()
sig0, n0 = nlp_signature(cd), nodes(cd)
ok("the round-trip file gives the same graph hash as the default", nlp_signature(build(f_round)) == sig0)
ok("pp terms nonzero and INFLPRES = NOMPRES: same graph hash", nlp_signature(build(f_pp)) == sig0)
c_dpi = build(None, {"INFLPRES": P1})
ok("pp terms 0 and dpi = 0.1: same graph hash and the same SX node count",
   nlp_signature(c_dpi) == sig0 and nodes(c_dpi) == n0)
c2, c3 = build(f_pp), build(f_pp, {"INFLPRES": P1})
ok("+10 % pressure changes the graph hash", nlp_signature(c3) != sig0)

S2, S3 = [state(c2, **o) for o in OPS], [state(c3, **o) for o in OPS]
for ax, fac in (("x", WANT[0]), ("y", WANT[2])):
    nm = [f"mu_{w}_{ax}" for w in W]
    e = float(np.max(np.abs(values(c3, nm, S3) / values(c2, nm, S2) - fac)))
    ok(f"mu_*_{ax} = {fac:.6f} x the p0 value at 3 operating states (max abs err {e:.1e} <= 1e-12)", e <= 1e-12)

c2h, c3h = build(f_pp, HY0), build(f_pp, {**HY0, "INFLPRES": P1})
e = float(np.max(np.abs(slopes(c3h, "x")[0] / slopes(c2h, "x")[0] - WANT[1])))
ok(f"dfx/dkappa at zero slip = {WANT[1]:.6f} x the p0 value (Kxk, max abs err {e:.1e} <= 1e-6)", e <= 1e-6)
sy2, fz = slopes(c2h, "y")
e = float(np.max(np.abs(slopes(c3h, "y")[0] / sy2 - kya_ratio(c2h, fz, PP["ppy1"], PP["ppy2"]))))
ok(f"dfy/dalpha at zero slip = Kya(p) / Kya(p0) at each load (max abs err {e:.1e} <= 1e-6)", e <= 1e-6)

# Zero peak: the force is the shift. Camber is on, so SVy has its camber part.
SHIFT = {"pDx1": 0.0, "pDx2": 0.0, "pDy1": 0.0, "pDy2": 0.0,
         "pVx1": 0.03, "pVx2": 0.02, "gamma_fl": -2.0, "gamma_rl": -1.5}
a, b = build(f_pp, SHIFT), build(f_pp, {**SHIFT, "INFLPRES": P1})
nm = [f"fx_{w}" for w in W] + [f"fy_{w}" for w in W]
Z = [state(a, 30.0, load=ld) for ld in (ONE, LOADS_B)]
va, vb = values(a, nm, Z), values(b, nm, Z)
ok(f"SVx and SVy at zero slip do not move with pressure (zero-peak probe, min |F| {np.min(np.abs(va)):.0f} N)",
   np.min(np.abs(va)) > 1.0 and float(np.max(np.abs(vb / va - 1.0))) <= 1e-12)

# Camber stiffness. At zero slip, fy follows Kyg0 gamma. Kyg0 carries (1 + ppy5 dpi).
CAM = {**HY0, "pVy1": 0.0, "pVy2": 0.0, "pVy3": 0.0, "pVy4": 0.0, "gamma_fl": -2.0, "gamma_rl": -1.5}
f_cam = write_tir(T("cam.tir"), {**MF0, "pKy6": -0.3, "ppy5": 0.15})
a, b = build(f_cam, CAM), build(f_cam, {**CAM, "INFLPRES": P1})
Z = [state(a, 30.0, load=ONE)]
fy = [f"fy_{w}" for w in W]
e = float(np.max(np.abs(values(b, fy, Z) / values(a, fy, Z) - (1 + 0.15 * DPI))))
ok(f"camber stiffness: fy at zero slip with camber = (1 + ppy5 dpi) x the p0 value (max abs err {e:.1e} <= 1e-3)",
   e <= 1e-3)

# INFLPRES as a casadi symbol. optimise_design and setup_gradients use one.
cs = Ctx()
quiet(userOpts, cs, circuit="Sturn", tyre_set="file:" + f_pp)
p_sym = ca.SX.sym("p")
cs.mf.INFLPRES = p_sym
quiet(vehModel, cs)
ms = cs.m23
x0, u0, k0 = state(c3, **OPS[0])
Fs = ca.Function("Fs", [ms.x, ms.u, ms.pv, p_sym], [ms.dx])
Fn = ca.Function("Fn", [c3.m23.x, c3.m23.u, c3.m23.pv], [c3.m23.dx])
e = float(np.max(np.abs(np.array(Fs(x0, u0, k0, P1)) - np.array(Fn(x0, u0, k0)))))
ok(f"INFLPRES as a casadi symbol gives the dx of the numeric model (max abs diff {e:.1e} <= 1e-12)", e <= 1e-12)
J = ca.Function("J", [ms.x, ms.u, ms.pv, p_sym], [ca.jacobian(ms.dx, p_sym)])
ok("... and dx depends on it", float(np.max(np.abs(np.array(J(x0, u0, k0, P1))))) > 0.0)
cz = Ctx()
quiet(userOpts, cz, circuit="Sturn")
cz.mf.INFLPRES = p_sym
quiet(vehModel, cz)
ok("... but with every pp term 0 it does not enter dx", ca.jacobian(cz.m23.dx, p_sym).nnz() == 0)

# =============================================================================
print("5. QSS screen")


def qss_run(**kw):
    return quiet(MLTP_screen, circuit="Sturn", save=False, verbose=False, **kw)


lap0 = qss_run().data.lap_time
ok("QSS lap with the round-trip file equals the default lap",
   qss_run(tyre_set="file:" + f_round).data.lap_time == lap0)
ok("QSS lap with the pp terms at INFLPRES = NOMPRES equals the default lap",
   qss_run(tyre_set="file:" + f_pp).data.lap_time == lap0)
lap_p = qss_run(tyre_set="file:" + f_pp, vp_overrides={"INFLPRES": P1}).data.lap_time
lap_s = qss_run(vp_overrides={"LMUX": WANT[0], "LMUY": WANT[2]}).data.lap_time
ok(f"QSS lap at +10 % pressure ({lap_p:.5f} s) equals the lap with LMUX {WANT[0]:.5f} and LMUY {WANT[2]:.5f}"
   f" (within 1e-9 s), and is longer than the default ({lap0:.5f} s)",
   abs(lap_p - lap_s) <= TOL_S and lap_p > lap0)
STIFF0 = {"ppx1": 0.0, "ppx2": 0.0, "ppy1": 0.0, "ppy2": 0.0, "ppy5": 0.0}
lap_k = qss_run(tyre_set="file:" + f_pp, vp_overrides={"INFLPRES": P1, **STIFF0}).data.lap_time
ok("the stiffness terms (ppx1, ppx2, ppy1, ppy2, ppy5) do not move the QSS lap", abs(lap_k - lap_p) <= TOL_S)


def tag_of(path):
    """Return the setup tag of a QSS run on the tyre file at path. Return '' when the run has no tag."""
    return getattr(qss_run(tyre_set="file:" + path).data, "setup_tag", "")


# Two same-name files in two folders. The stem keeps only the base name. The setup tag tells them apart.
dir_a, dir_b = T("tyres_a"), T("tyres_b")
os.makedirs(dir_a)
os.makedirs(dir_b)
pa = write_tir(os.path.join(dir_a, "same.tir"), MF0)
pb = write_tir(os.path.join(dir_b, "same.tir"), MF0)
ta, tb = tag_of(pa), tag_of(pb)
ok(f"two same-name files in two folders get different setup tags ({ta!r} and {tb!r}), one path gives one tag",
   bool(ta) and bool(tb) and ta != tb and tag_of(pa) == ta)

# setup_knobs compares an override with the value in the file, and keeps the absolute path of the file.
ka = setup_knobs({"tyre_set": "file:" + f_fz, "vp_overrides": {"Fz0": 3200.0}})
kb = setup_knobs({"tyre_set": "file:" + f_pp, "vp_overrides": {"ppy1": PP["ppy1"]}})
ok("setup_knobs: an override equal to the value in the file (Fz0 3200 = FNOMIN, ppy1 0.4) is no knob, "
   "the file adds only tyre_file", ka == {"tyre_file": os.path.abspath(f_fz)}
   and kb == {"tyre_file": os.path.abspath(f_pp)})
os.chdir(tmp.name)
try:
    tag_rel = knob_tag(setup_knobs({"tyre_set": "file:pp.tir"}))
finally:
    os.chdir(_bootstrap.ROOT)
ok("setup_knobs: a relative and an absolute spelling of one file give one tag", tag_rel == knob_tag(kb))

# A setup sweep hashes the tyre file with the data, so a resume after an edit to the file reuses no old row.
f_sw = write_tir(T("sweep.tir"), MF0)
SWEEP = dict(circuit="Sturn", n_samples=2, tyre_set="file:" + f_sw, plot=False, verbose=False, screen_workers=1)
fp1 = quiet(SS.setup_sweep, [("alpha_RW", 4.0, 16.0)], confirm=False, results_root=T("sweeps"), **SWEEP).fingerprint
with open(f_sw, "a", encoding="ascii") as fh:
    fh.write("$ edited in place\n")
fp2 = quiet(SS.setup_sweep, [("alpha_RW", 4.0, 16.0)], confirm=False, results_root=T("sweeps"), **SWEEP).fingerprint
ok("setup_sweep: an edit to the tyre file in place gives a new fingerprint (QSS-only sweeps, no NLP)", fp2 != fp1)


class Stop(Exception):
    """Stops a sweep before its first NLP solve."""


specs = []


def stub_runner(P, spec, inline, say):
    """Stand-in for setup_sweep._Runner: keep the worker spec, then stop the sweep."""
    specs.append(spec)
    raise Stop


real_runner, SS._Runner = SS._Runner, stub_runner
try:
    quiet(SS.setup_sweep, [("alpha_RW", 4.0, 16.0)], results_root=T("sweeps_nlp"), workers=1, pin_hsl=False, **SWEEP)
except Stop:
    pass
finally:
    SS._Runner = real_runner
SS._worker_init(specs[0])
ok("... a sweep worker gets the tyre file and hashes it as the plan does, a named set adds nothing to the hash",
   SS._W.get("model_hash") == SS._model_hash("Sturn", tyre_set="file:" + f_sw)
   and SS._model_hash("Sturn", tyre_set="CopyB") == SS._model_hash("Sturn"))

# =============================================================================
print("6. parity export")


def export(ov):
    """Run validation.py_export.params with vp_overrides. params() reads its userOpts keywords from PARITY."""
    saved = py_export.PARITY
    py_export.PARITY = {**saved, "vp_overrides": ov}
    try:
        return quiet(py_export.params, "Sturn", 30)["mf"]
    finally:
        py_export.PARITY = saved


mf_d = export(None)
ok("at default the mf block holds the 59 MATLAB coefficients and no pressure key",
   len(mf_d) == 59 and not any(k.startswith("pp") or k in ("NOMPRES", "INFLPRES") for k in mf_d))
mf_p = export({"ppy3": -0.1})
ok("with one pp term nonzero the mf block holds all nine pp terms and both pressures",
   all(k in mf_p for k in list(PP) + ["NOMPRES", "INFLPRES"]) and mf_p["ppy3"] == -0.1)
mf_i = export({"INFLPRES": 240000.0})
ok("with INFLPRES != NOMPRES and every pp term 0 the mf block also holds all nine pp terms and both pressures",
   all(k in mf_i for k in list(PP) + ["NOMPRES", "INFLPRES"]) and mf_i["INFLPRES"] == 240000.0)

# =============================================================================
print("7. MLTP on a tyre file, fake IPOPT (the NLP is built, nothing is solved)")


def fake_solver(ca_, nlp, opts, info=None):
    """Stand-in for transcription._make_solver: x = x0, zero duals, status Solve_Succeeded."""
    if info is not None:
        info["linear_solver"] = "fake"

    def solver(**k):
        return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Solve_Succeeded"}
    return solver


kw = dict(circuit="Sturn", tyre_set="file:" + f_pp, plot=False, results_dir=T("Results"))
real_make, TR._make_solver = TR._make_solver, fake_solver
try:
    c_def = quiet(mltp_mod.MLTP, save=True, **kw)
    c_hop = quiet(mltp_mod.MLTP, save=False, continue_from="auto", vp_overrides={"INFLPRES": P1}, **kw)
finally:
    TR._make_solver = real_make
want = f"Sturn_Static_ATDOn_EM4Off_tir_pp_{knob_tag({'tyre_file': os.path.abspath(f_pp)})}.mat"
ok(f"continue_from='auto' on a tyre file starts from the default result of that file ({want}) with full+duals",
   os.path.basename(c_def.out_path) == want and c_hop.elapsed["warm_start"] == "full+duals")

tmp.cleanup()
print("ALL tir TESTS PASSED")
