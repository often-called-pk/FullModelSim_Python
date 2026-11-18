"""ChassisSim circuit source (Phase 1, Task 1): functions/chassissim.py, the userOpts grip and track hooks,
the result stems and the plot folder. Synthetic install folder in a tempdir, a fake IPOPT (no solve).
The last section reads a real ChassisSim install only when CHASSISSIM_DIR is set, else it prints SKIP.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_chassissim_circuit.py
    CHASSISSIM_DIR="<ChassisSim install folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_chassissim_circuit.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import io
import os
import tempfile
import warnings

import numpy as np

import functions.transcription as T
from app.results import plot_dir
from functions import chassissim as cs
from functions.context import Ctx
from functions.importfile import circuit_slug, result_stem
from MLTP import MLTP
from MLTP_screen import MLTP_screen
from userOpts import setup_knobs, userOpts

N_OK = 0


def ok(name, cond):
    global N_OK
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name
    N_OK += 1


def quiet(fn, *a, **kw):
    """Call fn with stdout and warnings swallowed, so the output stays short."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def fake(ca, nlp, opts, info=None):
    """Stand-in for functions.transcription._make_solver: returns the start point, no iterations."""
    def solver(**k):
        return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
    if info is not None:
        info["linear_solver"] = "fake"
    return solver


T._make_solver = fake

# ---- synthetic data: 301 samples over 1200 m, k in [0.0035, 0.0079] (gentle, one left-positive lap) ----
S = np.linspace(0.0, 1200.0, 301)
K = (2.0 * np.pi / 1200.0) * (1.0 + 0.5 * np.sin(2.0 * np.pi * S / 1200.0))


def write_curv(path, s, k):
    """'k s' rows, CRLF, tabs and spaces mixed, one blank line at the end."""
    lines = []
    for i, (kk, ss) in enumerate(zip(k, s)):
        sep = "\t" if i % 2 else "   "
        lines.append(f"{float(kk)!r}{sep}{float(ss)!r}")
    with open(path, "w", encoding="latin-1", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n\r\n")


def write_text(path, text):
    with open(path, "w", encoding="latin-1", newline="") as f:
        f.write(text)


INI = (
    "[Total Circ prop]\r\nTotal circuits = 4\r\n\r\n"
    "[Circuit0]\r\nCircuit name = Alpha Park\r\nCircuit file name = tr_alpha park.txt\r\n"
    "Base Grip factor = 1.1\r\nCircuit bp name = bump_alpha.dat\r\nCircuit bp flag = 1\r\n\r\n"
    "[Circuit1]\r\nCircuit name = Beta Ring\r\nCircuit file name = tr_beta.txt\r\n"
    "Base Grip factor = 1.0\r\nCircuit bp name = bump_beta.dat\r\nCircuit bp flag = 0\r\n"
    "Circuit gsf name = gsf_beta.dat\r\nCircuit gsf flag = 0\r\n"
    "Circuit alt rcamb name = alt_beta.txt\r\nCircuit alt rcamb flag = 1\r\n\r\n"
    "[Circuit2]\r\nCircuit name = Default Bump\r\nCircuit file name = tr_defbump.txt\r\n"
    "Base Grip factor = 1.0\r\nCircuit bp name = default.dat\r\nCircuit bp flag = 1\r\n\r\n"
    "[Circuit3]\r\nCircuit name = Ghost Ring\r\nCircuit file name = tr_ghost.txt\r\n"
    "Base Grip factor = 1.0\r\nCircuit bp name = bump_ghost.dat\r\nCircuit bp flag = 0\r\n"
)


def build_install(root):
    """Fake install: Circ_prop.ini + Circuits/ with four listed circuits and two orphans."""
    circ = os.path.join(root, "Circuits")
    os.makedirs(circ)
    write_text(os.path.join(root, "Circ_prop.ini"), INI)
    write_curv(os.path.join(circ, "tr_alpha park.txt"), S, K)
    write_curv(os.path.join(circ, "tr_beta.txt"), S, K)
    write_curv(os.path.join(circ, "tr_defbump.txt"), S, K)
    write_curv(os.path.join(circ, "tr_Orphan.txt"), S, K)
    # bump: FL FR RL RR in mm, then distance in m (5 columns)
    write_text(os.path.join(circ, "bump_alpha.dat"),
               "1\t2\t3\t4\t0\r\n5 6 7 8 600\r\n9\t10\t11\t12\t1200\r\n")
    # alt_rcamb: base altitude on line 1 (trailing tabs), then s, pitch, camber
    alt = "12.5\t\t\r\n0.0\t0.001\t0.002\r\n600.0\t-0.003\t0.004\r\n1200.0\t0.0\t0.0\r\n"
    write_text(os.path.join(circ, "alt_beta.txt"), alt)
    write_text(os.path.join(circ, "alt_rcamb_Orphan.txt"), alt)
    return root


def set_env(value):
    if value is None:
        os.environ.pop("CHASSISSIM_DIR", None)
    else:
        os.environ["CHASSISSIM_DIR"] = value


with tempfile.TemporaryDirectory() as tmp:
    root = build_install(os.path.join(tmp, "ChassisSim"))
    circ = os.path.join(root, "Circuits")
    other = os.path.join(tmp, "Other")
    os.makedirs(other)
    print("== readers")
    s_a, k_a = cs.read_curvature(os.path.join(circ, "tr_alpha park.txt"))
    ok("read_curvature: len 301 from CRLF mixed tab/space file", len(s_a) == 301 and len(k_a) == 301)
    ok("read_curvature: s values exact", np.array_equal(s_a, S))
    ok("read_curvature: k values exact (file sign kept)", np.array_equal(k_a, K))
    ok("read_curvature: k positive as written", bool(np.all(k_a > 0)))

    bad = os.path.join(tmp, "bad_s.txt")
    write_curv(bad, [0.0, 4.0, 4.0, 8.0], [0.01, 0.01, 0.01, 0.01])
    try:
        cs.read_curvature(bad)
        raised = False
    except ValueError:
        raised = True
    ok("read_curvature: repeated s raises ValueError", raised)

    three = os.path.join(tmp, "three_cols.txt")
    write_text(three, "0.01 0.0 5\r\n0.01 4.0 5\r\n0.01 8.0 5\r\n")
    try:
        cs.read_curvature(three)
        raised = False
    except ValueError:
        raised = True
    ok("read_curvature: 3 columns raise ValueError", raised)

    blocks = cs.read_circ_prop(os.path.join(root, "Circ_prop.ini"))
    names = [b["Circuit name"] for b in blocks]
    ok("read_circ_prop: four circuit blocks, total section skipped", names == ["Alpha Park", "Beta Ring", "Default Bump", "Ghost Ring"])
    ok("read_circ_prop: 5-key block has 5 keys", len(blocks[0]) == 5)
    ok("read_circ_prop: 9-key block has 9 keys", len(blocks[1]) == 9)
    ok("read_circ_prop: alt keys kept in the 9-key block", blocks[1]["Circuit alt rcamb flag"] == "1")
    ok("read_circ_prop: grip is read as text", blocks[0]["Base Grip factor"] == "1.1")

    base, a_s, a_pitch, a_camber = cs.read_alt_camber(os.path.join(circ, "alt_beta.txt"))
    ok("read_alt_camber: base altitude 12.5 from line 1 with trailing tabs", base == 12.5)
    ok("read_alt_camber: s column", np.array_equal(a_s, [0.0, 600.0, 1200.0]))
    ok("read_alt_camber: pitch is column 2", np.array_equal(a_pitch, [0.001, -0.003, 0.0]))
    ok("read_alt_camber: camber is column 3", np.array_equal(a_camber, [0.002, 0.004, 0.0]))

    bad_alt = os.path.join(tmp, "bad_alt.txt")
    write_text(bad_alt, "10.0\r\n0.0\t0.1\r\n600.0\t0.1\r\n1200.0\t0.1\t0.2\r\n")
    try:
        cs.read_alt_camber(bad_alt)
        raised = False
    except ValueError:
        raised = True
    ok("read_alt_camber: a 2-column row raises ValueError", raised)

    bp_s, bp_h = cs.read_bump(os.path.join(circ, "bump_alpha.dat"))
    ok("read_bump: distance is column 5", np.array_equal(bp_s, [0.0, 600.0, 1200.0]))
    ok("read_bump: heights in m, FL is column 1", np.allclose(bp_h[:, 0], [0.001, 0.005, 0.009], rtol=0, atol=1e-12))
    ok("read_bump: RR is column 4", np.allclose(bp_h[:, 3], [0.004, 0.008, 0.012], rtol=0, atol=1e-12))
    ok("read_bump: shape (n, 4)", bp_h.shape == (3, 4))

    print("== load_circuit")
    alpha = quiet(cs.load_circuit, "Alpha Park", root)
    ok("load_circuit: s equals the file s", np.array_equal(alpha.s, S))
    ok("load_circuit: k is the file k with the sign flipped", np.array_equal(alpha.k, -K))
    ok("load_circuit: first k is negative (left-positive model sign)", alpha.k[0] < 0)
    ok("load_circuit: grip from the ini is 1.1", alpha.grip == 1.1)
    ok("load_circuit: bump path with flag 1", alpha.bump is not None and alpha.bump.endswith("bump_alpha.dat"))
    ok("load_circuit: no alt keys gives alt None", alpha.alt is None)
    ok("load_circuit: name lookup ignores case", np.array_equal(quiet(cs.load_circuit, "alpha PARK", root).s, S))
    beta = quiet(cs.load_circuit, "Beta Ring", root)
    ok("load_circuit: grip 1.0 is a float 1.0", type(beta.grip) is float and beta.grip == 1.0)
    ok("load_circuit: bump flag 0 gives bump None", beta.bump is None)
    ok("load_circuit: alt path with flag 1", beta.alt is not None and beta.alt.endswith("alt_beta.txt"))
    dflt = quiet(cs.load_circuit, "Default Bump", root)
    ok("load_circuit: default.dat gives bump None even with flag 1", dflt.bump is None)
    orphan = quiet(cs.load_circuit, "Orphan", root)
    ok("load_circuit: orphan tr_Orphan.txt found, grip 1.0", type(orphan.grip) is float and orphan.grip == 1.0)
    ok("load_circuit: orphan alt_rcamb_Orphan.txt found", orphan.alt is not None and orphan.alt.endswith("alt_rcamb_Orphan.txt"))
    ok("load_circuit: orphan without bp file gives bump None", orphan.bump is None)
    try:
        quiet(cs.load_circuit, "Nowhere", root)
        msg = ""
    except ValueError as e:
        msg = str(e)
    ok("load_circuit: unknown name raises ValueError naming a known circuit", "Alpha Park" in msg)
    try:
        quiet(cs.load_circuit, "Ghost Ring", root)
        raised = False
    except FileNotFoundError:
        raised = True
    ok("load_circuit: listed but absent curvature file raises FileNotFoundError", raised)

    print("== install_dir (env restored afterwards)")
    saved_env = os.environ.get("CHASSISSIM_DIR")
    try:
        set_env(None)
        ok("install_dir: argument folder returned", cs.install_dir(root) == root)
        try:
            cs.install_dir(None)
            raised = False
        except ValueError:
            raised = True
        ok("install_dir: no argument and no env raises ValueError", raised)
        set_env(other)
        ok("install_dir: env used when no argument", cs.install_dir(None) == other)
        ok("install_dir: argument beats env", cs.install_dir(root) == root)
        try:
            cs.install_dir(os.path.join(tmp, "missing"))
            raised = False
        except ValueError:
            raised = True
        ok("install_dir: missing folder raises ValueError", raised)
    finally:
        set_env(saved_env)
    ok("install_dir: env restored", os.environ.get("CHASSISSIM_DIR") == saved_env)

    print("== userOpts grip")
    ctx = Ctx()
    quiet(userOpts, ctx, circuit="chassissim:Alpha Park", chassissim_dir=root)
    ok("userOpts: grip 1.1 gives LMUX 1.1", ctx.mf.LMUX == 1.1)
    ok("userOpts: grip 1.1 gives LMUY 1.1", ctx.mf.LMUY == 1.1)
    ctx = Ctx()
    quiet(userOpts, ctx, circuit="chassissim:Beta Ring", chassissim_dir=root)
    ok("userOpts: grip 1.0 leaves LMUX a float 1.0", type(ctx.mf.LMUX) is float and ctx.mf.LMUX == 1.0)
    ok("userOpts: grip 1.0 leaves LMUY a float 1.0", type(ctx.mf.LMUY) is float and ctx.mf.LMUY == 1.0)

    print("== stems and plot folder")
    stem = result_stem("chassissim:Eastern Creek, AU", "Static_ATDOn_EM4Off")
    ok("result_stem: chassissim name slugged", stem == "cs_Eastern_Creek_AU_Static_ATDOn_EM4Off")
    ok("result_stem: Sturn unchanged", result_stem("Sturn", "Static_ATDOn_EM4Off") == "Sturn_Static_ATDOn_EM4Off")
    ok("circuit_slug: space becomes underscore", circuit_slug("chassissim:Alpha Park") == "cs_Alpha_Park")
    ok("circuit_slug: plain circuit unchanged", circuit_slug("BCN") == "BCN")
    folder = plot_dir("OUT", "chassissim:Eastern Creek, AU", "Static", "On", "Off")
    ok("plot_dir: folder is the slug, then the cfg part",
       folder == os.path.join("OUT", "Plots", "cs_Eastern_Creek_AU", "Static_ATDOn_EM4Off"))
    ok("setup_knobs: chassissim_dir is not a setup knob", setup_knobs({"chassissim_dir": "x"}) == {})

    print("== MLTP build with fake IPOPT")
    res = quiet(MLTP, circuit="chassissim:Alpha Park", chassissim_dir=root, save=False, plot=False,
                linear_solver="mumps", warm_start=None)
    ok("MLTP: builds and returns a ctx", res is not None and hasattr(res, "track"))
    ok("MLTP: built track s equals the file s", np.array_equal(res.track.s, S))
    ok("MLTP: built track k equals the negated file k", np.array_equal(res.track.k, -K))
    ok("MLTP: built track k is not the file sign", not np.array_equal(res.track.k, K))
    ok("MLTP: grip 1.1 reaches ctx.mf.LMUX", res.mf.LMUX == 1.1)
    ok("MLTP: grip 1.1 reaches ctx.mf.LMUY", res.mf.LMUY == 1.1)

# ---- vendor section: real ChassisSim install, only when CHASSISSIM_DIR is set ----
REAL = os.environ.get("CHASSISSIM_DIR")
if not REAL or not os.path.isdir(REAL):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    print("== vendor install")
    vcirc = os.path.join(REAL, "Circuits")
    loaded = 0
    for b in cs.read_circ_prop(cs._find(REAL, "Circ_prop.ini")):
        if cs._find(vcirc, b.get("Circuit file name", "")) is None:
            continue
        tr = cs.load_circuit(b["Circuit name"], REAL)
        ok(f"vendor block '{b['Circuit name']}' loads with increasing s", bool(np.all(np.diff(tr.s) > 0)))
        loaded += 1
    ok("vendor: at least one block has a curvature file", loaded >= 1)
    ec = cs.load_circuit("Eastern Creek", REAL)
    ok("Eastern Creek: s[-1] == 3824.181", abs(ec.s[-1] - 3824.181) < 1e-9)
    ok("Eastern Creek: 971 points", len(ec.s) == 971)
    ok("Eastern Creek: net turning > 0 (left) after the flip", float(np.sum(ec.k[:-1] * np.diff(ec.s))) > 0)
    ok("Eastern Creek: bump path ends bump_profile_ec.dat", ec.bump is not None and ec.bump.endswith("bump_profile_ec.dat"))
    ok("Eastern Creek: alt None", ec.alt is None)
    bh = cs.load_circuit("Brands Hatch", REAL)
    ok("Brands Hatch (orphan): alt file found", bh.alt is not None)
    scr = quiet(MLTP_screen, circuit="chassissim:Eastern Creek", chassissim_dir=REAL, save=False, verbose=False)
    lap = float(scr.data.lap_time)
    ok("MLTP_screen Eastern Creek: lap finite and positive", np.isfinite(lap) and lap > 0)

print(f"test_chassissim_circuit: {N_OK} checks PASS")
