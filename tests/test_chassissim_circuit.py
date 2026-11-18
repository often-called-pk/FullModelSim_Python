"""ChassisSim circuit source (Phase 1, Task 1): functions/chassissim.py, the userOpts grip and track hooks,
the result stems, the plot folder and the init file name. Synthetic install folder in a tempdir, a fake
IPOPT (no solve). The last section reads a real ChassisSim install only when CHASSISSIM_DIR is set, else
it prints SKIP.

    /home/pk/.venvs/fullmodelsim/bin/python tests/test_chassissim_circuit.py
    CHASSISSIM_DIR="<ChassisSim install folder>" /home/pk/.venvs/fullmodelsim/bin/python tests/test_chassissim_circuit.py
"""
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
import contextlib
import hashlib
import io
import os
import shutil
import tempfile
import warnings

import casadi as ca
import numpy as np

import functions.transcription as T
from app.results import plot_dir
from functions import chassissim as cs
from functions.context import Ctx
from functions.importfile import circuit_slug, result_stem
from MLTP import MLTP
from MLTP_initial import MLTP_initial
from MLTP_screen import MLTP_screen
from setup_sweep import _model_hash
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


def raises(exc, fn, *a, **kw):
    """True when fn(*a, **kw) raises exc (output swallowed)."""
    try:
        quiet(fn, *a, **kw)
    except exc:
        return True
    return False


GRAPHS = []  # sha256 of each NLP function (w -> f, g) handed to the fake solver, in build order


def fake(ca_, nlp, opts, info=None):
    """Stand-in for functions.transcription._make_solver: returns the start point, no iterations."""
    def solver(**k):
        fg = ca.Function("fg", [nlp["x"]], [nlp["f"], nlp["g"]])
        GRAPHS.append(hashlib.sha256(fg.serialize().encode()).hexdigest())
        return {"x": k["x0"], "f": 0.0, "lam_g": np.zeros(nlp["g"].numel()), "lam_x": np.zeros(nlp["x"].numel())}
    solver.stats = lambda: {"iter_count": 0, "return_status": "Fake"}
    if info is not None:
        info["linear_solver"] = "fake"
    return solver


T._make_solver = fake

# ---- synthetic data: 301 samples over 1200 m, gentle; file k > 0 is a right turn (ChassisSim sign) ----
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
    "[Circuit1]\r\nCircuit name = Beta Ring\r\nCircuit file name = TR_BETA.TXT\r\n"
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
    # bump: FL FR RL RR in mm, then distance in m (5 columns). Flagged-off and default.dat files exist too.
    bump = "1\t2\t3\t4\t0\r\n5 6 7 8 600\r\n9\t10\t11\t12\t1200\r\n"
    write_text(os.path.join(circ, "bump_alpha.dat"), bump)
    write_text(os.path.join(circ, "bump_beta.dat"), bump)
    write_text(os.path.join(circ, "default.dat"), bump)
    # alt_rcamb: base altitude on line 1 (trailing tabs), then s, pitch, camber
    alt = "12.5\t\t\r\n0.0\t0.001\t0.002\r\n600.0\t-0.003\t0.004\r\n1200.0\t0.0\t0.0\r\n"
    write_text(os.path.join(circ, "alt_beta.txt"), alt)
    write_text(os.path.join(circ, "alt_rcamb_Orphan.txt"), alt)
    return root


def blk(i, name, f, g):
    return (f"[Circuit{i}]\r\nCircuit name = {name}\r\nCircuit file name = {f}\r\nBase Grip factor = {g}\r\n"
            "Circuit bp name = default.dat\r\nCircuit bp flag = 0\r\n\r\n")


def build_grip(root):
    """Install with three circuits on one curvature: grip 1.2 (G12), 1.1 (G11), 1.0 (G10)."""
    circ = os.path.join(root, "Circuits")
    os.makedirs(circ)
    write_text(os.path.join(root, "Circ_prop.ini"),
               "[Total Circ prop]\r\nTotal circuits = 3\r\n\r\n"
               + blk(0, "G12", "tr_g12.txt", "1.2") + blk(1, "G11", "tr_g11.txt", "1.1") + blk(2, "G10", "tr_g10.txt", "1.0"))
    for n in ("g12", "g11", "g10"):
        write_curv(os.path.join(circ, f"tr_{n}.txt"), S, K)
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

    bad = os.path.join(tmp, "bad_s.txt")
    write_curv(bad, [0.0, 4.0, 4.0, 8.0], [0.01, 0.01, 0.01, 0.01])
    ok("read_curvature: repeated s raises ValueError", raises(ValueError, cs.read_curvature, bad))

    two = os.path.join(tmp, "two_rows.txt")
    write_curv(two, [0.0, 4.0], [0.01, 0.01])
    ok("read_curvature: a 2-row file raises ValueError", raises(ValueError, cs.read_curvature, two))

    nan_s = os.path.join(tmp, "nan_s.txt")
    write_text(nan_s, "0.01 0.0\r\n0.01 nan\r\n0.01 8.0\r\n0.01 12.0\r\n")
    ok("read_curvature: a NaN distance raises ValueError", raises(ValueError, cs.read_curvature, nan_s))

    three = os.path.join(tmp, "three_cols.txt")
    write_text(three, "0.01 0.0 5\r\n0.01 4.0 5\r\n0.01 8.0 5\r\n")
    ok("read_curvature: 3 columns raise ValueError", raises(ValueError, cs.read_curvature, three))

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
    ok("read_alt_camber: a 2-column row raises ValueError", raises(ValueError, cs.read_alt_camber, bad_alt))

    alt3 = os.path.join(tmp, "alt_line1_three.txt")
    write_text(alt3, "10.0 11.0 12.0\r\n0.0\t0.1\t0.2\r\n600.0\t0.1\t0.2\r\n1200.0\t0.1\t0.2\r\n")
    ok("read_alt_camber: line 1 with 3 numbers raises ValueError", raises(ValueError, cs.read_alt_camber, alt3))

    bp_s, bp_h = cs.read_bump(os.path.join(circ, "bump_alpha.dat"))
    ok("read_bump: distance is column 5", np.array_equal(bp_s, [0.0, 600.0, 1200.0]))
    h_expect = np.array([[0.001, 0.002, 0.003, 0.004], [0.005, 0.006, 0.007, 0.008], [0.009, 0.010, 0.011, 0.012]])
    ok("read_bump: whole height array in m, columns FL FR RL RR in order",
       bp_h.shape == (3, 4) and np.allclose(bp_h, h_expect, rtol=0, atol=1e-12))

    print("== load_circuit")
    alpha = quiet(cs.load_circuit, "Alpha Park", root)
    ok("load_circuit: s equals the file s", np.array_equal(alpha.s, S))
    ok("load_circuit: k is the file k with the sign flipped", np.array_equal(alpha.k, -K))
    ok("load_circuit: grip from the ini is 1.1", alpha.grip == 1.1)
    ok("load_circuit: bump path with flag 1", alpha.bump is not None and alpha.bump.endswith("bump_alpha.dat"))
    ok("load_circuit: no alt keys gives alt None", alpha.alt is None)
    ok("load_circuit: name lookup ignores case", np.array_equal(quiet(cs.load_circuit, "alpha PARK", root).s, S))
    beta = quiet(cs.load_circuit, "Beta Ring", root)
    ok("load_circuit: ini file name TR_BETA.TXT finds tr_beta.txt", np.array_equal(beta.s, S))
    ok("load_circuit: grip 1.0 is a float 1.0", type(beta.grip) is float and beta.grip == 1.0)
    ok("load_circuit: bump flag 0 gives bump None though bump_beta.dat exists", beta.bump is None)
    ok("load_circuit: alt path with flag 1", beta.alt is not None and beta.alt.endswith("alt_beta.txt"))
    dflt = quiet(cs.load_circuit, "Default Bump", root)
    ok("load_circuit: default.dat gives bump None though the file exists with flag 1", dflt.bump is None)
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
    listed_names = msg.split("are: ")[1].rstrip(".").split(", ") if "are: " in msg else []
    ok("load_circuit: known-circuit list has no duplicates", len(listed_names) == 5 and len(set(listed_names)) == len(listed_names))
    ok("load_circuit: listed base name 'beta' is not an orphan", raises(ValueError, cs.load_circuit, "beta", root))
    ok("load_circuit: listed base name 'defbump' is not an orphan", raises(ValueError, cs.load_circuit, "defbump", root))
    ok("load_circuit: listed but absent curvature file raises FileNotFoundError",
       raises(FileNotFoundError, cs.load_circuit, "Ghost Ring", root))
    ok("load_circuit: a cs_dir without Circuits/ raises ValueError", raises(ValueError, cs.load_circuit, "Alpha Park", other))

    rel = os.path.relpath(root, os.getcwd())
    alpha_rel = quiet(cs.load_circuit, "Alpha Park", rel)
    ok("load_circuit: relative cs_dir gives an absolute bump path",
       alpha_rel.bump is not None and os.path.isabs(alpha_rel.bump) and alpha_rel.bump.endswith("bump_alpha.dat"))

    print("== install_dir (env restored afterwards)")
    saved_env = os.environ.get("CHASSISSIM_DIR")
    try:
        set_env(None)
        ok("install_dir: argument folder returned", cs.install_dir(root) == root)
        ok("install_dir: no argument and no env raises ValueError", raises(ValueError, cs.install_dir, None))
        set_env(other)
        ok("install_dir: env used when no argument", cs.install_dir(None) == other)
        ok("install_dir: argument beats env", cs.install_dir(root) == root)
        ok("install_dir: missing folder raises ValueError", raises(ValueError, cs.install_dir, os.path.join(tmp, "missing")))
    finally:
        set_env(saved_env)

    print("== userOpts grip")
    ctx = Ctx()
    quiet(userOpts, ctx, circuit="chassissim:Alpha Park", chassissim_dir=root)
    ok("userOpts: grip 1.1 sets ctx.grip 1.1", ctx.grip == 1.1)
    ok("userOpts: grip 1.1 leaves ctx.mf.LMUX a float 1.0", type(ctx.mf.LMUX) is float and ctx.mf.LMUX == 1.0)
    ok("userOpts: grip 1.1 leaves ctx.mf.LMUY a float 1.0", type(ctx.mf.LMUY) is float and ctx.mf.LMUY == 1.0)
    ctx = Ctx()
    quiet(userOpts, ctx, circuit="chassissim:Beta Ring", chassissim_dir=root)
    ok("userOpts: grip 1.0 sets ctx.grip 1.0", ctx.grip == 1.0)
    ok("userOpts: grip 1.0 leaves ctx.mf.LMUX a float 1.0", type(ctx.mf.LMUX) is float and ctx.mf.LMUX == 1.0)
    ok("userOpts: grip 1.0 leaves ctx.mf.LMUY a float 1.0", type(ctx.mf.LMUY) is float and ctx.mf.LMUY == 1.0)
    ctx = Ctx()
    quiet(userOpts, ctx, circuit="chassissim:Alpha Park", chassissim_dir=root, vp_overrides={"LMUY": 0.9})
    ok("userOpts: LMUY override 0.9 stays 0.9 in ctx.mf", ctx.mf.LMUY == 0.9)
    ok("userOpts: grip 1.1 stays in ctx.grip with the LMUY override", ctx.grip == 1.1)
    alpha_c = quiet(cs.load_circuit, "Alpha Park", root)
    ok("load_circuit: curvature is the absolute curvature file path",
       os.path.isabs(alpha_c.curvature) and alpha_c.curvature.endswith("tr_alpha park.txt"))

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
    ok("MLTP: grip 1.1 is kept in ctx.grip", res.grip == 1.1)
    ok("MLTP: ctx.mf.LMUX stays a float 1.0 with grip 1.1", type(res.mf.LMUX) is float and res.mf.LMUX == 1.0)

    plots = os.path.join(tmp, "Plots")
    quiet(MLTP, circuit="chassissim:Alpha Park", chassissim_dir=root, save=False, plot=True,
          plots_dir=plots, linear_solver="mumps", warm_start=None)
    pdir = plot_dir(tmp, "chassissim:Alpha Park", "Static", "On", "Off")
    ok("plotSDI: writes racing_line.html into the plot_dir folder", os.path.isfile(os.path.join(pdir, "racing_line.html")))

    print("== MLTP_initial file names")
    res_dir = os.path.join(tmp, "Results")
    quiet(MLTP_initial, circuit="chassissim:Beta Ring", chassissim_dir=root, save=True,
          results_dir=res_dir, linear_solver="mumps")
    quiet(MLTP_initial, circuit="chassissim:Beta Ring", chassissim_dir=root, save=True,
          results_dir=res_dir, linear_solver="mumps", tyre_set="CopyB")
    ok("MLTP_initial: writes init_cs_Beta_Ring.mat and init_cs_Beta_Ring_CopyB.mat only",
       sorted(os.listdir(res_dir)) == ["init_cs_Beta_Ring.mat", "init_cs_Beta_Ring_CopyB.mat"])

    print("== grip: equivalence, product, model hash, QA error")
    gdir = build_grip(os.path.join(tmp, "Grip"))
    g12, g11, g10 = "chassissim:G12", "chassissim:G11", "chassissim:G10"

    def qss_lap(circuit, **kw):
        return float(quiet(MLTP_screen, circuit=circuit, chassissim_dir=gdir, save=False, verbose=False, **kw).data.lap_time)

    def nlp_hash(circuit, **kw):
        GRAPHS.clear()
        quiet(MLTP, circuit=circuit, chassissim_dir=gdir, save=False, plot=False, warm_start=None,
              linear_solver="mumps", **kw)
        return GRAPHS[-1]  # the 23-state NLP is built last

    ctx = Ctx()
    quiet(userOpts, ctx, circuit=g12, chassissim_dir=gdir)
    ok("grip 1.2 circuit: ctx.grip 1.2, ctx.mf.LMUX and LMUY stay float 1.0",
       ctx.grip == 1.2 and type(ctx.mf.LMUX) is float and ctx.mf.LMUX == 1.0 and ctx.mf.LMUY == 1.0)
    la = qss_lap(g12)
    lb = qss_lap(g10, vp_overrides={"LMUX": 1.2, "LMUY": 1.2})
    ok("grip 1.2 == grip 1.0 with LMUX = LMUY = 1.2: QSS lap bit for bit", la == lb)
    ok("grip 1.2 changes the QSS lap against grip 1.0 alone", la != qss_lap(g10))
    ha = nlp_hash(g12)
    hb = nlp_hash(g10, vp_overrides={"LMUX": 1.2, "LMUY": 1.2})
    ok("grip 1.2 == grip 1.0 with LMUX = LMUY = 1.2: 23-state graph hash equal", ha == hb)
    ok("grip 1.2 changes the 23-state graph against grip 1.0 alone", ha != nlp_hash(g10))

    lp = qss_lap(g11, vp_overrides={"LMUY": 0.9})
    lq = qss_lap(g10, vp_overrides={"LMUX": 1.1, "LMUY": 0.9 * 1.1})
    ctx = Ctx()
    quiet(userOpts, ctx, circuit=g11, chassissim_dir=gdir, vp_overrides={"LMUY": 0.9})
    ok("grip 1.1 with LMUY 0.9: ctx.mf.LMUY == 0.9", ctx.mf.LMUY == 0.9)
    ok("grip 1.1 with LMUY 0.9 == grip 1.0 with LMUX 1.1, LMUY 0.9*1.1: QSS lap bit for bit", lp == lq)

    curv = os.path.join(gdir, "Circuits", "tr_g12.txt")
    ini_p = os.path.join(gdir, "Circ_prop.ini")
    with open(curv, "rb") as f:
        curv_bytes = f.read()
    with open(ini_p, "rb") as f:
        ini_bytes = f.read()
    h0 = _model_hash(g12, chassissim_dir=gdir)
    ok("model hash: equal when nothing changes", h0 == _model_hash(g12, chassissim_dir=gdir))
    write_curv(curv, S, K * 1.01)
    ok("model hash: changes with the curvature file bytes", _model_hash(g12, chassissim_dir=gdir) != h0)
    with open(curv, "wb") as f:
        f.write(curv_bytes)
    ok("model hash: back to h0 after restoring the curvature file", _model_hash(g12, chassissim_dir=gdir) == h0)
    with open(ini_p, "wb") as f:
        f.write(ini_bytes + b"; changed\r\n")
    ok("model hash: changes with Circ_prop.ini", _model_hash(g12, chassissim_dir=gdir) != h0)
    with open(ini_p, "wb") as f:
        f.write(ini_bytes)
    ok("model hash: back to h0 after restoring Circ_prop.ini", _model_hash(g12, chassissim_dir=gdir) == h0)
    with tempfile.TemporaryDirectory() as moved:
        gcopy = shutil.copytree(gdir, os.path.join(moved, "elsewhere"))
        ok("model hash: a copy of the install in another folder gives the same hash",
           _model_hash(g12, chassissim_dir=gcopy) == h0)

    nan_root = os.path.join(tmp, "NanInstall")
    os.makedirs(os.path.join(nan_root, "Circuits"))
    write_text(os.path.join(nan_root, "Circ_prop.ini"),
               "[Total Circ prop]\r\nTotal circuits = 1\r\n\r\n" + blk(0, "Nan Track", "tr_nan.txt", "1.0"))
    write_text(os.path.join(nan_root, "Circuits", "tr_nan.txt"),
               "nan\t0.0\r\n" + "".join(f"{float(k)!r}\t{float(s)!r}\r\n" for k, s in zip(K[1:], S[1:])))
    try:
        quiet(userOpts, Ctx(), circuit="chassissim:Nan Track", chassissim_dir=nan_root)
        msg = ""
    except ValueError as e:
        msg = str(e)
    nan_file = os.path.join(os.path.abspath(nan_root), "Circuits", "tr_nan.txt")
    ok("QA error: nan curvature raises ValueError naming the curvature file", nan_file in msg)
    ok("QA error: message ends with 'Check the curvature file.'", "Check the curvature file." in msg)
    ok("QA error: message has no k_from_xy hint", "k_from_xy" not in msg)

# ---- vendor section: real ChassisSim install, only when CHASSISSIM_DIR is set ----
REAL = os.environ.get("CHASSISSIM_DIR")
if not REAL or not os.path.isdir(REAL):
    print("SKIP: CHASSISSIM_DIR not set")
else:
    print("== vendor install")
    vcirc = os.path.join(REAL, "Circuits")
    loaded = 0
    for b in cs.read_circ_prop(os.path.join(REAL, "Circ_prop.ini")):
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
