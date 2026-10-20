"""MATLAB side of the validation (docs/validation_matlab_vs_python.md): patched scratch copies of the
MATLAB original, run with ``matlab -batch``. Standard library only; the original is read, never edited.

    python validation/matlab_batch.py export --circuit BCN --opt-ds 10 [--w extra_w.mat] --out DIR
    python validation/matlab_batch.py solve --circuit BCN --tier ship --rep 1 --out DIR [--opt-ds 20] [--timeout 2700]

``export`` builds the 23-state NLP of MLTP.m at IPOPT max_iter 0 (the 7-state init does 0 iterations, the
23-state NLP is built and never solved) and writes DIR/matlab_params_<circuit>_ds<ds>.json and
DIR/matlab_nlp_<circuit>_ds<ds>.mat (layout: validation/matlab/vexport.m). The track and DATA_AA sha256 are
added to the JSON here.

``solve`` is one real MLTP.m run as shipped (doc section 7, tag mat_ship) in a fresh copy: IPOPT max_wall_time
1800 s added (runner cap), the 7-state init captured behind MLTP.m:27, MLTP.m cut behind data.t_opt (plots,
SDI and figures are not timed) as MLTP_solve.m, then validation/matlab/vsolve.m writes
DIR/<circuit>_mat_<tier>_r<rep>.mat + .json (layout: the header of vsolve.m). This script adds to the JSON the
external wall of the whole ``matlab -batch`` call, the MATLAB startup estimate, rc, timed_out, git sha and the
edited lines of the copy. ``--max-iter 0`` is the smoke test: everything is built, nothing is solved.
Exit code 0 = result files written (IPOPT status is in the JSON), 1 = run failed or killed at ``--timeout``.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRISTINE = Path(os.environ.get("MATLAB_REPO", r"D:\IRP\FullModel_4EM_Suspension_FullTyre"))
SCRATCH = Path(os.environ.get("VALIDATION_SCRATCH", r"C:\Users\ASUS\.claude\jobs\9b76333d\tmp\validation"))
NL = "\r\n"                                        # the MATLAB sources are CRLF
WALL_CAP = 1800                                    # IPOPT max_wall_time of every solve: runner cap, doc section 10
RUN_INIT = "run('MLTP_initial.m');"                # MLTP.m:27
T_OPT = "data.t_opt = [0 cumsum(full([dt_opt_val{:}]))];"   # MLTP.m:439, the lap
INIT_CAPTURE = (                                   # MLTP.m restarts elapsedTime and solver right behind the init
    "vsolve_init = struct('stats', solver.stats(), 'elapsedTime', elapsedTime, 'lap', data.init.t_opt(end));"
    "   % TMP-COPY ADDITION: the 7-state init, captured before MLTP.m restarts its timer",
    "save('vsolve_init.mat', '-struct', 'vsolve_init');   % TMP-COPY ADDITION",
)


def _sub_once(text, pattern, repl, what):
    """re.sub that insists the anchor occurs exactly once (a silent no-op patch is impossible)."""
    new, n = re.subn(pattern, lambda m: repl(m) if callable(repl) else repl, text, flags=re.M)
    if n != 1:
        raise RuntimeError(f"patch '{what}': anchor matched {n} times, expected exactly 1")
    return new


def _edit(path, *patches):
    """Apply (pattern, repl, what) patches to a file. Bytes round-trip through latin-1, so every
    line that is not patched stays byte-identical (userOpts.m holds non-UTF-8 characters)."""
    text = Path(path).read_bytes().decode("latin-1")
    for pattern, repl, what in patches:
        text = _sub_once(text, pattern, repl, what)
    Path(path).write_bytes(text.encode("latin-1"))


def make_copy(case, circuit, opt_ds=None, max_iter=None):
    """Fresh copy of the MATLAB original in SCRATCH/matlab/<case>, patched for ``circuit`` (and
    ``opt_ds`` / IPOPT ``max_iter`` when given). Writes MLTP_build.m = MLTP.m up to and including the
    nlpsol line (no solve). Returns the case directory."""
    dst = SCRATCH / "matlab" / case
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(PRISTINE, dst, ignore=shutil.ignore_patterns(".git", "Plots", "Barcelona"))

    def after(*added):                # keep the anchor line, add lines behind it
        return lambda m: m.group(0) + "".join(NL + a for a in added)

    patches = [
        (r"^circuit = '[A-Za-z0-9_]+';[^\r\n]*", f"circuit = '{circuit}';   % TMP-COPY EDIT (original: 'BCN')", "circuit"),
        (r"^[ \t]*track = load\('Circuits/Barcelona_circuit_fromassetto\.mat'\);[^\r\n]*",
         after("    case 'NBR'", "        track = load('Circuits/Nurburgring_circuit.mat');   % TMP-COPY ADDITION"), "NBR case"),
    ]
    if opt_ds is not None:
        patches.append((r"^OPT_ds = 10;", f"OPT_ds = {opt_ds:g};   % TMP-COPY EDIT (original: 10)", "OPT_ds"))
    if max_iter is not None:
        patches.append((r"^opts\.ipopt\.compl_inf_tol = 1e-4;[^\r\n]*",
                        after("opts_shipped = opts;   % TMP-COPY ADDITION: options as shipped, kept for the dump",
                              f"opts.ipopt.max_iter = {max_iter};   % TMP-COPY ADDITION: override (shipped 6000)"), "max_iter"))
    _edit(dst / "userOpts.m", *patches)
    _edit(dst / "vehModel.m", (r"^load\('DATA_AA\.mat'\);[^\r\n]*",
                               after("vx_vp_pre = vp;   % TMP-COPY ADDITION: vp before the aero block below (validation dump)"),
                               "vehModel DATA_AA"))

    mltp = (PRISTINE / "MLTP.m").read_bytes().decode("latin-1")
    anchor = "solver = nlpsol('solver', 'ipopt', nlp, opts);"
    if mltp.count(anchor) != 1:
        raise RuntimeError(f"MLTP.m: nlpsol anchor found {mltp.count(anchor)} times, expected exactly 1")
    (dst / "MLTP_build.m").write_bytes((mltp[:mltp.index(anchor) + len(anchor)] + NL).encode("latin-1"))
    return dst


def run(case_dir, command, timeout_s):
    """``matlab -batch`` with ``command`` in ``case_dir``; output to case_dir/run.log. Returns (wall_s, returncode);
    returncode -9 = killed at the timeout (the whole process tree)."""
    case_dir = Path(case_dir)
    matlab = shutil.which("matlab")
    if not matlab:
        raise RuntimeError("matlab is not on PATH")
    t0 = time.perf_counter()
    with open(case_dir / "run.log", "wb") as log:
        p = subprocess.Popen([matlab, "-batch", f"cd('{case_dir.as_posix()}'); {command}"],
                             cwd=case_dir, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            p.wait()
            rc = -9
    return time.perf_counter() - t0, rc


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(circuit, opt_ds, out, w=None, timeout_s=3600):
    """Parameter + NLP export (see module docstring). Returns (params_file, nlp_file, wall_s)."""
    tag = f"{circuit}_ds{opt_ds:g}"
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    case = make_copy(f"export_{tag}", circuit, opt_ds=opt_ds, max_iter=0)
    shutil.copy(HERE / "matlab" / "vexport.m", case)
    params_file, nlp_file = out / f"matlab_params_{tag}.json", out / f"matlab_nlp_{tag}.mat"
    for f in (params_file, nlp_file):
        f.unlink(missing_ok=True)
    cfg = dict(circuit=circuit, params_file=params_file.as_posix(), nlp_file=nlp_file.as_posix(),
               extra_w=Path(w).resolve().as_posix() if w else "", seed=1)
    (case / "vexport_cfg.json").write_text(json.dumps(cfg))
    wall, rc = run(case, "MLTP_build; vexport", timeout_s)
    if rc != 0 or not (params_file.exists() and nlp_file.exists()):
        tail = (case / "run.log").read_bytes().decode("latin-1")[-2500:]
        raise RuntimeError(f"MATLAB export failed (rc {rc}, {wall:.0f} s); tail of {case / 'run.log'}:\n{tail}")
    # hashes of the files MATLAB loaded, computed here
    userOpts = (case / "userOpts.m").read_bytes().decode("latin-1")
    m = re.search(r"case '%s'[^\r\n]*\r?\n\s*track = load\('([^']+)'\)" % re.escape(circuit), userOpts)
    P = json.loads(params_file.read_text())
    P["track_sha256"] = _sha256(case / m.group(1)) if m else None      # None: synthetic track
    P["aero_sha256"] = _sha256(case / "DATA_AA.mat")
    params_file.write_text(json.dumps(P, indent=1))
    return params_file, nlp_file, wall


def _mltp_solve(case):
    """Write case/MLTP_solve.m: MLTP.m with the 7-state init captured behind ``run('MLTP_initial.m')`` and
    everything behind ``data.t_opt`` cut (plots, SDI and figure saving are not timed, doc section 9)."""
    t = (case / "MLTP.m").read_bytes().decode("latin-1")        # make_copy leaves MLTP.m pristine
    for anchor in (RUN_INIT, T_OPT):
        if t.count(anchor) != 1:
            raise RuntimeError(f"MLTP.m: anchor {anchor!r} found {t.count(anchor)} times, expected exactly 1")
    i = t.index(RUN_INIT) + len(RUN_INIT)
    j = t.index(NL, t.index(T_OPT)) + len(NL)                   # end of the t_opt line
    t = (t[:i] + "".join(NL + c for c in INIT_CAPTURE) + t[i:j]
         + "% TMP-COPY: MLTP.m cut here (the original goes on with plots, SDI and figure saving)" + NL)
    (case / "MLTP_solve.m").write_bytes(t.encode("latin-1"))


def _edited_lines(case):
    """Every line this runner edited or added in the copy (all carry the marker TMP-COPY), as 'file:line: text'."""
    return [f"{p.name}:{n}: {' '.join(line.split())}" for p in sorted(case.glob("*.m"))
            for n, line in enumerate(p.read_bytes().decode("latin-1").split("\n"), 1) if "TMP-COPY" in line]


def _git(*args):
    r = subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def solve(circuit, tier, rep, out, opt_ds=None, max_iter=None, timeout_s=2700):
    """One MLTP.m solve as shipped (module docstring). Writes out/<circuit>_mat_<tier>_r<rep>.mat + .json and
    returns (record, ok, json_file); ok = the result files exist. A failed or killed run still gets its .json.
    Skipped: seeding MATLAB from another code's w* (xseed), add when a lap gap > 0.01% appears (doc section 8)."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{circuit}_mat_{tier}_r{rep}"
    mat_file, json_file = out / f"{stem}.mat", out / f"{stem}.json"
    for f in (mat_file, json_file):
        f.unlink(missing_ok=True)
    case = make_copy(f"solve_{stem}", circuit, opt_ds=opt_ds, max_iter=max_iter)
    _edit(case / "userOpts.m", (r"^%%-Constraints on the rate of inputs[^\r\n]*",       # first line behind the shipped options
          lambda m: f"opts.ipopt.max_wall_time = {WALL_CAP};   % TMP-COPY ADDITION: runner cap, not in the original" + NL + m.group(0),
          "max_wall_time"))
    _mltp_solve(case)
    shutil.copy(HERE / "matlab" / "vsolve.m", case)
    (case / "vsolve_cfg.json").write_text(json.dumps(dict(circuit=circuit, mat_file=mat_file.as_posix(),
                                                          json_file=json_file.as_posix())))
    started, t_call = datetime.now(timezone.utc), time.time()
    # vs_up is created by the first statement MATLAB runs: its mtime minus t_call = process start to user code
    wall, rc = run(case, "fclose(fopen('vs_up','w')); MLTP_solve; vsolve", timeout_s)
    up = case / "vs_up"
    ok = rc == 0 and mat_file.exists() and json_file.exists()
    rec = json.loads(json_file.read_text(encoding="utf-8")) if json_file.exists() else {}
    rec.update(circuit=circuit, code="mat", tier=tier, rep=rep, ok=ok, started_utc=started.isoformat(timespec="seconds"),
               wall_external_s=wall, startup_s=up.stat().st_mtime - t_call if up.exists() else None,
               rc=rc, timed_out=rc == -9, git_sha=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain")),
               args=dict(opt_ds=opt_ds, max_iter=max_iter, timeout_s=timeout_s), runner_caps=dict(max_wall_time=WALL_CAP),
               patches=_edited_lines(case), case_dir=case.as_posix())
    json_file.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if not ok:
        print((case / "run.log").read_bytes().decode("latin-1")[-2500:], file=sys.stderr)
    return rec, ok, json_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    ex = sub.add_parser("export", help="build the NLP at max_iter 0 and dump parameters + NLP numbers")
    ex.add_argument("--circuit", required=True)
    ex.add_argument("--opt-ds", type=float, required=True)
    ex.add_argument("--w", help=".mat with extra evaluation points (numeric variables with n_w rows)")
    ex.add_argument("--out", required=True)
    ex.add_argument("--timeout", type=int, default=3600, help="seconds")
    sv = sub.add_parser("solve", help="one MLTP.m solve as shipped: <circuit>_mat_<tier>_r<rep>.mat + .json")
    sv.add_argument("--circuit", required=True, help="BCN, NBR, Sturn, ...")
    sv.add_argument("--tier", required=True, help="part of the file names, 'ship' for the shipped settings")
    sv.add_argument("--rep", type=int, required=True)
    sv.add_argument("--opt-ds", type=float, help="OPT_ds override (shipped 10; the failure-policy retry uses 20)")
    sv.add_argument("--max-iter", type=int, help="smoke test only: IPOPT max_iter override (0 = build, no solve)")
    sv.add_argument("--out", required=True)
    sv.add_argument("--timeout", type=int, default=2700, help="seconds, kills the whole matlab -batch call")
    a = ap.parse_args(argv)
    if a.mode == "solve":
        rec, ok, json_file = solve(a.circuit, a.tier, a.rep, a.out, a.opt_ds, a.max_iter, a.timeout)
        st, up = rec.get("stats", {}), rec["startup_s"]
        print(f"{'ok' if ok else 'FAILED'}: rc {rec['rc']}{' (killed at the timeout)' if rec['timed_out'] else ''}, "
              f"MATLAB wall {rec['wall_external_s']:.0f} s (startup {'n/a' if up is None else f'{up:.1f} s'}), "
              f"lap {rec.get('lap')}, {st.get('iter_count')} iterations, {st.get('return_status')}\n{json_file}")
        return 0 if ok else 1
    params_file, nlp_file, wall = export(a.circuit, a.opt_ds, a.out, a.w, a.timeout)
    print(f"MATLAB wall {wall:.0f} s\n{params_file}\n{nlp_file}")


if __name__ == "__main__":
    sys.exit(main())
