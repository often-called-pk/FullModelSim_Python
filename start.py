"""One command setup: uv, packages, checks, a model check, quick tests, the GUI.

Usage: python start.py [--yes] [--no-gui] [--no-tests]
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
UV_DOCS = "https://docs.astral.sh/uv/getting-started/installation/"
UV_SH = "https://astral.sh/uv/install.sh"
UV_PS = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
APT = ("libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 "
       "libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libegl1 libgl1 libfontconfig1 "
       "libdbus-1-3")
DNF = ("xcb-util-cursor libxkbcommon-x11 xcb-util-wm xcb-util-image xcb-util-keysyms "
       "xcb-util-renderutil mesa-libEGL mesa-libGL fontconfig dbus-libs")
VERIFY = (
    "from functions.context import Ctx; from Powertrain import Powertrain; "
    "from vehParams import vehParams; from userOpts import userOpts; from vehModel import vehModel; "
    "ctx = Ctx(); Powertrain(ctx); vehParams(ctx); userOpts(ctx, circuit='Sturn'); vehModel(ctx); "
    "assert ctx.m23.nx == 23, ctx.m23.nx; print('model build: nx', ctx.m23.nx, 'nu', ctx.m23.nu); "
    "from MLTP_screen import MLTP_screen; "
    "r = MLTP_screen(circuit='Sturn', save=False, verbose=False); "
    "print('QSS lap on Sturn: %.3f s' % r.data.lap_time)"
)
HSL_PROBE = (
    "import warnings; warnings.simplefilter('ignore'); import functions.hsl as h; "
    "d = h.resolve_hsl_dir(); p = h.hsllib_path(d) if d else None; "
    "ok = bool(p) and h.probe_linear_solver('ma57', p); "
    "print('Coin-HSL: found, ' + p if ok else "
    "'Coin-HSL: found, does not load: solves use MUMPS' if p else "
    "'Coin-HSL: not found, solves use MUMPS (slower, no licence)')"
)
GUI_HSL = (
    "import os\n"
    "try:\n"
    "    from app.paths import settings\n"
    "    v = settings().value('hsl_dir') or ''\n"
    "    print('GUI Coin-HSL field: ' + (v or 'not set') + ('' if not v or os.path.isdir(v) else ' (not a folder)'))\n"
    "except ImportError:\n"
    "    print('GUI Coin-HSL field: not checked (PySide6 does not import)')\n"
)


def fail(message, hint):
    """Print an error with a fix hint and exit with code 1."""
    print(f"error: {message}")
    print(f"fix: {hint}")
    sys.exit(1)


def find_uv():
    """Return the uv path, or None. Looks on PATH and in the installer folders."""
    found = shutil.which("uv")
    if found:
        return found
    env, home = os.environ.get, Path.home()
    dirs = [env("UV_INSTALL_DIR"), env("UV_INSTALL_DIR") and Path(env("UV_INSTALL_DIR")) / "bin",
            env("XDG_BIN_HOME"), env("XDG_DATA_HOME") and Path(env("XDG_DATA_HOME")).parent / "bin",
            home / ".local" / "bin", home / ".cargo" / "bin"]
    for d in dirs:
        p = Path(d) / ("uv.exe" if WIN else "uv") if d else None
        if p and p.exists():
            return str(p)
    return None


def install_uv():
    """Run the official uv installer. Exit with a hint when it fails."""
    by_hand = "check the network, or install uv by hand: " + UV_DOCS
    if WIN:
        code = subprocess.run(UV_PS, shell=True, check=False).returncode
    else:
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "install.sh")
            try:
                urllib.request.urlretrieve(UV_SH, script)
            except OSError as exc:
                print(exc)
                fail("could not download the uv installer", by_hand)
            code = subprocess.run(["sh", script], check=False).returncode
    if code != 0:
        fail("the uv installer failed", by_hand)


def ask(question):
    """Return the answer in lower case. A closed input counts as 'n'."""
    try:
        return input(question).strip().lower()
    except (EOFError, RuntimeError):  # RuntimeError: stdin is closed
        print("\nno input is available. Use --yes to answer yes.")
        return "n"


def xcb_missing(uv):
    """Return the libraries that the Qt xcb plugin needs and the loader cannot find.

    Linux only. Return None when the check cannot run.
    """
    if platform.system() != "Linux" or not shutil.which("ldd"):
        return None
    here = subprocess.run([uv, "run", "--no-sync", "python", "-c",
                           "import PySide6, os; print(os.path.dirname(PySide6.__file__))"],
                          capture_output=True, text=True, check=False).stdout.strip()
    plugin = Path(here) / "Qt" / "plugins" / "platforms" / "libqxcb.so"
    if not here or not plugin.exists():
        return None
    out = subprocess.run(["ldd", str(plugin)], capture_output=True, text=True, check=False).stdout
    return [line.split()[0] for line in out.splitlines() if "not found" in line]


def run_py(uv, *cmd):
    """Run a command in the project environment, without a new sync."""
    return subprocess.run([uv, "run", "--no-sync", *cmd], check=False).returncode


def main():
    """Run the eight steps in order."""
    ap = argparse.ArgumentParser(description="Set up the project and open the GUI.")
    ap.add_argument("--yes", action="store_true", help="answer yes to every question")
    ap.add_argument("--no-gui", action="store_true", help="set up and check only")
    ap.add_argument("--no-tests", action="store_true", help="skip the quick tests")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with child output
    os.chdir(ROOT)

    print("[1/8] uv")
    uv = find_uv()
    new_uv = uv is None
    if uv is None:
        print("uv is missing. The official installer is:\n  " + (UV_PS if WIN else UV_SH))
        if not args.yes and ask("Install it now? [Y/n] ") not in ("", "y", "yes"):
            fail("uv is required", f"install uv ({UV_DOCS}) and run this script again")
        install_uv()
        uv = find_uv()
        if uv is None:
            fail("uv installed but not found", "open a new terminal and run this script again")
    print("ok:", subprocess.run([uv, "--version"], capture_output=True, text=True, check=False).stdout.strip())

    print("[2/8] Python and packages (uv sync)")
    if subprocess.run([uv, "sync", "--locked"], check=False).returncode != 0:
        fail("uv sync failed", "check the network. If uv.lock is out of date, run: uv lock")
    print("ok")

    print("[3/8] Qt system libraries")
    miss = xcb_missing(uv)
    if miss is None:
        print("not checked (the check needs Linux, ldd and the Qt xcb plugin)")
    elif not miss:
        print("ok")
    else:
        print("missing:", ", ".join(miss))
        print("  Debian/Ubuntu: sudo apt install " + APT)
        print("  Fedora: sudo dnf install " + DNF)
        print("warning: the GUI may not start. The solver still works.")

    print("[4/8] Data/DATA_AA.mat")
    if (ROOT / "Data" / "DATA_AA.mat").exists():
        print("ok")
    else:
        fail("Data/DATA_AA.mat is missing", "restore it with: git checkout -- Data/DATA_AA.mat")

    print("[5/8] Optional parts (they never stop the setup)")
    run_py(uv, "python", "-c", HSL_PROBE)
    run_py(uv, "python", "-c", GUI_HSL)
    var = "EXT_SIM_DIR" if os.environ.get("EXT_SIM_DIR") else "CHASSISSIM_DIR"
    cs = os.environ.get(var)
    if not cs:
        print("EXT_SIM_DIR: not set (external circuits and cars are unavailable)")
    else:
        print(f"{var}: {cs}" + ("" if os.path.isdir(cs) else " (not a folder)"))
    cc = [c for c in ("gcc", "clang", "cl") if shutil.which(c)]
    print(f"C compiler: {cc[0]}" if cc else "C compiler: none (jit=True builds without JIT)")

    print("[6/8] Verify the model")
    if run_py(uv, "python", "-c", VERIFY) != 0:
        fail("the model check failed", "run: uv sync --reinstall, then run this script again")
    print("ok")

    print("[7/8] Quick tests")
    tests_ok = True
    if args.no_tests:
        print("skipped (--no-tests)")
    else:
        tests_ok = run_py(uv, "python", "tests/run_tests.py", "--quick") == 0
        print("ok" if tests_ok else "tests failed, see the list above. The GUI may still run.")

    print("[8/8] GUI")
    gui_code = 0
    if args.no_gui:
        print("skipped (--no-gui)")
    else:
        print("starting the GUI, close its window to finish")
        gui_code = run_py(uv, "fullmodelsim")
        if gui_code != 0:
            print(f"the GUI stopped with code {gui_code}. Without a display, use --no-gui. "
                  "If Qt libraries are missing, install the packages of step 3.")
    print("Later, start the GUI with: uv run fullmodelsim")
    if new_uv:
        print("uv is new on this machine. Open a new terminal before you use the uv command.")
    sys.exit(0 if tests_ok and gui_code == 0 else 1)


if __name__ == "__main__":
    main()
