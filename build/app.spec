# PyInstaller spec for FullModelSim (onedir), Windows and Linux.
# Build from repo root:  uv run pyinstaller build/app.spec
#   set COINHSL_DIR to bundle Coin-HSL (.dll on Windows, .so on Linux) into hsl/; set FMS_CONSOLE=1
#   for a debug build with a console window (Windows only; the default is windowed).
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.dirname(os.path.abspath(SPECPATH))

# Windowed by default for a clean end-user launch; FMS_CONSOLE=1 forces a console
# so a startup crash's traceback is visible while debugging.
CONSOLE = os.environ.get("FMS_CONSOLE", "0") == "1"

# Optional drop-in icon: place build/app.ico and it is picked up automatically.
_icon = os.path.join(ROOT, "build", "app.ico")
ICON = _icon if os.path.isfile(_icon) else None

WIN = sys.platform == "win32"
# Version resource and icon exist on Windows only.
VERSION_FILE = os.path.join(ROOT, "build", "version_info.txt") if WIN else None
if not WIN:
    ICON = None

# Read-only resources, next to the frozen root (sys._MEIPASS). Only the files that the app reads.
# Keep CIRCUIT_FILES equal to the values of userOpts._REAL_CIRCUITS.
CIRCUIT_FILES = ["Barcelona_circuit.mat", "Barcelona_circuit_s1.mat", "Barcelona_circuit_s2.mat",
                 "Barcelona_circuit_s3.mat", "Jarama_circuit.mat", "Spa_circuit.mat",
                 "Nurburgring_circuit.mat"]
datas = [(os.path.join(ROOT, "Circuits", fn), "Circuits") for fn in CIRCUIT_FILES] + [
    (os.path.join(ROOT, "Data", "DATA_AA.mat"), "Data"),
    (os.path.join(ROOT, "app", "presets"), "app/presets"),
]
binaries = []
hiddenimports = collect_submodules("scipy", filter=lambda name: ".tests" not in name) + ["headless_solve"]

# CasADi ships its own compiled libraries (IPOPT, MUMPS): collect all of them. The C headers
# (casadi/include) serve the JIT only. The frozen app runs without JIT, so they stay out.
# The PyInstaller hooks of PySide6 collect QtCore, QtGui, QtWidgets and the platform plugins.
for pkg in ("casadi", "plotly"):
    d, b, h = collect_all(pkg)
    datas += [x for x in d if "/include" not in x[1].replace(os.sep, "/")]
    binaries += b; hiddenimports += h

# Optional: with COINHSL_DIR set at build time, the Coin-HSL libraries go into the `hsl`
# subfolder (functions/hsl.py looks there in a frozen app). The CI never sets it.
HSL_EXT = ".dll" if WIN else ".so"
hsl_dir = os.environ.get("COINHSL_DIR")
if hsl_dir and os.path.isdir(hsl_dir):
    for fn in os.listdir(hsl_dir):
        if fn.lower().endswith(HSL_EXT) or (not WIN and ".so." in fn.lower()):
            binaries.append((os.path.join(hsl_dir, fn), "hsl"))

a = Analysis(
    [os.path.join(ROOT, "app", "main.py")],   # absolute: relative paths resolve against the spec dir (build/), not ROOT
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[os.path.join(ROOT, "build", "rthook_casadi.py")],
    excludes=[],
)
# Linux: leave out the Qt GTK3 platform theme (platformthemes/libqgtk3.so) and each library that only
# it needs (GTK, GDK, pango, cairo, atk). Qt then uses its own style. A library that another binary
# needs stays (glib and gio stay: QtCore needs them).
if sys.platform.startswith("linux"):
    from PyInstaller.depend import bindepend

    def _needs(paths):
        """Real paths of all libraries that the files in `paths` need, direct or not."""
        seen, todo = set(), list(paths)
        while todo:
            for _name, path in bindepend.get_imports(todo.pop()):
                real = path and os.path.realpath(path)
                if real and real not in seen:
                    seen.add(real)
                    todo.append(real)
        return seen

    _src = {os.path.realpath(src): dest for dest, src, _kind in a.binaries}
    _gtk = {s for s, dest in _src.items() if os.path.basename(dest) == "libqgtk3.so"}
    _drop = _needs(_gtk)
    _drop = _gtk | (_drop - _needs(s for s in _src if s not in _drop and s not in _gtk))
    a.binaries = [b for b in a.binaries if os.path.realpath(b[1]) not in _drop]

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="FullModelSim",
    console=CONSOLE and WIN,        # windowed by default; the IPOPT log streams into the GUI log pane
    icon=ICON,
    version=VERSION_FILE,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="FullModelSim")
