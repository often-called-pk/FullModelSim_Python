# PyInstaller spec for the FullModelSim windows-app (onedir).
# Build from repo root:  venv\Scripts\pyinstaller.exe build\windows-app.spec
#   set COINHSL_DIR to bundle Coin-HSL; set FMS_CONSOLE=1 for a debug build with
#   a console window (default is windowed: no terminal pops up on launch).
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.dirname(os.path.abspath(SPECPATH))

# Windowed by default for a clean end-user launch; FMS_CONSOLE=1 forces a console
# so a startup crash's traceback is visible while debugging.
CONSOLE = os.environ.get("FMS_CONSOLE", "0") == "1"

# Optional drop-in icon: place build/app.ico and it is picked up automatically.
_icon = os.path.join(ROOT, "build", "app.ico")
ICON = _icon if os.path.isfile(_icon) else None

VERSION_FILE = os.path.join(ROOT, "build", "version_info.txt")

# Bundle read-only resources next to the frozen root (sys._MEIPASS).
datas = [
    (os.path.join(ROOT, "Circuits"), "Circuits"),
    (os.path.join(ROOT, "Data"), "Data"),
    (os.path.join(ROOT, "app", "presets"), "app/presets"),
]
binaries = []
hiddenimports = collect_submodules("scipy") + ["headless_solve"]

# CasADi ships its own compiled libs (IPOPT/MUMPS) — collect everything.
for pkg in ("casadi", "PySide6", "plotly"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# Best-effort: bundle Coin-HSL DLLs if COINHSL_DIR is set at build time.
hsl_dir = os.environ.get("COINHSL_DIR")
if hsl_dir and os.path.isdir(hsl_dir):
    for fn in os.listdir(hsl_dir):
        if fn.lower().endswith(".dll"):
            binaries.append((os.path.join(hsl_dir, fn), "."))

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
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="FullModelSim",
    console=CONSOLE,        # windowed by default; the IPOPT log streams into the GUI log pane
    icon=ICON,
    version=VERSION_FILE,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="FullModelSim")
