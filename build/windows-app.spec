# PyInstaller spec for the FullModelSim windows-app.
# Build from repo root:  venv\Scripts\pyinstaller.exe build\windows-app.spec
# Onedir first (easier DLL debugging); flip EXE(console=...) / onefile later.
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None
ROOT = os.path.abspath(os.getcwd())

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
    ["app/main.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="FullModelSim", console=True,        # console=True so the IPOPT log is visible while debugging
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="FullModelSim")
