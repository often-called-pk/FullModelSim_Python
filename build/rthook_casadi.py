"""PyInstaller runtime hook: make casadi's bundled DLLs findable.

collect_all("casadi") places the SWIG extension `_casadi.pyd` at the bundle
root (sys._MEIPASS) because casadi imports it as a top-level module, but
casadi's runtime DLLs (libcasadi.dll plus the dynamically-loaded linsol /
integrator / conic plugins) land under sys._MEIPASS/casadi/. The Windows loader
does not search that subfolder when loading a .pyd from the root, so casadi
fails to import with "DLL load failed while importing _casadi". Registering the
casadi/ directory on the DLL search path before casadi is imported resolves
both the direct libcasadi.dll dependency and the plugins casadi LoadLibrary's at
runtime. No-op in a dev (non-frozen) run.
"""
import os
import sys

_meipass = getattr(sys, "_MEIPASS", None)
if _meipass:
    _casadi_dir = os.path.join(_meipass, "casadi")
    if os.path.isdir(_casadi_dir):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_casadi_dir)
            except OSError:
                pass
        os.environ["PATH"] = _casadi_dir + os.pathsep + os.environ.get("PATH", "")
