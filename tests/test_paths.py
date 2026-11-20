"""app/paths: user_presets_dir resolution, the settings store (FULLMODELSIM_SETTINGS or the org / app
store) and the default output folder. The org / app store is only read here, never written."""
import os, sys, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import _bootstrap  # repo root -> sys.path[0] and cwd (see tests/_bootstrap.py)
from PySide6.QtCore import QSettings
from app.paths import user_presets_dir, user_cars_dir, default_output_dir, settings, ORG_NAME, APP_NAME

def ok(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

def same_file(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

ok("presets under output root",
   user_presets_dir() == os.path.join(default_output_dir(), "presets"))
ok("presets basename is 'presets'", os.path.basename(user_presets_dir()) == "presets")

# ---- settings store ---------------------------------------------------------
# _bootstrap sets FULLMODELSIM_SETTINGS to a temp ini of this process. Restore whatever it was.
_env_saved = os.environ.get("FULLMODELSIM_SETTINGS")
try:
    with tempfile.TemporaryDirectory(prefix="paths_settings_") as tmp:
        ini = os.path.join(tmp, "prefs.ini")
        os.environ["FULLMODELSIM_SETTINGS"] = ini
        s = settings()
        ok("env set -> settings() is an ini store at that path",
           isinstance(s, QSettings) and s.format() == QSettings.IniFormat and same_file(s.fileName(), ini))
        s.setValue("hsl_dir", "/opt/coinhsl/bin")
        s.sync()
        del s
        ok("the value is written to that ini file",
           os.path.isfile(ini) and QSettings(ini, QSettings.IniFormat).value("hsl_dir") == "/opt/coinhsl/bin")

        os.environ.pop("FULLMODELSIM_SETTINGS", None)
        native = settings()
        ok("env unset -> settings() is the org / app store (same file as QSettings(org, app))",
           native.fileName() == QSettings(ORG_NAME, APP_NAME).fileName()
           and not same_file(native.fileName(), ini))
        del native
finally:
    if _env_saved is None:
        os.environ.pop("FULLMODELSIM_SETTINGS", None)
    else:
        os.environ["FULLMODELSIM_SETTINGS"] = _env_saved

# ---- default output folder --------------------------------------------------
ok("default output folder ends with FullModelSim",
   os.path.basename(os.path.normpath(default_output_dir())) == "FullModelSim")

# _bootstrap sets FULLMODELSIM_DOCUMENTS to a temp folder of this process.
ok("env FULLMODELSIM_DOCUMENTS is the Documents folder (output, presets and cars follow it)",
   same_file(default_output_dir(), os.path.join(os.environ["FULLMODELSIM_DOCUMENTS"], "FullModelSim"))
   and user_cars_dir().startswith(default_output_dir()))

print("\nALL paths TESTS PASSED")
