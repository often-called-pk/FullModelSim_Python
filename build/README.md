# Building the FullModelSim Windows app

## Build (onedir)

```powershell
venv\Scripts\Activate.ps1
# optional: bundle Coin-HSL DLLs so MA57/MA97 work in the frozen exe
$env:COINHSL_DIR = "C:\path\to\coinhsl\bin"
# optional: $env:FMS_CONSOLE = "1"   # debug build WITH a console window
venv\Scripts\pyinstaller.exe build\windows-app.spec
```

Output: `dist\FullModelSim\FullModelSim.exe` (onedir; resources under
`dist\FullModelSim\_internal\`).

### Build options

- **Windowed by default (`console=False`).** No terminal window pops up on
  launch; the IPOPT log streams into the GUI log pane instead (verified to reach
  the QProcess pipe even though the exe is GUI-subsystem). Set `FMS_CONSOLE=1` at
  build time for a debug build with a console so a startup-crash traceback is
  visible.
- **Version metadata** (`build/version_info.txt`) is embedded — visible under the
  exe's right-click → Properties → Details (ProductName, FileVersion 1.0.0.0, …).
- **Icon** is a drop-in: place `build/app.ico` and it is picked up automatically
  (none shipped yet → default PyInstaller icon).
- Because the windowed exe shows a **modal dialog** on an *unhandled* exception,
  `headless_solve.main` catches solve failures, prints the traceback to stdout
  (so it lands in the GUI log), and exits non-zero — no dialog from the solve
  subprocess.

## Packaging notes (why the spec is the way it is)

- **Absolute entry-point.** `Analysis([os.path.join(ROOT, "app", "main.py")])`.
  PyInstaller resolves a *relative* script path against the spec's own directory
  (`build\`), so a bare `"app/main.py"` fails with "script not found".
- **casadi DLL runtime hook (`build/rthook_casadi.py`).** `collect_all("casadi")`
  drops the SWIG extension `_casadi.pyd` at the bundle root but casadi's runtime
  DLLs (`libcasadi.dll` + the linsol/integrator/conic plugins) under
  `_internal\casadi\`. The Windows loader does not search that subfolder for a
  `.pyd` loaded from the root, so casadi fails to import. The runtime hook adds
  `_internal\casadi\` to the DLL search path before casadi is imported.
- **HSL in a frozen app (implemented in `functions/hsl.py`).** When `COINHSL_DIR`
  is set at build time the spec bundles the Coin-HSL DLLs at the bundle root, and
  `resolve_hsl_dir()` includes `sys._MEIPASS` (the frozen root) in its search
  order: `COINHSL_DIR env > explicit arg > frozen _MEIPASS > seeded default`. If
  HSL still fails to load, the solve falls back to MUMPS automatically — the app
  always solves.

## Acceptance floor (manual, on a clean Windows VM with no Python)

1. Copy `dist\FullModelSim\` to the VM and launch `FullModelSim.exe`.
2. Main tab: circuit `Sturn`, Aero `Static`, ATD on, linear solver `mumps`
   (the MUMPS floor needs no Coin-HSL bundle); use `ma57` to exercise HSL.
3. Run. The IPOPT iteration log streams into the log pane.
4. On finish: summary shows a lap time; the Output tab lists plots; double-click
   opens one in the browser. Results/Plots are written under
   `%USERPROFILE%\Documents\FullModelSim`.

### Verified (frozen run on the dev machine, HSL bundle)

`FullModelSim.exe --headless cfg.json` (the subprocess the GUI Run button
spawns) with `circuit=Sturn, linear_solver=ma57`:

- casadi imports from the bundle (runtime hook); IPOPT 3.14.11 runs **`ma57`**
  loaded from the bundled Coin-HSL (no MUMPS fallback) for both the warm-start
  and the full 23-state solve;
- `EXIT: Optimal Solution Found.`, lap time 25.574 s;
- results + 9 Plotly plots written under `%USERPROFILE%\Documents\FullModelSim`.

### Self-containment (sanitized-environment run)

Re-running the frozen `--headless` solve with `PATH` stripped to the standard
Windows system dirs and `COINHSL_DIR` unset still imports casadi and runs
**`ma57` from the bundled Coin-HSL** (no MUMPS fallback, no DLL errors) at both
the warm-start and full-solve banners — i.e. neither casadi nor HSL leaks a DLL
from the dev `PATH`; the bundle is self-contained.

A literal clean-VM run (no Python/casadi/HSL installed) remains the gold-standard
final check — the sanitized run strongly approximates it but a real bare machine
is the only way to prove the total absence of host-environment leakage.
