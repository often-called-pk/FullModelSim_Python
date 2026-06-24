# Building the FullModelSim Windows app

## Build (onedir)

```powershell
venv\Scripts\Activate.ps1
# optional: bundle Coin-HSL DLLs so MA57/MA97 work in the frozen exe
$env:COINHSL_DIR = "C:\path\to\coinhsl\bin"
venv\Scripts\pyinstaller.exe build\windows-app.spec
```

Output: `dist\FullModelSim\FullModelSim.exe` (onedir; resources under
`dist\FullModelSim\_internal\`).

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

A literal clean-VM run (no Python/casadi/HSL installed) remains the gold-standard
final check — the dev-machine frozen run proves the bundle is self-contained but
cannot prove the total absence of host-environment leakage.
